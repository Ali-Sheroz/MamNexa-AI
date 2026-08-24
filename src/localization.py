"""OpenCV lesion localization for MamNexa AI (Phase II).

Turns the U-Net's binary suspicion mask (see ``segmentation``) into human-facing
localization: it traces the boundary of each connected suspicious region, drops
noise-sized specks, numbers the regions deterministically, and draws numbered
outlines over the (PHI-free) grayscale image for review.

Clinical guardrails
-------------------
Every region is labeled an **"AI-Identified Suspicious Area #N"** — never a
"tumor", "lesion confirmed", or "cancer". The numbering is a reading aid for a
professional, not a diagnosis. The output image is decision-support only.

Determinism
-----------
Regions are ordered by descending area, then top-to-bottom, then left-to-right,
so the same mask always yields the same numbering (important for reproducible
reports and for referring to "Region 2" consistently across the UI and PDF).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import (
    CONTOUR_COLOR_RGB,
    LABEL_COLOR_RGB,
    LESION_LABEL_PREFIX,
    MIN_LESION_AREA_PX,
)


# ---------------------------------------------------------------------------
# Mask normalization
# ---------------------------------------------------------------------------
def _to_binary_uint8(mask: np.ndarray) -> np.ndarray:
    """Coerce any mask (bool / {0,1} / {0,255} / (H,W,1)) to uint8 {0,255} (H,W)."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}")
    return ((m > 0).astype(np.uint8)) * 255


# ---------------------------------------------------------------------------
# Contour extraction + numbering
# ---------------------------------------------------------------------------
def extract_lesion_contours(
    mask: np.ndarray, min_area: int = MIN_LESION_AREA_PX
) -> list[np.ndarray]:
    """Find external contours of suspicious regions, drop specks, order them.

    ``min_area`` (pixels) filters out isolated noise; the remaining contours are
    returned in the deterministic reading order described in the module docstring.
    """
    binary = _to_binary_uint8(mask)
    # RETR_EXTERNAL: outer boundaries only (we don't split a region by its holes).
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c) >= float(min_area)]

    def _order_key(c: np.ndarray) -> tuple[float, int, int]:
        x, y, _, _ = cv2.boundingRect(c)
        return (-cv2.contourArea(c), y, x)  # largest first, then top-to-bottom, left-to-right

    kept.sort(key=_order_key)
    return kept


def summarize_lesions(contours: list[np.ndarray]) -> list[dict[str, Any]]:
    """Build a per-region metadata table with guardrail-compliant labels."""
    table: list[dict[str, Any]] = []
    for region_id, c in enumerate(contours, start=1):
        area = float(cv2.contourArea(c))
        x, y, w, h = cv2.boundingRect(c)
        moments = cv2.moments(c)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:  # degenerate (line-like) contour: fall back to bbox center
            cx, cy = x + w / 2.0, y + h / 2.0
        table.append(
            {
                "region_id": region_id,
                "label": f"{LESION_LABEL_PREFIX} #{region_id}",
                "area_px": round(area, 1),
                "centroid": (round(cx, 1), round(cy, 1)),
                "bbox": (int(x), int(y), int(w), int(h)),
            }
        )
    return table


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Return a writable (H, W, 3) uint8 copy of a grayscale or RGB image."""
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img.copy()


def draw_numbered_contours(
    base_image: np.ndarray,
    contours: list[np.ndarray],
    contour_color: tuple[int, int, int] = CONTOUR_COLOR_RGB,
    label_color: tuple[int, int, int] = LABEL_COLOR_RGB,
    thickness: int = 2,
) -> np.ndarray:
    """Draw each region's boundary and its number onto an RGB copy of the image.

    ``base_image`` is treated as RGB (grayscale is broadcast to 3 channels); the
    color tuples are RGB. Region numbers get a dark disc behind them for
    legibility on bright dense tissue. The input is not modified.
    """
    canvas = _ensure_rgb_uint8(base_image)
    if contours:
        cv2.drawContours(canvas, contours, -1, contour_color, thickness, lineType=cv2.LINE_AA)
    for region_id, c in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = int(x + w / 2.0), int(y + h / 2.0)
        cv2.circle(canvas, (cx, cy), 11, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas, str(region_id), (cx - 6, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA,
        )
    return canvas


def localize_lesions(
    mask: np.ndarray, base_image: np.ndarray, min_area: int = MIN_LESION_AREA_PX
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """One-shot: mask + image -> (numbered overlay RGB uint8, lesion metadata table)."""
    contours = extract_lesion_contours(mask, min_area=min_area)
    overlay = draw_numbered_contours(base_image, contours)
    table = summarize_lesions(contours)
    return overlay, table


def save_overlay_png(rgb_image: np.ndarray, out_path: str | Path) -> Path:
    """Save an RGB uint8 overlay to PNG (converts RGB->BGR for cv2)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(rgb_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)
    return out_path
