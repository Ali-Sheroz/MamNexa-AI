"""Local verification tests for src/localization.py (Phase II contours).

Uses hand-built binary masks (two filled rectangles + a sub-threshold speck) for
deterministic contour assertions, plus one end-to-end pass that runs a synthetic
DICOM through preprocessing and localizes a mask over the real grayscale image.

Run under pytest:  python -m pytest tests/test_localization.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

from src import localization as loc  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src.config import IMAGE_SIZE, LESION_LABEL_PREFIX  # noqa: E402
from synthetic_dicom import make_synthetic_dicom  # noqa: E402


def _two_blobs_plus_speck() -> np.ndarray:
    """224x224 mask: a big square (top-left), a smaller one (bottom-right), a 2px speck."""
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (70, 70), 1, thickness=-1)     # 50x50 -> largest
    cv2.rectangle(mask, (150, 160), (180, 190), 1, thickness=-1)  # 30x30 -> second
    mask[5, 5] = 1  # isolated speck, area ~0 -> below MIN_LESION_AREA_PX
    return mask


# ---------------------------------------------------------------------------
# Contour extraction + numbering
# ---------------------------------------------------------------------------
def test_extract_drops_speck_and_keeps_two_regions() -> None:
    contours = loc.extract_lesion_contours(_two_blobs_plus_speck())
    assert len(contours) == 2


def test_numbering_is_area_ordered_and_guardrail_safe() -> None:
    contours = loc.extract_lesion_contours(_two_blobs_plus_speck())
    table = loc.summarize_lesions(contours)

    assert [row["region_id"] for row in table] == [1, 2]
    # Largest region is #1.
    assert table[0]["area_px"] > table[1]["area_px"]
    # Guardrail-compliant labels, no forbidden vocabulary.
    for row in table:
        assert row["label"].startswith(LESION_LABEL_PREFIX)
        assert "cancer" not in row["label"].lower()
        assert "tumor" not in row["label"].lower()
    # Metadata is well-formed.
    for row in table:
        assert row["area_px"] > 0
        assert len(row["bbox"]) == 4
        assert len(row["centroid"]) == 2


def test_accepts_255_scale_and_bool_masks() -> None:
    m01 = _two_blobs_plus_speck()
    assert len(loc.extract_lesion_contours(m01 * 255)) == 2
    assert len(loc.extract_lesion_contours(m01.astype(bool))) == 2


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def test_draw_numbered_contours_marks_the_image() -> None:
    mask = _two_blobs_plus_speck()
    base = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    contours = loc.extract_lesion_contours(mask)

    overlay = loc.draw_numbered_contours(base, contours)
    assert overlay.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert overlay.dtype == np.uint8
    # Something was actually drawn (boundaries + numbers), and the input is intact.
    assert np.any(overlay != base)
    assert np.all(base == 0)


def test_localize_lesions_end_to_end_on_synthetic_dicom(tmp_path: Path) -> None:
    path = make_synthetic_dicom(tmp_path / "loc.dcm")
    clean = pp.dicom_to_clean_array(path, size=IMAGE_SIZE)
    base = pp.array_to_uint8(clean.array)

    overlay, table = loc.localize_lesions(_two_blobs_plus_speck(), base)
    assert overlay.shape == base.shape
    assert len(table) == 2


def test_empty_mask_yields_no_regions() -> None:
    empty = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    overlay, table = loc.localize_lesions(empty, np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8))
    assert table == []
    assert overlay.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
