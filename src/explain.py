"""Explainability orchestrator for MamNexa AI (Phase II).

Runs the full Phase-II analysis for a single preprocessed image and returns one
guardrail-checked bundle. This is the single seam Phase III's Streamlit UI and
PDF report will call, so all the framing lives in one place:

    classifier  -> Model Malignancy Suspicion Index  (how suspicious)
    Grad-CAM     -> attention heatmap overlay          (why)
    U-Net        -> suspicion probability mask
    OpenCV       -> numbered AI-Identified Suspicious Areas (where)

Every piece of text in the returned bundle is swept against the forbidden
diagnostic vocabulary (``model._FORBIDDEN_PHRASES``) before returning, so a
guardrail regression fails loud here rather than reaching a user.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from .config import (
    DECISION_DISCLAIMER,
    GRADCAM_OVERLAY_ALPHA,
    MASK_THRESHOLD,
    MIN_LESION_AREA_PX,
    SUSPICION_THRESHOLD,
)
from .gradcam import gradcam_overlay
from .localization import localize_lesions
from .model import _FORBIDDEN_PHRASES, interpret_suspicion, suspicion_index
from .preprocessing import array_to_uint8
from .segmentation import binarize_mask, predict_mask


def _collect_text(result: dict[str, Any]) -> list[str]:
    """Gather every human-facing string in the result bundle for guardrail check."""
    texts: list[str] = [str(result.get("disclaimer", ""))]
    texts.extend(str(v) for v in result.get("interpretation", {}).values())
    texts.extend(str(lesion.get("label", "")) for lesion in result.get("lesions", []))
    return texts


def _assert_all_text_safe(result: dict[str, Any]) -> None:
    """Fail loud if any output text contains forbidden diagnostic phrasing."""
    blob = " ".join(_collect_text(result)).lower()
    hits = [p for p in _FORBIDDEN_PHRASES if p in blob]
    if hits:
        raise ValueError(f"Guardrail violation - forbidden phrasing in explanation: {hits}")


def explain_case(
    image: np.ndarray,
    classifier: tf.keras.Model,
    segmenter: tf.keras.Model,
    *,
    suspicion_threshold: float = SUSPICION_THRESHOLD,
    mask_threshold: float = MASK_THRESHOLD,
    min_area: int = MIN_LESION_AREA_PX,
    gradcam_alpha: float = GRADCAM_OVERLAY_ALPHA,
) -> dict[str, Any]:
    """Produce the full, guardrail-compliant Phase-II explanation for one image.

    ``image`` is an ``(H, W, 3)`` float32 array in ``[0, 1]`` (a
    ``preprocessing.dicom_to_clean_array`` output). Returns a dict with the
    suspicion index and its cautious interpretation, a Grad-CAM attention
    overlay, the U-Net suspicion mask, and a numbered lesion overlay + table.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected a single (H, W, 3) image, got shape {arr.shape}")
    base_uint8 = array_to_uint8(arr)

    # 1. How suspicious? (Model Malignancy Suspicion Index + cautious wording.)
    index = float(suspicion_index(classifier, arr)[0])
    interpretation = interpret_suspicion(index, threshold=suspicion_threshold)

    # 2. Why? (Grad-CAM attention over the classifier.)
    gradcam_img, heatmap = gradcam_overlay(classifier, arr, base_image=base_uint8, alpha=gradcam_alpha)

    # 3. Where? (U-Net mask -> numbered OpenCV contours.)
    prob_mask = predict_mask(segmenter, arr)
    binary_mask = binarize_mask(prob_mask, threshold=mask_threshold)
    lesion_overlay, lesions = localize_lesions(binary_mask, base_uint8, min_area=min_area)

    result: dict[str, Any] = {
        "suspicion_index": index,
        "interpretation": interpretation,
        "attention_heatmap": heatmap,
        "gradcam_overlay": gradcam_img,
        "mask_probability": prob_mask,
        "suspicion_mask": binary_mask,
        "lesion_overlay": lesion_overlay,
        "lesions": lesions,
        "num_suspicious_regions": len(lesions),
        "disclaimer": DECISION_DISCLAIMER,
    }
    _assert_all_text_safe(result)
    return result
