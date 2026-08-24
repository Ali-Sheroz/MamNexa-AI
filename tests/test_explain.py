"""Local verification tests for src/explain.py (Phase II orchestrator).

Runs the full explainability bundle (classifier + Grad-CAM + U-Net + contours)
on a synthetic-DICOM image with randomly-initialized models, checking the bundle
structure, artifact shapes, and — most importantly — that every human-facing
string passes the clinical guardrail sweep.

Run under pytest:  python -m pytest tests/test_explain.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

from src import explain  # noqa: E402
from src import model as m  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src import segmentation as seg  # noqa: E402
from src.config import IMAGE_SIZE  # noqa: E402
from synthetic_dicom import make_synthetic_dicom  # noqa: E402


@pytest.fixture(scope="module")
def models():
    classifier, _ = m.build_efficientnet_b0(weights=None)
    m.compile_model(classifier)
    segmenter = seg.build_unet()
    seg.compile_unet(segmenter)
    return classifier, segmenter


@pytest.fixture(scope="module")
def clean_image(tmp_path_factory):
    path = make_synthetic_dicom(tmp_path_factory.mktemp("exp") / "case.dcm")
    return pp.dicom_to_clean_array(path, size=IMAGE_SIZE).array


def test_explain_case_bundle_structure(models, clean_image) -> None:
    classifier, segmenter = models
    result = explain.explain_case(clean_image, classifier, segmenter)

    expected_keys = {
        "suspicion_index", "interpretation", "attention_heatmap", "gradcam_overlay",
        "mask_probability", "suspicion_mask", "lesion_overlay", "lesions",
        "num_suspicious_regions", "disclaimer",
    }
    assert expected_keys.issubset(result.keys())

    assert 0.0 <= result["suspicion_index"] <= 1.0
    assert result["gradcam_overlay"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert result["lesion_overlay"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert result["suspicion_mask"].shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert result["attention_heatmap"].shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert isinstance(result["lesions"], list)
    assert result["num_suspicious_regions"] == len(result["lesions"])


def test_explain_case_interpretation_uses_cautious_vocabulary(models, clean_image) -> None:
    classifier, segmenter = models
    result = explain.explain_case(clean_image, classifier, segmenter)

    interp = result["interpretation"]
    assert interp["recommendation"] == "Requires Professional Review"
    assert interp["index_name"] == "Model Malignancy Suspicion Index"

    # Sweep the entire bundle's text for forbidden diagnostic phrasing.
    blob = " ".join(
        [str(result["disclaimer"])]
        + [str(v) for v in interp.values()]
        + [str(row["label"]) for row in result["lesions"]]
    ).lower()
    for phrase in m._FORBIDDEN_PHRASES:
        assert phrase not in blob


def test_explain_case_rejects_batched_input(models) -> None:
    classifier, segmenter = models
    batched = np.zeros((2, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        explain.explain_case(batched, classifier, segmenter)


def test_guardrail_sweep_catches_injected_forbidden_text() -> None:
    poisoned = {
        "disclaimer": "",
        "interpretation": {"assessment": "this is confirmed cancer"},
        "lesions": [],
    }
    with pytest.raises(ValueError):
        explain._assert_all_text_safe(poisoned)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
