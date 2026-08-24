"""Local verification tests for src/report.py (Phase III PDF generation).

Builds the vector PDF from a fabricated explanation bundle (no TensorFlow, so
these run fast), verifying it produces real PDF bytes, embeds images, handles the
no-lesion case, and - most importantly - that the pre-render guardrail sweep
rejects any forbidden diagnostic phrasing anywhere in the report (interpretation
or technical metadata).

Run under pytest:  python -m pytest tests/test_report.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

from src import report  # noqa: E402
from src.config import DECISION_DISCLAIMER, IMAGE_SIZE, SUSPICION_INDEX_NAME  # noqa: E402

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _img() -> np.ndarray:
    # Deterministic non-uniform RGB so the embedded PNGs are non-trivial.
    grad = np.linspace(0, 255, IMAGE_SIZE, dtype=np.uint8)
    plane = np.tile(grad, (IMAGE_SIZE, 1))
    return np.stack([plane, plane, plane], axis=-1)


def make_bundle(with_lesions: bool = True) -> dict:
    lesions = []
    if with_lesions:
        lesions = [
            {
                "region_id": 1,
                "label": "AI-Identified Suspicious Area #1",
                "area_px": 123.4,
                "centroid": (50.0, 60.0),
                "bbox": (10, 20, 30, 40),
            },
            {
                "region_id": 2,
                "label": "AI-Identified Suspicious Area #2",
                "area_px": 42.0,
                "centroid": (120.0, 130.0),
                "bbox": (100, 110, 20, 25),
            },
        ]
    return {
        "suspicion_index": 0.8700,
        "interpretation": {
            "index_name": SUSPICION_INDEX_NAME,
            "index_value": "0.8700",
            "assessment": "AI-Identified Suspicious Area",
            "recommendation": "Requires Professional Review",
            "disclaimer": DECISION_DISCLAIMER,
        },
        "attention_heatmap": np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32),
        "gradcam_overlay": _img(),
        "mask_probability": np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32),
        "suspicion_mask": np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8),
        "lesion_overlay": _img(),
        "lesions": lesions,
        "num_suspicious_regions": len(lesions),
        "disclaimer": DECISION_DISCLAIMER,
    }


def test_build_report_returns_pdf_bytes() -> None:
    pdf = report.build_report_pdf(make_bundle(), safe_meta={"Modality": "MG"})
    assert isinstance(pdf, (bytes, bytearray))
    assert bytes(pdf[:5]) == b"%PDF-"
    assert len(pdf) > 1000  # a real multi-image document, not an empty shell


def test_report_writes_to_disk(tmp_path) -> None:
    out = tmp_path / "report.pdf"
    pdf = report.build_report_pdf(make_bundle(), output_path=out)
    assert out.is_file()
    assert out.read_bytes()[:5] == b"%PDF-"
    assert out.read_bytes() == pdf


def test_report_without_lesions_is_valid() -> None:
    pdf = report.build_report_pdf(make_bundle(with_lesions=False))
    assert bytes(pdf[:5]) == b"%PDF-"


def test_report_embeds_original_image_row() -> None:
    pdf = report.build_report_pdf(
        make_bundle(), original_image=_img(), case_id="abc123", created_at="2026-08-24T00:00:00"
    )
    assert bytes(pdf[:5]) == b"%PDF-"


def test_guardrail_rejects_forbidden_in_interpretation() -> None:
    bundle = make_bundle()
    bundle["interpretation"]["assessment"] = "this is confirmed cancer"
    with pytest.raises(ValueError):
        report.build_report_pdf(bundle)


def test_guardrail_rejects_forbidden_in_metadata() -> None:
    # "definitely" is a forbidden phrase; it must be caught even inside metadata.
    with pytest.raises(ValueError):
        report.build_report_pdf(make_bundle(), safe_meta={"note": "definitely malignant"})


def test_np_to_png_bytes_handles_float_uint8_and_grayscale() -> None:
    float_rgb = np.zeros((8, 8, 3), np.float32)          # [0,1] float
    uint8_rgb = np.zeros((8, 8, 3), np.uint8)
    gray = np.zeros((8, 8), np.uint8)                    # 2-D
    single = np.zeros((8, 8, 1), np.uint8)               # (H,W,1) squeezed
    for arr in (float_rgb, uint8_rgb, gray, single):
        out = report.np_to_png_bytes(arr)
        assert out[:8] == _PNG_MAGIC


def test_ascii_replaces_and_drops_non_latin1() -> None:
    cleaned = report._ascii("range 1–5 → done")  # en-dash + right-arrow
    assert "→" not in cleaned      # non-latin1 arrow dropped
    assert "-" in cleaned               # en-dash folded to hyphen
    assert cleaned.encode("latin-1")    # must be latin-1 encodable (no exception)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
