"""Regression tests for the scientific-honesty guarantees added in the review.

TensorFlow-free and fast. They lock in three things the review required:

* the molecular module VALIDATES a cohort (rejects single-sample / no-overlap
  matrices, flags constant + duplicate genes) and never relabels a user upload
  as verified TCGA-BRCA;
* provenance is carried verbatim through ``analyze_sample`` into the bundle;
* the PDF report threads model status + data provenance into its guardrail sweep
  and its honesty block, so a demonstration PDF is self-describing.

Run:  python -m pytest tests/test_honesty_regressions.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

from src import model_status as ms  # noqa: E402
from src import molecular as mol  # noqa: E402
from src import report  # noqa: E402
from src.config import DECISION_DISCLAIMER, IMAGE_SIZE, SUSPICION_INDEX_NAME  # noqa: E402


# ---------------------------------------------------------------------------
# Molecular cohort validation
# ---------------------------------------------------------------------------
def test_single_sample_cohort_is_rejected() -> None:
    """A z-score is undefined for one sample; validate_cohort must refuse it."""
    expr = mol.synthetic_expression_matrix(n_samples=1)
    with pytest.raises(ValueError, match="sample"):
        mol.validate_cohort(expr)
    # analyze_sample validates first, so it too must raise (not silently score).
    with pytest.raises(ValueError):
        mol.analyze_sample(expr)


def test_matrix_without_curated_gene_overlap_is_rejected() -> None:
    expr = pd.DataFrame(
        np.random.default_rng(0).normal(size=(4, 5)),
        index=["FOO", "BAR", "BAZ", "QUX"],
        columns=[f"S{i}" for i in range(5)],
    )
    with pytest.raises(ValueError, match="curated"):
        mol.validate_cohort(expr)


def test_validate_cohort_flags_constant_and_duplicate_and_small() -> None:
    expr = mol.synthetic_expression_matrix(n_samples=5, seed=1)
    expr.loc["CCND1"] = 7.0  # force a constant gene
    expr = pd.concat([expr, expr.iloc[:, [0]]], axis=1)  # duplicate a sample col
    report_ = mol.validate_cohort(expr)
    joined = " ".join(report_["notes"]).lower()
    assert report_["constant_genes"] >= 1 and "constant" in joined
    assert report_["duplicate_samples"] and "duplicate" in joined
    assert "small cohort" in joined  # < 10 samples


def test_provenance_is_recorded_verbatim_and_upload_never_relabeled() -> None:
    expr = mol.synthetic_expression_matrix(n_samples=8)
    up = mol.analyze_sample(expr, provenance=mol.PROVENANCE_USER)
    assert up["provenance"] == mol.PROVENANCE_USER
    assert "user-provided" in up["provenance"].lower()
    # The user-provided bundle must NOT claim verified TCGA-BRCA anywhere.
    assert mol.PROVENANCE_TCGA_VERIFIED.lower() not in up["summary"].lower()

    syn = mol.analyze_sample(expr, provenance=mol.PROVENANCE_SYNTHETIC)
    assert syn["provenance"] == mol.PROVENANCE_SYNTHETIC
    assert "not ssgsea" in syn["method"].lower()


def test_method_string_is_not_ssgsea() -> None:
    expr = mol.synthetic_expression_matrix(n_samples=6)
    bundle = mol.analyze_sample(expr)
    assert "ssgsea" not in bundle["method"].lower().replace("not ssgsea", "")


# ---------------------------------------------------------------------------
# PDF report honesty block + provenance threading
# ---------------------------------------------------------------------------
def _bundle() -> dict:
    grad = np.tile(np.linspace(0, 255, IMAGE_SIZE, dtype=np.uint8), (IMAGE_SIZE, 1))
    img = np.stack([grad, grad, grad], axis=-1)
    return {
        "suspicion_index": 0.5,
        "interpretation": {
            "index_name": SUSPICION_INDEX_NAME,
            "index_value": "0.5000",
            "assessment": "AI-Identified Suspicious Area",
            "recommendation": "Requires Professional Review",
            "disclaimer": DECISION_DISCLAIMER,
        },
        "gradcam_overlay": img,
        "lesion_overlay": img,
        "lesions": [],
        "num_suspicious_regions": 0,
        "disclaimer": DECISION_DISCLAIMER,
    }


def test_report_collects_status_and_provenance_text_for_sweep() -> None:
    demo = ms.ModelStatus(name="classifier", state=ms.STATE_UNTRAINED, checkpoint="x")
    texts = report._collect_report_text(
        _bundle(), None, "case1", "2026-01-01T00:00:00",
        model_statuses=[demo], data_provenance="Synthetic demo mammogram",
    )
    blob = " ".join(texts).lower()
    assert "no predictive meaning" in blob
    assert "synthetic demo mammogram" in blob


def test_demo_report_builds_and_provenance_is_swept_for_guardrails() -> None:
    demo = ms.ModelStatus(name="classifier", state=ms.STATE_UNTRAINED, checkpoint="x")
    pdf = report.build_report_pdf(
        _bundle(), model_statuses=[demo],
        data_provenance="Synthetic demo mammogram (built-in; not a real image)",
    )
    assert bytes(pdf[:5]) == b"%PDF-"
    # A forbidden phrase sneaked in via provenance must still be caught.
    with pytest.raises(ValueError):
        report.build_report_pdf(
            _bundle(), model_statuses=[demo], data_provenance="no cancer here",
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
