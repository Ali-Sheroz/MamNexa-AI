"""Local verification tests for src/molecular.py (Phase IV molecular extension).

These run on a deterministic SYNTHETIC TCGA-BRCA-like matrix (no download, no
TensorFlow), verifying the pathway-scoring mechanics, the cohort-relative
z-scoring, the explainable top-gene surfacing, and -- most importantly -- that
every emitted string passes the clinical guardrail sweep and that the imaging
<-> molecular presentation is explicitly NON patient-matched.

Run under pytest:  python -m pytest tests/test_molecular.py -v
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

from src import molecular as mol  # noqa: E402
from src.config import FORBIDDEN_PHRASES, PATHWAY_ACTIVITY_THRESHOLD  # noqa: E402


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return mol.synthetic_expression_matrix(n_samples=30, seed=7)


def test_pathway_gene_sets_are_populated_and_relevant() -> None:
    assert mol.PATHWAY_GENE_SETS, "no pathways defined"
    for name, genes in mol.PATHWAY_GENE_SETS.items():
        assert genes and all(isinstance(g, str) and g == g.upper() for g in genes)
    assert "CCND1" in mol.PATHWAY_GENE_SETS["Cell-Cycle Control"]
    dna = mol.PATHWAY_GENE_SETS["DNA Damage Response & Repair"]
    assert {"BRCA1", "BRCA2", "TP53"}.issubset(set(dna))


def test_synthetic_matrix_is_reproducible_and_shaped() -> None:
    a = mol.synthetic_expression_matrix(n_samples=12, seed=3)
    b = mol.synthetic_expression_matrix(n_samples=12, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert a.shape[1] == 12
    # Every curated pathway gene must exist in the matrix.
    all_genes = {g for gs in mol.PATHWAY_GENE_SETS.values() for g in gs}
    assert all_genes.issubset(set(a.index))


def test_zscore_is_standardized_and_handles_constant_genes() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(5, 2, size=(6, 20)), index=[f"G{i}" for i in range(6)])
    df.loc["CONST"] = 3.0  # a constant gene
    z = mol.zscore_genes(df)
    # Non-constant genes: mean ~0, std (ddof=0) ~1.
    non_const = z.drop(index="CONST")
    assert np.allclose(non_const.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(non_const.std(axis=1, ddof=0), 1.0, atol=1e-9)
    # Constant gene must not produce NaN/inf -- it maps to all zeros.
    assert np.allclose(z.loc["CONST"], 0.0)


def test_score_pathway_subset_and_missing_genes(cohort) -> None:
    z = mol.zscore_genes(cohort)
    sample = cohort.columns[0]

    ps = mol.score_pathway_for_sample(
        z, sample, mol.PATHWAY_GENE_SETS["Cell-Cycle Control"], name="Cell-Cycle Control"
    )
    assert 0 < ps.genes_used <= ps.genes_total
    assert len(ps.top_genes) <= 5
    assert ps.direction in {"up", "down", "neutral"}

    # A pathway whose genes are entirely absent must not crash.
    missing = mol.score_pathway_for_sample(z, sample, ["NOT_A_GENE_1", "NOT_A_GENE_2"], name="X")
    assert missing.genes_used == 0 and missing.score == 0.0 and missing.direction == "neutral"


def test_analyze_sample_bundle_is_complete_and_guardrail_safe(cohort) -> None:
    bundle = mol.analyze_sample(cohort)
    assert set(bundle) >= {
        "sample", "cohort_size", "genes_measured", "pathway_scores", "summary", "disclaimer"
    }
    assert len(bundle["pathway_scores"]) == len(mol.PATHWAY_GENE_SETS)
    assert bundle["cohort_size"] == cohort.shape[1]

    # Disclaimer must assert the non-patient-matched, research-only framing.
    disc = bundle["disclaimer"].lower()
    assert "not" in disc and "patient" in disc

    # Sweep ALL molecular text for forbidden diagnostic phrasing.
    blob = " ".join(mol._collect_molecular_text(bundle)).lower()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in blob


def test_interpret_activity_labels_are_cautious() -> None:
    up_dir, up_label = mol.interpret_pathway_activity(PATHWAY_ACTIVITY_THRESHOLD + 1.0)
    dn_dir, dn_label = mol.interpret_pathway_activity(-(PATHWAY_ACTIVITY_THRESHOLD + 1.0))
    mid_dir, _ = mol.interpret_pathway_activity(0.0)
    assert up_dir == "up" and dn_dir == "down" and mid_dir == "neutral"
    for label in (up_label, dn_label):
        low = label.lower()
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in low


def test_link_imaging_to_molecular_is_explicitly_non_matched(cohort) -> None:
    bundle = mol.analyze_sample(cohort)
    link = mol.link_imaging_to_molecular(0.83, bundle)
    narrative = link["narrative"].lower()
    assert "not patient-matched" in narrative or ("not" in narrative and "patient-matched" in narrative)
    assert "not a combined diagnosis" in narrative
    assert "requires professional review" in narrative
    # Guardrail-safe.
    blob = " ".join(mol._collect_molecular_text(link)).lower()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in blob


def test_guardrail_rejects_injected_forbidden_text() -> None:
    poisoned = {
        "summary": "",
        "disclaimer": "",
        "pathway_scores": [
            mol.PathwayScore(
                name="X", score=0.0, direction="neutral",
                activity_label="this is confirmed cancer",  # forbidden
                genes_used=0, genes_total=0, top_genes=[],
            )
        ],
    }
    with pytest.raises(ValueError):
        mol._assert_molecular_text_safe(poisoned)


def test_load_expression_matrix_from_csv(tmp_path) -> None:
    csv = tmp_path / "expr.csv"
    csv.write_text(
        "gene,S1,S2,S3\n"
        "brca1,5.0,6.0,7.0\n"
        "ccnd1,1.0,2.0,3.0\n"
        "brca1,9.0,9.0,9.0\n",  # duplicate -> first kept
        encoding="utf-8",
    )
    df = mol.load_expression_matrix(csv)
    assert list(df.columns) == ["S1", "S2", "S3"]
    assert "BRCA1" in df.index and "CCND1" in df.index      # upper-cased
    assert df.loc["BRCA1", "S1"] == 5.0                     # first duplicate kept
    assert df.index.is_unique


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
