"""TCGA-BRCA transcriptomic pathway analysis for MamNexa AI (Phase IV).

Turns a breast-cancer RNA-seq expression matrix into interpretable, *cohort-
relative* **biological pathway activity** signals -- e.g. cell-cycle regulation
and DNA damage response -- to provide molecular research context alongside the
Phase I-III imaging analysis.

Scientific-integrity guardrail (critical)
-----------------------------------------
The imaging pipeline (CBIS-DDSM) and this transcriptomic module (TCGA-BRCA) use
**separate cohorts that are NOT patient-matched**. Nothing here describes the
biology of the patient whose mammogram was uploaded. The molecular panel is
reference context for research and education only; it never confirms, rules out,
or diagnoses disease. All language stays cautious and review-oriented, and every
emitted string is swept against :data:`config.FORBIDDEN_PHRASES`.

Method
------
Single-sample scoring is deliberately simple and transparent (no black box):

1. z-score every gene across the reference cohort (mean 0, unit variance);
2. a pathway's activity score for a sample is the **mean z** of its measured
   genes (a lightweight single-sample gene-set score in the spirit of ssGSEA);
3. |mean z| beyond ``PATHWAY_ACTIVITY_THRESHOLD`` is flagged as an *elevated* or
   *reduced* cohort-relative research signal, with the top contributing genes
   surfaced so the score is fully explainable.

Gene sets are small, hand-curated panels of well-established breast-cancer genes
(KEGG / MSigDB-Hallmark-inspired); they are illustrative, not exhaustive.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    FORBIDDEN_PHRASES,
    MOLECULAR_DISCLAIMER,
    MOLECULAR_TOP_GENES,
    PATHWAY_ACTIVITY_THRESHOLD,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Curated pathway gene sets (illustrative; KEGG / MSigDB-Hallmark-inspired).
# Symbols are HGNC upper-case. These are small teaching panels, not exhaustive
# signatures -- enough to demonstrate explainable pathway-level scoring.
# ---------------------------------------------------------------------------
PATHWAY_GENE_SETS: dict[str, list[str]] = {
    "Cell-Cycle Control": [
        "CCND1", "CCNE1", "CDK4", "CDK6", "RB1", "E2F1", "CDKN2A",
        "CCNB1", "CDK1", "CDC20", "MKI67", "AURKA", "PLK1", "BUB1",
    ],
    "DNA Damage Response & Repair": [
        "BRCA1", "BRCA2", "ATM", "ATR", "TP53", "CHEK1", "CHEK2",
        "RAD51", "PARP1", "MDM2", "PALB2", "XRCC1", "ERCC1", "MRE11",
    ],
    "PI3K-AKT-mTOR Signaling": [
        "PIK3CA", "AKT1", "AKT2", "PTEN", "MTOR", "TSC1", "TSC2",
        "RPS6KB1", "PIK3R1",
    ],
    "Hormone Receptor (Luminal) Signaling": [
        "ESR1", "PGR", "GATA3", "FOXA1", "XBP1", "BCL2", "TFF1",
    ],
    "HER2 / ERBB Signaling": [
        "ERBB2", "ERBB3", "EGFR", "GRB7", "ERBB4",
    ],
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class PathwayScore:
    """One pathway's cohort-relative activity for a single sample."""

    name: str
    score: float                       # mean z of measured genes
    direction: str                     # "up" | "down" | "neutral"
    activity_label: str                # cautious, guardrail-safe wording
    genes_used: int                    # measured genes found in the matrix
    genes_total: int                   # genes in the curated set
    top_genes: list[tuple[str, float]] # most influential (gene, z), |z| desc

    def as_row(self) -> dict[str, Any]:
        """Flatten to a display/table-friendly dict."""
        top = ", ".join(f"{g} ({z:+.2f})" for g, z in self.top_genes) or "-"
        return {
            "Pathway": self.name,
            "Activity score (mean z)": round(self.score, 3),
            "Signal": self.activity_label,
            "Genes measured": f"{self.genes_used}/{self.genes_total}",
            "Top contributing genes": top,
        }


# ---------------------------------------------------------------------------
# Expression loading + normalization
# ---------------------------------------------------------------------------
def load_expression_matrix(source: Any, sep: str = ",", index_col: int = 0) -> pd.DataFrame:
    """Load a (genes x samples) expression matrix from a path or file-like object.

    The first column is the gene symbol (used as the index); remaining columns are
    samples. Symbols are upper-cased and de-duplicated; non-numeric cells become 0.
    """
    df = pd.read_csv(source, sep=sep, index_col=index_col)
    df.index = df.index.astype(str).str.strip().str.upper()
    df = df[~df.index.duplicated(keep="first")]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if df.shape[1] == 0:
        raise ValueError("Expression matrix has no sample columns.")
    return df


def zscore_genes(expr: pd.DataFrame) -> pd.DataFrame:
    """Z-score each gene (row) across samples. Constant genes map to all zeros."""
    mean = expr.mean(axis=1)
    std = expr.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = expr.sub(mean, axis=0).div(std, axis=0)
    return z.fillna(0.0)


# ---------------------------------------------------------------------------
# Scoring + cautious interpretation
# ---------------------------------------------------------------------------
def interpret_pathway_activity(
    score: float, threshold: float = PATHWAY_ACTIVITY_THRESHOLD
) -> tuple[str, str]:
    """Map a mean-z score to a (direction, cautious label). Never diagnostic."""
    if score >= threshold:
        return "up", "Elevated pathway activity (cohort-relative research signal)"
    if score <= -threshold:
        return "down", "Reduced pathway activity (cohort-relative research signal)"
    return "neutral", "Near cohort-average activity"


def score_pathway_for_sample(
    zexpr: pd.DataFrame,
    sample: str,
    genes: list[str],
    *,
    threshold: float = PATHWAY_ACTIVITY_THRESHOLD,
    top_n: int = MOLECULAR_TOP_GENES,
    name: str = "",
) -> PathwayScore:
    """Score one pathway for one sample from a gene-by-sample z-score matrix."""
    present = [g for g in genes if g in zexpr.index]
    if not present:
        return PathwayScore(
            name=name, score=0.0, direction="neutral",
            activity_label="No measured genes for this pathway in the matrix",
            genes_used=0, genes_total=len(genes), top_genes=[],
        )
    zvals = zexpr.loc[present, sample].astype(float)
    score = float(zvals.mean())
    direction, label = interpret_pathway_activity(score, threshold)
    ranked = zvals.reindex(zvals.abs().sort_values(ascending=False).index)
    top_genes = [(str(g), float(z)) for g, z in ranked.head(top_n).items()]
    return PathwayScore(
        name=name, score=score, direction=direction, activity_label=label,
        genes_used=len(present), genes_total=len(genes), top_genes=top_genes,
    )


def analyze_sample(
    expr: pd.DataFrame,
    sample: str | None = None,
    *,
    pathways: dict[str, list[str]] | None = None,
    threshold: float = PATHWAY_ACTIVITY_THRESHOLD,
    top_n: int = MOLECULAR_TOP_GENES,
) -> dict[str, Any]:
    """Produce the full, guardrail-safe molecular pathway bundle for one sample.

    ``sample`` defaults to the first column. Returns a dict with per-pathway
    :class:`PathwayScore` objects, a cautious summary, cohort context, and the
    non-patient-matched disclaimer.
    """
    pathways = pathways or PATHWAY_GENE_SETS
    if expr.shape[1] == 0:
        raise ValueError("Expression matrix has no samples to analyze.")
    sample = sample if sample is not None else str(expr.columns[0])
    if sample not in expr.columns:
        raise KeyError(f"Sample {sample!r} is not a column in the expression matrix.")

    zexpr = zscore_genes(expr)
    scores = [
        score_pathway_for_sample(
            zexpr, sample, genes, threshold=threshold, top_n=top_n, name=name
        )
        for name, genes in pathways.items()
    ]

    n_up = sum(s.direction == "up" for s in scores)
    n_down = sum(s.direction == "down" for s in scores)
    summary = (
        f"Cohort-relative molecular pathway context for sample {sample}: "
        f"{n_up} pathway(s) show elevated and {n_down} show reduced research signals "
        f"versus the {expr.shape[1]}-sample TCGA-BRCA reference cohort. These are "
        "hypothesis-generating research observations, not findings about the imaged "
        "patient, and Require Professional Review."
    )

    bundle: dict[str, Any] = {
        "sample": sample,
        "cohort_size": int(expr.shape[1]),
        "genes_measured": int(expr.shape[0]),
        "pathway_scores": scores,
        "summary": summary,
        "disclaimer": MOLECULAR_DISCLAIMER,
    }
    _assert_molecular_text_safe(bundle)
    return bundle


def link_imaging_to_molecular(
    suspicion_index: float, molecular_bundle: dict[str, Any]
) -> dict[str, Any]:
    """Present imaging suspicion and molecular context together -- WITHOUT linking them.

    Explicitly states the two modalities are separate, non-patient-matched cohorts.
    The output is a research/education narrative, never a combined diagnosis.
    """
    idx = float(suspicion_index)
    elevated = [s.name for s in molecular_bundle.get("pathway_scores", []) if s.direction == "up"]
    context = (
        f"The imaging Model Malignancy Suspicion Index for the uploaded image is "
        f"{idx:.4f}. The molecular pathway context below is drawn from a separate "
        "TCGA-BRCA reference cohort and is NOT from this patient. Imaging "
        "(CBIS-DDSM) and transcriptomic (TCGA-BRCA) data are not patient-matched; "
        "they are shown side by side only to illustrate, for research and education, "
        "how mammographic suspicion and tumor-biology pathways such as cell-cycle "
        "regulation and DNA damage response are studied. This is not a combined "
        "diagnosis and Requires Professional Review."
    )
    result = {
        "suspicion_index": idx,
        "narrative": context,
        "related_pathways": elevated,
        "disclaimer": MOLECULAR_DISCLAIMER,
    }
    _assert_molecular_text_safe(result)
    return result


# ---------------------------------------------------------------------------
# Guardrail sweep
# ---------------------------------------------------------------------------
def _collect_molecular_text(obj: dict[str, Any]) -> list[str]:
    """Gather every human-facing string in a molecular bundle/link result."""
    texts: list[str] = [str(obj.get("summary", "")), str(obj.get("disclaimer", "")),
                        str(obj.get("narrative", ""))]
    for s in obj.get("pathway_scores", []):
        texts.extend([s.name, s.activity_label, s.direction])
    texts.extend(str(p) for p in obj.get("related_pathways", []))
    return texts


def _assert_molecular_text_safe(obj: dict[str, Any]) -> None:
    """Fail loud if any molecular text contains forbidden diagnostic phrasing."""
    blob = " ".join(_collect_molecular_text(obj)).lower()
    hits = [p for p in FORBIDDEN_PHRASES if p in blob]
    if hits:
        raise ValueError(f"Guardrail violation - forbidden phrasing in molecular output: {hits}")


# ---------------------------------------------------------------------------
# Synthetic reference cohort (offline demo + tests; no TCGA download needed)
# ---------------------------------------------------------------------------
def synthetic_expression_matrix(
    n_samples: int = 40, seed: int = RANDOM_SEED, n_background: int = 200
) -> pd.DataFrame:
    """Build a deterministic, structured (genes x samples) matrix for offline use.

    Genes are the union of all curated pathway genes plus ``n_background`` filler
    genes. Samples are drawn from a few latent groups with pathway-correlated
    shifts, so pathway scores vary meaningfully in the demo. This is SYNTHETIC
    data for wiring/verification -- it is not real TCGA-BRCA expression.
    """
    rng = np.random.default_rng(seed)
    pathway_genes = sorted({g for gs in PATHWAY_GENE_SETS.values() for g in gs})
    background = [f"BG{i:04d}" for i in range(n_background)]
    genes = pathway_genes + background

    base = rng.normal(6.0, 1.0, size=(len(genes), n_samples))  # log2-like baseline
    gene_pos = {g: i for i, g in enumerate(genes)}

    # Three latent "profiles" that up/down-shift specific pathways for structure.
    groups = rng.integers(0, 3, size=n_samples)
    profile_shifts = {
        0: {"Cell-Cycle Control": 1.5, "DNA Damage Response & Repair": 1.2},
        1: {"Hormone Receptor (Luminal) Signaling": 1.4, "PI3K-AKT-mTOR Signaling": 0.8},
        2: {"HER2 / ERBB Signaling": 1.6, "Cell-Cycle Control": 0.6},
    }
    for s in range(n_samples):
        for pathway, shift in profile_shifts[int(groups[s])].items():
            for g in PATHWAY_GENE_SETS[pathway]:
                base[gene_pos[g], s] += shift

    columns = [f"TCGA-SYN-{i:03d}" for i in range(n_samples)]
    return pd.DataFrame(base, index=genes, columns=columns)
