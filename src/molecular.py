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

Method (what this actually computes)
------------------------------------
The score is a **mean cohort-relative gene z-score** -- deliberately simple and
transparent, NOT a validated pathway-activation measurement:

1. z-score every gene across the reference cohort (mean 0, unit variance within
   this cohort);
2. a pathway's score for a sample is the **arithmetic mean of the z-scores** of
   that pathway's measured genes;
3. |mean z| beyond ``PATHWAY_ACTIVITY_THRESHOLD`` is surfaced as an *elevated*
   or *reduced* cohort-relative signal, with the top contributing genes shown so
   the number is fully explainable.

This is explicitly **not ssGSEA** and not any rank-based or competitive gene-set
enrichment method: there is no ranking, no enrichment statistic, no null model,
and no significance testing. It is a plain descriptive average of standardized
expression, and it is meaningful only *relative to the specific reference cohort
supplied*. A different cohort yields different z-scores for the same sample.

Gene sets are small, hand-curated panels of well-established breast-cancer genes
(KEGG / MSigDB-Hallmark-inspired); they are illustrative, not exhaustive, and are
not validated signatures.

Input expectations
------------------
* A (genes x samples) matrix: first column = HGNC gene symbol, remaining columns
  = samples. Symbols are upper-cased; duplicate symbols keep the first row.
* Values are treated as already-normalized expression (e.g. log2 TPM/FPKM or
  RSEM). No within-matrix normalization beyond the cohort z-scoring is applied.
* At least :data:`MIN_COHORT_SAMPLES` samples are required: a z-score across one
  sample is undefined, so single-sample "cohorts" are rejected.
* Provenance is caller-supplied and never inferred: an uploaded matrix is labeled
  "user-provided (unverified)" and is NEVER relabeled as verified TCGA-BRCA.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    FORBIDDEN_PHRASES,
    MIN_COHORT_SAMPLES,
    MOLECULAR_DISCLAIMER,
    MOLECULAR_TOP_GENES,
    PATHWAY_ACTIVITY_THRESHOLD,
    RANDOM_SEED,
)

# Provenance labels. Provenance is caller-supplied, never inferred from content.
PROVENANCE_SYNTHETIC = "synthetic (deterministic demo cohort)"
PROVENANCE_USER = "user-provided (unverified)"
PROVENANCE_TCGA_VERIFIED = "TCGA-BRCA (verified)"

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
    samples. Symbols are upper-cased and de-duplicated (first row kept); non-numeric
    cells become 0. Structural problems raise ``ValueError`` with a clear message.
    """
    df = pd.read_csv(source, sep=sep, index_col=index_col)
    if df.shape[1] == 0:
        raise ValueError(
            "Expression matrix has no sample columns. Expected the first column to "
            "be gene symbols and each remaining column to be a sample."
        )
    df.index = df.index.astype(str).str.strip().str.upper()
    # Drop unnamed / empty gene symbols that pandas may read as 'NAN'/''.
    df = df[(df.index != "") & (df.index != "NAN")]
    df = df[~df.index.duplicated(keep="first")]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if df.shape[0] == 0:
        raise ValueError("Expression matrix has no usable gene rows after cleaning.")
    return df


def validate_cohort(
    expr: pd.DataFrame,
    *,
    pathways: dict[str, list[str]] | None = None,
    min_samples: int = MIN_COHORT_SAMPLES,
) -> dict[str, Any]:
    """Check an expression matrix is usable and report what was found.

    Raises ``ValueError`` for fatal problems (too few samples, no genes, no
    overlap with any curated pathway). Returns a report dict of non-fatal notes
    (constant genes, duplicate columns, curated-gene coverage) the UI can surface
    so the user understands the limits of the cohort-relative score.
    """
    pathways = pathways or PATHWAY_GENE_SETS
    n_genes, n_samples = expr.shape

    if n_samples < min_samples:
        raise ValueError(
            f"Cohort has only {n_samples} sample(s); at least {min_samples} are "
            "required. A cohort-relative z-score is undefined for a single sample "
            "(no spread to standardize against). Provide a larger reference cohort."
        )
    if n_genes == 0:
        raise ValueError("Expression matrix has no gene rows.")

    curated = {g for gs in pathways.values() for g in gs}
    measured = curated & set(expr.index)
    if not measured:
        raise ValueError(
            "None of the curated pathway genes were found in this matrix. Check "
            "that the first column contains HGNC gene symbols (e.g. BRCA1, CCND1)."
        )

    # Non-fatal quality notes.
    duplicate_samples = [c for c in expr.columns if list(expr.columns).count(c) > 1]
    constant_genes = int((expr.std(axis=1, ddof=0) == 0).sum())
    notes: list[str] = []
    if constant_genes:
        notes.append(
            f"{constant_genes} gene(s) are constant across the cohort and "
            "contribute a z-score of 0 (no cohort-relative information)."
        )
    if duplicate_samples:
        notes.append(
            f"Duplicate sample column name(s): {sorted(set(duplicate_samples))}. "
            "Pathway scoring uses the first matching column."
        )
    if n_samples < 10:
        notes.append(
            f"Small cohort ({n_samples} samples): z-scores are unstable and easily "
            "dominated by individual samples. Interpret with caution."
        )

    return {
        "n_genes": int(n_genes),
        "n_samples": int(n_samples),
        "curated_genes_total": len(curated),
        "curated_genes_measured": len(measured),
        "constant_genes": constant_genes,
        "duplicate_samples": sorted(set(duplicate_samples)),
        "notes": notes,
    }


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
        return "up", "Higher cohort-relative mean z-score"
    if score <= -threshold:
        return "down", "Lower cohort-relative mean z-score"
    return "neutral", "Near cohort-average mean z-score"


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
    provenance: str = PROVENANCE_USER,
    min_samples: int = MIN_COHORT_SAMPLES,
) -> dict[str, Any]:
    """Produce the full, guardrail-safe molecular pathway bundle for one sample.

    ``sample`` defaults to the first column. ``provenance`` is a caller-supplied
    label describing where the cohort came from (synthetic / user-provided /
    verified TCGA-BRCA) -- it is recorded verbatim and NEVER inferred from the
    data. Validates the cohort first (raises ``ValueError`` on a single-sample or
    otherwise unusable matrix). Returns a dict with per-pathway
    :class:`PathwayScore` objects, a cautious summary, cohort context, quality
    notes, provenance, and the non-patient-matched disclaimer.
    """
    pathways = pathways or PATHWAY_GENE_SETS
    report = validate_cohort(expr, pathways=pathways, min_samples=min_samples)

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
        f"Cohort-relative molecular pathway context for sample {sample} "
        f"(mean gene z-score per pathway): {n_up} pathway(s) show elevated and "
        f"{n_down} show reduced signals versus the {expr.shape[1]}-sample reference "
        f"cohort [provenance: {provenance}]. These are descriptive, "
        "hypothesis-generating research observations, not findings about the imaged "
        "patient, and Require Professional Review."
    )

    bundle: dict[str, Any] = {
        "sample": sample,
        "cohort_size": int(expr.shape[1]),
        "genes_measured": int(expr.shape[0]),
        "provenance": provenance,
        "method": "mean cohort-relative gene z-score (not ssGSEA; not validated)",
        "quality_notes": report["notes"],
        "curated_gene_coverage": f"{report['curated_genes_measured']}/{report['curated_genes_total']}",
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
    provenance = molecular_bundle.get("provenance", "")
    if provenance == PROVENANCE_SYNTHETIC:
        cohort_source = (
            "The molecular pathway context is drawn from a separate synthetic "
            "reference cohort and is not derived from the imaged sample."
        )
    elif provenance == PROVENANCE_USER:
        cohort_source = (
            "The molecular pathway context is drawn from a separate user-provided "
            "expression cohort."
        )
    else:
        cohort_source = (
            "The molecular pathway context is drawn from a separate reference "
            "cohort and is not derived from the imaged sample."
        )
    context = (
        f"The imaging Model Malignancy Suspicion Index for the uploaded image is "
        f"{idx:.4f}. {cohort_source} Imaging (CBIS-DDSM) and transcriptomic "
        "(TCGA-BRCA) data are not patient-matched; they are shown side by side "
        "only to illustrate, for research and education, how mammographic "
        "suspicion and tumor-biology pathways such as cell-cycle regulation and "
        "DNA damage response are studied. This is not a combined diagnosis and "
        "Requires Professional Review."
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
                        str(obj.get("narrative", "")), str(obj.get("method", "")),
                        str(obj.get("provenance", ""))]
    texts.extend(str(n) for n in obj.get("quality_notes", []))
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
