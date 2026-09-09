# Molecular pathway analysis — what it computes and its limits

The Phase-IV molecular panel turns a breast-cancer RNA-seq expression matrix into
interpretable, **cohort-relative pathway activity** signals. This document is the
canonical description of the method and its expected input.

## The method (and what it is NOT)

The score is a **mean cohort-relative gene z-score** — deliberately simple and
transparent:

1. z-score every gene across the reference cohort (mean 0, unit variance *within
   this cohort*);
2. a pathway's score for a sample is the **arithmetic mean of the z-scores** of
   that pathway's measured genes;
3. `|mean z|` beyond `PATHWAY_ACTIVITY_THRESHOLD` is surfaced as an *elevated* or
   *reduced* cohort-relative signal, with the top contributing genes shown so the
   number is fully explainable.

> This is **explicitly not ssGSEA** and not any rank-based or competitive gene-set
> enrichment method. There is no ranking, no enrichment statistic, no null model,
> and no significance testing. It is a plain descriptive average of standardized
> expression, meaningful only *relative to the specific reference cohort supplied*.
> A different cohort yields different z-scores for the same sample. It is **not a
> validated measurement of pathway activation.**

The gene sets are small, hand-curated panels of well-established breast-cancer
genes (KEGG / MSigDB-Hallmark-inspired). They are illustrative teaching panels,
not exhaustive or validated signatures.

## Input format

A `(genes × samples)` matrix (CSV or TSV):

- **First column** = HGNC gene symbol (e.g. `BRCA1`, `CCND1`). Symbols are
  upper-cased; duplicate symbols keep the first row.
- **Remaining columns** = samples. Values are treated as already-normalized
  expression (e.g. log2 TPM/FPKM or RSEM). No normalization beyond cohort
  z-scoring is applied.
- **At least `MIN_COHORT_SAMPLES` (3) samples** are required — a z-score across a
  single sample is undefined, so single-sample "cohorts" are rejected.

Example:

```
gene,SAMPLE_A,SAMPLE_B,SAMPLE_C,SAMPLE_D
BRCA1,5.1,6.3,4.8,5.9
CCND1,2.0,1.4,3.1,2.7
TP53,7.2,7.0,6.6,7.4
```

## Validation and quality notes

`molecular.validate_cohort` refuses fatally-broken input and reports non-fatal
caveats the UI surfaces:

- **Rejected (raises `ValueError`):** fewer than the minimum samples, no gene
  rows, or zero overlap with any curated pathway gene.
- **Flagged (non-fatal notes):** constant genes (z-score 0, no information),
  duplicate sample columns (first match used), and small cohorts (< 10 samples →
  unstable z-scores).

## Provenance — never inferred

Provenance is **caller-supplied and never inferred from the data**:

- `synthetic (deterministic demo cohort)` — the built-in offline demo matrix.
- `user-provided (unverified)` — anything you upload. It is **never** relabeled
  as verified TCGA-BRCA.
- `TCGA-BRCA (verified)` — reserved for genuinely verified TCGA-BRCA data.

## The critical scientific-integrity guardrail

The imaging pipeline (CBIS-DDSM) and this transcriptomic module (TCGA-BRCA) use
**separate cohorts that are NOT patient-matched**. Nothing in the molecular panel
describes the biology of the patient whose mammogram was uploaded. The two are
shown side by side only to illustrate, for research and education, how
mammographic suspicion and tumor-biology pathways are studied — never as a
combined diagnosis. Every output Requires Professional Review.
