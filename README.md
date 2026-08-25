# MamNexa AI

**An open-source research framework developed in Python for exploring explainable AI methods in mammographic analysis and molecular breast-cancer research..**

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.11 / 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Status: Research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#status--maturity)

> ⚠️ **Research decision-support only — NOT a medical device and NOT a diagnostic tool.**
> MamNexa AI does not, and cannot, confirm or rule out cancer. Every output is a
> model-generated *research signal* that **Requires Professional Review** by a
> qualified clinician. See [Clinical & scientific guardrails](#clinical--scientific-guardrails).

---

## Table of contents

- [What MamNexa AI is](#what-mamnexa-ai-is)
- [Clinical & scientific guardrails](#clinical--scientific-guardrails)
- [Architecture at a glance](#architecture-at-a-glance)
- [The four phases](#the-four-phases)
- [Repository layout](#repository-layout)
- [Quick start — run the Streamlit app locally](#quick-start--run-the-streamlit-app-locally)
- [Optional: enable the Supabase backend](#optional-enable-the-supabase-backend)
- [Right to Erasure](#right-to-erasure)
- [Running the verification tests](#running-the-verification-tests)
- [Datasets](#datasets)
- [Status & maturity](#status--maturity)
- [License](#license)
- [Citation & academic context](#citation--academic-context)

---

## What MamNexa AI is

MamNexa AI is a **portfolio / research** framework that demonstrates, end to end,
how an explainable AI pipeline for breast-cancer research can be built responsibly:

- **Imaging analysis** — a mammogram is preprocessed from DICOM (with all patient
  identifiers stripped), scored by an EfficientNet-B0 classifier, localized by a
  U-Net segmenter, and explained with Grad-CAM attention heatmaps and numbered,
  contoured regions of interest.
- **Molecular context** — a TCGA-BRCA transcriptomic module maps gene-expression
  profiles onto well-established biological pathways (cell-cycle control, DNA
  damage response & repair, PI3K–AKT–mTOR, hormone-receptor signaling, HER2/ERBB)
  to provide *research* context on tumor biology.
- **Interface & governance** — a local Streamlit dashboard ties it together, emits
  a vector PDF pre-analysis report, and implements a strict **Right to Erasure**.

Everything runs **locally, in memory**. The model architectures are open and free;
no part of this project is a paid product or contains monetization hooks.

## Clinical & scientific guardrails

These are hard constraints baked into the code (see `src/config.py` →
`FORBIDDEN_PHRASES`, and the `_assert_*_safe` sweeps in `model.py`, `explain.py`,
`report.py`, and `molecular.py`). They are enforced by automated tests.

| Principle | How it is enforced |
|---|---|
| **Not diagnostic.** The system never states a diagnosis, and never confirms or excludes malignancy. | A shared forbidden-phrase list is swept over **every** human-facing string (UI, PDF, model output) *before* it is displayed or drawn; a violation raises an error rather than shipping the text. |
| **Cautious mandated vocabulary.** | Outputs use *"Model Malignancy Suspicion Index"*, *"AI-Identified Suspicious Area"*, and *"Requires Professional Review"* — never verdict language. |
| **No patient data leakage across dataset splits.** | Patient-level `StratifiedGroupKFold` splitting with an independent `assert_no_patient_leakage` gate (`src/data_split.py`). |
| **Imaging and molecular data are separate cohorts.** | The mammography (CBIS-DDSM) and transcriptomic (TCGA-BRCA) data are **NOT patient-matched**. The molecular panel is explicitly presented as reference context — never as the imaged patient's biology, and never as a combined diagnosis. |
| **Right to Erasure.** | A user-triggered hard purge deletes the stored object(s) *and* the metadata row, then re-reads both to verify nothing remains. |
| **Privacy by default.** | DICOM headers are stripped to technical-only metadata during preprocessing; the app runs fully offline in an ephemeral in-memory mode when no backend is configured. |

## Architecture at a glance

```
                 ┌──────────────────────────── Streamlit dashboard (app.py) ────────────────────────────┐
                 │                                                                                        │
   DICOM / PNG ──┼─▶ preprocessing ─▶ classifier (EfficientNet-B0) ─┐                                     │
   upload        │   (PHI stripped)     └─▶ Model Malignancy         │                                     │
                 │                          Suspicion Index          ├─▶ explain_case() ─▶ visual bundle:  │
                 │      └────────────▶ U-Net segmenter ─────────────┤     • Grad-CAM attention overlay    │
                 │                     (suspicious-region mask)      │     • numbered AI-Identified areas  │
                 │                                                   │     • cautious interpretation       │
                 │                                                   └─────────────┬──────────────────────┤
                 │                                                                 ▼                      │
   Expression ───┼─▶ molecular.analyze_sample() ─▶ pathway activity ──▶ side-by-side (NOT combined)       │
   matrix (TCGA) │   (cohort-relative z-scores)     (cell-cycle, DDR, ...)      research context          │
                 │                                                                                        │
                 │   PDF report (fpdf2, vector) ◀──────────────────────────┘                              │
                 │   Supabase / ephemeral store  +  Right-to-Erasure hard purge                           │
                 └────────────────────────────────────────────────────────────────────────────────────────┘
```

## The four phases

The project was built in **strict, isolated phases**; each was verified before the
next began.

- **Phase I — Imaging Baseline.** CBIS-DDSM preparation, DICOM decoding with VOI
  LUT / MONOCHROME1 handling, PHI stripping, patient-level splits, and an
  EfficientNet-B0 transfer-learning classifier.
  `src/preprocessing.py`, `src/data_split.py`, `src/dataset.py`, `src/model.py`, `src/train.py`
- **Phase II — Localization & Explainability.** U-Net segmentation of suspicious
  regions, OpenCV contour extraction / numbering, and Grad-CAM attention heatmaps,
  unified by a single `explain_case()` orchestration seam.
  `src/segmentation.py`, `src/localization.py`, `src/gradcam.py`, `src/explain.py`
- **Phase III — Interface & Erasure.** The Streamlit dashboard, a Supabase backend
  (PostgreSQL metadata + S3 objects) with an automatic offline ephemeral fallback,
  a vector PDF pre-analysis report, and the Right-to-Erasure hard-purge flow.
  `app.py`, `src/backend.py`, `src/report.py`, `.env.example`
- **Phase IV — Molecular Extension.** TCGA-BRCA transcriptomic analysis mapped to
  biological pathways, surfaced in the dashboard alongside the imaging heatmaps as
  **non-patient-matched** research context.
  `src/molecular.py`

## Repository layout

```
MamNexa-AI/
├── app.py                  # Streamlit dashboard (entry point)
├── CLAUDE.md               # Project directives & guardrails (source of truth)
├── README.md               # You are here
├── requirements.txt        # Pinned dependencies (one environment for all phases)
├── .env.example            # Supabase config template (copy to .env)
├── scripts/
│   ├── setup_env.sh        # venv bootstrap (Linux / macOS / Colab)
│   └── setup_env.bat        # venv bootstrap (Windows)
├── src/
│   ├── config.py           # Central config, mandated vocabulary, FORBIDDEN_PHRASES
│   ├── preprocessing.py    # DICOM → clean array + PHI stripping (Phase I)
│   ├── data_split.py       # Patient-level splits + leakage gate (Phase I)
│   ├── dataset.py          # tf.data pipeline (Phase I)
│   ├── model.py            # EfficientNet-B0 classifier (Phase I)
│   ├── train.py            # Training CLI (Phase I)
│   ├── segmentation.py     # U-Net (Phase II)
│   ├── localization.py     # OpenCV contour extraction / numbering (Phase II)
│   ├── gradcam.py          # Grad-CAM attention (Phase II)
│   ├── explain.py          # explain_case() orchestrator (Phase II)
│   ├── backend.py          # Supabase + ephemeral store, Right-to-Erasure (Phase III)
│   ├── report.py           # fpdf2 vector PDF report (Phase III)
│   └── molecular.py        # TCGA-BRCA pathway analysis (Phase IV)
├── tests/                  # pytest suite (runs on synthetic data — no downloads)
└── artifacts/              # Generated splits, exported models, sample report
```

## Quick start — run the Streamlit app locally

The demo runs **fully offline**. No dataset download, no cloud account, and no GPU
are required — if trained model weights are absent, the app loads the real
architectures with random weights and clearly flags that outputs verify the
*pipeline mechanics only* and are not clinically meaningful.

### 1. Prerequisites

- **Python 3.11 or 3.12** (a real CPython interpreter). On Windows, the Microsoft
  Store `python.exe` stub does not count — install from
  [python.org](https://www.python.org/downloads/windows/) or
  `winget install Python.Python.3.12`.

### 2. Create the environment and install dependencies

**Linux / macOS / Colab shell:**

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

**Windows (PowerShell or cmd):**

```bash
scripts\setup_env.bat
```

Or do it manually on any platform:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Launch the dashboard

With the virtual environment activated:

```bash
streamlit run app.py
```

If Python is not on your PATH (a common Windows setup), invoke the venv
interpreter directly:

```bash
.venv/Scripts/python.exe -m streamlit run app.py
```

Streamlit will open the app in your browser (default `http://localhost:8501`).

### 4. Use it

1. Upload a mammogram — **DICOM (`.dcm`)** is preferred (full PHI-stripping
   pipeline); **PNG/JPG** is accepted as a demo convenience.
2. Click **Run analysis** to compute the Model Malignancy Suspicion Index, the
   Grad-CAM attention overlay, and the numbered AI-Identified Suspicious Areas.
3. Explore the **Molecular Pathway Context** panel — use the built-in synthetic
   TCGA-BRCA demo cohort or upload your own expression matrix (genes × samples),
   pick a sample, and view cohort-relative pathway activity. This context is a
   **separate, non-patient-matched** research cohort by design.
4. **Download the PDF** pre-analysis report, optionally **store** the analysis to
   the backend, and exercise the **Right to Erasure** to purge it.

## Optional: enable the Supabase backend

By default the app runs in **ephemeral in-memory mode** — full store / fetch /
erase semantics work, but data lives only for the session. To persist to Supabase:

```bash
cp .env.example .env        # then edit .env with your project URL + key
```

Create, once in your Supabase project:

- a PostgreSQL table **`analyses`** (`id` text PK, `created_at` timestamptz,
  `image_path` text, `pdf_path` text, plus the non-identifying scoring columns), and
- a Storage bucket **`mamnexa-temp`** for *temporary* image/PDF objects.

Table/bucket names are configurable in `src/config.py` (`SUPABASE_TABLE`,
`SUPABASE_BUCKET`). Credentials are read from the environment only and are never
hard-coded; `.env` is gitignored and must never be committed.

## Right to Erasure

When a user requests deletion, MamNexa AI performs a **hard purge**, in order:

1. delete the stored object(s) (image + PDF) from the storage bucket,
2. delete the PostgreSQL metadata row, then
3. re-read both to **verify** nothing remains (`fully_erased = row_deleted and verified_gone`).

The result — including which artifacts were deleted and whether erasure was
verified — is surfaced back in the UI.

## Running the verification tests

The entire suite runs on **synthetic DICOMs and a synthetic expression matrix** —
it verifies pipeline *mechanics* and the guardrails, and needs no dataset download
or network access.

```bash
.venv/Scripts/python.exe -m pytest tests -q      # Windows (venv interpreter)
# or, with the environment activated on any platform:
python -m pytest tests -v
```

Each test file is also directly runnable (e.g.
`python -m pytest tests/test_molecular.py -v`). The guardrail tests specifically
assert that forbidden diagnostic phrasing can never reach the UI, the PDF, or the
molecular panel, and that the imaging↔molecular presentation stays explicitly
non-patient-matched.

## Datasets

- **CBIS-DDSM** (Curated Breast Imaging Subset of DDSM) — mammography, for the
  imaging pipeline.
- **TCGA-BRCA** — breast-cancer transcriptomics, for the molecular pathway module.

These are **separate cohorts and are not patient-matched.** No dataset is bundled
with this repository; the demo and tests use synthetic stand-ins so the project
runs without any download. Obtain the real datasets from their official sources
under their respective data-use terms.

## Status & maturity

This is a **research prototype for portfolio evaluation**, not clinical software.
The pipelines, interface, reporting, governance, and molecular analysis are
implemented and test-verified end to end **on synthetic data**. The neural
networks have **not** been trained on the real datasets here — with weights
absent, the app runs the architectures on random weights and says so plainly.
Nothing in this project should be used to make, support, or influence any clinical
decision.

## License

Licensed under the **Creative Commons Attribution-NonCommercial 4.0 International
License (CC BY-NC 4.0)**.

You are free to share and adapt the material for **non-commercial** purposes, with
attribution. **Commercial use is not permitted.** Full text:
<https://creativecommons.org/licenses/by-nc/4.0/>.

*MamNexa AI is not a medical device.*

## Citation & academic context

MamNexa AI was developed as a research portfolio project in support of graduate
study in **Medical Biotechnology**. If you reference this framework, please cite
the repository and the CC BY-NC 4.0 license. The design intentionally foregrounds
**explainability, scientific integrity, and data governance** as first-class
requirements rather than afterthoughts.
