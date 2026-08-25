# MamNexa AI

**An open-source research framework developed in Python for exploring explainable AI methods in mammographic analysis and molecular breast-cancer research.**

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.11 / 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Status: Research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)](#status--maturity)

> ⚠️ ** Research and educational prototype only — NOT a medical device, diagnostic tool, or clinical decision-support system.
MamNexa AI is designed to demonstrate an explainable AI framework for breast-cancer research. It does not confirm, rule out, or diagnose cancer. Outputs are model-generated research signals intended solely for educational and research evaluation and must not be used to make or influence clinical decisions.
Any real-world mammographic finding requires interpretation by a qualified healthcare professional. See Clinical & scientific guardrails. See [Clinical & scientific guardrails](#clinical--scientific-guardrails).

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

MamNexa AI is a **python-based research and educational framework** that demonstrates how an explainable AI pipeline for breast-cancer research can be designed and implemented responsibly:

- **Mammographic Imaging Framework** —supports DICOM preprocessing with removal of patient-identifying metadata and implements an EfficientNet-B0 architecture for classification, U-Net for lesion segmentation/localization, and Grad-CAM for visual explainability. The current neural-network architectures have not yet been trained or validated on the intended real-world mammography dataset, so current outputs must not be interpreted as clinically meaningful predictions.
- **Molecular research framework** — implements a transcriptomic-analysis workflow designed for TCGA-BRCA gene-expression data and biologically relevant pathways, including cell-cycle regulation, DNA-damage response and repair, PI3K–AKT–mTOR signaling, hormone-receptor signaling, and HER2/ERBB signaling. The molecular and mammography datasets represent separate, non-patient-matched cohorts and are not combined to generate patient-level conclusions.
- **Interface and data governance** — a Streamlit-based research interface integrates the implemented components, supports generation of a preliminary-analysis PDF report, and includes privacy-oriented data handling and a **Right to Erasure**.

MamNexa AI is an academic research prototype, not a medical device, diagnostic system, or clinical decision-support tool.

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

Each software phase was implementation-tested on synthetic data before integration with the next phase; this does not constitute model or clinical validation.

- **Phase I — Imaging Baseline.** CBIS-DDSM preparation, DICOM decoding with VOI
  LUT / MONOCHROME1 handling, PHI stripping, patient-level splits, and an
  EfficientNet-B0 transfer-learning classifier.
  `src/preprocessing.py`, `src/data_split.py`, `src/dataset.py`, `src/model.py`, `src/train.py`
- **Phase II — Localization & Explainability.** U-Net segmentation of suspicious
  regions, OpenCV contour extraction / numbering, and Grad-CAM attention heatmaps,
  unified by a single `explain_case()` orchestration seam.
  `src/segmentation.py`, `src/localization.py`, `src/gradcam.py`, `src/explain.py`
- **Phase III — Interface & Erasure.** The Streamlit dashboard, a Supabase backend
  (PostgreSQL metadata + object storage) with an automatic offline ephemeral fallback,
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

- MamNexa AI is designed around two publicly available, de-identified research datasets representing separate components of the framework:

CBIS-DDSM (Curated Breast Imaging Subset of DDSM) — the intended real-world dataset for the mammographic imaging pipeline, including development and evaluation of breast-lesion classification, localization, and explainability components.

TCGA-BRCA (The Cancer Genome Atlas — Breast Invasive Carcinoma) — the intended molecular dataset for the transcriptomic and biological-pathway analysis module.

Important Dataset Distinction
CBIS-DDSM and TCGA-BRCA represent independent, non-patient-matched cohorts. Mammographic images from CBIS-DDSM are therefore not linked to the gene-expression profiles of TCGA-BRCA patients, and MamNexa AI does not treat these data as belonging to the same individuals or use them for patient-level multimodal prediction.

The current repository does not bundle either real-world dataset. The implemented pipelines and automated tests use synthetic stand-ins to verify software functionality without requiring large dataset downloads.

Full training and evaluation on the intended real-world datasets remain future experimental work. No real-data performance or clinical-validity claims are made in the current version.

Researchers wishing to reproduce or extend the real-data experiments should obtain the datasets independently from their official sources and comply with the applicable access conditions, licenses, and data-use requirements.
## Status & maturity

## Status & Maturity

MamNexa AI is a **research and educational prototype developed for academic and portfolio evaluation**. It is **not clinical software, a medical device, or a diagnostic system**.

The software pipelines, user interface, reporting workflow, data-governance safeguards, and molecular-analysis framework have been implemented and **verified end-to-end using synthetic test data**. This verification demonstrates software functionality and pipeline integration; it does **not** constitute model, diagnostic, or clinical validation.

The neural-network architectures have **not yet been trained or evaluated on the intended real-world datasets**. When trained weights are unavailable, the application may use randomly initialized weights solely to demonstrate pipeline functionality. Any resulting outputs are explicitly identified as **non-interpretable demonstration outputs**.

Accordingly, **no claims are made regarding model accuracy, sensitivity, specificity, diagnostic performance, or clinical validity**. Nothing produced by the current version of MamNexa AI should be used to diagnose, confirm, exclude, support, or influence any clinical decision.

### Current Development Status

| Component | Status |
|---|---|
| Software architecture | ✅ Implemented |
| Synthetic end-to-end testing | ✅ Completed |
| Safety and terminology guardrails | ✅ Implemented and tested |
| Patient-level splitting logic | ✅ Implemented and tested |
| DICOM preprocessing pipeline | ✅ Implemented |
| EfficientNet-B0 architecture | ✅ Implemented |
| U-Net architecture | ✅ Implemented |
| Grad-CAM explainability pipeline | ✅ Implemented |
| Streamlit research interface | ✅ Implemented |
| PDF reporting workflow | ✅ Implemented |
| Data-governance / erasure workflow | ✅ Implemented |
| Molecular-analysis framework | ✅ Implemented |
| Real CBIS-DDSM model training | ⏳ Future work |
| Real TCGA-BRCA analysis | ⏳ Future work |
| Real-data performance evaluation | ⏳ Not established |
| External dataset validation | ⏳ Not performed |
| Clinical validation | ❌ Outside current project scope |.

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
