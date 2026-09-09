"""Central configuration for MamNexa AI Phase I.

Keeping paths, image geometry, and the label mapping in one place ensures the
preprocessing module and the data-partitioning script agree on conventions
(image size, class encoding) so there is no silent drift between them.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Repository layout ------------------------------------------------------
# All paths are derived from this file's location so the code is CWD-agnostic.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"          # raw CBIS-DDSM lives here (gitignored)
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
SPLITS_DIR: Path = ARTIFACTS_DIR / "splits"     # generated train/val/test manifests
MODELS_DIR: Path = ARTIFACTS_DIR / "models"      # exported weights + metric reports

# --- Model checkpoints (SINGLE source of truth) -----------------------------
# Training (src/train.py) EXPORTS to these exact paths and the dashboard
# (app.py) LOADS from them, so the two can never drift into the "training wrote
# X, the app expected Y" mismatch. Override via environment variables to point
# the dashboard at checkpoints stored elsewhere WITHOUT renaming files:
#     MAMNEXA_CLASSIFIER_CHECKPOINT=/path/to/model.keras streamlit run app.py
# The classifier basename matches what train.py saves; there is no U-Net
# trainer yet (see docs/MODEL_STATUS.md), so the segmenter checkpoint is a
# documented placeholder path that simply will not exist until one is added.
CLASSIFIER_CHECKPOINT_NAME: str = "efficientnetb0_baseline.keras"
SEGMENTER_CHECKPOINT_NAME: str = "unet.keras"


def _checkpoint_path(env_var: str, default_name: str) -> Path:
    """Resolve a checkpoint path from an env override, else the default location."""
    override = os.environ.get(env_var)
    return Path(override) if override else MODELS_DIR / default_name


CLASSIFIER_CHECKPOINT: Path = _checkpoint_path(
    "MAMNEXA_CLASSIFIER_CHECKPOINT", CLASSIFIER_CHECKPOINT_NAME
)
SEGMENTER_CHECKPOINT: Path = _checkpoint_path(
    "MAMNEXA_SEGMENTER_CHECKPOINT", SEGMENTER_CHECKPOINT_NAME
)

# --- Image geometry ---------------------------------------------------------
# EfficientNet-B0's native input resolution is 224x224x3.
IMAGE_SIZE: int = 224
NUM_CHANNELS: int = 3

# --- Label conventions ------------------------------------------------------
# CBIS-DDSM 'pathology' values collapse to a binary malignancy target.
# BENIGN_WITHOUT_CALLBACK is a benign finding that did not warrant recall, so
# it maps to the benign (negative) class alongside BENIGN.
POSITIVE_CLASS: str = "MALIGNANT"
BINARY_LABEL_MAP: dict[str, int] = {
    "MALIGNANT": 1,
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
}
# Human-readable, guardrail-compliant class names (no "cancer"/"safe" wording).
CLASS_NAMES: dict[int, str] = {
    0: "Benign-appearing finding",
    1: "AI-Identified Suspicious Area",
}

# --- Split configuration ----------------------------------------------------
DEFAULT_TEST_SIZE: float = 0.15
DEFAULT_VAL_SIZE: float = 0.15
RANDOM_SEED: int = 42

# --- Training configuration (EfficientNet-B0 baseline) ----------------------
# The manifest column holding the DICOM path (checked in order).
IMAGE_PATH_COLUMNS: tuple[str, ...] = (
    "image_file_path",
    "image_path",
    "cropped_image_file_path",
)

BATCH_SIZE: int = 16              # conservative; safe on CPU-only Windows TF
INITIAL_EPOCHS: int = 10         # stage 1: train the new head, base frozen
FINE_TUNE_EPOCHS: int = 10       # stage 2: unfreeze upper base, low LR
INITIAL_LR: float = 1e-3
FINE_TUNE_LR: float = 1e-5
# EfficientNet-B0 has ~237 layers; unfreeze from here up during fine-tuning.
FINE_TUNE_AT: int = 150
DROPOUT_RATE: float = 0.3
EARLY_STOPPING_PATIENCE: int = 5

# Guardrail framing for every model output (see CLAUDE.md mandatory vocabulary).
SUSPICION_INDEX_NAME: str = "Model Malignancy Suspicion Index"
DECISION_DISCLAIMER: str = (
    "Research decision-support output only. This is NOT a diagnosis and does "
    "not confirm or rule out cancer. All findings Require Professional Review."
)
# Phrasing that must never appear in any human-facing output (UI / PDF / model).
# Centralized here so every layer (model.py, explain.py, report.py) enforces the
# same CLAUDE.md mandatory-vocabulary guardrail.
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "confirmed cancer",
    "definitely",
    "safe tissue",
    "no cancer",
    "cancer free",
    "cancer-free",
    "benign confirmed",
    "malignant confirmed",
)
# Threshold above which a case is labeled an AI-Identified Suspicious Area.
SUSPICION_THRESHOLD: float = 0.5

# --- Phase II: Localization & Explainability --------------------------------
# U-Net tissue segmentation. Its input matches the classifier's contract (a
# preprocessing.dicom_to_clean_array output): (IMAGE_SIZE, IMAGE_SIZE, 3) float
# in [0, 1]. Its output is a single-channel suspicion mask in [0, 1] -- an
# "AI-Identified Suspicious Area" probability map, never a tumor/cancer verdict.
UNET_BASE_FILTERS: int = 32       # encoder width at the top level (doubles each down-block)
UNET_DEPTH: int = 4               # down/up-sampling stages; 224 -> 14 at the bottleneck
UNET_LR: float = 1e-4
SEG_OUTPUT_LAYER_NAME: str = "suspicious_region_mask"
MASK_THRESHOLD: float = 0.5       # prob >= this belongs to an AI-Identified Suspicious Area

# OpenCV contour extraction / lesion numbering.
MIN_LESION_AREA_PX: int = 20      # drop specks smaller than this (noise) at 224x224
CONTOUR_COLOR_RGB: tuple[int, int, int] = (0, 255, 0)    # region boundary (RGB)
LABEL_COLOR_RGB: tuple[int, int, int] = (255, 255, 0)    # region-number text (RGB)
# Guardrail-compliant label for each detected region (numbered #1, #2, ...).
LESION_LABEL_PREFIX: str = "AI-Identified Suspicious Area"

# Grad-CAM attention overlay.
GRADCAM_OVERLAY_ALPHA: float = 0.4   # heatmap opacity when blended over the image

# --- Phase III: Interface, Backend & Erasure --------------------------------
# Supabase: PostgreSQL row per analysis + an S3 bucket for TEMPORARY image/PDF
# storage. Credentials come from the environment (never hard-coded); when they
# are absent the app runs in a local, in-memory ephemeral mode instead.
SUPABASE_URL_ENV: str = "SUPABASE_URL"
SUPABASE_KEY_ENV: str = "SUPABASE_KEY"
SUPABASE_TABLE: str = "analyses"
SUPABASE_BUCKET: str = "mamnexa-temp"

# Reporting (fpdf2). All copy uses the mandated cautious vocabulary.
REPORT_TITLE: str = "MamNexa AI - Pre-Analysis Research Report"
LICENSE_NOTICE: str = (
    "MamNexa AI - CC BY-NC 4.0 (non-commercial research use). Not a medical device."
)

# --- Phase IV: Molecular Extension (TCGA-BRCA transcriptomics) ---------------
# Single-sample pathway scoring is COHORT-RELATIVE: each gene is z-scored across
# the reference cohort, then a pathway's score is the mean z of its measured
# genes. |mean z| above this threshold is surfaced as a (cautious) research
# signal -- never a diagnosis.
PATHWAY_ACTIVITY_THRESHOLD: float = 0.5
MOLECULAR_TOP_GENES: int = 5          # contributing genes to surface per pathway
# A z-score needs spread across samples; a single sample cannot be z-scored. We
# require a minimum cohort so "cohort-relative" is meaningful, not noise.
MIN_COHORT_SAMPLES: int = 3
# Scientific-integrity guardrail: imaging (CBIS-DDSM) and transcriptomics
# (TCGA-BRCA) are SEPARATE, non-patient-matched cohorts. The molecular panel is
# reference context for research/education, never this patient's tumor biology.
MOLECULAR_DISCLAIMER: str = (
    "Molecular pathway context is computed from a separate TCGA-BRCA reference "
    "cohort and is NOT derived from this patient's tissue. It is not patient-matched "
    "to the uploaded image and does not describe or diagnose this patient. "
    "Research and education context only - Requires Professional Review."
)
