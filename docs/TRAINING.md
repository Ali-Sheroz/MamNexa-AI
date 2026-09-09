# Training on real data (future experimental work)

The repository is a **verified software prototype**; it does not bundle datasets
and makes no trained-model claims. This document is the practical route to real
training when you have the data and compute. Nothing here has been run at scale
in this project — treat it as the intended procedure, not a reproduced result.

> **Resources.** The classifier is EfficientNet-B0 (~5M params). Head-only
> training on CBIS-DDSM is feasible on CPU for a smoke test but slow; real
> fine-tuning wants a GPU. None of this is required to run the demo.

## 1. Obtain CBIS-DDSM

Download the Curated Breast Imaging Subset of DDSM from its official source (TCIA)
and comply with its data-use terms. Place it under `data/CBIS-DDSM/` (gitignored).
You need the mass/calcification description CSVs and the DICOM images they
reference.

## 2. Build patient-level splits (no leakage)

Scientific integrity requires that no patient appears in more than one split.
`src/data_split.py` performs `StratifiedGroupKFold` grouping by patient ID and
runs an independent `assert_no_patient_leakage` gate:

```bash
.venv/Scripts/python.exe -m src.data_split
```

This writes `artifacts/splits/{train,val,test}.csv`.

## 3. Train the EfficientNet-B0 baseline

```bash
# Fast end-to-end smoke test on a handful of images (verifies the pipeline):
.venv/Scripts/python.exe -m src.train --data-root data/CBIS-DDSM --limit 20 \
    --initial-epochs 1 --fine-tune-epochs 0 --batch-size 4

# A fuller run (two-stage transfer learning):
.venv/Scripts/python.exe -m src.train --data-root data/CBIS-DDSM \
    --initial-epochs 10 --fine-tune-epochs 10
```

Training is two-stage: train the new head with the ImageNet base frozen, then
fine-tune the upper base layers at a low learning rate (BatchNorm layers stay
frozen). Metrics are imbalance-aware (AUC-ROC, AUC-PR).

## 4. Where the checkpoint goes

`src/train.py` exports to `config.CLASSIFIER_CHECKPOINT_NAME` under
`artifacts/models/`, which is the **exact path the dashboard loads**. Nothing to
rename — launch the app and its classifier status flips from `untrained` to
`loaded`. See [MODEL_STATUS.md](MODEL_STATUS.md).

## 5. The segmenter

There is **no U-Net trainer yet**. The architecture, loss (BCE + Dice), and
metrics (Dice, IoU) exist in `src/segmentation.py`, and `load_trained_unet` will
load a checkpoint from `config.SEGMENTER_CHECKPOINT` if one is present, but adding
the training loop (with pixel-level masks) is future work.

## 6. `loaded` ≠ validated

A loaded checkpoint is not a validated model. Before any performance or clinical
claim: evaluate on the held-out test split, report AUC-ROC / AUC-PR with
confidence intervals, and — for any real-world use — pursue external validation
and clinical review, all of which are outside this project's scope.
