# Model status & output honesty

This document explains exactly what the numbers and images produced by MamNexa AI
mean today, and why. It is referenced from `src/config.py` and is the canonical
description of the checkpoint-loading and output-provenance behavior.

## The short version

**The repository ships without trained model weights.** When you run the app or
the pipeline without supplying a checkpoint, the neural networks run on **random
(untrained) weights**. Every resulting score, heatmap, and region is a
**demonstration output with no predictive meaning** — it verifies that the
software pipeline works end to end, nothing more.

## The three checkpoint states

Each model (the EfficientNet-B0 classifier and the U-Net segmenter) is reported
independently as one of three states, defined in
[`src/model_status.py`](../src/model_status.py):

| State | When it happens | What outputs mean |
|---|---|---|
| `untrained` | No checkpoint file at the configured path | **Demonstration output — no predictive meaning.** Random weights. |
| `loaded` | A checkpoint file was found and loaded successfully | Produced from a checkpoint. The app does **not** verify how it was trained; treat performance as unestablished until independently validated. |
| `load_failed` | A checkpoint file exists but could not be deserialized | Surfaced as an **error**. The app builds an untrained model to stay alive, but the failure is shown — it is **never** silently masked as a normal demo. |

The overall analysis mode is the worst of the component states: a single
untrained or failed model makes the whole run a demonstration, because a combined
output built on an untrained component is not interpretable.

## Why a corrupt checkpoint is not silently replaced

A tempting shortcut is: "if loading fails, just fall back to random weights." We
deliberately do **not** do that. A user who placed a checkpoint on disk expects
that checkpoint to be used; silently substituting random weights would hide a
real problem (a corrupt file, a TensorFlow/Keras version mismatch) behind a
plausible-looking demo. Instead, `load_trained_model` /`load_trained_unet` raise
a typed `CheckpointLoadError`, and the UI shows the error.

- A **missing** file raises `FileNotFoundError` → the caller may fall back to a
  clearly-labeled untrained demo.
- A **present-but-corrupt** file raises `CheckpointLoadError` → surfaced as an
  error, not a demo.

## Checkpoint paths (single source of truth)

Training and inference share the same path constants so they can never drift into
a "training wrote X, the app expected Y" mismatch:

- `config.CLASSIFIER_CHECKPOINT` — default `artifacts/models/efficientnetb0_baseline.keras`
- `config.SEGMENTER_CHECKPOINT` — default `artifacts/models/unet.keras`

`src/train.py` exports to `CLASSIFIER_CHECKPOINT_NAME`, which is the exact
basename the dashboard loads. Both paths are overridable by environment variable
**without renaming files**:

```bash
MAMNEXA_CLASSIFIER_CHECKPOINT=/path/to/model.keras \
MAMNEXA_SEGMENTER_CHECKPOINT=/path/to/unet.keras \
  streamlit run app.py
```

> There is currently **no U-Net trainer** in the repository. `SEGMENTER_CHECKPOINT`
> is a documented placeholder path that simply will not exist until a segmentation
> trainer is added, so the segmenter runs untrained by default.

## How to move from demo to trained

See [TRAINING.md](TRAINING.md) for the classifier training path. Once a real
`.keras` checkpoint is present at the configured path, the app loads it
automatically and the status flips to `loaded`. Note again that `loaded` means
"a file loaded", not "a validated model" — real evaluation on held-out data is
required before any performance claim.
