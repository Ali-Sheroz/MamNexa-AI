"""Train & evaluate the EfficientNet-B0 baseline (Phase I).

Pipeline
--------
1. Load the patient-level split manifests (artifacts/splits/*.csv).
2. Build tf.data pipelines (augmented train; clean val/test).
3. Stage 1 - train the classifier head with the EfficientNet base frozen.
4. Stage 2 - fine-tune the upper base layers at a low learning rate.
5. Evaluate on the held-out test split with imbalance-aware metrics.
6. Export the model + weights and write a guardrail-compliant report.

Every result is framed as the "Model Malignancy Suspicion Index" and as
research decision-support requiring professional review - never as a diagnosis.

Example
-------
    python -m src.train --data-root data/CBIS-DDSM \
        --initial-epochs 10 --fine-tune-epochs 10

For a fast end-to-end smoke test on a handful of images:
    python -m src.train --data-root data/CBIS-DDSM --limit 20 \
        --initial-epochs 1 --fine-tune-epochs 0 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from .config import (
    BATCH_SIZE,
    CLASSIFIER_CHECKPOINT_NAME,
    DECISION_DISCLAIMER,
    EARLY_STOPPING_PATIENCE,
    FINE_TUNE_AT,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LR,
    INITIAL_EPOCHS,
    INITIAL_LR,
    MODELS_DIR,
    RANDOM_SEED,
    SPLITS_DIR,
    SUSPICION_INDEX_NAME,
    SUSPICION_THRESHOLD,
)
from .dataset import build_dataset, compute_class_weights
from .model import build_efficientnet_b0, compile_model, enable_fine_tuning

BANNER = "=" * 78


def _load_split(splits_dir: Path, name: str, limit: int | None) -> pd.DataFrame:
    path = Path(splits_dir) / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing split manifest: {path}. Run `python -m src.data_split` first."
        )
    df = pd.read_csv(path)
    if limit is not None:
        df = df.iloc[:limit].copy()
    return df


def _make_callbacks(output_dir: Path) -> list[tf.keras.callbacks.Callback]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "efficientnetb0_best.keras"),
            monitor="val_auc", mode="max", save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7,
        ),
    ]


def evaluate_on_test(
    model: tf.keras.Model, test_df: pd.DataFrame, test_ds: tf.data.Dataset
) -> dict:
    """Compute imbalance-aware metrics on the (unshuffled) test split.

    y_true is taken from the manifest in file order; test_ds is not shuffled, so
    predictions align row-for-row.
    """
    y_true = test_df["label"].astype(int).to_numpy()
    y_score = model.predict(test_ds, verbose=0).reshape(-1)
    y_pred = (y_score >= SUSPICION_THRESHOLD).astype(int)

    # roc_auc / average_precision require both classes to be present.
    both_classes = len(np.unique(y_true)) == 2
    report = {
        "n_test_images": int(len(y_true)),
        "positive_class": "AI-Identified Suspicious Area",
        "negative_class": "Benign-appearing finding",
        "suspicion_threshold": SUSPICION_THRESHOLD,
        "auc_roc": float(roc_auc_score(y_true, y_score)) if both_classes else None,
        "auc_pr": float(average_precision_score(y_true, y_score)) if both_classes else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["Benign-appearing", "AI-Identified Suspicious"],
            output_dict=True, zero_division=0,
        ),
    }
    return report


def write_report(report: dict, output_dir: Path) -> None:
    """Persist the metrics as JSON and a human-readable, guardrail-framed text."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    cr = report["classification_report"]
    lines = [
        BANNER,
        "MamNexa AI - EfficientNet-B0 Baseline - Evaluation Report",
        BANNER,
        f"!! {DECISION_DISCLAIMER}",
        "",
        f"Score reported: {SUSPICION_INDEX_NAME} (0-1), threshold "
        f"{report['suspicion_threshold']} -> 'AI-Identified Suspicious Area'.",
        f"Test images: {report['n_test_images']}",
        "",
        f"AUC-ROC (suspicious vs. benign-appearing): {report['auc_roc']}",
        f"AUC-PR: {report['auc_pr']}",
        "",
        "Confusion matrix [rows=true, cols=pred] "
        "(0=Benign-appearing, 1=AI-Identified Suspicious):",
        f"  {report['confusion_matrix']}",
        "",
        "Per-class (precision / recall / f1):",
        f"  Benign-appearing        : "
        f"{cr['Benign-appearing']['precision']:.3f} / "
        f"{cr['Benign-appearing']['recall']:.3f} / "
        f"{cr['Benign-appearing']['f1-score']:.3f}",
        f"  AI-Identified Suspicious: "
        f"{cr['AI-Identified Suspicious']['precision']:.3f} / "
        f"{cr['AI-Identified Suspicious']['recall']:.3f} / "
        f"{cr['AI-Identified Suspicious']['f1-score']:.3f}",
        "",
        "Interpretation is advisory only. Every case Requires Professional Review.",
        BANNER,
    ]
    text = "\n".join(lines)
    (output_dir / "metrics.txt").write_text(text)
    print("\n" + text)


def train(args: argparse.Namespace) -> None:
    tf.keras.utils.set_random_seed(args.seed)
    output_dir = Path(args.output_dir)

    print(BANNER)
    print("MamNexa AI - EfficientNet-B0 Baseline Training (Phase I)")
    print(f"!! {DECISION_DISCLAIMER}")
    print(BANNER)

    # --- Data ---------------------------------------------------------------
    train_df = _load_split(args.splits_dir, "train", args.limit)
    val_df = _load_split(args.splits_dir, "val", args.limit)
    test_df = _load_split(args.splits_dir, "test", args.limit)
    print(f"Images  -> train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    train_ds = build_dataset(train_df, args.data_root, args.batch_size, training=True, seed=args.seed)
    val_ds = build_dataset(val_df, args.data_root, args.batch_size, training=False)
    test_ds = build_dataset(test_df, args.data_root, args.batch_size, training=False)

    class_weights = compute_class_weights(train_df)
    print(f"Balanced class weights: {class_weights}")

    # --- Model --------------------------------------------------------------
    weights = None if args.weights == "none" else "imagenet"
    model, base_model = build_efficientnet_b0(weights=weights)
    compile_model(model, learning_rate=INITIAL_LR)
    callbacks = _make_callbacks(output_dir)

    # --- Stage 1: head only -------------------------------------------------
    print(f"\n[Stage 1] Training head (base frozen) for {args.initial_epochs} epoch(s)...")
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.initial_epochs,
        class_weight=class_weights, callbacks=callbacks, verbose=args.verbose,
    )

    # --- Stage 2: fine-tune -------------------------------------------------
    if args.fine_tune_epochs > 0:
        print(f"\n[Stage 2] Fine-tuning from layer {FINE_TUNE_AT} "
              f"for {args.fine_tune_epochs} epoch(s) at LR {FINE_TUNE_LR}...")
        enable_fine_tuning(model, base_model, FINE_TUNE_AT, FINE_TUNE_LR)
        model.fit(
            train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
            class_weight=class_weights, callbacks=callbacks, verbose=args.verbose,
        )

    # --- Export -------------------------------------------------------------
    # Export to the SHARED checkpoint name (src/config.py) that the dashboard
    # loads, so training output is picked up with no manual renaming.
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / CLASSIFIER_CHECKPOINT_NAME
    weights_path = model_path.with_suffix(".weights.h5")
    model.save(str(model_path))
    model.save_weights(str(weights_path))
    print(f"\nExported model  -> {model_path}")
    print(f"Exported weights-> {weights_path}")
    print(
        "This checkpoint is loaded automatically by app.py "
        f"(config.CLASSIFIER_CHECKPOINT). Place it under {MODELS_DIR} or set "
        "MAMNEXA_CLASSIFIER_CHECKPOINT to its path."
    )

    # --- Evaluate -----------------------------------------------------------
    print("\nEvaluating on held-out test split...")
    report = evaluate_on_test(model, test_df, test_ds)
    write_report(report, output_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the EfficientNet-B0 baseline classifier.")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    p.add_argument("--data-root", type=Path, required=True,
                   help="Root dir the manifest image paths are relative to.")
    p.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--initial-epochs", type=int, default=INITIAL_EPOCHS)
    p.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    p.add_argument("--limit", type=int, default=None,
                   help="Use only the first N rows of each split (smoke test).")
    p.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--verbose", type=int, default=2)
    return p


if __name__ == "__main__":  # pragma: no cover
    train(_build_arg_parser().parse_args())
