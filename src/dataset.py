"""tf.data input pipeline for the EfficientNet-B0 baseline (Phase I).

Bridges the patient-level split manifests (artifacts/splits/*.csv) to batched,
model-ready tensors. Every image is loaded through
``preprocessing.dicom_to_clean_array`` so the PHI-stripping and normalization
guarantees established in Phase I apply uniformly to training and evaluation -
there is no second, divergent image path.

Images leave this pipeline as float32 in [0, 1]; the EfficientNet rescaling to
[0, 255] lives *inside* the model (see model.py), so the exported model accepts
the exact same [0, 1] arrays that preprocessing produces. That keeps training
and Phase III inference on one contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .config import BATCH_SIZE, IMAGE_PATH_COLUMNS, IMAGE_SIZE, NUM_CHANNELS, RANDOM_SEED
from .preprocessing import dicom_to_clean_array

AUTOTUNE = tf.data.AUTOTUNE

# Cached augmentation stack (built lazily so importing this module stays cheap).
_augmenter: tf.keras.Sequential | None = None


# ---------------------------------------------------------------------------
# Manifest -> resolved file paths
# ---------------------------------------------------------------------------
def _find_path_column(df: pd.DataFrame) -> str:
    for col in IMAGE_PATH_COLUMNS:
        if col in df.columns:
            return col
    raise KeyError(
        f"No image-path column found. Looked for {IMAGE_PATH_COLUMNS}; "
        f"columns present: {list(df.columns)}"
    )


def resolve_image_paths(df: pd.DataFrame, data_root: str | Path) -> list[str]:
    """Resolve each manifest row to an absolute DICOM path.

    CBIS-DDSM's recorded 'image file path' does not always match the file on
    disk after download, so if the literal path is missing we fall back to the
    first ``*.dcm`` under its parent folder. If nothing is found we still return
    the literal path - the loader will then raise a clear per-file error.
    """
    data_root = Path(data_root)
    path_col = _find_path_column(df)
    resolved: list[str] = []
    for rel in df[path_col].astype(str):
        literal = data_root / rel
        if literal.is_file():
            resolved.append(str(literal))
            continue
        parent = literal.parent
        if parent.is_dir():
            candidates = sorted(parent.rglob("*.dcm"))
            if candidates:
                resolved.append(str(candidates[0]))
                continue
        resolved.append(str(literal))
    return resolved


# ---------------------------------------------------------------------------
# Loading (Python -> graph via tf.py_function)
# ---------------------------------------------------------------------------
def _py_load_image(path_tensor: tf.Tensor) -> np.ndarray:
    """Eager loader: DICOM path -> (H, W, 3) float32 in [0, 1]. Fails loud."""
    path = path_tensor.numpy().decode("utf-8")
    clean = dicom_to_clean_array(path, size=IMAGE_SIZE)
    return clean.array.astype(np.float32)


def _load(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    img = tf.py_function(func=_py_load_image, inp=[path], Tout=tf.float32)
    img.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    label = tf.cast(label, tf.float32)
    return img, label


def _get_augmenter() -> tf.keras.Sequential:
    """Light, physiologically reasonable augmentation for mammograms.

    Horizontal flip is safe (laterality is a separate label); small rotation,
    zoom, and contrast jitter improve robustness without inventing anatomy.
    """
    global _augmenter
    if _augmenter is None:
        _augmenter = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal", seed=RANDOM_SEED),
                tf.keras.layers.RandomRotation(0.05, seed=RANDOM_SEED),
                tf.keras.layers.RandomZoom(0.1, seed=RANDOM_SEED),
                tf.keras.layers.RandomContrast(0.1, seed=RANDOM_SEED),
            ],
            name="augmentation",
        )
    return _augmenter


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def build_dataset(
    df: pd.DataFrame,
    data_root: str | Path,
    batch_size: int = BATCH_SIZE,
    training: bool = False,
    limit: int | None = None,
    seed: int = RANDOM_SEED,
) -> tf.data.Dataset:
    """Build a batched tf.data pipeline from a split manifest DataFrame.

    ``limit`` truncates to the first N rows (handy for a fast smoke test).
    Augmentation and shuffling are applied only when ``training`` is True.
    """
    if limit is not None:
        df = df.iloc[:limit].copy()
    if len(df) == 0:
        raise ValueError("Manifest is empty; nothing to build a dataset from.")

    paths = resolve_image_paths(df, data_root)
    labels = df["label"].astype(np.float32).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if training:
        augmenter = _get_augmenter()
        ds = ds.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
    return ds.prefetch(AUTOTUNE)


def compute_class_weights(df: pd.DataFrame) -> dict[int, float]:
    """Balanced class weights from the training labels (CBIS-DDSM is skewed)."""
    y = df["label"].astype(int).to_numpy()
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}
