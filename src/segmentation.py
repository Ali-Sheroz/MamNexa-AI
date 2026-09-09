"""U-Net tissue segmentation for MamNexa AI (Phase II).

Purpose
-------
The Phase-I EfficientNet-B0 baseline answers *"how suspicious is this image?"*
(a single Model Malignancy Suspicion Index). Phase II answers *"where?"* — it
produces a per-pixel **AI-Identified Suspicious Area** probability map so a
reviewer can see which tissue drove the concern.

Contract
--------
* **Input:** the exact tensor ``preprocessing.dicom_to_clean_array`` yields —
  ``(IMAGE_SIZE, IMAGE_SIZE, 3)`` float32 in ``[0, 1]``. Sharing one input
  contract with the classifier means a single preprocessing path feeds both
  models; there is no divergent second pipeline.
* **Output:** a single-channel mask ``(IMAGE_SIZE, IMAGE_SIZE, 1)`` in ``[0, 1]``
  via a sigmoid. This is a *suspicion probability map*, NOT a segmentation of
  "cancer". Downstream (``localization``) turns it into numbered regions.

Clinical guardrails
-------------------
The mask is decision-support only. Nothing here confirms or rules out cancer;
the output layer is named ``suspicious_region_mask`` and every derived artifact
is framed as an "AI-Identified Suspicious Area / Requires Professional Review".

Architecture
------------
A classic symmetric U-Net (Ronneberger et al., 2015): a contracting encoder of
``UNET_DEPTH`` down-blocks (two 3x3 convs + BatchNorm + ReLU, then 2x2 max-pool),
a bottleneck, and an expanding decoder that up-samples and concatenates the
mirrored skip connection before two more convs. Filter counts start at
``UNET_BASE_FILTERS`` and double each level. Dice + binary-crossentropy loss is
used because suspicious regions are a small fraction of the breast, so raw
per-pixel BCE alone is dominated by the background.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .config import (
    IMAGE_SIZE,
    MASK_THRESHOLD,
    NUM_CHANNELS,
    SEG_OUTPUT_LAYER_NAME,
    UNET_BASE_FILTERS,
    UNET_DEPTH,
    UNET_LR,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU.

    'same' padding keeps spatial dims intact so encoder/decoder feature maps
    line up exactly for the skip-connection concatenations.
    """
    for i in (1, 2):
        x = layers.Conv2D(
            filters, 3, padding="same", use_bias=False,
            kernel_initializer="he_normal", name=f"{name}_conv{i}",
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
    return x


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_unet(
    input_size: int = IMAGE_SIZE,
    base_filters: int = UNET_BASE_FILTERS,
    depth: int = UNET_DEPTH,
    num_channels: int = NUM_CHANNELS,
) -> tf.keras.Model:
    """Build a symmetric U-Net that maps a [0,1] image to a [0,1] suspicion mask.

    ``input_size`` must be divisible by ``2**depth`` so every 2x max-pool halves
    a whole number of pixels and the decoder can restore the exact input size.
    """
    if input_size % (2 ** depth) != 0:
        raise ValueError(
            f"input_size={input_size} must be divisible by 2**depth={2 ** depth} "
            f"so the encoder/decoder resolutions stay aligned."
        )

    inputs = tf.keras.Input(shape=(input_size, input_size, num_channels), name="image_0_1")

    # --- Encoder: remember each pre-pool activation for the skip connection. ---
    skips: list[tf.Tensor] = []
    x = inputs
    for level in range(depth):
        filters = base_filters * (2 ** level)
        conv = _conv_block(x, filters, name=f"enc{level}")
        skips.append(conv)
        x = layers.MaxPooling2D(2, name=f"enc{level}_pool")(conv)

    # --- Bottleneck. ---
    x = _conv_block(x, base_filters * (2 ** depth), name="bottleneck")

    # --- Decoder: up-sample, concat mirrored skip, then convolve. ---
    for level in reversed(range(depth)):
        filters = base_filters * (2 ** level)
        x = layers.Conv2DTranspose(
            filters, 2, strides=2, padding="same", name=f"dec{level}_upconv",
        )(x)
        x = layers.Concatenate(name=f"dec{level}_concat")([x, skips[level]])
        x = _conv_block(x, filters, name=f"dec{level}")

    outputs = layers.Conv2D(
        1, 1, activation="sigmoid", name=SEG_OUTPUT_LAYER_NAME,
    )(x)

    return tf.keras.Model(inputs, outputs, name="mamnexa_unet")


# ---------------------------------------------------------------------------
# Loss / metrics (imbalance-aware: suspicious pixels are a small minority)
# ---------------------------------------------------------------------------
def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """Soft Dice overlap in [0, 1] (1.0 = perfect). Smoothing avoids /0 on empty masks."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """1 - Dice; directly optimizes region overlap."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Binary-crossentropy + Dice.

    BCE gives smooth pixel-wise gradients; Dice counters the heavy class
    imbalance (background >> suspicious tissue) that would otherwise let the
    model score well by predicting "nothing suspicious" everywhere.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


def iou_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """Soft Jaccard / intersection-over-union overlap in [0, 1] (1.0 = perfect)."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def compile_unet(model: tf.keras.Model, learning_rate: float = UNET_LR) -> tf.keras.Model:
    """Compile the U-Net with the imbalance-aware loss and overlap metrics.

    Dice and IoU are the headline metrics (both robust to class imbalance);
    precision/recall are reported for completeness.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=bce_dice_loss,
        metrics=[
            dice_coefficient,
            iou_coefficient,
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_mask(model: tf.keras.Model, image: np.ndarray) -> np.ndarray:
    """Return the suspicion probability mask for one image.

    Accepts an ``(H, W, 3)`` float32 image in ``[0, 1]`` (a
    ``preprocessing.dicom_to_clean_array`` output). Returns an ``(H, W)`` float32
    array in ``[0, 1]`` — the per-pixel AI-Identified Suspicious Area probability.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected a single (H, W, 3) image, got shape {arr.shape}")
    pred = model.predict(arr[None, ...], verbose=0)
    return pred[0, ..., 0].astype(np.float32)


def binarize_mask(mask: np.ndarray, threshold: float = MASK_THRESHOLD) -> np.ndarray:
    """Threshold a probability mask into a uint8 {0, 1} foreground mask."""
    return (np.asarray(mask, dtype=np.float32) >= threshold).astype(np.uint8)


class CheckpointLoadError(RuntimeError):
    """A U-Net checkpoint was present but could not be loaded (corrupt/incompatible).

    Raised instead of silently falling back to random weights.
    """


def load_trained_unet(path: str | Path) -> tf.keras.Model:
    """Load an exported U-Net, wiring up the custom loss/metric objects.

    Phase III loads this under ``@st.cache_resource`` for local inference. Raises
    :class:`FileNotFoundError` if the file is absent and :class:`CheckpointLoadError`
    if it exists but cannot be deserialized -- never a silent random-weight fallback.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Segmenter checkpoint not found: {path}")
    try:
        return tf.keras.models.load_model(
            str(path),
            custom_objects={
                "bce_dice_loss": bce_dice_loss,
                "dice_loss": dice_loss,
                "dice_coefficient": dice_coefficient,
                "iou_coefficient": iou_coefficient,
            },
        )
    except Exception as exc:  # noqa: BLE001 - re-raise as an explicit, typed failure
        raise CheckpointLoadError(
            f"Segmenter checkpoint at {path} exists but failed to load "
            f"({type(exc).__name__}: {exc}). It may be corrupt or built with an "
            f"incompatible TensorFlow/Keras version."
        ) from exc
