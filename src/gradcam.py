"""Grad-CAM attention heatmaps for MamNexa AI (Phase II).

Grad-CAM (Selvaraju et al., 2017) makes the Phase-I EfficientNet-B0 classifier
*explainable*: it back-propagates the Model Malignancy Suspicion Index to the
last convolutional feature map and weights each channel by its gradient, yielding
a coarse heatmap of the image regions that most increased the model's suspicion.
Overlaid on the (PHI-free) image, it shows a reviewer *why* the model responded.

Contract
--------
* Input image: ``(H, W, 3)`` float32 in ``[0, 1]`` (a
  ``preprocessing.dicom_to_clean_array`` output) — the same tensor the model was
  trained on. The model owns its own ``Rescaling`` layer, so no extra scaling.
* Output heatmap: float32 in ``[0, 1]`` at the conv feature-map resolution
  (7x7 for EfficientNet-B0 at 224px); ``resize_heatmap`` lifts it to image size.

Clinical guardrails
-------------------
A heatmap indicates model *attention*, not tissue pathology. It is decision
support requiring professional review; it neither confirms nor rules out cancer.
"""
from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

from .config import GRADCAM_OVERLAY_ALPHA, IMAGE_SIZE
from .preprocessing import array_to_uint8


# ---------------------------------------------------------------------------
# Locating the target feature map
# ---------------------------------------------------------------------------
def _find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Return the last layer that emits a 4-D (spatial) feature map.

    Name-independent so it works for EfficientNet-B0 (``top_activation``) or any
    other convolutional backbone without hard-coding a layer name.
    """
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except (AttributeError, RuntimeError):  # pragma: no cover - multi-node/edge layers
            continue
        if len(shape) == 4:
            return layer
    raise ValueError("No 4-D convolutional feature map found in the model.")


# ---------------------------------------------------------------------------
# Heatmap computation
# ---------------------------------------------------------------------------
def make_gradcam_heatmap(
    model: tf.keras.Model,
    image: np.ndarray,
    last_conv_layer_name: str | None = None,
) -> np.ndarray:
    """Compute the raw Grad-CAM heatmap for one image.

    Returns a float32 array in ``[0, 1]`` at the feature-map resolution. For the
    single-sigmoid suspicion head, the "score" being explained is the suspicion
    index itself (higher heatmap == more responsible for higher suspicion).
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]

    conv_layer = (
        model.get_layer(last_conv_layer_name)
        if last_conv_layer_name
        else _find_last_conv_layer(model)
    )
    grad_model = tf.keras.models.Model(model.inputs, [conv_layer.output, model.output])

    x = tf.convert_to_tensor(arr)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        # Single sigmoid unit -> explain the suspicion score directly.
        score = preds[:, 0]
    grads = tape.gradient(score, conv_out)

    # Channel importance = mean gradient over the spatial dims.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]  # drop batch -> (h, w, channels)
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)  # (h, w)

    heatmap = tf.nn.relu(heatmap)  # only regions that push suspicion UP
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))  # -> [0, 1]
    return heatmap.numpy().astype(np.float32)


def resize_heatmap(heatmap: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Bilinearly upscale a small heatmap to ``size`` x ``size``, clipped to [0, 1]."""
    resized = cv2.resize(
        np.asarray(heatmap, dtype=np.float32), (size, size), interpolation=cv2.INTER_LINEAR
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def overlay_heatmap(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = GRADCAM_OVERLAY_ALPHA,
) -> np.ndarray:
    """Blend an image-sized heatmap over a grayscale/RGB image (JET colormap).

    ``base_image`` is uint8, grayscale or RGB; ``heatmap`` is float ``[0, 1]`` at
    the same H x W. Returns an ``(H, W, 3)`` uint8 RGB overlay.
    """
    base = np.asarray(base_image)
    if base.ndim == 2:
        base = np.stack([base, base, base], axis=-1)
    base = base.astype(np.float32)

    heat_u8 = np.clip(np.asarray(heatmap, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    blended = (1.0 - alpha) * base + alpha * colored_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


def gradcam_overlay(
    model: tf.keras.Model,
    image: np.ndarray,
    base_image: np.ndarray | None = None,
    alpha: float = GRADCAM_OVERLAY_ALPHA,
    last_conv_layer_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """End-to-end: image -> (RGB uint8 attention overlay, image-sized heatmap [0,1]).

    If ``base_image`` is omitted, the overlay is drawn on the model input itself
    (converted from [0, 1] float to uint8).
    """
    arr = np.asarray(image, dtype=np.float32)
    size = arr.shape[0]
    heatmap = make_gradcam_heatmap(model, arr, last_conv_layer_name=last_conv_layer_name)
    heatmap_resized = resize_heatmap(heatmap, size=size)
    if base_image is None:
        base_image = array_to_uint8(arr)
    overlay = overlay_heatmap(base_image, heatmap_resized, alpha=alpha)
    return overlay, heatmap_resized
