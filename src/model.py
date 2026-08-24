"""EfficientNet-B0 baseline classifier for MamNexa AI (Phase I).

Architecture
------------
Input is a float32 image in [0, 1] (exactly what ``preprocessing`` produces).
A ``Rescaling(255.0)`` layer converts to the [0, 255] range EfficientNet's
built-in normalization expects, so the exported model owns its whole
preprocessing contract and Phase III inference needs no extra scaling.

    inputs[0,1] -> Rescaling(255) -> EfficientNetB0(base) -> GAP
                -> Dropout -> Dense(1, sigmoid) = Model Malignancy Suspicion Index

Transfer learning is two-stage: train the new head with the base frozen, then
fine-tune the upper base layers at a low learning rate (BatchNorm layers stay
frozen to keep their ImageNet running statistics stable).

Clinical guardrails
-------------------
The single sigmoid output is the **Model Malignancy Suspicion Index** in [0, 1].
``interpret_suspicion`` is the ONLY sanctioned way to turn that number into
words: it emits cautious, review-oriented phrasing and never the forbidden
vocabulary ("confirmed cancer", "definitely", "safe tissue", ...).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

from .config import (
    DECISION_DISCLAIMER,
    DROPOUT_RATE,
    FORBIDDEN_PHRASES,
    IMAGE_SIZE,
    INITIAL_LR,
    NUM_CHANNELS,
    SUSPICION_INDEX_NAME,
    SUSPICION_THRESHOLD,
)

OUTPUT_LAYER_NAME = "malignancy_suspicion"

# Phrasing that must never appear in any human-facing model output. Sourced from
# config so every layer (model, explain, report, UI) enforces the identical list.
_FORBIDDEN_PHRASES = FORBIDDEN_PHRASES


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_efficientnet_b0(
    input_size: int = IMAGE_SIZE,
    dropout_rate: float = DROPOUT_RATE,
    weights: str | None = "imagenet",
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build the baseline model.

    Returns ``(model, base_model)``. The base is returned so the caller can
    freeze/unfreeze it across the two transfer-learning stages. On construction
    the base is frozen (stage-1 ready). Pass ``weights=None`` in tests to avoid
    the ImageNet weight download.
    """
    inputs = tf.keras.Input(shape=(input_size, input_size, NUM_CHANNELS), name="image_0_1")
    # [0,1] -> [0,255] for EfficientNet's internal Normalization layer.
    x = layers.Rescaling(255.0, name="to_efficientnet_range")(inputs)

    base_model = EfficientNetB0(
        include_top=False,
        weights=weights,
        input_tensor=x,
        pooling="avg",
    )
    base_model.trainable = False  # stage 1: head-only training

    x = layers.Dropout(dropout_rate, name="head_dropout")(base_model.output)
    outputs = layers.Dense(1, activation="sigmoid", name=OUTPUT_LAYER_NAME)(x)

    model = tf.keras.Model(inputs, outputs, name="mamnexa_efficientnetb0")
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float = INITIAL_LR) -> tf.keras.Model:
    """Compile for binary classification with metrics suited to imbalance.

    AUC-ROC and AUC-PR are the headline metrics; accuracy is reported but is a
    poor summary on skewed data.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def enable_fine_tuning(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    fine_tune_at: int,
    learning_rate: float,
) -> tf.keras.Model:
    """Stage 2: unfreeze base layers from ``fine_tune_at`` up, recompile at low LR.

    BatchNormalization layers are kept frozen throughout so their ImageNet
    running statistics are not corrupted by small medical-imaging batches.
    """
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return compile_model(model, learning_rate=learning_rate)


# ---------------------------------------------------------------------------
# Inference + guardrail interpretation
# ---------------------------------------------------------------------------
def suspicion_index(model: tf.keras.Model, image: np.ndarray) -> np.ndarray:
    """Return the Model Malignancy Suspicion Index/indices in [0, 1].

    Accepts a single (H, W, 3) image or a batch (N, H, W, 3), both float32 in
    [0, 1]. Returns a 1-D array of length N.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    preds = model.predict(arr, verbose=0)
    return preds.reshape(-1)


def interpret_suspicion(index: float, threshold: float = SUSPICION_THRESHOLD) -> dict[str, str]:
    """Turn a suspicion index into guardrail-compliant, review-oriented wording.

    This is the only sanctioned formatter for model output. It deliberately
    frames results as decision-support requiring professional review and never
    emits definitive diagnostic language.
    """
    index = float(index)
    assessment = (
        "AI-Identified Suspicious Area"
        if index >= threshold
        else "Benign-appearing finding (low model suspicion)"
    )
    result = {
        "index_name": SUSPICION_INDEX_NAME,
        "index_value": f"{index:.4f}",
        "assessment": assessment,
        "recommendation": "Requires Professional Review",
        "disclaimer": DECISION_DISCLAIMER,
    }
    _assert_guardrail_safe(result)
    return result


def _assert_guardrail_safe(result: dict[str, str]) -> None:
    """Fail loud if any output text contains forbidden diagnostic phrasing."""
    blob = " ".join(result.values()).lower()
    hits = [p for p in _FORBIDDEN_PHRASES if p in blob]
    if hits:
        raise ValueError(f"Guardrail violation - forbidden phrasing in output: {hits}")


def load_trained_model(path: str | Path) -> tf.keras.Model:
    """Load an exported .keras model (used by Phase III with @st.cache_resource)."""
    return tf.keras.models.load_model(str(path))
