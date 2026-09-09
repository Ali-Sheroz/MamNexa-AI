"""Local verification tests for the EfficientNet-B0 baseline (Phase I).

Covers three concerns:
  1. Architecture    - shapes, output-layer name, [0,1] input contract, transfer
                       -learning freeze/unfreeze (BatchNorm stays frozen).
  2. Data pipeline   - synthetic DICOMs flow through preprocessing into correctly
                       -shaped, in-range batches; balanced class weights.
  3. Clinical guard  - interpret_suspicion emits only cautious, review-oriented
                       vocabulary and the guardrail assert rejects forbidden text.

Models are built with ``weights=None`` so no ImageNet download is needed.

Run under pytest:  python -m pytest tests/test_model.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

# Make the project root and the tests dir importable (pytest or direct run).
_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

from src import model as m  # noqa: E402
from src.config import FINE_TUNE_AT, IMAGE_SIZE, NUM_CHANNELS, SUSPICION_THRESHOLD  # noqa: E402
from src.dataset import build_dataset, compute_class_weights  # noqa: E402
from synthetic_dicom import make_synthetic_dicom  # noqa: E402


# ---------------------------------------------------------------------------
# Shared, read-only model (built once; the freeze/unfreeze test uses its own).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def frozen_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    model, base_model = m.build_efficientnet_b0(weights=None)
    m.compile_model(model)
    return model, base_model


# ---------------------------------------------------------------------------
# 1. Architecture
# ---------------------------------------------------------------------------
def test_input_output_shapes_and_output_layer_name(frozen_model) -> None:
    model, _ = frozen_model
    assert model.input_shape == (None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    assert model.output_shape == (None, 1)
    # The single sigmoid unit is the Model Malignancy Suspicion Index.
    out_layer = model.get_layer(m.OUTPUT_LAYER_NAME)
    assert out_layer.activation.__name__ == "sigmoid"


def test_base_is_frozen_on_build(frozen_model) -> None:
    _, base_model = frozen_model
    assert base_model.trainable is False


def test_forward_pass_on_0_1_input_stays_in_probability_range(frozen_model) -> None:
    model, _ = frozen_model
    rng = np.random.default_rng(0)
    batch = rng.random((2, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)
    preds = model.predict(batch, verbose=0)
    assert preds.shape == (2, 1)
    assert preds.min() >= 0.0 and preds.max() <= 1.0


def test_enable_fine_tuning_unfreezes_upper_base_but_keeps_batchnorm_frozen() -> None:
    # Fresh model so we don't mutate the shared fixture's trainability.
    model, base_model = m.build_efficientnet_b0(weights=None)
    m.compile_model(model)
    m.enable_fine_tuning(model, base_model, FINE_TUNE_AT, learning_rate=1e-5)

    # Everything below the cut stays frozen.
    assert all(not layer.trainable for layer in base_model.layers[:FINE_TUNE_AT])

    # Above the cut: at least one ordinary layer trains, and every BatchNorm
    # layer (anywhere) remains frozen to protect its ImageNet statistics.
    upper = base_model.layers[FINE_TUNE_AT:]
    assert any(
        layer.trainable
        for layer in upper
        if not isinstance(layer, tf.keras.layers.BatchNormalization)
    )
    bn_layers = [
        layer for layer in base_model.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    assert bn_layers, "expected BatchNormalization layers in EfficientNet-B0"
    assert all(not bn.trainable for bn in bn_layers)


# ---------------------------------------------------------------------------
# 2. Inference helper + data pipeline
# ---------------------------------------------------------------------------
def test_suspicion_index_single_and_batch(frozen_model) -> None:
    model, _ = frozen_model
    single = np.zeros((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=np.float32)
    out_single = m.suspicion_index(model, single)
    assert out_single.shape == (1,)

    batch = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=np.float32)
    out_batch = m.suspicion_index(model, batch)
    assert out_batch.shape == (3,)
    assert out_batch.min() >= 0.0 and out_batch.max() <= 1.0


def test_build_dataset_yields_model_ready_batches(tmp_path: Path) -> None:
    # Four synthetic DICOMs on disk; manifest points at them by filename.
    for i in range(4):
        make_synthetic_dicom(tmp_path / f"img{i}.dcm")
    df = pd.DataFrame(
        {
            "image_file_path": [f"img{i}.dcm" for i in range(4)],
            "label": [0, 1, 0, 1],
        }
    )
    ds = build_dataset(df, tmp_path, batch_size=2, training=False)
    x, y = next(iter(ds))

    assert tuple(x.shape) == (2, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    assert tuple(y.shape) == (2,)
    assert x.dtype == tf.float32
    # The whole DICOM -> preprocessing -> tensor path stays in [0, 1].
    assert float(tf.reduce_min(x)) >= 0.0
    assert float(tf.reduce_max(x)) <= 1.0


def test_compute_class_weights_upweights_minority() -> None:
    df = pd.DataFrame({"label": [0] * 8 + [1] * 2})
    weights = compute_class_weights(df)
    assert set(weights) == {0, 1}
    # The rarer positive (suspicious) class must carry more weight.
    assert weights[1] > weights[0]


# ---------------------------------------------------------------------------
# 3. Clinical guardrails
# ---------------------------------------------------------------------------
def test_interpret_high_index_uses_cautious_suspicious_vocabulary() -> None:
    result = m.interpret_suspicion(0.92, threshold=SUSPICION_THRESHOLD)
    assert result["assessment"] == "AI-Identified Suspicious Area"
    assert result["recommendation"] == "Requires Professional Review"
    # No forbidden diagnostic phrasing anywhere in the output.
    blob = " ".join(result.values()).lower()
    for phrase in m._FORBIDDEN_PHRASES:
        assert phrase not in blob


def test_interpret_low_index_is_illustrative_not_definitive() -> None:
    result = m.interpret_suspicion(0.03, threshold=SUSPICION_THRESHOLD)
    assert result["assessment"] == "Assessment: Illustrative output only — untrained model"
    # Must never claim the tissue is safe / cancer-free.
    blob = " ".join(result.values()).lower()
    assert "safe tissue" not in blob
    assert "no cancer" not in blob


def test_guardrail_assert_raises_on_forbidden_phrase() -> None:
    poisoned = {
        "index_name": "Model Malignancy Suspicion Index",
        "assessment": "This is confirmed cancer",  # forbidden
        "recommendation": "Requires Professional Review",
    }
    with pytest.raises(ValueError):
        m._assert_guardrail_safe(poisoned)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
