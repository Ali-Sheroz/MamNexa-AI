"""Local verification tests for src/segmentation.py (Phase II U-Net).

Verifies the architecture contract (shapes, [0,1] output, output-layer name),
the imbalance-aware Dice/IoU metrics, and single-image inference driven by a
synthetic DICOM run through the real preprocessing path. Built with random
weights (no training needed) since these check *mechanics*, not accuracy.

Run under pytest:  python -m pytest tests/test_segmentation.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

from src import preprocessing as pp  # noqa: E402
from src import segmentation as seg  # noqa: E402
from src.config import IMAGE_SIZE, NUM_CHANNELS, SEG_OUTPUT_LAYER_NAME  # noqa: E402
from synthetic_dicom import make_synthetic_dicom  # noqa: E402


@pytest.fixture(scope="module")
def unet():
    model = seg.build_unet()
    seg.compile_unet(model)
    return model


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
def test_unet_io_shapes_and_output_layer(unet) -> None:
    assert unet.input_shape == (None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    # Single-channel mask at full input resolution.
    assert unet.output_shape == (None, IMAGE_SIZE, IMAGE_SIZE, 1)
    out_layer = unet.get_layer(SEG_OUTPUT_LAYER_NAME)
    assert out_layer.activation.__name__ == "sigmoid"


def test_unet_is_compiled_with_dice(unet) -> None:
    assert unet.optimizer is not None
    metric_names = [m.name for m in unet.metrics] if unet.metrics else []
    # Names populate after a step; at minimum the loss/metric fns are wired.
    assert unet.loss is seg.bce_dice_loss


def test_build_unet_rejects_indivisible_input() -> None:
    # 220 is not divisible by 2**4=16, so encoder/decoder would misalign.
    with pytest.raises(ValueError):
        seg.build_unet(input_size=220, depth=4)


# ---------------------------------------------------------------------------
# Inference on a real (synthetic) DICOM
# ---------------------------------------------------------------------------
def test_predict_mask_shape_and_range(unet, tmp_path: Path) -> None:
    path = make_synthetic_dicom(tmp_path / "seg.dcm")
    clean = pp.dicom_to_clean_array(path, size=IMAGE_SIZE)

    prob = seg.predict_mask(unet, clean.array)
    assert prob.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert prob.dtype == np.float32
    assert prob.min() >= 0.0 and prob.max() <= 1.0

    binary = seg.binarize_mask(prob, threshold=0.5)
    assert binary.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 1})


def test_predict_mask_rejects_batched_input(unet) -> None:
    batched = np.zeros((2, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=np.float32)
    with pytest.raises(ValueError):
        seg.predict_mask(unet, batched)


# ---------------------------------------------------------------------------
# Overlap metrics behave sensibly
# ---------------------------------------------------------------------------
def _mask(pattern: np.ndarray) -> np.ndarray:
    return pattern.reshape(1, pattern.shape[0], pattern.shape[1], 1).astype(np.float32)


def test_dice_and_iou_perfect_overlap() -> None:
    m = _mask(np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    assert float(seg.dice_coefficient(m, m)) > 0.99
    assert float(seg.iou_coefficient(m, m)) > 0.99
    assert float(seg.dice_loss(m, m)) < 0.01


def test_dice_and_iou_disjoint_masks() -> None:
    a = _mask(np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    b = _mask(np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]))
    # No overlap -> only the smoothing term survives, so both are near zero.
    assert float(seg.dice_coefficient(a, b)) < 0.2
    assert float(seg.iou_coefficient(a, b)) < 0.2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
