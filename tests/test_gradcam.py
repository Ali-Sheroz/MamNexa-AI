"""Local verification tests for src/gradcam.py (Phase II explainability).

Builds the Phase-I classifier with random weights (no ImageNet download) and
checks the Grad-CAM mechanics end-to-end on an image derived from a synthetic
DICOM: locating the conv feature map, producing a normalized heatmap, upscaling
it, and blending a valid RGB attention overlay.

Run under pytest:  python -m pytest tests/test_gradcam.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

from src import gradcam as gc  # noqa: E402
from src import model as m  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src.config import IMAGE_SIZE, NUM_CHANNELS  # noqa: E402
from synthetic_dicom import make_synthetic_dicom  # noqa: E402


@pytest.fixture(scope="module")
def classifier():
    model, _ = m.build_efficientnet_b0(weights=None)
    m.compile_model(model)
    return model


@pytest.fixture(scope="module")
def clean_image(tmp_path_factory):
    path = make_synthetic_dicom(tmp_path_factory.mktemp("gc") / "cam.dcm")
    return pp.dicom_to_clean_array(path, size=IMAGE_SIZE).array


def test_finds_a_4d_conv_feature_map(classifier) -> None:
    layer = gc._find_last_conv_layer(classifier)
    assert len(layer.output.shape) == 4  # (batch, h, w, channels)


def test_heatmap_is_normalized_and_low_resolution(classifier, clean_image) -> None:
    heatmap = gc.make_gradcam_heatmap(classifier, clean_image)
    assert heatmap.ndim == 2
    # EfficientNet-B0 at 224px pools down to a 7x7 final feature map.
    assert heatmap.shape == (7, 7)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0


def test_resize_heatmap_matches_image_size(classifier, clean_image) -> None:
    heatmap = gc.make_gradcam_heatmap(classifier, clean_image)
    resized = gc.resize_heatmap(heatmap, size=IMAGE_SIZE)
    assert resized.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert resized.min() >= 0.0 and resized.max() <= 1.0


def test_overlay_is_valid_rgb_image(classifier, clean_image) -> None:
    overlay, heatmap_resized = gc.gradcam_overlay(classifier, clean_image)
    assert overlay.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert overlay.dtype == np.uint8
    assert heatmap_resized.shape == (IMAGE_SIZE, IMAGE_SIZE)


def test_overlay_blends_toward_the_heatmap(classifier) -> None:
    # A uniform gray base with a full-strength heatmap must shift color (JET).
    base = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=np.uint8)
    heatmap = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    blended = gc.overlay_heatmap(base, heatmap, alpha=0.5)
    assert blended.shape == base.shape
    assert np.any(blended != base)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
