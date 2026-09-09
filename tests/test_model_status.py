"""Regression tests for src/model_status.py and checkpoint-load honesty.

These are TensorFlow-free (except the explicit corrupt-checkpoint case) so they
run fast in CI. They lock in the guarantees the repository review asked for:

* checkpoint path constants are shared between training and the dashboard, and
  are environment-overridable without renaming files;
* a MISSING checkpoint raises FileNotFoundError (caller may fall back to demo);
* a PRESENT-but-corrupt checkpoint raises CheckpointLoadError -- it is NEVER
  silently replaced with random weights;
* demonstration outputs carry the mandated "no predictive meaning" banner.

Run:  python -m pytest tests/test_model_status.py -v
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

from src import model_status as ms  # noqa: E402


# ---------------------------------------------------------------------------
# ModelStatus semantics
# ---------------------------------------------------------------------------
def test_untrained_status_is_demo_with_no_predictive_meaning() -> None:
    s = ms.ModelStatus(name="classifier", state=ms.STATE_UNTRAINED, checkpoint="x.keras")
    assert s.is_demo is True
    assert s.is_trained_source is False
    assert "no predictive meaning" in s.banner.lower()
    assert "demo" in s.score_label.lower()


def test_loaded_status_is_not_demo_but_provenance_unverified() -> None:
    s = ms.ModelStatus(name="classifier", state=ms.STATE_LOADED, checkpoint="x.keras")
    assert s.is_demo is False
    assert s.is_trained_source is True
    # Loaded != validated: the banner must still caveat provenance.
    assert "not" in s.banner.lower() and "verif" in s.banner.lower()


def test_load_failed_is_demo_and_never_claims_a_prediction() -> None:
    s = ms.ModelStatus(name="segmenter", state=ms.STATE_LOAD_FAILED, checkpoint="x.keras")
    assert s.is_demo is True
    assert s.is_trained_source is False


def test_overall_mode_reflects_worst_component() -> None:
    loaded = ms.ModelStatus("clf", ms.STATE_LOADED, "a")
    untrained = ms.ModelStatus("seg", ms.STATE_UNTRAINED, "b")
    failed = ms.ModelStatus("seg", ms.STATE_LOAD_FAILED, "b")

    assert "demonstration" in ms.overall_mode([untrained, untrained]).lower()
    assert "mixed" in ms.overall_mode([loaded, untrained]).lower()
    assert "error" in ms.overall_mode([loaded, failed]).lower()
    assert "loaded" in ms.overall_mode([loaded, loaded]).lower()


# ---------------------------------------------------------------------------
# Shared, overridable checkpoint paths
# ---------------------------------------------------------------------------
def test_checkpoint_paths_default_under_models_dir() -> None:
    from src import config

    assert config.CLASSIFIER_CHECKPOINT.name == config.CLASSIFIER_CHECKPOINT_NAME
    assert config.CLASSIFIER_CHECKPOINT.parent == config.MODELS_DIR
    assert config.SEGMENTER_CHECKPOINT.name == config.SEGMENTER_CHECKPOINT_NAME


def test_checkpoint_path_env_override(monkeypatch, tmp_path) -> None:
    target = tmp_path / "my_model.keras"
    monkeypatch.setenv("MAMNEXA_CLASSIFIER_CHECKPOINT", str(target))
    import src.config as config

    reloaded = importlib.reload(config)
    try:
        assert reloaded.CLASSIFIER_CHECKPOINT == target
    finally:
        monkeypatch.delenv("MAMNEXA_CLASSIFIER_CHECKPOINT", raising=False)
        importlib.reload(reloaded)  # restore defaults for other tests


def test_training_and_dashboard_share_the_classifier_name() -> None:
    """train.py must export to the exact basename the dashboard loads."""
    from src import config

    train_src = (Path(config.PROJECT_ROOT) / "src" / "train.py").read_text(encoding="utf-8")
    assert "CLASSIFIER_CHECKPOINT_NAME" in train_src
    # The old hard-coded, mismatched name must be gone.
    assert 'output_dir / "efficientnetb0_baseline.keras"' not in train_src


# ---------------------------------------------------------------------------
# Checkpoint loading: missing vs corrupt (TensorFlow required)
# ---------------------------------------------------------------------------
def test_missing_classifier_checkpoint_raises_filenotfound(tmp_path) -> None:
    from src import model as m

    with pytest.raises(FileNotFoundError):
        m.load_trained_model(tmp_path / "nope.keras")


def test_corrupt_classifier_checkpoint_raises_load_error_not_silent(tmp_path) -> None:
    from src import model as m

    bad = tmp_path / "corrupt.keras"
    bad.write_bytes(b"this is not a valid keras file")
    with pytest.raises(m.CheckpointLoadError):
        m.load_trained_model(bad)


def test_missing_segmenter_checkpoint_raises_filenotfound(tmp_path) -> None:
    from src import segmentation as seg

    with pytest.raises(FileNotFoundError):
        seg.load_trained_unet(tmp_path / "nope.keras")


def test_corrupt_segmenter_checkpoint_raises_load_error(tmp_path) -> None:
    from src import segmentation as seg

    bad = tmp_path / "corrupt.keras"
    bad.write_bytes(b"garbage")
    with pytest.raises(seg.CheckpointLoadError):
        seg.load_trained_unet(bad)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
