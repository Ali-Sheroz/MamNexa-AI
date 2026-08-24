"""Local verification tests for src/preprocessing.py (Phase I).

Self-contained: synthesizes DICOM files in memory (uncompressed, so no external
codec is required) with deliberately planted PHI, then checks that preprocessing
produces a correctly normalized, PHI-free image.

Run under pytest:      python -m pytest tests/test_preprocessing.py -v
Or directly:           python tests/test_preprocessing.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make the project root and the tests dir importable (pytest or direct run).
_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

from src import preprocessing as pp  # noqa: E402
from src.config import IMAGE_SIZE  # noqa: E402
from synthetic_dicom import PHI_VALUES as _PHI_VALUES  # noqa: E402
from synthetic_dicom import make_synthetic_dicom  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_clean_array_shape_dtype_and_range(tmp_path: Path) -> None:
    path = make_synthetic_dicom(tmp_path / "a.dcm")
    clean = pp.dicom_to_clean_array(path, size=IMAGE_SIZE)

    assert clean.array.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert clean.array.dtype == np.float32
    assert clean.array.min() >= 0.0 and clean.array.max() <= 1.0
    # A non-constant gradient must produce real contrast, not a flat image.
    assert clean.array.max() - clean.array.min() > 0.5


def test_no_phi_in_clean_output(tmp_path: Path) -> None:
    path = make_synthetic_dicom(tmp_path / "b.dcm")
    clean = pp.dicom_to_clean_array(path)

    # No PHI keyword should be a metadata key...
    assert not (set(clean.safe_meta) & set(pp.PHI_TAG_KEYWORDS))
    # ...and no planted PHI value should appear anywhere in the metadata values.
    serialized = " ".join(str(v) for v in clean.safe_meta.values())
    for val in _PHI_VALUES.values():
        assert val not in serialized
    # Technical field is retained.
    assert clean.safe_meta.get("Modality") == "MG"


def test_deidentify_dataset_removes_phi_keeps_technical(tmp_path: Path) -> None:
    path = make_synthetic_dicom(tmp_path / "c.dcm")
    ds = pp.load_dicom(path)

    # Sanity: PHI is present before scrubbing.
    assert ds.PatientID == "SECRET-12345"

    clean = pp.deidentify_dataset(ds)
    for kw in _PHI_VALUES:
        assert kw not in clean, f"{kw} survived de-identification"
    assert clean.PatientIdentityRemoved == "YES"
    assert clean.DeidentificationMethod == pp.DEIDENTIFICATION_METHOD
    # Technical content preserved.
    assert clean.Modality == "MG"
    assert int(clean.Rows) == 64


def test_monochrome1_is_inverted(tmp_path: Path) -> None:
    """MONOCHROME1 must be polarity-flipped relative to MONOCHROME2.

    Uses real DICOM files (via make_synthetic_dicom) so that ``pixel_array`` has
    a valid transfer syntax; a bare header-less Dataset cannot be decoded.
    """
    raw = np.array([[0, 1], [2, 3]], dtype=np.uint16)

    p2 = make_synthetic_dicom(
        tmp_path / "mono2.dcm", photometric="MONOCHROME2", rows=2, cols=2, pixels=raw
    )
    p1 = make_synthetic_dicom(
        tmp_path / "mono1.dcm", photometric="MONOCHROME1", rows=2, cols=2, pixels=raw
    )

    out2 = pp.apply_luts(pp.load_dicom(p2))
    out1 = pp.apply_luts(pp.load_dicom(p1))

    assert np.allclose(out2, raw.astype(np.float32))              # unchanged
    assert np.allclose(out1, raw.max() - raw.astype(np.float32))  # inverted


def test_normalize_constant_image_is_zeros() -> None:
    const = np.full((10, 10), 7.0, dtype=np.float32)
    out = pp.normalize_intensity(const)
    assert np.all(out == 0.0)
    assert not np.isnan(out).any()


def test_clean_image_guard_rejects_phi_metadata() -> None:
    arr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        pp.CleanImage(array=arr, safe_meta={"PatientID": "oops"})


# ---------------------------------------------------------------------------
# Direct-run harness (no pytest required)
# ---------------------------------------------------------------------------
def _run_directly() -> int:
    tests = [
        test_normalize_constant_image_is_zeros,
        test_clean_image_guard_rejects_phi_metadata,
    ]
    tmp_tests = [
        test_clean_array_shape_dtype_and_range,
        test_no_phi_in_clean_output,
        test_deidentify_dataset_removes_phi_keeps_technical,
        test_monochrome1_is_inverted,
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for fn in tests:
            try:
                fn()
                print(f"PASS  {fn.__name__}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {fn.__name__}: {exc}")
        for fn in tmp_tests:
            try:
                fn(tmp)
                print(f"PASS  {fn.__name__}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_directly())
