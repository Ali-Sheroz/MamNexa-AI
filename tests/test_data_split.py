"""Local verification tests for src/data_split.py (Phase I).

The central guarantee under test: **no patient id may appear in more than one
partition.** Also verifies label mapping, row conservation, and that the
independent leakage detector actually fails on a deliberately leaked split.

Run under pytest:      python -m pytest tests/test_data_split.py -v
Or directly:           python tests/test_data_split.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Make the project root importable whether run via pytest or directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_split as ds  # noqa: E402


def make_synthetic_metadata(n_patients: int = 24, images_per_patient: int = 3) -> pd.DataFrame:
    """Build a CBIS-DDSM-like frame: consistent per-patient pathology, multi-view.

    Uses the *raw* CBIS column spellings (with spaces) so we also exercise
    column normalization in load_metadata via the CSV round-trip.
    """
    rows = []
    for i in range(n_patients):
        pid = f"P_{i:05d}"
        # Alternate malignant / benign at the patient level for clean stratification.
        pathology = "MALIGNANT" if i % 2 == 0 else "BENIGN"
        for v in range(images_per_patient):
            rows.append(
                {
                    "patient_id": pid,
                    "pathology": pathology,
                    "image view": "CC" if v % 2 == 0 else "MLO",
                    "image file path": f"{pid}/img_{v}.dcm",
                }
            )
    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_load_metadata_maps_labels_and_drops_unknown(tmp_path: Path) -> None:
    df = make_synthetic_metadata()
    # Add a row with an out-of-vocabulary pathology that must be dropped.
    df = pd.concat(
        [df, pd.DataFrame([{"patient_id": "P_99999", "pathology": "FOO",
                            "image view": "CC", "image file path": "x.dcm"}])],
        ignore_index=True,
    )
    csv = _write_csv(df, tmp_path / "meta.csv")

    loaded = ds.load_metadata([csv])
    assert "label" in loaded.columns
    assert set(loaded["label"].unique()) <= {0, 1}
    # BENIGN_WITHOUT_CALLBACK maps to 0; MALIGNANT to 1.
    assert loaded[loaded["pathology"].str.upper() == "MALIGNANT"]["label"].eq(1).all()
    # The bogus 'FOO' row was dropped.
    assert "P_99999" not in set(loaded["patient_id"])


def test_split_has_zero_patient_leakage(tmp_path: Path) -> None:
    df = make_synthetic_metadata(n_patients=24, images_per_patient=3)
    csv = _write_csv(df, tmp_path / "meta.csv")
    loaded = ds.load_metadata([csv])

    splits = ds.split_patient_level(loaded, test_size=0.2, val_size=0.2, seed=42)

    # Pairwise patient-set intersections must all be empty.
    p = {k: set(v["patient_id"]) for k, v in splits.items()}
    assert p["train"] & p["val"] == set()
    assert p["train"] & p["test"] == set()
    assert p["val"] & p["test"] == set()

    # Every patient landed somewhere, and the union equals the input patients.
    assert p["train"] | p["val"] | p["test"] == set(loaded["patient_id"])


def test_split_conserves_all_rows(tmp_path: Path) -> None:
    df = make_synthetic_metadata()
    csv = _write_csv(df, tmp_path / "meta.csv")
    loaded = ds.load_metadata([csv])

    splits = ds.split_patient_level(loaded, test_size=0.2, val_size=0.2, seed=7)
    total = sum(len(v) for v in splits.values())
    assert total == len(loaded)


def test_both_classes_present_in_train(tmp_path: Path) -> None:
    df = make_synthetic_metadata()
    csv = _write_csv(df, tmp_path / "meta.csv")
    loaded = ds.load_metadata([csv])

    splits = ds.split_patient_level(loaded, test_size=0.2, val_size=0.2, seed=1)
    assert set(splits["train"]["label"].unique()) == {0, 1}


def test_leakage_detector_raises_on_shared_patient() -> None:
    """The independent guard must fail loudly if a patient straddles splits."""
    shared = pd.DataFrame(
        [{"patient_id": "P_00001", "label": 1},
         {"patient_id": "P_00002", "label": 0}]
    )
    other = pd.DataFrame([{"patient_id": "P_00001", "label": 1}])  # P_00001 leaks!
    with pytest.raises(AssertionError):
        ds.assert_no_patient_leakage({"train": shared, "test": other, "val": other.iloc[0:0]})


def test_write_splits_creates_manifests(tmp_path: Path) -> None:
    df = make_synthetic_metadata()
    csv = _write_csv(df, tmp_path / "meta.csv")
    loaded = ds.load_metadata([csv])
    splits = ds.split_patient_level(loaded, test_size=0.2, val_size=0.2, seed=3)

    out = tmp_path / "splits"
    ds.write_splits(splits, out_dir=out)
    for name in ("train", "val", "test", "split_summary"):
        assert (out / f"{name}.csv").is_file()
    # The written manifest carries the split label.
    train_written = pd.read_csv(out / "train.csv")
    assert (train_written["split"] == "train").all()


# ---------------------------------------------------------------------------
# Direct-run harness (no pytest required)
# ---------------------------------------------------------------------------
def _run_directly() -> int:
    tmp_tests = [
        test_load_metadata_maps_labels_and_drops_unknown,
        test_split_has_zero_patient_leakage,
        test_split_conserves_all_rows,
        test_both_classes_present_in_train,
        test_write_splits_creates_manifests,
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for fn in tmp_tests:
            try:
                fn(tmp)
                print(f"PASS  {fn.__name__}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {fn.__name__}: {exc}")
        try:
            test_leakage_detector_raises_on_shared_patient()
            print("PASS  test_leakage_detector_raises_on_shared_patient")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  test_leakage_detector_raises_on_shared_patient: {exc}")
    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_directly())
