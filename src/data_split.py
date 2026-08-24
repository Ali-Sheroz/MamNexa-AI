"""Patient-level dataset partitioning for CBIS-DDSM (Phase I).

Scientific-integrity mandate
----------------------------
Images from the same patient must NEVER appear in more than one partition. A
mammogram and its contralateral/other-view images share anatomy and acquisition
characteristics; letting them straddle train and test inflates metrics and
invalidates the evaluation. This module therefore splits at the *patient* level,
not the image level, and then independently *verifies* that no patient id is
shared across splits before writing anything.

Strategy
--------
``sklearn.model_selection.StratifiedGroupKFold`` produces folds that (a) keep
every group (patient) wholly within a single fold and (b) balance the class
distribution across folds. We carve the test partition as one fold, then repeat
on the remainder to carve validation, leaving the rest as train. The class
balance is preserved end-to-end while the patient grouping is never broken.

The malignancy label is derived from the CBIS-DDSM ``pathology`` column via
``config.BINARY_LABEL_MAP`` (MALIGNANT -> 1; BENIGN and BENIGN_WITHOUT_CALLBACK
-> 0).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import (
    BINARY_LABEL_MAP,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    RANDOM_SEED,
    SPLITS_DIR,
)

# Column-name aliases so we tolerate the raw CBIS-DDSM headers (which contain
# spaces) as well as already-normalized variants.
_PATIENT_ID_ALIASES = ("patient_id", "patientid", "patient")
_PATHOLOGY_ALIASES = ("pathology",)


# ---------------------------------------------------------------------------
# Loading & normalization
# ---------------------------------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and underscore-join column names in place."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _find_column(df: pd.DataFrame, aliases: tuple[str, ...], role: str) -> str:
    for alias in aliases:
        if alias in df.columns:
            return alias
    raise KeyError(
        f"Could not find the {role} column. Looked for {aliases}; "
        f"available columns are {list(df.columns)}"
    )


def load_metadata(csv_paths: list[str | Path]) -> pd.DataFrame:
    """Load and concatenate one or more CBIS-DDSM description CSVs.

    Adds a normalized ``label`` column (int) and standardizes the patient-id
    column name to ``patient_id``. Rows whose pathology is not recognized are
    dropped with a warning rather than silently mislabeled.
    """
    if not csv_paths:
        raise ValueError("At least one CSV path is required.")

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {path}")
        frames.append(_normalize_columns(pd.read_csv(path)))

    df = pd.concat(frames, ignore_index=True)

    patient_col = _find_column(df, _PATIENT_ID_ALIASES, "patient id")
    pathology_col = _find_column(df, _PATHOLOGY_ALIASES, "pathology")
    if patient_col != "patient_id":
        df = df.rename(columns={patient_col: "patient_id"})

    df["patient_id"] = df["patient_id"].astype(str).str.strip()

    pathology_norm = df[pathology_col].astype(str).str.strip().str.upper()
    df["label"] = pathology_norm.map(BINARY_LABEL_MAP)

    unmapped = int(df["label"].isna().sum())
    if unmapped:
        bad_values = sorted(pathology_norm[df["label"].isna()].unique())
        print(
            f"WARNING: dropping {unmapped} row(s) with unrecognized pathology "
            f"values {bad_values}. Update BINARY_LABEL_MAP if these are valid."
        )
        df = df[df["label"].notna()].copy()

    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def _safe_n_splits(desired_fraction: float, groups: np.ndarray, labels: np.ndarray) -> int:
    """Pick an n_splits that yields ~desired_fraction and is actually feasible.

    StratifiedGroupKFold needs n_splits <= number of groups, and each class must
    have at least n_splits groups to be represented in every fold. We clamp to
    the most restrictive of those bounds (and to a minimum of 2).
    """
    desired = max(2, round(1.0 / desired_fraction))
    n_groups = len(np.unique(groups))

    # Groups-per-class is the binding constraint for stratification feasibility.
    groups_per_class = [
        len(np.unique(groups[labels == c])) for c in np.unique(labels)
    ]
    min_groups_per_class = min(groups_per_class) if groups_per_class else n_groups

    return max(2, min(desired, n_groups, min_groups_per_class))


def _carve_one_fold(
    df: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (larger_remainder, one_fold ~= fraction) by patient.

    Returns row-disjoint, patient-disjoint frames.
    """
    groups = df["patient_id"].to_numpy()
    labels = df["label"].to_numpy()
    n_splits = _safe_n_splits(fraction, groups, labels)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    remainder_idx, fold_idx = next(
        sgkf.split(np.zeros(len(df)), y=labels, groups=groups)
    )
    return df.iloc[remainder_idx].copy(), df.iloc[fold_idx].copy()


def split_patient_level(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Partition into train/val/test with zero patient overlap.

    ``test_size`` and ``val_size`` are fractions of the *whole* dataset. Because
    the split is grouped and stratified, achieved fractions are approximate; the
    exact per-split composition is reported by :func:`summarize_splits`.
    """
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError(
            f"Invalid split sizes: test={test_size}, val={val_size} "
            f"(each must be in (0,1) and together < 1)."
        )

    # 1) Carve test off the whole set.
    trainval_df, test_df = _carve_one_fold(df, test_size, seed)

    # 2) Carve val off the remainder. Convert val_size to a fraction *of the
    #    remainder* so the final val share of the whole set stays ~val_size.
    val_fraction_of_remainder = val_size / (1.0 - test_size)
    train_df, val_df = _carve_one_fold(trainval_df, val_fraction_of_remainder, seed + 1)

    splits = {"train": train_df, "val": val_df, "test": test_df}
    assert_no_patient_leakage(splits)  # hard gate before anyone uses the result
    return splits


# ---------------------------------------------------------------------------
# Verification (the mandate) & reporting
# ---------------------------------------------------------------------------
def assert_no_patient_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Raise AssertionError if any patient id appears in more than one split.

    This is an independent check on the *output* - it does not trust the
    splitter. It is deliberately strict: any overlap aborts the pipeline.
    """
    names = list(splits)
    patient_sets = {name: set(splits[name]["patient_id"]) for name in names}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = patient_sets[a] & patient_sets[b]
            assert not overlap, (
                f"PATIENT-LEVEL LEAKAGE between '{a}' and '{b}': "
                f"{len(overlap)} shared patient id(s), e.g. {sorted(overlap)[:5]}"
            )


def summarize_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a per-split summary: #images, #patients, and class balance."""
    rows = []
    total_images = sum(len(d) for d in splits.values())
    for name, d in splits.items():
        n_images = len(d)
        n_pos = int((d["label"] == 1).sum())
        rows.append(
            {
                "split": name,
                "images": n_images,
                "patients": d["patient_id"].nunique(),
                "malignant": n_pos,
                "benign": n_images - n_pos,
                "malignant_frac": round(n_pos / n_images, 4) if n_images else 0.0,
                "image_share": round(n_images / total_images, 4) if total_images else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path = SPLITS_DIR) -> None:
    """Write each split to ``out_dir/<name>.csv`` with a ``split`` column added."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, d in splits.items():
        out = d.copy()
        out.insert(0, "split", name)
        out.to_csv(out_dir / f"{name}.csv", index=False)
    summarize_splits(splits).to_csv(out_dir / "split_summary.csv", index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(
        description="Create patient-level train/val/test splits for CBIS-DDSM."
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or more CBIS-DDSM description CSVs (e.g. mass + calc train sets).",
    )
    parser.add_argument("--out", type=Path, default=SPLITS_DIR)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    df = load_metadata(args.csv)
    print(f"Loaded {len(df)} labeled images from {df['patient_id'].nunique()} patients.")

    splits = split_patient_level(
        df, test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )
    write_splits(splits, out_dir=args.out)

    print("\nPatient-level split (verified: zero cross-split patient overlap):")
    print(summarize_splits(splits).to_string(index=False))
    print(f"\nManifests written to {Path(args.out).resolve()}")


if __name__ == "__main__":  # pragma: no cover
    _main()
