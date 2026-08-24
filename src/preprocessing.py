"""DICOM preprocessing for MamNexa AI (Phase I).

Responsibilities
----------------
1. Load a DICOM study robustly (handles compressed transfer syntaxes via GDCM /
   pylibjpeg when installed).
2. Convert stored pixel values to a physically meaningful, display-consistent
   grayscale image (Modality LUT -> VOI LUT -> MONOCHROME1 inversion).
3. Normalize intensities to a model-agnostic float32 range in [0, 1].
4. Resize and expand to the 3-channel tensor EfficientNet-B0 expects.
5. **Mathematically strip HIPAA / PHI metadata.**

On "mathematically stripping" PHI
---------------------------------
DICOM interleaves Protected Health Information (patient name, ID, birth date,
addresses, private vendor tags, ...) with the pixel data inside one file. This
module offers two complementary guarantees:

* ``dicom_to_clean_array`` decouples the raw pixel matrix from the header
  entirely. What leaves this function is a NumPy array plus a hand-picked set of
  purely *technical* fields (rows, columns, modality, view). By construction the
  returned object carries **no** identifiers - the PHI never makes it past the
  function boundary.
* ``deidentify_dataset`` is for the case where a de-identified DICOM must be
  kept: it deletes the standard PHI tag set, strips private and overlay groups,
  and stamps the de-identification provenance flags.

NOTE ON BURNED-IN PHI: some mammograms embed patient text directly in the pixel
data (an overlay burned into the image). Header scrubbing cannot remove that;
``detect_possible_burned_in_annotation`` surfaces the DICOM flag so a caller can
route such images to manual review. Pixel-level redaction is out of scope here.

Clinical framing: outputs are research decision-support inputs only. Nothing in
this module confirms or rules out cancer.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.tag import Tag

# apply_voi_lut moved between pydicom versions; support both import locations.
try:  # pydicom 2.x
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except ImportError:  # pragma: no cover - pydicom 3.x fallback
    from pydicom.pixels import apply_voi_lut  # type: ignore

import cv2

from .config import IMAGE_SIZE

# ---------------------------------------------------------------------------
# PHI tag inventory
# ---------------------------------------------------------------------------
# Practical subset of the DICOM PS3.15 Basic Application Level Confidentiality
# Profile. Keyword-based so it is readable and version-stable. This is applied
# only in the "keep a de-identified DICOM" path; the array path drops the whole
# header regardless.
PHI_TAG_KEYWORDS: tuple[str, ...] = (
    "PatientName",
    "PatientID",
    "OtherPatientIDs",
    "OtherPatientIDsSequence",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientMotherBirthName",
    "PatientBirthName",
    "IssuerOfPatientID",
    "EthnicGroup",
    "PatientComments",
    "AdditionalPatientHistory",
    "MilitaryRank",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy",
    "OperatorsName",
    "PhysiciansOfRecord",
    "RequestingPhysician",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "StationName",
    "AccessionNumber",
    "StudyID",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "StudyTime",
    "SeriesTime",
    "AcquisitionTime",
    "ContentTime",
    "DeviceSerialNumber",
    "ProtocolName",
    "PerformedProcedureStepID",
    "RequestAttributesSequence",
    "ScheduledProcedureStepID",
)

# Technical fields that are safe to retain because they carry no identity but
# are useful for QA, stratification, and reproducibility.
SAFE_TECHNICAL_KEYWORDS: tuple[str, ...] = (
    "Modality",
    "Rows",
    "Columns",
    "PhotometricInterpretation",
    "BitsAllocated",
    "BitsStored",
    "PixelRepresentation",
    "BodyPartExamined",
    "ViewPosition",
    "ImageLaterality",
    "PresentationLUTShape",
)

DEIDENTIFICATION_METHOD = "MamNexa Phase I basic PHI tag removal"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class CleanImage:
    """A PHI-free preprocessed image plus non-identifying technical metadata."""

    array: np.ndarray  # float32, shape (H, W, 3), values in [0, 1]
    safe_meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Fail loud if an upstream change ever lets identifiers slip through.
        leaked = set(self.safe_meta) & set(PHI_TAG_KEYWORDS)
        if leaked:
            raise ValueError(f"PHI keyword(s) present in safe_meta: {sorted(leaked)}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_dicom(path: str | Path) -> Dataset:
    """Read a DICOM file and eagerly decode its pixel data.

    Raises a clear error if the transfer syntax cannot be decoded (usually a
    missing codec: install python-gdcm / pylibjpeg-* from requirements.txt).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DICOM file not found: {path}")
    ds = pydicom.dcmread(str(path), force=False)
    try:
        _ = ds.pixel_array  # trigger decode now so codec errors surface here
    except Exception as exc:  # noqa: BLE001 - re-raise with actionable guidance
        raise RuntimeError(
            f"Failed to decode pixel data for {path.name}. If the transfer syntax "
            f"is compressed, ensure python-gdcm / pylibjpeg-* are installed. "
            f"Original error: {exc}"
        ) from exc
    return ds


# ---------------------------------------------------------------------------
# Intensity handling
# ---------------------------------------------------------------------------
def apply_luts(ds: Dataset) -> np.ndarray:
    """Convert stored pixels to a display-consistent float array.

    Order of operations follows the DICOM display pipeline:
      1. VOI LUT / windowing (``apply_voi_lut`` also applies the Modality LUT
         or rescale slope/intercept when present).
      2. MONOCHROME1 inversion so that, in the output, higher value == brighter
         tissue (the MONOCHROME2 convention). Mammography sources mix both
         photometric interpretations; normalizing this here prevents polarity
         inconsistencies from reaching the model.
    """
    try:
        arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    except Exception:  # noqa: BLE001 - fall back to a plain rescale
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
    if photometric == "MONOCHROME1":
        # MONOCHROME1: minimum stored value displays as white -> invert.
        arr = float(arr.max()) - arr
    return arr


def normalize_intensity(
    arr: np.ndarray,
    clip_percentiles: tuple[float, float] | None = (1.0, 99.0),
) -> np.ndarray:
    """Scale an image to float32 in [0, 1].

    Percentile clipping (default 1st-99th) tames the extreme outliers common in
    mammography (dense implants, calcifications, unexposed borders) so the
    useful tissue contrast is not crushed into a narrow band by min-max scaling.
    A constant image is returned as all zeros rather than producing NaNs.
    """
    arr = arr.astype(np.float32)
    if clip_percentiles is not None:
        lo_p, hi_p = clip_percentiles
        lo = float(np.percentile(arr, lo_p))
        hi = float(np.percentile(arr, hi_p))
        if hi > lo:
            arr = np.clip(arr, lo, hi)

    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def resize_image(arr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize a 2-D image to ``size`` x ``size``.

    INTER_AREA is used when shrinking (the common case for full-field
    mammograms) because it avoids the aliasing that INTER_LINEAR introduces on
    heavy downscales; INTER_LINEAR is used when upscaling.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {arr.shape}")
    h, w = arr.shape
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    # cv2.resize takes (width, height).
    return cv2.resize(arr, (size, size), interpolation=interp)


def to_three_channel(arr: np.ndarray) -> np.ndarray:
    """Stack a single-channel image into 3 identical channels (H, W, 3)."""
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {arr.shape}")
    return np.stack([arr, arr, arr], axis=-1)


# ---------------------------------------------------------------------------
# PHI handling
# ---------------------------------------------------------------------------
def detect_possible_burned_in_annotation(ds: Dataset) -> bool:
    """Return True if the header suggests PHI may be burned into the pixels.

    This does not read the pixels; it reports the ``BurnedInAnnotation`` flag (or
    its absence). Callers should route flagged images to manual review because
    header scrubbing cannot remove text baked into the image itself.
    """
    flag = str(getattr(ds, "BurnedInAnnotation", "")).upper()
    return flag != "NO"  # "YES" or missing/unknown -> treat as possible


def deidentify_dataset(ds: Dataset) -> Dataset:
    """Return a copy of ``ds`` with PHI removed (for the keep-DICOM path).

    Removes the PHI tag inventory, all private tags, and overlay groups, then
    stamps the de-identification provenance flags required by DICOM PS3.15.
    """
    clean = copy.deepcopy(ds)

    for keyword in PHI_TAG_KEYWORDS:
        if keyword in clean:
            delattr(clean, keyword)

    # Private (odd-group) tags frequently carry vendor-specific identifiers.
    clean.remove_private_tags()

    # Overlay planes (groups 0x6000-0x60FF) can contain burned-in text overlays.
    for group in range(0x6000, 0x6100, 2):
        for elem in (0x3000, 0x0010, 0x0011, 0x0022, 0x0040, 0x0050):
            tag = Tag(group, elem)
            if tag in clean:
                del clean[tag]

    # Provenance flags.
    clean.PatientIdentityRemoved = "YES"
    clean.DeidentificationMethod = DEIDENTIFICATION_METHOD
    return clean


def extract_safe_metadata(ds: Dataset) -> dict[str, Any]:
    """Pull only non-identifying technical fields from a dataset."""
    meta: dict[str, Any] = {}
    for keyword in SAFE_TECHNICAL_KEYWORDS:
        if keyword in ds:
            value = getattr(ds, keyword)
            # Coerce pydicom value types to plain Python for safe serialization.
            meta[keyword] = value.value if hasattr(value, "value") else value
    return meta


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
def dicom_to_clean_array(
    path: str | Path,
    size: int = IMAGE_SIZE,
    clip_percentiles: tuple[float, float] | None = (1.0, 99.0),
) -> CleanImage:
    """Full pipeline: DICOM file -> PHI-free (H, W, 3) float32 image in [0, 1].

    The returned :class:`CleanImage` contains the pixel array and only technical
    metadata. Patient identifiers are never copied out of the header, so the
    result is safe to cache, upload to temporary storage, or hand to a model.
    """
    ds = load_dicom(path)
    gray = apply_luts(ds)
    gray = normalize_intensity(gray, clip_percentiles=clip_percentiles)
    gray = resize_image(gray, size=size)
    rgb = to_three_channel(gray).astype(np.float32)

    safe_meta = extract_safe_metadata(ds)
    safe_meta["possible_burned_in_annotation"] = detect_possible_burned_in_annotation(ds)
    return CleanImage(array=rgb, safe_meta=safe_meta)


def array_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a [0, 1] float image to uint8 [0, 255] for PNG export/QA."""
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def save_png(arr: np.ndarray, out_path: str | Path) -> Path:
    """Save a [0, 1] float image (2-D or 3-channel) to PNG. Header-free by design."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = array_to_uint8(arr)
    if img.ndim == 3:
        # cv2 writes BGR; our channels are identical grayscale, so order is moot.
        cv2.imwrite(str(out_path), img)
    else:
        cv2.imwrite(str(out_path), img)
    return out_path


# ---------------------------------------------------------------------------
# CLI: batch-preprocess a directory of DICOM files into PHI-free PNGs.
# ---------------------------------------------------------------------------
def _main() -> None:  # pragma: no cover - thin CLI wrapper
    import argparse

    from tqdm import tqdm

    parser = argparse.ArgumentParser(
        description="Preprocess DICOM files into PHI-free normalized PNGs."
    )
    parser.add_argument("input_dir", type=Path, help="Directory to search for *.dcm files")
    parser.add_argument("output_dir", type=Path, help="Where to write PNGs")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()

    dcm_files = sorted(args.input_dir.rglob("*.dcm"))
    if not dcm_files:
        print(f"No .dcm files found under {args.input_dir}")
        return

    flagged: list[str] = []
    for dcm in tqdm(dcm_files, desc="Preprocessing"):
        try:
            clean = dicom_to_clean_array(dcm, size=args.size)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {dcm.name}: {exc}")
            continue
        if clean.safe_meta.get("possible_burned_in_annotation"):
            flagged.append(dcm.name)
        rel = dcm.relative_to(args.input_dir).with_suffix(".png")
        save_png(clean.array, args.output_dir / rel)

    if flagged:
        print(
            f"\nWARNING: {len(flagged)} image(s) may contain burned-in PHI and "
            f"need manual review: {flagged[:10]}{' ...' if len(flagged) > 10 else ''}"
        )


if __name__ == "__main__":  # pragma: no cover
    _main()
