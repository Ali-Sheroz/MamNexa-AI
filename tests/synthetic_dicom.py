"""Shared test helper: build minimal uncompressed DICOM files with planted PHI.

Uncompressed (Explicit VR Little Endian) so no external codec is needed to
decode the pixels during tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

# Standard "Secondary Capture Image Storage" SOP Class UID, hardcoded because
# the named constant's import path is not stable across pydicom releases.
SC_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.7"

# PHI values planted into synthetic files; must never survive preprocessing.
PHI_VALUES = {
    "PatientName": "DOE^JANE",
    "PatientID": "SECRET-12345",
    "PatientBirthDate": "19700101",
    "InstitutionName": "General Hospital",
    "ReferringPhysicianName": "SMITH^JOHN",
    "AccessionNumber": "ACC-999",
}


def make_synthetic_dicom(
    path: Path,
    photometric: str = "MONOCHROME2",
    rows: int = 64,
    cols: int = 64,
    pixels: np.ndarray | None = None,
) -> Path:
    """Write a minimal uncompressed DICOM with planted PHI to ``path``."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SC_STORAGE_UID
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    for kw, val in PHI_VALUES.items():
        setattr(ds, kw, val)

    ds.Modality = "MG"
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.Rows = rows
    ds.Columns = cols
    ds.BurnedInAnnotation = "NO"

    if pixels is None:
        pixels = np.linspace(0, 4095, rows * cols, dtype=np.float32)
        pixels = pixels.reshape(rows, cols).astype(np.uint16)
    ds.PixelData = pixels.astype(np.uint16).tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path), write_like_original=False)
    return path
