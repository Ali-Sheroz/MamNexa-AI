"""Supabase backend for MamNexa AI (Phase III).

Persists a small, **non-identifying** metadata row per analysis (PostgreSQL) and
stores the associated image/PDF in an S3 bucket for *temporary* review. The
headline feature is the **Right to Erasure**: ``purge_analysis`` hard-deletes
both the storage objects and the database row, then verifies nothing remains.

Two modes, one API
------------------
* **Supabase mode** — used when ``SUPABASE_URL`` / ``SUPABASE_KEY`` are in the
  environment (or a client is injected). Talks to real PostgreSQL + S3 storage.
* **Ephemeral mode** — the automatic fallback when no credentials are present.
  Everything lives in process memory and vanishes when the app closes. This is
  not a stub: it implements the full store/fetch/purge semantics, so the demo
  runs offline and the erasure guarantees are exercised end-to-end in tests.

The public methods (`store_analysis`, `purge_analysis`, ...) are mode-agnostic;
only the low-level put/get/delete primitives branch on the mode. That keeps the
erasure logic — the part that must be correct — identical and tested in both.

Privacy note: by the time data reaches this layer it is already PHI-free
(Phase-I ``dicom_to_clean_array`` never copies identifiers out of the header).
We still store only a technical metadata subset and never the raw DICOM.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import (
    SUPABASE_BUCKET,
    SUPABASE_KEY_ENV,
    SUPABASE_TABLE,
    SUPABASE_URL_ENV,
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class StoredAnalysis:
    """A persisted analysis: its id, metadata row, and object storage paths."""

    analysis_id: str
    row: dict[str, Any]
    image_path: str | None = None
    pdf_path: str | None = None


@dataclass
class PurgeResult:
    """Outcome of a Right-to-Erasure purge."""

    analysis_id: str
    row_deleted: bool
    objects_deleted: list[str] = field(default_factory=list)
    verified_gone: bool = False

    @property
    def fully_erased(self) -> bool:
        return self.row_deleted and self.verified_gone


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
class SupabaseBackend:
    """Metadata + temporary object storage with a strict hard-purge erasure flow."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        table: str = SUPABASE_TABLE,
        bucket: str = SUPABASE_BUCKET,
        url: str | None = None,
        key: str | None = None,
    ) -> None:
        self.table_name = table
        self.bucket = bucket
        self._client = client or self._maybe_build_client(url, key)
        self.ephemeral = self._client is None

        # In-memory stores backing ephemeral mode.
        self._mem_rows: dict[str, dict[str, Any]] = {}
        self._mem_objects: dict[str, bytes] = {}

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def _maybe_build_client(url: str | None, key: str | None) -> Any | None:
        """Build a real Supabase client if creds are available, else return None.

        Absent credentials (or an uninstalled SDK) is not an error: the backend
        transparently falls back to ephemeral mode.
        """
        # Best-effort: honor a local .env if python-dotenv is installed. Never
        # overrides already-set environment variables and is a no-op if absent.
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:  # pragma: no cover - dotenv is in requirements.txt
            pass

        url = url or os.environ.get(SUPABASE_URL_ENV)
        key = key or os.environ.get(SUPABASE_KEY_ENV)
        if not url or not key:
            return None
        try:
            from supabase import create_client
        except ImportError:  # pragma: no cover - SDK is in requirements.txt
            return None
        return create_client(url, key)

    # -- id / path helpers ---------------------------------------------------
    @staticmethod
    def new_analysis_id() -> str:
        return uuid.uuid4().hex

    def _object_prefix(self, analysis_id: str) -> str:
        return f"{analysis_id}"

    def _image_object_path(self, analysis_id: str) -> str:
        return f"{self._object_prefix(analysis_id)}/image.png"

    def _pdf_object_path(self, analysis_id: str) -> str:
        return f"{self._object_prefix(analysis_id)}/report.pdf"

    # -- storage primitives (mode-specific) ----------------------------------
    def _put_object(self, path: str, data: bytes, content_type: str) -> None:
        if self.ephemeral:
            self._mem_objects[path] = bytes(data)
            return
        self._client.storage.from_(self.bucket).upload(
            path, bytes(data), {"content-type": content_type, "upsert": "true"}
        )

    def _remove_objects(self, paths: list[str]) -> list[str]:
        """Delete objects; return the paths that were actually present/removed."""
        if not paths:
            return []
        if self.ephemeral:
            removed = [p for p in paths if self._mem_objects.pop(p, None) is not None]
            return removed
        self._client.storage.from_(self.bucket).remove(paths)
        return list(paths)

    def _object_exists(self, path: str) -> bool:
        if self.ephemeral:
            return path in self._mem_objects
        # List the analysis prefix and check membership (Storage has no stat()).
        prefix, _, name = path.rpartition("/")
        listing = self._client.storage.from_(self.bucket).list(prefix)
        return any(entry.get("name") == name for entry in listing)

    # -- database primitives (mode-specific) ---------------------------------
    def _insert_row(self, row: dict[str, Any]) -> None:
        if self.ephemeral:
            self._mem_rows[row["id"]] = dict(row)
            return
        self._client.table(self.table_name).insert(row).execute()

    def _fetch_row(self, analysis_id: str) -> dict[str, Any] | None:
        if self.ephemeral:
            return self._mem_rows.get(analysis_id)
        resp = self._client.table(self.table_name).select("*").eq("id", analysis_id).execute()
        data = getattr(resp, "data", None) or []
        return data[0] if data else None

    def _delete_row(self, analysis_id: str) -> bool:
        if self.ephemeral:
            return self._mem_rows.pop(analysis_id, None) is not None
        self._client.table(self.table_name).delete().eq("id", analysis_id).execute()
        return True

    # -- public API ----------------------------------------------------------
    def store_analysis(
        self,
        metadata: dict[str, Any],
        image_png: bytes | None = None,
        pdf_bytes: bytes | None = None,
        analysis_id: str | None = None,
    ) -> StoredAnalysis:
        """Persist one analysis: upload image/PDF (if given) and insert the row.

        ``metadata`` must already be PHI-free; only technical/scoring fields are
        expected (suspicion index, region count, safe DICOM meta). The stored row
        records the object paths so ``purge_analysis`` can later erase them.
        """
        analysis_id = analysis_id or self.new_analysis_id()

        image_path = pdf_path = None
        if image_png is not None:
            image_path = self._image_object_path(analysis_id)
            self._put_object(image_path, image_png, "image/png")
        if pdf_bytes is not None:
            pdf_path = self._pdf_object_path(analysis_id)
            self._put_object(pdf_path, pdf_bytes, "application/pdf")

        row: dict[str, Any] = {
            "id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image_path": image_path,
            "pdf_path": pdf_path,
            **_sanitize_metadata(metadata),
        }
        self._insert_row(row)
        return StoredAnalysis(
            analysis_id=analysis_id, row=row, image_path=image_path, pdf_path=pdf_path
        )

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        return self._fetch_row(analysis_id)

    def purge_analysis(self, analysis_id: str) -> PurgeResult:
        """Right to Erasure: hard-delete the row AND its storage objects.

        Objects are removed first (so a mid-failure never leaves orphaned files
        that the row no longer references), then the row, then we re-read both to
        VERIFY the erasure actually took effect.
        """
        row = self._fetch_row(analysis_id)
        object_paths = (
            [p for p in (row.get("image_path"), row.get("pdf_path")) if p] if row else []
        )

        objects_deleted = self._remove_objects(object_paths)
        row_deleted = self._delete_row(analysis_id)

        # Verification pass: nothing may survive a purge.
        row_gone = self._fetch_row(analysis_id) is None
        objects_gone = all(not self._object_exists(p) for p in object_paths)

        return PurgeResult(
            analysis_id=analysis_id,
            row_deleted=row_deleted,
            objects_deleted=objects_deleted,
            verified_gone=row_gone and objects_gone,
        )


# ---------------------------------------------------------------------------
# Metadata hygiene
# ---------------------------------------------------------------------------
# Belt-and-suspenders: even though upstream data is PHI-free, never let a known
# identifier keyword be written to the database.
_BLOCKED_METADATA_KEYS = frozenset(
    {
        "PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime",
        "PatientAddress", "PatientTelephoneNumbers", "AccessionNumber",
        "ReferringPhysicianName", "InstitutionName", "InstitutionAddress",
        "StudyID", "OtherPatientIDs", "name", "patient", "patient_id",
    }
)


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop any reserved or identity-bearing keys before persisting."""
    reserved = {"id", "created_at", "image_path", "pdf_path"}
    clean: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if key in reserved or key in _BLOCKED_METADATA_KEYS:
            continue
        clean[key] = value
    return clean
