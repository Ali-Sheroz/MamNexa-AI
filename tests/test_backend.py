"""Local verification tests for src/backend.py (Phase III storage + erasure).

These exercise the **ephemeral in-memory** backend (the automatic fallback when
no Supabase credentials are present), which implements the full store / fetch /
purge semantics. The headline guarantee under test is the **Right to Erasure**:
``purge_analysis`` must hard-delete both the metadata row and the stored objects
and then verify nothing survives.

Run under pytest:  python -m pytest tests/test_backend.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent))
sys.path.insert(0, str(_TESTS_DIR))

from src.backend import SupabaseBackend  # noqa: E402


@pytest.fixture()
def backend() -> SupabaseBackend:
    # No client and no creds -> ephemeral mode.
    return SupabaseBackend(url=None, key=None)


def test_defaults_to_ephemeral_without_credentials(backend) -> None:
    assert backend.ephemeral is True


def test_new_analysis_id_is_unique_hex(backend) -> None:
    a, b = backend.new_analysis_id(), backend.new_analysis_id()
    assert a != b
    assert len(a) == 32 and all(c in "0123456789abcdef" for c in a)


def test_store_and_get_roundtrip(backend) -> None:
    stored = backend.store_analysis(
        metadata={"suspicion_index": 0.87, "num_suspicious_regions": 2},
        image_png=b"\x89PNG-fake",
        pdf_bytes=b"%PDF-fake",
    )
    assert stored.image_path and stored.pdf_path
    assert stored.image_path.startswith(stored.analysis_id)

    row = backend.get_analysis(stored.analysis_id)
    assert row is not None
    assert row["id"] == stored.analysis_id
    assert row["image_path"] == stored.image_path
    assert row["pdf_path"] == stored.pdf_path
    assert row["suspicion_index"] == 0.87
    assert "created_at" in row


def test_store_without_artifacts_leaves_paths_none(backend) -> None:
    stored = backend.store_analysis(metadata={"suspicion_index": 0.1})
    assert stored.image_path is None and stored.pdf_path is None
    assert backend.get_analysis(stored.analysis_id) is not None


def test_store_sanitizes_phi_and_reserved_keys(backend) -> None:
    stored = backend.store_analysis(
        metadata={
            "PatientName": "DOE^JANE",      # PHI - must be dropped
            "PatientID": "123",             # PHI - must be dropped
            "id": "ATTACKER_OVERRIDE",      # reserved - must not override analysis id
            "created_at": "1999-01-01",     # reserved - must be backend-generated
            "suspicion_index": 0.5,         # legitimate technical field - kept
        }
    )
    row = backend.get_analysis(stored.analysis_id)
    assert "PatientName" not in row and "PatientID" not in row
    assert row["id"] == stored.analysis_id          # not the attacker's value
    assert row["created_at"] != "1999-01-01"
    assert row["suspicion_index"] == 0.5


def test_purge_hard_deletes_row_and_objects(backend) -> None:
    stored = backend.store_analysis(
        metadata={"suspicion_index": 0.9},
        image_png=b"img",
        pdf_bytes=b"pdf",
    )
    result = backend.purge_analysis(stored.analysis_id)

    assert result.fully_erased is True
    assert result.row_deleted is True
    assert set(result.objects_deleted) == {stored.image_path, stored.pdf_path}

    # Nothing may survive: row gone, objects gone.
    assert backend.get_analysis(stored.analysis_id) is None
    assert not backend._object_exists(stored.image_path)
    assert not backend._object_exists(stored.pdf_path)
    assert backend._mem_rows == {} and backend._mem_objects == {}


def test_purge_unknown_id_is_safe_and_not_fully_erased(backend) -> None:
    result = backend.purge_analysis("does-not-exist")
    assert result.row_deleted is False
    assert result.objects_deleted == []
    assert result.fully_erased is False


def test_purge_is_idempotent(backend) -> None:
    stored = backend.store_analysis(metadata={"x": 1}, image_png=b"i", pdf_bytes=b"p")
    first = backend.purge_analysis(stored.analysis_id)
    second = backend.purge_analysis(stored.analysis_id)
    assert first.fully_erased is True
    # Second purge finds nothing to delete but must not error.
    assert second.row_deleted is False
    assert second.objects_deleted == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
