"""Tests unitaires — noms de fichiers PDF mission institution."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from services.institutions.mission_report_context import (
    build_mission_pdf_filename,
    make_unique_mission_pdf_filenames,
)


def _transport(
    *,
    tid: int = 994,
    last_name: str = "STOFER-THOMI",
    first_name: str = "Eliane Francine",
    mission_date: date | None = date(2026, 6, 14),
):
    patient = SimpleNamespace(last_name=last_name, first_name=first_name)
    return SimpleNamespace(
        id=tid,
        patient=patient,
        external_reference=None,
        mission_date=mission_date,
    )


def test_build_mission_pdf_filename_audit_prioritizes_date_and_patient():
    tr = _transport()
    assert (
        build_mission_pdf_filename(tr, variant="audit")
        == "2026-06-14_STOFER-THOMI-Eliane-Francine_Rapport-mission.pdf"
    )


def test_build_mission_pdf_filename_operational():
    tr = _transport()
    assert (
        build_mission_pdf_filename(tr, variant="operational")
        == "2026-06-14_STOFER-THOMI-Eliane-Francine_Bon-transport.pdf"
    )


def test_build_mission_pdf_filename_disambiguate_adds_id_suffix():
    tr = _transport()
    assert (
        build_mission_pdf_filename(tr, variant="audit", disambiguate=True)
        == "2026-06-14_STOFER-THOMI-Eliane-Francine_Rapport-mission_994.pdf"
    )


def test_make_unique_mission_pdf_filenames_collision_only_when_needed():
    same_day = [
        _transport(tid=994),
        _transport(tid=995),
    ]
    names = make_unique_mission_pdf_filenames(same_day, variant="operational")
    assert names[994] == "2026-06-14_STOFER-THOMI-Eliane-Francine_Bon-transport.pdf"
    assert names[995] == "2026-06-14_STOFER-THOMI-Eliane-Francine_Bon-transport_995.pdf"
