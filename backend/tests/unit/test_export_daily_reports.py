"""Tests unitaires — export ZIP des rapports de mission journaliers."""

from __future__ import annotations

import zipfile
from datetime import date
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from services.institutions.export_transports import build_daily_mission_reports_zip


def _transport(tid: int):
    patient = SimpleNamespace(last_name="DUPONT", first_name="Jean")
    return SimpleNamespace(
        id=tid,
        patient=patient,
        external_reference=None,
        mission_date=date(2026, 6, 14),
    )


def test_build_daily_mission_reports_zip_one_pdf_per_transport():
    institution = SimpleNamespace(id=1, name="Clinique test")
    requests = [_transport(101), _transport(205)]

    with (
        patch(
            "services.institutions.mission_report_context.collect_mission_report_context",
            return_value={},
        ),
        patch(
            "services.institutions.mission_report_pdf.build_mission_audit_report_pdf",
            return_value=b"%PDF-mock",
        ),
    ):
        zip_bytes = build_daily_mission_reports_zip(institution, requests)

    with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
        names = sorted(archive.namelist())
        assert names == [
            "2026-06-14_DUPONT-Jean_Rapport-mission.pdf",
            "2026-06-14_DUPONT-Jean_Rapport-mission_205.pdf",
        ]
        for name in names:
            assert archive.read(name) == b"%PDF-mock"
