"""Format unifié « NOM Prénom » pour les noms patients facture clinique S2."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from application.invoices.invoice_line_description import (
    format_patient_display_name_nom_prenom,
    resolve_s2_clinic_line_patient_name,
)
from repositories.invoice_repository import _merge_s2_clinic_line_meta_from_booking


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Eliane Francine STOFER-THOMI", "STOFER-THOMI Eliane Francine"),
        ("Khalid ALAOUI", "ALAOUI Khalid"),
        ("ALEXANDRE Pierre", "ALEXANDRE Pierre"),
        ("AVERSANO Salvatore", "AVERSANO Salvatore"),
        ("BENDER-BITTAR Chantal-marie", "BENDER-BITTAR Chantal-Marie"),
    ],
)
def test_format_patient_display_name_nom_prenom(raw: str, expected: str):
    assert format_patient_display_name_nom_prenom(raw) == expected


def test_institution_client_uses_booking_customer_name_formatted():
    client = SimpleNamespace(
        is_institution=True,
        user=SimpleNamespace(
            first_name="Clinique les hauts d'anières",
            last_name="INSTITUTION",
        ),
    )
    booking = SimpleNamespace(
        customer_name="Eliane Francine STOFER-THOMI",
        client_id=42,
    )
    assert (
        resolve_s2_clinic_line_patient_name(client, booking)  # type: ignore[arg-type]
        == "STOFER-THOMI Eliane Francine"
    )


def test_individual_client_uses_nom_prenom_format():
    client = SimpleNamespace(
        is_institution=False,
        user=SimpleNamespace(
            first_name="Pierre",
            last_name="Alexandre",
            username="palex",
        ),
    )
    booking = SimpleNamespace(customer_name=None, client_id=7)
    assert (
        resolve_s2_clinic_line_patient_name(client, booking)  # type: ignore[arg-type]
        == "ALEXANDRE Pierre"
    )


def test_merge_normalizes_patient_name_snapshot():
    line = MagicMock()
    line.line_meta = {"patient_name": "Khalid ALAOUI"}
    booking = SimpleNamespace(
        customer_name="Khalid ALAOUI",
        client_id=99,
        scheduled_time=None,
    )
    client = SimpleNamespace(is_institution=True, user=None)
    merged = _merge_s2_clinic_line_meta_from_booking(line, booking, client)
    assert merged["patient_name"] == "ALAOUI Khalid"


def test_merge_overwrites_wrong_institution_snapshot_in_line_meta():
    line = MagicMock()
    line.line_meta = {"patient_name": "INSTITUTION Clinique les hauts d'anières"}
    booking = SimpleNamespace(
        customer_name="Khalid ALAOUI",
        client_id=99,
        scheduled_time=None,
    )
    client = SimpleNamespace(is_institution=True, user=None)
    merged = _merge_s2_clinic_line_meta_from_booking(line, booking, client)
    assert merged["patient_name"] == "ALAOUI Khalid"
