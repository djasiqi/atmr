"""Format unifié « NOM Prénom » pour les noms patients facture clinique S2."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from application.invoices.invoice_line_description import (
    format_patient_display_name_nom_prenom,
    resolve_s2_clinic_line_patient_name,
)
from models.enums import InvoiceLineType
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


def test_material_delivery_institution_without_patient_has_no_client_label():
    """Livraison établissement : pas de « Client : » (contact ≠ bénéficiaire)."""
    client = SimpleNamespace(
        is_institution=True,
        user=SimpleNamespace(first_name="Clinique", last_name="X"),
    )
    booking = SimpleNamespace(
        customer_name="ALOISI Anne",
        client_id=42,
        mission_type="material_delivery",
        _get_institution_passenger_brief=lambda: None,
    )
    assert resolve_s2_clinic_line_patient_name(client, booking) == ""  # type: ignore[arg-type]


def test_material_delivery_for_institution_patient_shows_client():
    """Livraison pour un patient institution : « Client : NOM Prénom »."""
    client = SimpleNamespace(
        is_institution=True,
        user=SimpleNamespace(first_name="Clinique", last_name="X"),
    )
    booking = SimpleNamespace(
        customer_name="Eliane Francine STOFER-THOMI",
        client_id=42,
        mission_type="material_delivery",
        _get_institution_passenger_brief=lambda: {
            "institution_patient_id": 7,
            "first_name": "Eliane Francine",
            "last_name": "STOFER-THOMI",
        },
    )
    assert (
        resolve_s2_clinic_line_patient_name(client, booking)  # type: ignore[arg-type]
        == "STOFER-THOMI Eliane Francine"
    )


def test_merge_clears_patient_name_for_institution_delivery_without_patient():
    line = MagicMock()
    line.type = InvoiceLineType.MATERIAL_DELIVERY
    line.line_meta = {"patient_name": "ALOISI Anne"}
    booking = SimpleNamespace(
        customer_name="ALOISI Anne",
        client_id=99,
        scheduled_time=None,
        mission_type="material_delivery",
        _get_institution_passenger_brief=lambda: None,
    )
    client = SimpleNamespace(is_institution=True, user=None)
    merged = _merge_s2_clinic_line_meta_from_booking(line, booking, client)
    assert "patient_name" not in merged


def test_merge_keeps_patient_name_for_delivery_with_institution_patient():
    line = MagicMock()
    line.type = InvoiceLineType.MATERIAL_DELIVERY
    line.line_meta = {"patient_name": "ALOISI Anne"}
    booking = SimpleNamespace(
        customer_name="Eliane Francine STOFER-THOMI",
        client_id=99,
        scheduled_time=None,
        mission_type="material_delivery",
        _get_institution_passenger_brief=lambda: {
            "institution_patient_id": 7,
            "first_name": "Eliane Francine",
            "last_name": "STOFER-THOMI",
        },
    )
    client = SimpleNamespace(is_institution=True, user=None)
    merged = _merge_s2_clinic_line_meta_from_booking(line, booking, client)
    assert merged["patient_name"] == "STOFER-THOMI Eliane Francine"


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
