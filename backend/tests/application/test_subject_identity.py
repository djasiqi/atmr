"""Tests pour resolve_subject_identity (C1 — jamais client:23 institutionnel)."""

from __future__ import annotations

from types import SimpleNamespace

from application.invoices.subject_identity import (
    booking_is_institution_origin,
    resolve_subject_identity,
)
from models.enums import BookingCreatedVia


def test_institution_patient_id_resolved():
    b = SimpleNamespace(
        id=10,
        client_id=23,
        institution_patient_id=458,
        created_via=None,
        client=None,
        _resolve_source_transport_request=lambda: None,
    )
    r = resolve_subject_identity(b)
    assert r.key == "institution_patient:458"
    assert r.status == "resolved"
    assert r.subject_type == "institution_patient"
    assert r.subject_id == 458
    assert r.carrier_client_id == 23


def test_classic_client_resolved():
    b = SimpleNamespace(
        id=11,
        client_id=99,
        institution_patient_id=None,
        created_via=None,
        client=SimpleNamespace(is_institution=False, linked_institution_id=None),
        _resolve_source_transport_request=lambda: None,
    )
    r = resolve_subject_identity(b)
    assert r.key == "client:99"
    assert r.status == "resolved"
    assert r.subject_type == "client"
    assert r.carrier_client_id == 99


def test_institution_origin_without_patient_never_falls_back_to_client():
    """C1 : booking institutionnel incomplet = singleton needs_review, pas client:23."""
    b = SimpleNamespace(
        id=12,
        client_id=23,
        institution_patient_id=None,
        created_via=BookingCreatedVia.INSTITUTION_PORTAL,
        client=SimpleNamespace(is_institution=True, linked_institution_id=5),
        _resolve_source_transport_request=lambda: None,
    )
    assert booking_is_institution_origin(b) is True
    r = resolve_subject_identity(b)
    assert r.key == "legacy-institution-booking:12"
    assert r.status == "needs_review"
    assert r.subject_type == "legacy_institution_booking"
    assert r.key != "client:23"


def test_institution_via_client_flag_without_portal():
    b = SimpleNamespace(
        id=13,
        client_id=23,
        institution_patient_id=None,
        created_via=None,
        client=SimpleNamespace(is_institution=True, linked_institution_id=None),
        _resolve_source_transport_request=lambda: None,
    )
    r = resolve_subject_identity(b)
    assert r.status == "needs_review"
    assert r.key.startswith("legacy-institution-booking:")


def test_unknown_booking_without_client():
    b = SimpleNamespace(
        id=14,
        client_id=None,
        institution_patient_id=None,
        created_via=None,
        client=None,
        _resolve_source_transport_request=lambda: None,
    )
    r = resolve_subject_identity(b)
    assert r.key == "unknown-booking:14"
    assert r.status == "needs_review"
