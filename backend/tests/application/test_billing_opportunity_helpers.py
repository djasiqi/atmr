"""Tests parsing opportunité + recipient_status."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from application.invoices import institution_patient_resolution
from application.invoices.billing_opportunities import (
    build_opportunity_key,
    parse_billing_opportunity_key,
    pick_canonical_billing_party_id,
    resolve_recipient_status,
)
from application.invoices.institution_patient_resolution import (
    resolve_missing_institution_patient_ids,
)
from models.enums import BillingPartyType


def test_parse_institution_patient_opportunity_key():
    p = parse_billing_opportunity_key("institution_patient:458|billing_party:901")
    assert p.subject_type == "institution_patient"
    assert p.subject_id == 458
    assert p.billing_party_id == 901
    assert p.subject_key == "institution_patient:458"
    assert p.opportunity_key == "institution_patient:458|billing_party:901"


def test_parse_client_opportunity_key():
    p = parse_billing_opportunity_key("client:99|billing_party:12")
    assert p.subject_type == "client"
    assert p.subject_id == 99
    assert build_opportunity_key("client:99", 12) == "client:99|billing_party:12"


def test_parse_invalid_opportunity_key():
    with pytest.raises(ValueError, match="billing_opportunity_key invalide"):
        parse_billing_opportunity_key("client:23")


def test_recipient_ready_patient_from_structured_address():
    bp = SimpleNamespace(
        type=BillingPartyType.PATIENT,
        display_name="HERRERO Nicolas",
        billing_address=None,
    )
    ip = SimpleNamespace(
        address="Chemin des Ramiers 9",
        postal_code="1222",
        city="Vésenaz",
    )
    assert (
        resolve_recipient_status(
            billing_party=bp, institution_patient=ip, display_name="HERRERO Nicolas"
        )
        == "ready"
    )


def test_recipient_missing_patient_address():
    bp = SimpleNamespace(
        type=BillingPartyType.PATIENT,
        display_name="HERRERO Nicolas",
        billing_address=None,
    )
    ip = SimpleNamespace(address="Chemin X", postal_code=None, city=None)
    assert (
        resolve_recipient_status(
            billing_party=bp, institution_patient=ip, display_name="HERRERO Nicolas"
        )
        == "missing_billing_address"
    )


def test_recipient_ready_non_patient_billing_party():
    bp = SimpleNamespace(
        type=BillingPartyType.CURATORSHIP,
        display_name="OPAD",
        billing_address="Rue du Stand 1, 1204 Genève",
    )
    assert resolve_recipient_status(billing_party=bp) == "ready"


def test_recipient_missing_non_patient_empty_address():
    bp = SimpleNamespace(
        type=BillingPartyType.INSURANCE,
        display_name="Assurance X",
        billing_address="  ",
    )
    assert resolve_recipient_status(billing_party=bp) == "missing_billing_address"


def test_canonical_billing_party_prefers_most_frequent():
    bookings = [
        SimpleNamespace(billing_party_id=901),
        SimpleNamespace(billing_party_id=902),
        SimpleNamespace(billing_party_id=901),
    ]
    assert pick_canonical_billing_party_id(bookings) == 901


def test_canonical_billing_party_ties_on_lowest_id():
    bookings = [
        SimpleNamespace(billing_party_id=902),
        SimpleNamespace(billing_party_id=901),
    ]
    assert pick_canonical_billing_party_id(bookings) == 901


def test_canonical_billing_party_none_when_absent():
    bookings = [SimpleNamespace(billing_party_id=None)]
    assert pick_canonical_billing_party_id(bookings) is None


def _booking(booking_id: int, **kwargs):
    return SimpleNamespace(
        id=booking_id,
        institution_patient_id=kwargs.get("institution_patient_id"),
        parent_booking_id=kwargs.get("parent_booking_id"),
        route_group_id=kwargs.get("route_group_id"),
    )


def test_resolve_missing_ids_fills_bookings(monkeypatch):
    monkeypatch.setattr(
        institution_patient_resolution,
        "build_institution_patient_mapping",
        lambda ids, **_: ({int(i): 458 for i in ids}, set()),
    )
    bookings = [_booking(1), _booking(2), _booking(3, institution_patient_id=77)]

    assert resolve_missing_institution_patient_ids(bookings, persist=False) == 2
    assert [b.institution_patient_id for b in bookings] == [458, 458, 77]


def test_resolve_missing_ids_leaves_ambiguous_untouched(monkeypatch):
    monkeypatch.setattr(
        institution_patient_resolution,
        "build_institution_patient_mapping",
        lambda ids, **_: ({1: 458}, {2}),
    )
    bookings = [_booking(1), _booking(2)]

    assert resolve_missing_institution_patient_ids(bookings, persist=False) == 1
    assert bookings[0].institution_patient_id == 458
    assert bookings[1].institution_patient_id is None


def test_resolve_missing_ids_noop_when_all_set(monkeypatch):
    def _fail(*_args, **_kwargs):  # pragma: no cover - ne doit jamais être appelé
        raise AssertionError("aucune requête attendue")

    monkeypatch.setattr(
        institution_patient_resolution, "build_institution_patient_mapping", _fail
    )
    bookings = [_booking(1, institution_patient_id=10)]

    assert resolve_missing_institution_patient_ids(bookings, persist=False) == 0
