"""Tests parsing opportunité + recipient_status."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from application.invoices.billing_opportunities import (
    build_opportunity_key,
    parse_billing_opportunity_key,
    resolve_recipient_status,
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
    with pytest.raises(ValueError):
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
