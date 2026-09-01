"""Tests validation facturation assign (sans auto-repair)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from domain.billing.errors import BillingValidationError
from services.billing.booking_billing_guard import (
    assert_non_patient_billing_complete,
    billing_type_normalized,
    validate_booking_billing_ready_for_write,
)


def test_patient_ok():
    booking = SimpleNamespace(billed_to_type="patient", billed_to_company_id=None)
    validate_booking_billing_ready_for_write(booking)


def test_clinic_requires_company_and_bp():
    booking = SimpleNamespace(
        id=1,
        billed_to_type="clinic",
        billed_to_company_id=39947,
        billing_party_id=None,
    )
    with pytest.raises(BillingValidationError, match="billing_party_id"):
        validate_booking_billing_ready_for_write(booking)


def test_clinic_requires_company_id():
    booking = SimpleNamespace(
        id=1,
        billed_to_type="clinic",
        billed_to_company_id=None,
        billing_party_id=11,
    )
    with pytest.raises(BillingValidationError, match="billed_to_company_id"):
        validate_booking_billing_ready_for_write(booking)


def test_clinic_complete_ok():
    booking = SimpleNamespace(
        id=39042,
        billed_to_type="clinic",
        billed_to_company_id=39947,
        billing_party_id=11,
    )
    validate_booking_billing_ready_for_write(booking)


def test_normalize_patient_crlf_not_inconsistent():
    assert billing_type_normalized("\r\npatient\r") == "patient"
    booking = SimpleNamespace(
        billed_to_type="\r\npatient\r",
        billed_to_company_id=None,
        billing_party_id=None,
    )
    assert_non_patient_billing_complete(
        booking, context="test", require_billing_party_for_clinic=True
    )
