"""Tests facturation retour par leg (accept_offer legacy paths)."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from application.institutions.accept_offer import AcceptOfferUseCase
from domain.billing.errors import BillingValidationError


@dataclass
class _Leg:
    sequence_index: int = 0
    is_return_stop: bool = False
    destination_billing_override: str | None = None
    pickup_location: str = "A"
    dropoff_location: str = "B"
    pickup_lat: float | None = None
    pickup_lng: float | None = None
    dropoff_lat: float | None = None
    dropoff_lng: float | None = None
    scheduled_time: object | None = None
    time_confirmed: bool = False
    booking_id: int | None = None


class _TransportRequest:
    def __init__(
        self,
        *,
        billing_intent: str = "institution",
        legs: list[_Leg] | None = None,
        is_round_trip: bool = False,
        return_to_institution: bool = True,
    ):
        self.billing_intent = billing_intent
        self.id = 4464
        self.institution = SimpleNamespace(id=251, name="Clinique test")
        self.legs = legs or []
        self.is_round_trip = is_round_trip
        self.return_to_institution = return_to_institution
        self.return_date = None
        self.return_time = None
        self.patient_id = 1
        self.mission_type = "patient_transport"
        self.delivery_description = None
        self.notes = None


def test_apply_effective_billing_uses_leg_override_not_global():
    uc = AcceptOfferUseCase()
    tr = _TransportRequest(
        billing_intent="institution",
        legs=[
            _Leg(sequence_index=0),
            _Leg(
                sequence_index=1,
                is_return_stop=True,
                destination_billing_override="patient",
            ),
        ],
    )
    booking = SimpleNamespace(billed_to_type="clinic", billed_to_company_id=99)
    client = SimpleNamespace(id=10)

    with patch.object(
        uc,
        "_resolve_billed_to_company_id_before_flush",
        return_value=None,
    ) as mock_resolve:
        effective = uc._apply_effective_billing_for_leg(
            booking,
            tr.legs[1],
            tr,  # type: ignore[arg-type]
            company_id=1,
            institution_client=client,  # type: ignore[arg-type]
        )

    assert effective == "patient"
    assert booking.billed_to_type == "patient"
    mock_resolve.assert_called_once()
    assert mock_resolve.call_args.kwargs["billed_to_type"] == "patient"


def test_apply_effective_billing_null_override_uses_global_intent():
    uc = AcceptOfferUseCase()
    tr = _TransportRequest(
        billing_intent="institution",
        legs=[_Leg(sequence_index=0), _Leg(sequence_index=1, is_return_stop=True)],
    )
    booking = SimpleNamespace(billed_to_type="patient", billed_to_company_id=None)

    with patch.object(
        uc,
        "_resolve_billed_to_company_id_before_flush",
        return_value=39947,
    ):
        effective = uc._apply_effective_billing_for_leg(
            booking,
            tr.legs[1],
            tr,  # type: ignore[arg-type]
            company_id=1,
            institution_client=SimpleNamespace(id=10),
        )

    assert effective == "institution"
    assert booking.billed_to_type == "clinic"
    assert booking.billed_to_company_id == 39947


def test_finalize_billing_raises_when_clinic_incomplete():
    uc = AcceptOfferUseCase()
    uc._resolve_billing_party = MagicMock()  # type: ignore[method-assign]
    booking = SimpleNamespace(
        id=39042,
        billed_to_type="clinic",
        billed_to_company_id=39947,
        billing_party_id=None,
    )
    tr = _TransportRequest()

    with pytest.raises(BillingValidationError, match="billing_party_id"):
        uc._finalize_booking_billing_resolution(
            booking,  # type: ignore[arg-type]
            tr,  # type: ignore[arg-type]
            company_id=1,
            effective_intent="institution",
            context="retour test",
        )


def test_finalize_billing_patient_return_non_strict():
    uc = AcceptOfferUseCase()
    uc._resolve_billing_party = MagicMock()  # type: ignore[method-assign]
    booking = SimpleNamespace(
        id=2,
        billed_to_type="patient",
        billed_to_company_id=None,
        billing_party_id=None,
    )
    tr = _TransportRequest(billing_intent="institution")

    uc._finalize_booking_billing_resolution(
        booking,  # type: ignore[arg-type]
        tr,  # type: ignore[arg-type]
        company_id=1,
        effective_intent="patient",
        context="retour patient",
    )
    uc._resolve_billing_party.assert_called_once()
    assert uc._resolve_billing_party.call_args.kwargs["strict"] is False
