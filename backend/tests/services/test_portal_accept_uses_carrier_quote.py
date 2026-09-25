"""Acceptation PORTAL DV : offered_amount = grille transporteur, jamais booking.amount."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from application.companies.accept_reservation import AcceptReservationUseCase
from services.legal.portal_double_validation import FLOW_DOUBLE_VALIDATION_V2


def test_portal_accept_falls_back_to_company_quote_not_booking_amount():
    booking = SimpleNamespace(
        id=46759,
        status="pending",
        company_id=None,
        client_id=10,
        amount=50.0,
        portal_contract_flow=FLOW_DOUBLE_VALIDATION_V2,
        pickup_location="GE",
        dropoff_location="HUG",
        pickup_lat=46.2,
        pickup_lon=6.1,
        dropoff_lat=46.19,
        dropoff_lon=6.14,
        scheduled_time=None,
        is_round_trip=False,
    )
    company = SimpleNamespace(id=1, is_approved=True, name="Emmenez")
    client = SimpleNamespace(id=10)
    offer = SimpleNamespace(id=99)
    offer_result = SimpleNamespace(ok=True, offer=offer, error=None, status_code=None)

    company_query = MagicMock()
    company_query.get.return_value = company
    client_query = MagicMock()
    client_query.get.return_value = client

    with (
        patch("models.Company") as CompanyModel,
        patch("models.client.Client") as ClientModel,
        patch(
            "services.auth.portal_phone_verification.is_portal_client",
            return_value=True,
        ),
        patch(
            "services.legal.portal_double_validation.booking_uses_double_validation",
            return_value=True,
        ),
        patch(
            "services.pricing.portal_carrier_ceiling.estimate_portal_carrier_offer_amount",
            return_value=40.0,
        ) as estimate_fn,
        patch(
            "services.legal.portal_carrier_offer.create_portal_carrier_offer",
            return_value=offer_result,
        ) as create_offer,
    ):
        CompanyModel.query = company_query
        ClientModel.query = client_query

        result = AcceptReservationUseCase().execute(
            booking, company_id=1, offered_amount=None
        )

    assert result.ok is True
    assert result.portal_offer_pending_client is True
    estimate_fn.assert_called_once()
    kwargs = create_offer.call_args.kwargs
    assert float(kwargs["offered_amount"]) == 40.0
    assert float(kwargs["offered_amount"]) != float(booking.amount)
