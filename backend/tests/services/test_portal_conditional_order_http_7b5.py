"""7B.5 — HTTP accept : idempotence + concurrence + projection carrier_quote."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from application.companies.accept_reservation import AcceptReservationUseCase
from services.companies.booking_transfer_cache import attach_serialize_context_to_bookings
from services.legal.portal_double_validation import (
    ERROR_TRANSPORT_ALREADY_ASSIGNED,
    FLOW_CONDITIONAL_ORDER_V1,
)


def test_conditional_accept_idempotent_same_company_via_usecase():
    booking = SimpleNamespace(
        id=100,
        status="ACCEPTED",
        company_id=1,
        client_id=10,
        amount=40.0,
        portal_contract_flow=FLOW_CONDITIONAL_ORDER_V1,
    )
    company = SimpleNamespace(id=1, is_approved=True, name="Emmenez")
    client = SimpleNamespace(id=10)
    formed = SimpleNamespace(id=7, company_id=1, carrier_quote=Decimal("40.00"))
    form_result = SimpleNamespace(
        ok=True, contract=formed, error=None, status_code=None, idempotent_replay=True
    )

    with (
        patch("models.Company") as CompanyModel,
        patch("models.client.Client") as ClientModel,
        patch(
            "services.auth.portal_phone_verification.is_portal_client",
            return_value=True,
        ),
        patch(
            "services.legal.portal_double_validation.booking_uses_conditional_order",
            return_value=True,
        ),
        patch(
            "services.pricing.portal_carrier_ceiling.estimate_portal_carrier_offer_amount",
            return_value=40.0,
        ),
        patch(
            "services.legal.form_portal_transport_contract.form_portal_transport_contract_on_accept",
            return_value=form_result,
        ) as form_fn,
    ):
        CompanyModel.query.get.return_value = company
        ClientModel.query.get.return_value = client
        result = AcceptReservationUseCase().execute(booking, company_id=1)

    assert result.ok is True
    assert result.idempotent_replay is True
    form_fn.assert_called_once()


def test_conditional_accept_other_company_transport_already_assigned():
    booking = SimpleNamespace(
        id=100,
        status="ACCEPTED",
        company_id=1,
        client_id=10,
        amount=40.0,
        portal_contract_flow=FLOW_CONDITIONAL_ORDER_V1,
    )
    company = SimpleNamespace(id=64604, is_approved=True, name="A")
    client = SimpleNamespace(id=10)
    form_result = SimpleNamespace(
        ok=False,
        contract=None,
        error={
            "error": ERROR_TRANSPORT_ALREADY_ASSIGNED,
            "message": "Ce transport a déjà été accepté par une autre entreprise.",
        },
        status_code=409,
        idempotent_replay=False,
    )

    with (
        patch("models.Company") as CompanyModel,
        patch("models.client.Client") as ClientModel,
        patch(
            "services.auth.portal_phone_verification.is_portal_client",
            return_value=True,
        ),
        patch(
            "services.legal.portal_double_validation.booking_uses_conditional_order",
            return_value=True,
        ),
        patch(
            "services.pricing.portal_carrier_ceiling.estimate_portal_carrier_offer_amount",
            return_value=45.0,
        ),
        patch(
            "services.legal.form_portal_transport_contract.form_portal_transport_contract_on_accept",
            return_value=form_result,
        ),
    ):
        CompanyModel.query.get.return_value = company
        ClientModel.query.get.return_value = client
        result = AcceptReservationUseCase().execute(booking, company_id=64604)

    assert result.ok is False
    assert result.error["error"] == ERROR_TRANSPORT_ALREADY_ASSIGNED
    assert result.status_code == 409


def test_attach_carrier_quote_for_conditional_order_not_booking_amount(app):
    booking = SimpleNamespace(
        id=200,
        company_id=None,
        portal_contract_flow=FLOW_CONDITIONAL_ORDER_V1,
        amount=50.0,
    )
    with app.app_context():
        with patch(
            "services.pricing.portal_carrier_ceiling.estimate_portal_carrier_offer_amount",
            return_value=40.0,
        ):
            attach_serialize_context_to_bookings([booking], viewer_company_id=1)

    assert booking._company_suggested_amount == 40.0
    assert float(booking._company_suggested_amount) != float(booking.amount)


def test_attach_skips_legacy_booking_amount_path(app):
    booking = SimpleNamespace(
        id=201,
        company_id=None,
        portal_contract_flow="legacy",
        amount=50.0,
    )
    with app.app_context():
        attach_serialize_context_to_bookings([booking], viewer_company_id=1)
    assert getattr(booking, "_company_suggested_amount", None) is None
