"""7B.5 — conditional_order_v1 : caps, pool figé, formation atomique."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from services.legal.portal_channel_cancellation_caps import (
    ERROR_DIMENSION,
    ERROR_EXCEEDS_CAP,
    synthetic_test_channel_caps,
    validate_company_policy_against_channel,
)
from services.legal.portal_double_validation import (
    ERROR_CARRIER_NOT_IN_ORDER_POOL,
    ERROR_TRANSPORT_ALREADY_ASSIGNED,
    FLOW_CONDITIONAL_ORDER_V1,
)
from services.legal.portal_terms_catalog import (
    TERMS_OF_SERVICE_V21_SHA256,
    TRANSPORT_TERMS_V21_SHA256,
    PortalTermsActivationError,
    assert_activation_coordination,
    prepared_portal_terms_v21,
)


def test_terms_2_1_hashes_immutable():
    tos, transport = prepared_portal_terms_v21()
    assert tos.terms_version == "2.1"
    assert transport.terms_version == "2.1"
    assert tos.terms_hash == TERMS_OF_SERVICE_V21_SHA256
    assert transport.terms_hash == TRANSPORT_TERMS_V21_SHA256
    assert tos.status == "prepared"


def test_activation_flags_mutually_exclusive():
    with pytest.raises(PortalTermsActivationError):
        assert_activation_coordination(
            effective_version="1.0",
            double_validation_enabled=True,
            conditional_order_enabled=True,
        )
    with pytest.raises(PortalTermsActivationError):
        assert_activation_coordination(
            effective_version="2.1",
            double_validation_enabled=False,
            conditional_order_enabled=False,
        )
    assert_activation_coordination(
        effective_version="2.1",
        double_validation_enabled=False,
        conditional_order_enabled=True,
    )
    assert_activation_coordination(
        effective_version="2.0",
        double_validation_enabled=True,
        conditional_order_enabled=False,
    )


def test_channel_caps_company_below_and_equal_pass():
    channel = synthetic_test_channel_caps()
    below = {
        "enabled": True,
        "tiers": [{"type": "time", "hours_before": 24, "percent": 30}],
        "reason_overrides": {"NO_SHOW": {"billable": True}},
    }
    assert validate_company_policy_against_channel(
        company_policy=below, channel_body=channel
    ).ok
    equal = {
        "enabled": True,
        "tiers": [{"type": "time", "hours_before": 24, "percent": 50}],
    }
    assert validate_company_policy_against_channel(
        company_policy=equal, channel_body=channel
    ).ok
    zero = {"enabled": True, "tiers": [{"type": "time", "hours_before": 24, "percent": 0}]}
    assert validate_company_policy_against_channel(
        company_policy=zero, channel_body=channel
    ).ok


def test_channel_caps_company_above_rejected():
    channel = synthetic_test_channel_caps()
    above = {
        "enabled": True,
        "tiers": [{"type": "time", "hours_before": 24, "percent": 70}],
    }
    res = validate_company_policy_against_channel(
        company_policy=above, channel_body=channel
    )
    assert not res.ok
    assert res.error == ERROR_EXCEEDS_CAP


def test_channel_caps_extra_dimension_rejected():
    channel = synthetic_test_channel_caps()
    bad = {
        "enabled": True,
        "tiers": [{"type": "time", "hours_before": 24, "percent": 30}],
        "admin_cancellation_fee_chf": 25,
    }
    res = validate_company_policy_against_channel(
        company_policy=bad, channel_body=channel
    )
    assert not res.ok
    assert res.error == ERROR_DIMENSION
    assert "admin_cancellation_fee_chf" in (res.message or "")


def test_form_contract_rejects_carrier_outside_pool(app):
    from services.legal.form_portal_transport_contract import (
        form_portal_transport_contract_on_accept,
    )

    booking = SimpleNamespace(
        id=1,
        portal_contract_flow=FLOW_CONDITIONAL_ORDER_V1,
        status="PENDING",
        company_id=None,
        client_id=10,
        amount=50,
    )
    order = SimpleNamespace(
        eligible_carriers_snapshot=[
            {"company_id": 1, "legal_name": "A Sàrl"},
            {"company_id": 2, "legal_name": "B SA"},
        ],
        client_ceiling=Decimal("52.00"),
        channel_policy_version="v1",
        channel_policy_hash="abc",
        channel_policy_snapshot="caps",
        terms_of_service_version="2.1",
        terms_of_service_hash="t",
        transport_terms_version="2.1",
        transport_terms_hash="u",
    )

    with app.app_context():
        with (
            patch(
                "services.legal.form_portal_transport_contract.db"
            ) as mock_db,
            patch(
                "services.legal.form_portal_transport_contract.PortalTransportContractFormed"
            ) as MockFormed,
            patch(
                "services.legal.form_portal_transport_contract.PortalClientConditionalOrder"
            ) as MockOrder,
            patch(
                "services.legal.form_portal_transport_contract.booking_uses_conditional_order",
                return_value=True,
            ),
        ):
            mock_q = MagicMock()
            mock_q.filter_by.return_value.with_for_update.return_value.one.return_value = (
                booking
            )
            mock_db.session.query.return_value = mock_q
            MockFormed.query.filter_by.return_value.one_or_none.return_value = None
            MockOrder.query.filter_by.return_value.with_for_update.return_value.one_or_none.return_value = (
                order
            )

            result = form_portal_transport_contract_on_accept(
                booking=booking,
                company_id=99,  # D hors pool
                offered_amount=40,
            )
            assert not result.ok
            assert result.error["error"] == ERROR_CARRIER_NOT_IN_ORDER_POOL


def test_form_contract_idempotent_same_company(app):
    from services.legal.form_portal_transport_contract import (
        form_portal_transport_contract_on_accept,
    )

    booking = SimpleNamespace(
        id=1,
        portal_contract_flow=FLOW_CONDITIONAL_ORDER_V1,
        status="ACCEPTED",
        company_id=7,
    )
    existing = SimpleNamespace(company_id=7, carrier_quote=Decimal("40.00"), id=1)

    with app.app_context():
        with (
            patch(
                "services.legal.form_portal_transport_contract.db"
            ) as mock_db,
            patch(
                "services.legal.form_portal_transport_contract.PortalTransportContractFormed"
            ) as MockFormed,
            patch(
                "services.legal.form_portal_transport_contract.booking_uses_conditional_order",
                return_value=True,
            ),
        ):
            mock_q = MagicMock()
            mock_q.filter_by.return_value.with_for_update.return_value.one.return_value = (
                booking
            )
            mock_db.session.query.return_value = mock_q
            MockFormed.query.filter_by.return_value.one_or_none.return_value = existing

            result = form_portal_transport_contract_on_accept(
                booking=booking, company_id=7, offered_amount=40
            )
            assert result.ok
            assert result.idempotent_replay
            assert result.contract is existing


def test_form_contract_other_company_already_assigned(app):
    from services.legal.form_portal_transport_contract import (
        form_portal_transport_contract_on_accept,
    )

    booking = SimpleNamespace(
        id=1,
        portal_contract_flow=FLOW_CONDITIONAL_ORDER_V1,
        status="ACCEPTED",
        company_id=7,
    )
    existing = SimpleNamespace(company_id=7, carrier_quote=Decimal("40.00"), id=1)

    with app.app_context():
        with (
            patch(
                "services.legal.form_portal_transport_contract.db"
            ) as mock_db,
            patch(
                "services.legal.form_portal_transport_contract.PortalTransportContractFormed"
            ) as MockFormed,
            patch(
                "services.legal.form_portal_transport_contract.booking_uses_conditional_order",
                return_value=True,
            ),
        ):
            mock_q = MagicMock()
            mock_q.filter_by.return_value.with_for_update.return_value.one.return_value = (
                booking
            )
            mock_db.session.query.return_value = mock_q
            MockFormed.query.filter_by.return_value.one_or_none.return_value = existing

            result = form_portal_transport_contract_on_accept(
                booking=booking, company_id=8, offered_amount=38.40
            )
            assert not result.ok
            assert result.error["error"] == ERROR_TRANSPORT_ALREADY_ASSIGNED
