"""Gate restriction commerciale full sur création manuelle entreprise."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from application.companies.reservations.create_manual_booking import (
    CreateManualBookingError,
    CreateManualBookingUseCase,
)
from services.platform_billing.capabilities import (
    BillingAccessRestricted,
    BillingCapability,
)


def test_manual_booking_blocked_when_full_restriction():
    uc = CreateManualBookingUseCase()
    with patch(
        "services.platform_billing.capabilities.assert_billing_capability_allowed",
        side_effect=BillingAccessRestricted(
            BillingCapability.CREATE_OWN_PORTFOLIO_BOOKING, "full"
        ),
    ):
        with pytest.raises(CreateManualBookingError) as exc:
            uc.execute(
                company_id=467,
                validated_data={
                    "client_id": 1,
                    "scheduled_time": "2026-08-10T10:00:00",
                },
                client=object(),
                user=object(),
            )
    assert exc.value.status_code == 403
    assert exc.value.error_code == "billing_access_restricted"
    assert exc.value.details.get("billing_access_state") == "full"
    assert "nouvelle course impossible" in exc.value.message.lower()
    assert "recouvrement" in exc.value.message.lower()
    assert "022 512 02 03" in exc.value.message
    assert "info@lirie.ch" in exc.value.message


def test_manual_booking_calls_own_portfolio_gate():
    uc = CreateManualBookingUseCase()
    with patch(
        "services.platform_billing.capabilities.assert_billing_capability_allowed",
        side_effect=BillingAccessRestricted(
            BillingCapability.CREATE_OWN_PORTFOLIO_BOOKING, "full"
        ),
    ) as gate:
        with pytest.raises(CreateManualBookingError):
            uc.execute(
                company_id=42,
                validated_data={"client_id": 9, "scheduled_time": "2026-08-10T10:00:00"},
                client=object(),
                user=object(),
            )
    gate.assert_called_once_with(
        42, BillingCapability.CREATE_OWN_PORTFOLIO_BOOKING
    )
