"""Réponses 403 métier sur PUT driver booking status — codes stables."""

from constants.driver_api_errors import (
    BOOKING_ASSIGNED_TO_OTHER_DRIVER,
    BOOKING_COMPANY_FORBIDDEN,
)


def test_booking_assigned_to_other_driver_constant() -> None:
    assert BOOKING_ASSIGNED_TO_OTHER_DRIVER == "BOOKING_ASSIGNED_TO_OTHER_DRIVER"


def test_booking_company_forbidden_constant() -> None:
    assert BOOKING_COMPANY_FORBIDDEN == "BOOKING_COMPANY_FORBIDDEN"


def test_inc_forbidden_metrics_no_crash() -> None:
    from services.monitoring.driver_booking_metrics import (
        inc_booking_reassigned_fanout,
        inc_driver_booking_status_forbidden,
    )

    inc_driver_booking_status_forbidden(BOOKING_ASSIGNED_TO_OTHER_DRIVER)
    inc_booking_reassigned_fanout()
