from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterator
from unittest.mock import patch

from application.drivers.update_driver_booking_status import (
    UpdateDriverBookingStatusCommand,
    UpdateDriverBookingStatusUseCase,
)
from models import BookingStatus
from models.enums import AssignmentStatus, CancelReason


@contextmanager
def _patch_db_session_no_autoflush() -> Iterator[None]:
    """Évite RuntimeError hors app_context (db.session / Model.query)."""
    with (
        patch("ext.db.session") as mock_session,
        patch("models.invoice.CompanyBillingSettings") as mock_cbs,
    ):
        mock_session.no_autoflush = nullcontext()
        mock_session.query.return_value.filter_by.return_value.delete.return_value = 0
        # Pas de policy DB → compute_cancellation_fields utilise le legacy billable
        mock_cbs.query.filter_by.return_value.first.return_value = None
        yield


@dataclass
class _Booking:
    id: int
    company_id: int
    driver_id: int | None
    status: BookingStatus
    is_return: bool = False
    boarded_at: datetime | None = None
    completed_at: datetime | None = None
    parent_booking_id: int | None = None
    # Champs annulation standardisée (pour tests driver CANCEL)
    cancelled_at: datetime | None = None
    cancelled_by_role: str | None = None
    cancellation_reason_code: str | None = None
    cancellation_reason_text: str | None = None
    is_cancellation_billable: bool | None = None
    cancellation_display_label: str | None = None


class _BookingRepo:
    def __init__(
        self,
        booking: _Booking | None,
        bookings_by_id: dict[int, _Booking] | None = None,
        children_by_parent: dict[int, list[_Booking]] | None = None,
    ):
        self._booking = booking
        self._bookings_by_id = bookings_by_id or (
            {booking.id: booking} if booking is not None else {}
        )
        self._children_by_parent = children_by_parent or {}

    def find_model_by_id(self, booking_id: int):  # type: ignore[no-untyped-def]
        if self._bookings_by_id and booking_id in self._bookings_by_id:
            return self._bookings_by_id[booking_id]
        if self._booking is not None and self._booking.id == booking_id:
            return self._booking
        return None

    def find_children_by_parent_booking_id(self, parent_booking_id: int):  # type: ignore[no-untyped-def]
        return self._children_by_parent.get(parent_booking_id, [])


@dataclass
class _Assignment:
    id: int
    booking_id: int = 1
    driver_id: int | None = 10
    status: AssignmentStatus = AssignmentStatus.SCHEDULED


class _AssignmentRepo:
    def __init__(self, assignment: _Assignment | None):
        self._assignment = assignment
        self.deleted_dependents_for: list[int] = []

    def find_model_by_booking_id(self, booking_id: int):  # type: ignore[no-untyped-def]
        _ = booking_id
        return self._assignment

    def delete_dependent_records_for_assignment_id(self, assignment_id: int) -> None:
        self.deleted_dependents_for.append(assignment_id)


class _Db:
    def __init__(self) -> None:
        self.commits = 0
        self.deleted: list[object] = []

    def commit(self) -> None:
        self.commits += 1

    def delete(self, obj: object) -> None:
        self.deleted.append(obj)


def test_en_route_requires_assigned() -> None:
    booking = _Booking(id=1, company_id=1, driver_id=10, status=BookingStatus.PENDING)
    db = _Db()
    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BookingRepo(booking),
        assignment_repo=_AssignmentRepo(None),
        db_session=db,
        notify_booking_update_fn=lambda _driver_id, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _cid, _aid, _bid, _did: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )
    res = uc.execute(
        UpdateDriverBookingStatusCommand(
            booking_id=1,
            driver_id=10,
            payload={"status": "en_route"},
        )
    )
    assert res.status_code == 400
    assert db.commits == 0


def test_release_cancels_assignment_and_triggers_dispatch() -> None:
    booking = _Booking(id=1, company_id=7, driver_id=10, status=BookingStatus.ASSIGNED)
    assignment = _Assignment(id=123, booking_id=1, driver_id=10)
    assignment_repo = _AssignmentRepo(assignment)
    db = _Db()
    events: dict[str, Any] = {"emit": 0, "trigger": 0}

    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BookingRepo(booking),
        assignment_repo=assignment_repo,
        db_session=db,
        notify_booking_update_fn=lambda _driver_id, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _cid, _aid, _bid, _did: events.__setitem__(
            "emit", events["emit"] + 1
        ),
        maybe_trigger_dispatch_fn=lambda _cid, _action: events.__setitem__(
            "trigger", events["trigger"] + 1
        ),
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )

    # Mock publish_event pour qu'il échoue, forçant l'appel à emit_assignment_cancelled
    with (
        patch(
            "application.events.event_bus.publish_event",
            side_effect=Exception("Event publish failed"),
        ),
        _patch_db_session_no_autoflush(),
    ):
        res = uc.execute(
            UpdateDriverBookingStatusCommand(
                booking_id=1,
                driver_id=10,
                payload={
                    "status": "canceled",
                    "cancel_reason": CancelReason.RELEASE.value,
                },
            )
        )

    assert res.status_code == 200
    assert booking.status == BookingStatus.ACCEPTED
    assert booking.driver_id is None
    assert db.deleted  # assignment deleted
    assert assignment_repo.deleted_dependents_for == [123]
    assert events["emit"] == 1
    assert events["trigger"] == 1
    # RELEASE ne doit pas remplir les champs cancellation_*
    assert booking.cancellation_reason_code is None
    assert booking.cancelled_by_role is None


def test_release_accepts_legacy_reason_alias() -> None:
    booking = _Booking(id=1, company_id=7, driver_id=10, status=BookingStatus.ASSIGNED)
    assignment = _Assignment(id=123, booking_id=1, driver_id=10)
    db = _Db()

    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BookingRepo(booking),
        assignment_repo=_AssignmentRepo(assignment),
        db_session=db,
        notify_booking_update_fn=lambda _driver_id, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _cid, _aid, _bid, _did: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )

    with (
        patch("application.events.event_bus.publish_event"),
        _patch_db_session_no_autoflush(),
    ):
        res = uc.execute(
            UpdateDriverBookingStatusCommand(
                booking_id=1,
                driver_id=10,
                payload={
                    "status": "canceled",
                    "reason": "RELEASE",
                },
            )
        )

    assert res.status_code == 200
    assert booking.status == BookingStatus.ACCEPTED
    assert booking.driver_id is None
    assert booking.cancellation_reason_code is None


def test_driver_cancel_no_show_billable_and_label() -> None:
    """Driver CANCEL + NO_SHOW ⇒ billable True + label 'Client ne s'est pas présenté'."""
    booking = _Booking(
        id=1,
        company_id=7,
        driver_id=10,
        status=BookingStatus.ASSIGNED,
    )
    db = _Db()
    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BookingRepo(booking),
        assignment_repo=_AssignmentRepo(None),
        db_session=db,
        notify_booking_update_fn=lambda _driver_id, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _cid, _aid, _bid, _did: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )

    with (
        patch("application.events.event_bus.publish_event"),
        _patch_db_session_no_autoflush(),
    ):
        res = uc.execute(
            UpdateDriverBookingStatusCommand(
                booking_id=1,
                driver_id=10,
                payload={
                    "status": "canceled",
                    "cancel_reason": CancelReason.CANCEL.value,
                    "reason_code": "NO_SHOW",
                    "reason_text": None,
                },
            )
        )

    assert res.status_code == 200
    assert booking.status == BookingStatus.CANCELED
    assert booking.is_cancellation_billable is True
    assert booking.cancellation_display_label == "Client ne s'est pas présenté"
    assert booking.cancellation_reason_code == "NO_SHOW"
    assert booking.cancelled_by_role == "driver"
    assert booking.cancelled_at is not None


def test_driver_cancel_vehicle_issue_non_billable_and_label() -> None:
    """Driver CANCEL + VEHICLE_ISSUE ⇒ billable False + label 'Problème véhicule'."""
    booking = _Booking(
        id=1,
        company_id=7,
        driver_id=10,
        status=BookingStatus.ASSIGNED,
    )
    db = _Db()
    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BookingRepo(booking),
        assignment_repo=_AssignmentRepo(None),
        db_session=db,
        notify_booking_update_fn=lambda _driver_id, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _cid, _aid, _bid, _did: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )

    with (
        patch("application.events.event_bus.publish_event"),
        _patch_db_session_no_autoflush(),
    ):
        res = uc.execute(
            UpdateDriverBookingStatusCommand(
                booking_id=1,
                driver_id=10,
                payload={
                    "status": "canceled",
                    "cancel_reason": CancelReason.CANCEL.value,
                    "reason_code": "VEHICLE_ISSUE",
                    "reason_text": None,
                },
            )
        )

    assert res.status_code == 200
    assert booking.status == BookingStatus.CANCELED
    assert booking.is_cancellation_billable is False
    assert booking.cancellation_display_label == "Problème véhicule"
    assert booking.cancellation_reason_code == "VEHICLE_ISSUE"
    assert booking.cancelled_by_role == "driver"
    assert booking.cancelled_at is not None


def test_driver_cancel_does_not_overwrite_existing_cancellation_fields() -> None:
    """Si cancellation_reason_code déjà set, ne pas écraser (idempotence)."""
    booking = _Booking(
        id=1,
        company_id=7,
        driver_id=10,
        status=BookingStatus.ASSIGNED,
        cancellation_reason_code="NO_SHOW",
        cancelled_by_role="driver",
        is_cancellation_billable=True,
        cancellation_display_label="Client ne s'est pas présenté",
    )
    db = _Db()
    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BookingRepo(booking),
        assignment_repo=_AssignmentRepo(None),
        db_session=db,
        notify_booking_update_fn=lambda _driver_id, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _cid, _aid, _bid, _did: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )

    with (
        patch("application.events.event_bus.publish_event"),
        _patch_db_session_no_autoflush(),
    ):
        res = uc.execute(
            UpdateDriverBookingStatusCommand(
                booking_id=1,
                driver_id=10,
                payload={
                    "status": "canceled",
                    "cancel_reason": CancelReason.CANCEL.value,
                    "reason_code": "VEHICLE_ISSUE",  # Différent du déjà set
                    "reason_text": None,
                },
            )
        )

    assert res.status_code == 200
    assert booking.status == BookingStatus.CANCELED
    # Champs non écrasés (gardent les valeurs initiales)
    assert booking.cancellation_reason_code == "NO_SHOW"
    assert booking.is_cancellation_billable is True
    assert booking.cancellation_display_label == "Client ne s'est pas présenté"


def test_scope_reservation_parent_canceled_child_accepted_cancels_child() -> None:
    """scope=reservation : parent déjà CANCELED, child ACCEPTED → child est annulé."""
    parent = _Booking(
        id=51,
        company_id=1,
        driver_id=10,
        status=BookingStatus.CANCELED,
        is_return=False,
        parent_booking_id=None,
    )
    child = _Booking(
        id=52,
        company_id=1,
        driver_id=10,
        status=BookingStatus.ACCEPTED,
        is_return=True,
        parent_booking_id=51,
    )
    repo = _BookingRepo(
        booking=child,
        bookings_by_id={51: parent, 52: child},
        children_by_parent={51: [child]},
    )
    db = _Db()
    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=repo,
        assignment_repo=_AssignmentRepo(None),
        db_session=db,
        notify_booking_update_fn=lambda _d, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _c, _a, _b, _d: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )
    with _patch_db_session_no_autoflush():
        res = uc.execute(
            UpdateDriverBookingStatusCommand(
                booking_id=52,
                driver_id=10,
                payload={
                    "status": "canceled",
                    "cancel_reason": "CANCEL",
                    "reason_code": "NO_SHOW",
                    "scope": "reservation",
                },
            )
        )
    assert res.status_code == 200
    assert res.response.get("updated_booking_ids") == [52]
    assert 51 in res.response.get("skipped_booking_ids", [])
    assert child.status == BookingStatus.CANCELED
    assert parent.status == BookingStatus.CANCELED


def test_scope_reservation_parent_completed_child_accepted_cancels_only_child() -> None:
    """scope=reservation : parent COMPLETED, child ACCEPTED → seul le child est annulé."""
    parent = _Booking(
        id=51,
        company_id=1,
        driver_id=10,
        status=BookingStatus.COMPLETED,
        is_return=False,
        parent_booking_id=None,
    )
    child = _Booking(
        id=52,
        company_id=1,
        driver_id=10,
        status=BookingStatus.ACCEPTED,
        is_return=True,
        parent_booking_id=51,
    )
    repo = _BookingRepo(
        booking=child,
        bookings_by_id={51: parent, 52: child},
        children_by_parent={51: [child]},
    )
    db = _Db()
    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=repo,
        assignment_repo=_AssignmentRepo(None),
        db_session=db,
        notify_booking_update_fn=lambda _d, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _c, _a, _b, _d: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )
    with _patch_db_session_no_autoflush():
        res = uc.execute(
            UpdateDriverBookingStatusCommand(
                booking_id=52,
                driver_id=10,
                payload={
                    "status": "canceled",
                    "cancel_reason": "CANCEL",
                    "reason_code": "NO_SHOW",
                    "scope": "reservation",
                },
            )
        )
    assert res.status_code == 200
    assert res.response.get("updated_booking_ids") == [52]
    assert 51 in res.response.get("skipped_booking_ids", [])
    assert child.status == BookingStatus.CANCELED
    assert parent.status == BookingStatus.COMPLETED


def test_en_route_syncs_assignment_to_en_route_pickup() -> None:
    booking = _Booking(id=1, company_id=1, driver_id=10, status=BookingStatus.ASSIGNED)
    assignment = _Assignment(id=5, booking_id=1, driver_id=10)
    db = _Db()
    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BookingRepo(booking),
        assignment_repo=_AssignmentRepo(assignment),
        db_session=db,
        notify_booking_update_fn=lambda _d, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _c, _a, _b, _d: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )
    res = uc.execute(
        UpdateDriverBookingStatusCommand(
            booking_id=1,
            driver_id=10,
            payload={"status": "en_route"},
        )
    )
    assert res.status_code == 200
    assert booking.status == BookingStatus.EN_ROUTE
    assert assignment.status == AssignmentStatus.EN_ROUTE_PICKUP
    assert db.commits == 1


def test_arrived_syncs_assignment_when_present() -> None:
    booking = _Booking(id=1, company_id=1, driver_id=10, status=BookingStatus.EN_ROUTE)
    assignment = _Assignment(
        id=5,
        booking_id=1,
        driver_id=10,
        status=AssignmentStatus.EN_ROUTE_PICKUP,
    )
    db = _Db()
    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BookingRepo(booking),
        assignment_repo=_AssignmentRepo(assignment),
        db_session=db,
        notify_booking_update_fn=lambda _d, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _c, _a, _b, _d: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )
    res = uc.execute(
        UpdateDriverBookingStatusCommand(
            booking_id=1,
            driver_id=10,
            payload={"status": "ARRIVED"},
        )
    )
    assert res.status_code == 200
    assert assignment.status == AssignmentStatus.ARRIVED_PICKUP
    assert db.commits == 1


def test_arrived_idempotent_when_assignment_already_arrived() -> None:
    """Rejeu d'un `arrived` déjà persisté → 200 unchanged (aucun double effet)."""
    booking = _Booking(id=1, company_id=1, driver_id=10, status=BookingStatus.EN_ROUTE)
    assignment = _Assignment(
        id=5, booking_id=1, driver_id=10, status=AssignmentStatus.ARRIVED_PICKUP
    )
    db = _Db()
    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BookingRepo(booking),
        assignment_repo=_AssignmentRepo(assignment),
        db_session=db,
        notify_booking_update_fn=lambda _d, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _c, _a, _b, _d: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )
    res = uc.execute(
        UpdateDriverBookingStatusCommand(
            booking_id=1,
            driver_id=10,
            payload={"status": "ARRIVED"},
        )
    )
    assert res.status_code == 200
    assert res.response.get("unchanged") is True
    assert res.response.get("mission_milestone") == "ARRIVED"
    assert res.response.get("milestone_persisted") is True
    assert res.response.get("assignment_id") == 5
    assert booking.status == BookingStatus.EN_ROUTE
    assert assignment.status == AssignmentStatus.ARRIVED_PICKUP


def test_arrived_creates_missing_assignment_then_persists() -> None:
    """P0-C + SOT-1B : assignment manquant → recréé via ensure puis persisté."""
    booking = _Booking(id=1, company_id=1, driver_id=10, status=BookingStatus.EN_ROUTE)
    assignment_repo = _AssignmentRepo(None)
    db = _Db()

    def _fake_ensure(*, company_id: int, booking: Any, driver_id: int) -> None:
        _ = company_id
        assignment_repo._assignment = _Assignment(
            id=77, booking_id=booking.id, driver_id=driver_id
        )

    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BookingRepo(booking),
        assignment_repo=assignment_repo,
        db_session=db,
        notify_booking_update_fn=lambda _d, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _c, _a, _b, _d: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )
    with patch(
        "application.companies.assignment_binding.ensure_booking_assignment",
        side_effect=_fake_ensure,
    ):
        res = uc.execute(
            UpdateDriverBookingStatusCommand(
                booking_id=1,
                driver_id=10,
                payload={"status": "ARRIVED"},
            )
        )
    assert res.status_code == 200
    assert res.response.get("milestone_persisted") is True
    assert res.response.get("assignment_id") == 77
    assert assignment_repo._assignment is not None
    assert assignment_repo._assignment.status == AssignmentStatus.ARRIVED_PICKUP
    assert db.commits == 1


def test_arrived_never_returns_200_when_persistence_fails() -> None:
    """P0-C : ensure échoue ⇒ jamais de « 200 ARRIVED » mensonger."""
    booking = _Booking(id=1, company_id=1, driver_id=10, status=BookingStatus.EN_ROUTE)
    db = _Db()
    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BookingRepo(booking),
        assignment_repo=_AssignmentRepo(None),
        db_session=db,
        notify_booking_update_fn=lambda _d, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _c, _a, _b, _d: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )
    with patch(
        "application.companies.assignment_binding.ensure_booking_assignment",
        side_effect=RuntimeError("db down"),
    ):
        res = uc.execute(
            UpdateDriverBookingStatusCommand(
                booking_id=1,
                driver_id=10,
                payload={"status": "ARRIVED"},
            )
        )
    assert res.status_code == 500
    assert res.response.get("error_code") == "driver_transition_persistence_failed"
    assert res.response.get("retryable") is True
    assert db.commits == 0


def test_arrived_persistence_disabled_returns_503(
    monkeypatch: Any,
) -> None:
    import services.dispatch.assignment_status_sync as sync_mod

    monkeypatch.setattr(sync_mod, "ASSIGNMENT_STATUS_SYNC_ENABLED", False)
    booking = _Booking(id=1, company_id=1, driver_id=10, status=BookingStatus.EN_ROUTE)
    db = _Db()
    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BookingRepo(booking),
        assignment_repo=_AssignmentRepo(None),
        db_session=db,
        notify_booking_update_fn=lambda _d, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _c, _a, _b, _d: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )
    res = uc.execute(
        UpdateDriverBookingStatusCommand(
            booking_id=1,
            driver_id=10,
            payload={"status": "ARRIVED"},
        )
    )
    assert res.status_code == 503
    assert res.response.get("retryable") is True
    assert db.commits == 0


# ── P0-A : stale writes → 409, jamais appliqués ─────────────────────────────


def _make_uc(booking: _Booking, assignment: _Assignment | None = None) -> tuple[
    UpdateDriverBookingStatusUseCase, _Db
]:
    db = _Db()
    return (
        UpdateDriverBookingStatusUseCase(
            booking_repo=_BookingRepo(booking),
            assignment_repo=_AssignmentRepo(assignment),
            db_session=db,
            notify_booking_update_fn=lambda _d, _b: None,
            resolve_delays_fn=lambda _bid, _dt: None,
            emit_assignment_cancelled_fn=lambda _c, _a, _b, _d: None,
            maybe_trigger_dispatch_fn=None,
            now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
        ),
        db,
    )


def test_stale_arrived_after_in_progress_is_409() -> None:
    """C3 : vieux paquet `arrived` après IN_PROGRESS → 409, aucune régression."""
    booking = _Booking(
        id=1, company_id=1, driver_id=10, status=BookingStatus.IN_PROGRESS
    )
    assignment = _Assignment(
        id=5, booking_id=1, driver_id=10, status=AssignmentStatus.ONBOARD
    )
    uc, db = _make_uc(booking, assignment)
    res = uc.execute(
        UpdateDriverBookingStatusCommand(
            booking_id=1, driver_id=10, payload={"status": "ARRIVED"}
        )
    )
    assert res.status_code == 409
    assert res.response.get("error_code") == "driver_transition_stale"
    assert res.response.get("retryable") is False
    assert booking.status == BookingStatus.IN_PROGRESS
    assert assignment.status == AssignmentStatus.ONBOARD
    assert db.commits == 0


def test_stale_en_route_after_completed_is_409() -> None:
    """C3 : COMPLETED est terminal — un vieux `en_route` est rejeté en 409."""
    booking = _Booking(id=1, company_id=1, driver_id=10, status=BookingStatus.COMPLETED)
    uc, db = _make_uc(booking)
    res = uc.execute(
        UpdateDriverBookingStatusCommand(
            booking_id=1, driver_id=10, payload={"status": "en_route"}
        )
    )
    assert res.status_code == 409
    assert res.response.get("error_code") == "driver_transition_stale"
    assert booking.status == BookingStatus.COMPLETED
    assert db.commits == 0


def test_stale_in_progress_after_completed_is_409() -> None:
    booking = _Booking(id=1, company_id=1, driver_id=10, status=BookingStatus.COMPLETED)
    uc, db = _make_uc(booking)
    res = uc.execute(
        UpdateDriverBookingStatusCommand(
            booking_id=1, driver_id=10, payload={"status": "in_progress"}
        )
    )
    assert res.status_code == 409
    assert booking.status == BookingStatus.COMPLETED
    assert db.commits == 0


def test_completed_after_canceled_is_409() -> None:
    booking = _Booking(id=1, company_id=1, driver_id=10, status=BookingStatus.CANCELED)
    uc, db = _make_uc(booking)
    res = uc.execute(
        UpdateDriverBookingStatusCommand(
            booking_id=1, driver_id=10, payload={"status": "completed"}
        )
    )
    assert res.status_code == 409
    assert booking.status == BookingStatus.CANCELED
    assert db.commits == 0


def test_cancel_after_completed_is_409() -> None:
    booking = _Booking(id=1, company_id=1, driver_id=10, status=BookingStatus.COMPLETED)
    uc, db = _make_uc(booking)
    res = uc.execute(
        UpdateDriverBookingStatusCommand(
            booking_id=1,
            driver_id=10,
            payload={"status": "canceled", "cancel_reason": "CANCEL"},
        )
    )
    assert res.status_code == 409
    assert res.response.get("error_code") == "driver_transition_stale"
    assert booking.status == BookingStatus.COMPLETED
    assert db.commits == 0


def test_completed_success_returns_lifecycle_identity() -> None:
    """P1 : la réponse porte assignment_id + mission_revision."""
    booking = _Booking(
        id=1, company_id=1, driver_id=10, status=BookingStatus.IN_PROGRESS
    )
    assignment = _Assignment(
        id=5, booking_id=1, driver_id=10, status=AssignmentStatus.ONBOARD
    )
    uc, db = _make_uc(booking, assignment)
    res = uc.execute(
        UpdateDriverBookingStatusCommand(
            booking_id=1, driver_id=10, payload={"status": "completed"}
        )
    )
    assert res.status_code == 200
    assert booking.status == BookingStatus.COMPLETED
    assert res.response.get("assignment_id") == 5
    assert res.response.get("mission_revision", 0) >= 1
    assert assignment.status == AssignmentStatus.COMPLETED
    assert db.commits == 1
