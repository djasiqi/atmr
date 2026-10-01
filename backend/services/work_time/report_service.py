"""Chargement SQL et exposition du rapport de temps de travail."""

from __future__ import annotations

import logging
import os
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import joinedload

from domain.work_time.clock import as_utc, scheduled_zurich_date, zurich_date_of_instant
from domain.work_time.compensation import PolicyView
from domain.work_time.journeys import SegmentInput
from domain.work_time.periods import parse_ymd, zurich_period_bounds
from domain.work_time.report import (
    DurationDecisionView,
    apply_ledger_snapshot,
    build_work_time_report,
)
from domain.work_time.transport_time import AdjustmentView
from domain.work_time.transports import driver_arrived_on_site
from ext import db
from models.booking import Booking
from models.dispatch import Assignment
from models.driver import Driver
from models.driver_work_time import (
    DriverCompensationLedger,
    DriverCompensationPolicy,
    DriverManualWorkEntry,
    DriverWorkTimeAdjustment,
    DriverWorkTimeDurationDecision,
    DriverWorkTimePeriodClosure,
    DriverWorkTimeSettings,
)
from models.enums import BookingStatus
from shared.time_utils import LOCAL_TZ

_COMPLETED = (BookingStatus.COMPLETED, BookingStatus.RETURN_COMPLETED)
_OPEN = (
    BookingStatus.PENDING,
    BookingStatus.ACCEPTED,
    BookingStatus.ASSIGNED,
    BookingStatus.EN_ROUTE,
    BookingStatus.IN_PROGRESS,
)
logger = logging.getLogger(__name__)


def load_cutover(raw: str) -> datetime:
    """Parse la bascule configurée (ISO 8601, naïf = UTC)."""
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _status_value(booking: Booking) -> str:
    status = booking.status
    return str(getattr(status, "value", status))


def _to_segment(
    booking: Booking,
    *,
    cutover: datetime | None = None,
    force_route: bool = False,
) -> SegmentInput:
    if force_route or _needs_driving_route(booking, cutover):
        trace = _driving_route_trace(booking)
    else:
        trace = _idle_trace()
    return SegmentInput(
        booking_id=int(booking.id),
        driver_id=int(booking.driver_id) if booking.driver_id is not None else None,
        status=_status_value(booking),
        is_return=bool(booking.is_return),
        is_round_trip=bool(booking.is_round_trip),
        parent_booking_id=(
            int(booking.parent_booking_id) if booking.parent_booking_id else None
        ),
        route_group_id=booking.route_group_id,
        route_sequence_number=booking.route_sequence_number,
        pickup_location=booking.pickup_location or "",
        dropoff_location=booking.dropoff_location or "",
        scheduled_time=booking.scheduled_time,
        arrived_at=booking.arrived_at,
        boarded_at=booking.boarded_at,
        completed_at=booking.completed_at,
        duration_seconds=(
            int(booking.duration_seconds) if booking.duration_seconds else None
        ),
        pickup_lat=booking.pickup_lat,
        pickup_lon=booking.pickup_lon,
        dropoff_lat=booking.dropoff_lat,
        dropoff_lon=booking.dropoff_lon,
        route_provider=trace["route_provider"],
        route_duration_seconds=trace["route_duration_seconds"],
        route_distance_m=trace["route_distance_m"],
        routing_profile=trace["routing_profile"],
        routing_status=trace["routing_status"],
        route_calculated_at=trace["calculated_at"],
    )


class WorkTimeReportService:
    """Le même calcul alimente le résumé et le détail."""

    def __init__(self, cutover: datetime) -> None:
        self._cutover = cutover

    def build(
        self, company_id: int, period_from: str, period_to: str
    ) -> dict[str, Any]:
        """Rapport dynamique, ou ledger si la période exacte est clôturée."""
        report = self._dynamic(company_id, period_from, period_to)
        closure = self._active_closure(company_id, period_from, period_to)
        if closure is None:
            report["period_finalized"] = False
            return report
        ledger = (
            db.session.query(DriverCompensationLedger)
            .filter(DriverCompensationLedger.closure_id == closure.id)
            .all()
        )
        rows = [_ledger_dict(row) for row in ledger]
        apply_ledger_snapshot(report, rows)
        report["finalized_at"] = closure.finalized_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        return report

    def explain_booking(
        self, company_id: int, booking_id: int
    ) -> dict[str, Any] | None:
        """Jalons, règle et corrections d'une course de l'entreprise."""
        booking = db.session.get(Booking, booking_id)
        if booking is None or int(booking.company_id or 0) != int(company_id):
            return None
        siblings = _siblings(company_id, [booking])
        segments = [
            _to_segment(item, cutover=self._cutover)
            for item in _dedupe([booking, *siblings])
        ]
        ids = [segment.booking_id for segment in segments]
        estimate = _estimate_settings(company_id)
        report = build_work_time_report(
            segments=segments,
            adjustments=_latest_adjustments(ids),
            manuals=[],
            policies=_policies(company_id),
            cutover=self._cutover,
            period_from="2000-01-01",
            period_to="2100-01-01",
            duration_decisions=_latest_decisions(ids),
            route_margin_minutes=estimate["route_margin_minutes"],
            route_estimates_enabled=estimate["route_estimate_enabled"],
        )
        entries = [
            entry
            for days in report["days_by_driver"].values()
            for day in days
            for entry in day["entries"]
            if entry.get("booking_id") == booking_id
        ]
        history = (
            db.session.query(DriverWorkTimeAdjustment)
            .filter(
                DriverWorkTimeAdjustment.company_id == company_id,
                DriverWorkTimeAdjustment.booking_id == booking_id,
            )
            .order_by(DriverWorkTimeAdjustment.created_at.asc())
            .all()
        )
        return {
            "booking_id": booking_id,
            "cutover_at": report["cutover_at"],
            "milestones": {
                "scheduled_time": scheduled_zurich_date(booking.scheduled_time),
                "arrived_at": _iso_db(booking.arrived_at),
                "boarded_at": _iso_db(booking.boarded_at),
                "completed_at": _iso_db(booking.completed_at),
                "status": _status_value(booking),
            },
            "entries": entries,
            "adjustments": [_adjustment_dict(row) for row in history],
            "compensation_lines": [
                line
                for line in report["compensation_lines"]
                if line.get("attached_to_booking_id") == booking_id
                or any(
                    entry.get("journey_key") == line.get("journey_key")
                    for entry in entries
                )
            ],
        }

    def _dynamic(
        self, company_id: int, period_from: str, period_to: str
    ) -> dict[str, Any]:
        utc_start, utc_end, local_start, local_end = zurich_period_bounds(
            period_from, period_to
        )
        seeds = _period_bookings(company_id, utc_start, utc_end, local_start, local_end)
        siblings = _siblings(company_id, seeds)
        opened = _open_bookings(company_id, local_start, local_end)
        cancelled = _cancelled_bookings(
            company_id, utc_start, utc_end, local_start, local_end
        )
        bookings = _dedupe([*seeds, *siblings, *opened, *cancelled])
        ids = [int(booking.id) for booking in bookings]
        assignment_status = _assignment_status_by_booking(ids)
        manuals = (
            db.session.query(DriverManualWorkEntry)
            .filter(
                DriverManualWorkEntry.company_id == company_id,
                DriverManualWorkEntry.cancelled_at.is_(None),
                DriverManualWorkEntry.work_date >= parse_ymd(period_from),
                DriverManualWorkEntry.work_date <= parse_ymd(period_to),
            )
            .all()
        )
        estimate = _estimate_settings(company_id)
        report = build_work_time_report(
            segments=_segments(bookings, assignment_status, self._cutover),
            adjustments=_latest_adjustments(ids),
            manuals=[_manual_dict(row) for row in manuals],
            policies=_policies(company_id),
            cutover=self._cutover,
            period_from=period_from,
            period_to=period_to,
            duration_decisions=_latest_decisions(ids),
            route_margin_minutes=estimate["route_margin_minutes"],
            route_estimates_enabled=estimate["route_estimate_enabled"],
        )
        names = _display_names([row["driver_id"] for row in report["drivers"]])
        for row in report["drivers"]:
            row["display_name"] = names.get(row["driver_id"], "Chauffeur")
        return report

    def _active_closure(
        self, company_id: int, period_from: str, period_to: str
    ) -> DriverWorkTimePeriodClosure | None:
        row = (
            db.session.query(DriverWorkTimePeriodClosure)
            .filter(
                DriverWorkTimePeriodClosure.company_id == company_id,
                DriverWorkTimePeriodClosure.period_from == parse_ymd(period_from),
                DriverWorkTimePeriodClosure.period_to == parse_ymd(period_to),
            )
            .order_by(DriverWorkTimePeriodClosure.id.desc())
            .first()
        )
        if row is None or row.reopened_at is not None:
            return None
        return row


def _segments(
    bookings: list[Booking],
    assignment_status: dict[int, str],
    cutover: datetime | None = None,
) -> list[SegmentInput]:
    rows = []
    for booking in bookings:
        segment = _to_segment(booking, cutover=cutover)
        segment.assignment_status = assignment_status.get(int(booking.id))
        rows.append(segment)
    return rows


def _assignment_status_by_booking(booking_ids: list[int]) -> dict[int, str]:
    """Garde une preuve d'arrivée même si une ligne plus récente est annulée."""
    if not booking_ids:
        return {}
    rows = (
        db.session.query(Assignment.booking_id, Assignment.status, Assignment.id)
        .filter(Assignment.booking_id.in_(booking_ids))
        .order_by(Assignment.id.asc())
        .all()
    )
    chosen: dict[int, str] = {}
    for booking_id, status, _assignment_id in rows:
        token = str(getattr(status, "value", status) or "")
        previous = chosen.get(int(booking_id))
        if previous and driver_arrived_on_site(
            arrived_at=None, assignment_status=previous
        ):
            continue
        chosen[int(booking_id)] = token
    return chosen


def _cancelled_bookings(
    company_id: int,
    utc_start: datetime,
    utc_end: datetime,
    local_start: datetime,
    local_end: datetime,
) -> list[Booking]:
    """Annulations de la période. Le domaine décide lesquelles sont des transports."""
    return (
        db.session.query(Booking)
        .filter(
            Booking.company_id == company_id,
            Booking.driver_id.isnot(None),
            Booking.status == BookingStatus.CANCELED,
            or_(
                and_(
                    Booking.arrived_at.isnot(None),
                    Booking.arrived_at >= utc_start,
                    Booking.arrived_at < utc_end,
                ),
                and_(
                    Booking.scheduled_time >= local_start,
                    Booking.scheduled_time < local_end,
                ),
            ),
        )
        .all()
    )


def _period_bookings(
    company_id: int,
    utc_start: datetime,
    utc_end: datetime,
    local_start: datetime,
    local_end: datetime,
) -> list[Booking]:
    """Courses terminées de la période, y compris sans ``completed_at``.

    Une course à cheval est incluse si son intervalle réel chevauche la période,
    pour ventiler les minutes du bon jour civil.
    """
    start_ts = func.coalesce(Booking.arrived_at, Booking.boarded_at)
    return (
        db.session.query(Booking)
        .filter(
            Booking.company_id == company_id,
            Booking.driver_id.isnot(None),
            Booking.status.in_(_COMPLETED),
            or_(
                and_(
                    Booking.completed_at.isnot(None),
                    Booking.completed_at >= utc_start,
                    Booking.completed_at < utc_end,
                ),
                and_(
                    Booking.completed_at.is_(None),
                    Booking.scheduled_time >= local_start,
                    Booking.scheduled_time < local_end,
                ),
                and_(
                    Booking.completed_at.isnot(None),
                    Booking.completed_at > utc_start,
                    start_ts.isnot(None),
                    start_ts < utc_end,
                ),
            ),
        )
        .all()
    )


def _open_bookings(
    company_id: int, local_start: datetime, local_end: datetime
) -> list[Booking]:
    """Courses commencées ou assignées, prévues dans la période. Les annulées n'y sont pas."""
    return (
        db.session.query(Booking)
        .filter(
            Booking.company_id == company_id,
            Booking.driver_id.isnot(None),
            Booking.status.in_(_OPEN),
            Booking.scheduled_time >= local_start,
            Booking.scheduled_time < local_end,
        )
        .all()
    )


def _siblings(company_id: int, seeds: list[Booking]) -> list[Booking]:
    group_ids = {booking.route_group_id for booking in seeds if booking.route_group_id}
    anchors: set[int] = set()
    for booking in seeds:
        if booking.parent_booking_id:
            anchors.add(int(booking.parent_booking_id))
        if booking.is_round_trip or booking.is_return:
            anchors.add(int(booking.parent_booking_id or booking.id))
    if not group_ids and not anchors:
        return []
    clauses = []
    if group_ids:
        clauses.append(Booking.route_group_id.in_(group_ids))
    if anchors:
        clauses.append(Booking.id.in_(anchors))
        clauses.append(Booking.parent_booking_id.in_(anchors))
    return (
        db.session.query(Booking)
        .filter(Booking.company_id == company_id, or_(*clauses))
        .all()
    )


def _dedupe(bookings: list[Booking]) -> list[Booking]:
    seen: dict[int, Booking] = {}
    for booking in bookings:
        seen[int(booking.id)] = booking
    return list(seen.values())


_DRIVING_ROUTE_CACHE: dict[tuple[float, float, float, float, str], dict[str, Any]] = {}
_GEOCODE_CACHE: dict[str, tuple[float, float] | None] = {}


def _idle_trace() -> dict[str, Any]:
    return {
        "route_provider": None,
        "route_duration_seconds": None,
        "route_distance_m": None,
        "routing_profile": None,
        "routing_status": None,
        "calculated_at": None,
    }


def _unavailable_trace() -> dict[str, Any]:
    return {
        "route_provider": None,
        "route_duration_seconds": None,
        "route_distance_m": None,
        "routing_profile": None,
        "routing_status": "unavailable",
        "calculated_at": None,
    }


def _needs_driving_route(booking: Booking, cutover: datetime | None) -> bool:
    """Même cas que le domaine : terminé, sans arrivée, sans estimation historique."""
    if _status_value(booking) not in {"COMPLETED", "RETURN_COMPLETED"}:
        return False
    if booking.arrived_at is not None:
        return False
    completed = as_utc(booking.completed_at)
    if completed is None:
        return False
    return not (
        cutover is not None and completed < cutover and booking.boarded_at is not None
    )


def _osrm_base_url() -> str:
    return os.getenv("OSRM_BASE_URL") or os.getenv("UD_OSRM_URL") or "http://osrm:5000"


def _geocode_point(address: str | None) -> tuple[float, float] | None:
    label = (address or "").strip()
    if not label:
        return None
    key = label.casefold()
    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key]
    try:
        from services.geolocation.maps import geocode_address

        point = geocode_address(label, country="CH", language="fr")
    except Exception as exc:
        logger.info("Géocodage impossible pour %r : %s", label, exc)
        point = None
    coords = None
    if (
        isinstance(point, dict)
        and point.get("lat") is not None
        and point.get("lon") is not None
    ):
        coords = (float(point["lat"]), float(point["lon"]))
    _GEOCODE_CACHE[key] = coords
    return coords


def _resolve_booking_coords(
    booking: Booking,
) -> tuple[float, float, float, float] | None:
    pickup_lat = getattr(booking, "pickup_lat", None)
    pickup_lon = getattr(booking, "pickup_lon", None)
    dropoff_lat = getattr(booking, "dropoff_lat", None)
    dropoff_lon = getattr(booking, "dropoff_lon", None)
    pickup = (
        (float(pickup_lat), float(pickup_lon))
        if pickup_lat is not None and pickup_lon is not None
        else _geocode_point(getattr(booking, "pickup_location", None))
    )
    dropoff = (
        (float(dropoff_lat), float(dropoff_lon))
        if dropoff_lat is not None and dropoff_lon is not None
        else _geocode_point(getattr(booking, "dropoff_location", None))
    )
    if pickup is None or dropoff is None:
        return None
    return (pickup[0], pickup[1], dropoff[0], dropoff[1])


def _driving_route_trace(booking: Booking) -> dict[str, Any]:
    """Temps de trajet estimé, le même que la réservation.

    OSRM voiture fournit l'itinéraire. Sans GPS stocké, les adresses sont
    géocodées. La durée retenue est l'estimation améliorée (historique, sinon
    facteur urbain), pas le temps à vide. Un échec ou un repli heuristique ne
    donne aucune durée. Les minutes offertes au chauffeur sont ajoutées plus
    tard, à part.
    """
    coords = _resolve_booking_coords(booking)
    if coords is None:
        return _unavailable_trace()
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = coords
    key = (
        round(pickup_lat, 5),
        round(pickup_lon, 5),
        round(dropoff_lat, 5),
        round(dropoff_lon, 5),
        "driving",
    )
    cached = _DRIVING_ROUTE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from services.geolocation.historical_eta import (
            URBAN_TRAFFIC_FACTOR,
            get_improved_duration_estimate,
        )
        from services.geolocation.osrm import route_info

        result = route_info(
            (pickup_lat, pickup_lon),
            (dropoff_lat, dropoff_lon),
            base_url=_osrm_base_url(),
            profile="driving",
            timeout=8,
            overview="false",
        )
        if not isinstance(result, dict) or result.get("fallback"):
            return _unavailable_trace()
        duration = result.get("duration")
        distance = result.get("distance")
        if not isinstance(duration, (int, float)) or duration <= 0:
            return _unavailable_trace()
        if not isinstance(distance, (int, float)) or distance < 0:
            return _unavailable_trace()
        typical, _source = get_improved_duration_estimate(
            (pickup_lat, pickup_lon),
            (dropoff_lat, dropoff_lon),
            float(duration),
            traffic_factor=URBAN_TRAFFIC_FACTOR,
            use_weather=False,
        )
    except Exception as exc:
        logger.info(
            "Durée de trajet indisponible pour la course %s : %s", booking.id, exc
        )
        return _unavailable_trace()
    if not isinstance(typical, (int, float)) or typical <= 0:
        return _unavailable_trace()
    trace = {
        "route_provider": "osrm",
        "route_duration_seconds": max(1, round(float(typical))),
        "route_distance_m": round(float(distance)),
        "routing_profile": "driving",
        "routing_status": "ok",
        "calculated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _DRIVING_ROUTE_CACHE[key] = trace
    return trace


def _estimate_settings(company_id: int) -> dict[str, Any]:
    row = db.session.get(DriverWorkTimeSettings, company_id)
    if row is None:
        return {"route_margin_minutes": 5, "route_estimate_enabled": True}
    return {
        "route_margin_minutes": max(0, int(row.route_margin_minutes)),
        "route_estimate_enabled": bool(row.route_estimate_enabled),
    }


def _latest_decisions(booking_ids: list[int]) -> dict[int, DurationDecisionView]:
    if not booking_ids:
        return {}
    rows = (
        db.session.query(DriverWorkTimeDurationDecision)
        .filter(DriverWorkTimeDurationDecision.booking_id.in_(booking_ids))
        .order_by(DriverWorkTimeDurationDecision.id.asc())
        .all()
    )
    latest: dict[int, DurationDecisionView] = {}
    for row in rows:
        latest[int(row.booking_id)] = DurationDecisionView(
            validated_worked_minutes=int(row.validated_worked_minutes),
            source=row.source,
            proposed_worked_minutes=int(row.proposed_worked_minutes),
            route_minutes=row.route_minutes,
            margin_minutes=row.margin_minutes,
            route_provider=row.route_provider,
            reason=row.reason,
        )
    return latest


def _latest_adjustments(booking_ids: list[int]) -> dict[int, AdjustmentView]:
    if not booking_ids:
        return {}
    rows = (
        db.session.query(DriverWorkTimeAdjustment)
        .filter(DriverWorkTimeAdjustment.booking_id.in_(booking_ids))
        .order_by(
            DriverWorkTimeAdjustment.created_at.desc(),
            DriverWorkTimeAdjustment.id.desc(),
        )
        .all()
    )
    latest: dict[int, AdjustmentView] = {}
    for row in rows:
        if row.booking_id in latest:
            continue
        latest[int(row.booking_id)] = AdjustmentView(
            corrected_arrived_at=row.corrected_arrived_at,
            corrected_completed_at=row.corrected_completed_at,
            reason=row.reason,
            created_at=row.created_at,
            adjustment_id=int(row.id),
        )
    return latest


def _policies(company_id: int) -> list[PolicyView]:
    rows = (
        db.session.query(DriverCompensationPolicy)
        .filter(DriverCompensationPolicy.company_id == company_id)
        .all()
    )
    return [
        PolicyView(
            policy_id=int(row.id),
            driver_id=int(row.driver_id) if row.driver_id is not None else None,
            effective_from=row.effective_from,
            effective_until=row.effective_until,
            one_way_minutes=int(row.one_way_minutes),
            round_trip_minutes=int(row.round_trip_minutes),
            intermediate_stop_minutes=row.intermediate_stop_minutes,
            transport_flat_minutes=int(row.transport_flat_minutes),
            max_reasonable_minutes=row.max_reasonable_minutes,
            overlap_threshold_minutes=int(row.overlap_threshold_minutes or 1),
            work_type_rules=dict(row.work_type_rules_json or {}),
            mode=str(row.mode or "flat_per_trip"),
        )
        for row in rows
    ]


def _display_names(driver_ids: list[int]) -> dict[int, str]:
    if not driver_ids:
        return {}
    rows = (
        db.session.query(Driver)
        .options(joinedload(Driver.user))
        .filter(Driver.id.in_(driver_ids))
        .all()
    )
    names: dict[int, str] = {}
    for driver in rows:
        user = driver.user
        first = str(getattr(user, "first_name", "") or "").strip()
        last = str(getattr(user, "last_name", "") or "").strip().upper()
        names[int(driver.id)] = f"{first} {last}".strip() or "Chauffeur"
    return names


def _manual_dict(row: DriverManualWorkEntry) -> dict[str, Any]:
    return {
        "entry_id": int(row.id),
        "driver_id": int(row.driver_id),
        "work_type": row.work_type,
        "duration_minutes": int(row.duration_minutes),
        "work_date": row.work_date.isoformat(),
        "description": row.description,
        "pickup_location": row.pickup_location,
        "dropoff_location": row.dropoff_location,
        "booking_id": row.booking_id,
        "started_at": row.started_at,
        "ended_at": row.ended_at,
        "cancelled": row.cancelled_at is not None,
    }


def _iso_db(value: datetime | None) -> str | None:
    if value is None:
        return None
    aware = value if value.tzinfo else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _adjustment_dict(row: DriverWorkTimeAdjustment) -> dict[str, Any]:
    return {
        "id": int(row.id),
        "booking_id": int(row.booking_id),
        "driver_id": int(row.driver_id),
        "original_arrived_at": _iso_db(row.original_arrived_at),
        "original_completed_at": _iso_db(row.original_completed_at),
        "original_source": row.original_source,
        "corrected_arrived_at": _iso_db(row.corrected_arrived_at),
        "corrected_completed_at": _iso_db(row.corrected_completed_at),
        "reason": row.reason,
        "comment": row.comment,
        "created_at": _iso_db(row.created_at),
    }


def _ledger_dict(row: DriverCompensationLedger) -> dict[str, Any]:
    accounting = row.accounting_date.isoformat() if row.accounting_date else None
    return {
        "journey_key": row.journey_key,
        "line_key": row.line_key,
        "booking_id": row.booking_id,
        "manual_entry_id": row.manual_entry_id,
        "flat_minutes": row.flat_minutes,
        "driver_id": row.driver_id,
        "accounting_date": accounting,
        "compensated_minutes": int(row.compensated_minutes),
        "policy_id": row.policy_id,
        "rule_type": row.rule_type,
        "base_minutes": int(row.base_minutes or 0),
        "intermediate_stop_count": int(row.intermediate_stop_count or 0),
        "intermediate_stop_minutes": row.intermediate_stop_minutes,
        "classification_source": row.classification_source,
        "journey_status": row.journey_status,
        "compensation_status": row.compensation_status,
        "attached_to_booking_id": row.attached_to_booking_id,
        "generated_at": _iso_db(row.generated_at),
        "finalized_at": _iso_db(row.finalized_at),
    }


def work_date_in_zurich(started_at: datetime) -> date:
    """Jour civil de début d'une saisie manuelle."""
    aware = started_at if started_at.tzinfo else started_at.replace(tzinfo=UTC)
    return aware.astimezone(LOCAL_TZ).date()


def classification_hint(booking: Booking) -> str | None:
    """Jour d'affichage indicatif (explication), jamais une durée."""
    if booking.completed_at is not None:
        return zurich_date_of_instant(booking.completed_at)
    return scheduled_zurich_date(booking.scheduled_time)
