from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, cast

from flask import current_app, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource
from sqlalchemy import and_, or_
from sqlalchemy.orm import joinedload, selectinload

from ext import db, limiter, role_required
from models import (
    Assignment,
    AutonomousAction,
    Booking,
    Company,
    DispatchMode,
    Driver,
    Message,
    User,
    UserRole,
)
from models.enums import AssignmentStatus, BookingStatus, DriverType, SenderRole
from routes.companies import get_company_from_token
from services.agent_dispatch.orchestrator import get_agent_for_company, stop_agent_for_company
from services.agent_dispatch.tools import AgentTools
from services.unified_dispatch import settings as dispatch_settings
from services.unified_dispatch.heuristics import MAX_FAIRNESS_GAP
from services.unified_dispatch.queue import trigger_job
from services.unified_dispatch.realtime_optimizer import (
    check_opportunities_manual,
    get_optimizer_for_company,
)
from services.unified_dispatch.validation import check_existing_assignment_conflict
from shared.geo_utils import haversine_distance
from shared.time_utils import day_local_bounds, now_local, parse_local_naive

company_mobile_dispatch_ns = Namespace(
    "company_mobile_dispatch",
    description="API mobile entreprise pour le pilotage dispatch (v1)",
)

logger = logging.getLogger(__name__)


def _abort_from_company_error(error: dict[str, Any] | None, code: int | None) -> None:
    message = (error or {}).get("error") if isinstance(error, dict) else "Accès refusé"
    company_mobile_dispatch_ns.abort(code or 403, message)


def _get_current_company() -> Company:
    company, err, code = get_company_from_token()
    if err or company is None:
        _abort_from_company_error(err if isinstance(err, dict) else None, code)
        raise AssertionError("Company should not be None after abort") from None
    return company


def _get_company_context() -> tuple[Company, int]:
    company = _get_current_company()
    company_id_attr = getattr(company, "id", None)
    if company_id_attr is None:
        company_mobile_dispatch_ns.abort(400, "Company ID invalide.")
        raise AssertionError("Company ID should be defined after abort") from None
    try:
        company_id = int(company_id_attr)
    except (TypeError, ValueError) as exc:
        company_mobile_dispatch_ns.abort(400, "Company ID invalide.")
        raise AssertionError("Company ID should be convertible to int") from exc
    return company, company_id


def _get_current_user() -> User:
    """Récupère l'utilisateur courant à partir du token JWT."""
    identity = get_jwt_identity()
    if identity is None:
        company_mobile_dispatch_ns.abort(401, "Token invalide ou expiré.")
        raise AssertionError("JWT identity missing after abort") from None
    user = User.query.filter_by(public_id=identity).first()
    if user is None:
        company_mobile_dispatch_ns.abort(404, "Utilisateur introuvable.")
        raise AssertionError("User should exist after abort") from None
    return user


def _serialize_dispatch_settings(company: Company) -> Dict[str, Any]:
    """Retourne les paramètres de dispatch pertinents pour l'app mobile."""
    settings_obj = dispatch_settings.for_company(company)
    overrides = company.get_autonomous_config().get("dispatch_overrides", {}) or {}

    fairness_overrides = overrides.get("fairness", {}) or {}
    fairness_max_gap = fairness_overrides.get("max_gap", MAX_FAIRNESS_GAP)

    service_times = settings_obj.service_times
    emergency_policy = settings_obj.emergency

    return {
        "fairness": {
            "max_gap": int(fairness_max_gap),
        },
        "emergency": {
            "emergency_penalty": float(emergency_policy.emergency_penalty),
        },
        "service_times": {
            "pickup_service_min": int(service_times.pickup_service_min),
            "dropoff_service_min": int(service_times.dropoff_service_min),
            "min_transition_margin_min": int(service_times.min_transition_margin_min),
        },
    }


DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
ACTIVE_ASSIGNMENT_STATUSES = {
    AssignmentStatus.SCHEDULED,
    AssignmentStatus.EN_ROUTE_PICKUP,
    AssignmentStatus.ARRIVED_PICKUP,
    AssignmentStatus.ONBOARD,
    AssignmentStatus.EN_ROUTE_DROPOFF,
}


def _format_datetime(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _driver_display_name(driver: Driver) -> str:
    user = driver.user
    if user:
        first_name = (user.first_name or "").strip()
        last_name = (user.last_name or "").strip()
        full = f"{first_name} {last_name}".strip()
        if full:
            return full
        if user.username:
            return user.username
    return (
        getattr(driver, "name", None) or f"Chauffeur #{driver.id if getattr(driver, 'id', None) is not None else '?'}"
    )


def _serialize_driver(driver: Optional[Driver]) -> Optional[Dict[str, Any]]:
    if not driver:
        return None
    driver_type = driver.driver_type.value if isinstance(driver.driver_type, DriverType) else str(driver.driver_type)
    is_emergency = str(driver_type).upper() == DriverType.EMERGENCY.value
    return {
        "id": str(driver.id),
        "name": _driver_display_name(driver),
        "is_emergency": is_emergency,
    }


def _resolve_booking_status(booking: Booking) -> str:
    status_value = getattr(booking.status, "value", str(booking.status or "")).upper()
    if status_value in {"CANCELED", "CANCELLED"}:
        return "cancelled"
    if status_value in {"COMPLETED", "RETURN_COMPLETED"}:
        return "completed"
    driver_id_value = getattr(booking, "driver_id", None)
    if isinstance(driver_id_value, int):
        return "assigned"
    return "unassigned"


def _get_active_assignment(booking: Booking) -> Optional[Assignment]:
    assignments = getattr(booking, "assignments", []) or []
    for assignment in assignments:
        if assignment.status in ACTIVE_ASSIGNMENT_STATUSES:
            return assignment
    return None


def _build_ride_summary(booking: Booking) -> Dict[str, Any]:
    active_assignment = _get_active_assignment(booking)
    driver = booking.driver or (active_assignment.driver if active_assignment and active_assignment.driver else None)
    drop_eta = None
    if active_assignment and active_assignment.eta_dropoff_at:
        drop_eta = _format_datetime(active_assignment.eta_dropoff_at)

    delay_seconds = getattr(active_assignment, "delay_seconds", 0) if active_assignment else 0
    risk_delay = bool(booking.is_urgent) or (isinstance(delay_seconds, (int, float)) and delay_seconds > 15 * 60)

    distance_meters = getattr(booking, "distance_meters", None)
    distance_km = round(distance_meters / 1000.0, 1) if isinstance(distance_meters, (int, float)) else None

    client_priority = "HIGH" if getattr(booking, "is_urgent", False) else "NORMAL"
    client_id_value = getattr(booking, "client_id", None)

    summary: Dict[str, Any] = {
        "id": str(booking.id),
        "time": {
            "pickup_at": booking.scheduled_time.isoformat() if booking.scheduled_time else None,
            "drop_eta": drop_eta,
            "window_start": None,
            "window_end": None,
        },
        "client": {
            "id": str(client_id_value) if client_id_value is not None else "None",
            "name": getattr(booking, "customer_full_name", None) or booking.customer_name or "Client",
            "priority": client_priority,
        },
        "route": {
            "pickup_address": booking.pickup_location or "",
            "dropoff_address": booking.dropoff_location or "",
            "distance_km": distance_km,
        },
        "status": _resolve_booking_status(booking),
        "driver": _serialize_driver(driver),
        "flags": {
            "risk_delay": risk_delay,
            "prefs_respected": True,
            "fairness_score": None,
            "override_pending": False,
        },
    }
    return summary


def _compute_driver_suggestions(company_id: int, booking: Booking) -> List[Dict[str, Any]]:
    pickup_lat = getattr(booking, "pickup_lat", None)
    pickup_lon = getattr(booking, "pickup_lon", None)
    if pickup_lat is None or pickup_lon is None:
        return []

    available_drivers = (
        Driver.query.options(joinedload(Driver.user))
        .filter(
            Driver.company_id == company_id,
            Driver.is_active.is_(True),
            Driver.is_available.is_(True),
        )
        .all()
    )

    driver_id_value = getattr(booking, "driver_id", None)
    current_driver_id = driver_id_value if isinstance(driver_id_value, int) else None

    suggestions: List[Dict[str, Any]] = []
    for driver in available_drivers:
        if current_driver_id is not None and driver.id == current_driver_id:
            continue
        if driver.latitude is None or driver.longitude is None:
            continue
        try:
            distance_km = haversine_distance(
                float(pickup_lat),
                float(pickup_lon),
                float(driver.latitude),
                float(driver.longitude),
            )
        except Exception:
            continue

        score = round(1.0 / (1.0 + distance_km), 4)
        driver_type = (
            driver.driver_type.value if isinstance(driver.driver_type, DriverType) else str(driver.driver_type)
        )
        suggestions.append(
            {
                "driver_id": str(driver.id),
                "driver_name": _driver_display_name(driver),
                "score": score,
                "fairness_delta": None,
                "preferred_match": False,
                "is_emergency": str(driver_type).upper() == DriverType.EMERGENCY.value,
                "reason": f"Distance estimée {distance_km:.1f} km",
            }
        )

    suggestions.sort(key=lambda s: s["score"], reverse=True)
    return suggestions[:3]


def _build_ride_history(booking: Booking, assignment: Optional[Assignment]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    created_at = getattr(booking, "created_at", None)
    history.append(
        {
            "ts": created_at.isoformat() if isinstance(created_at, datetime) else now_local().isoformat(),
            "event": "created",
            "actor": "system",
            "details": {"status": getattr(booking.status, "value", str(booking.status))},
        }
    )
    if assignment:
        assigned_at = getattr(assignment, "updated_at", None) or getattr(assignment, "created_at", None)
        history.append(
            {
                "ts": assigned_at.isoformat() if isinstance(assigned_at, datetime) else now_local().isoformat(),
                "event": "assigned",
                "actor": "dispatcher",
                "details": {
                    "driver_id": assignment.driver_id,
                    "status": getattr(assignment.status, "value", str(assignment.status)),
                },
            }
        )
    return history


def _build_ride_conflicts(booking: Booking) -> List[Dict[str, Any]]:
    booking_driver_id = getattr(booking, "driver_id", None)
    if booking_driver_id is None or not booking.scheduled_time:
        return []
    has_conflict, message = check_existing_assignment_conflict(
        driver_id=int(booking_driver_id),
        scheduled_time=booking.scheduled_time,
        booking_id=int(booking.id),
        tolerance_minutes=30,
    )
    if not has_conflict:
        return []
    return [
        {
            "type": "temporal",
            "message": message or "Conflit temporel détecté",
            "blocking": True,
        }
    ]


def _log_mobile_action(tools: AgentTools, kind: str, payload: Dict[str, Any], reasoning: str) -> Optional[str]:
    try:
        result = tools.log_action(kind=kind, payload=payload, reasoning_brief=reasoning)
        return result.get("event_id")
    except Exception as exc:  # pragma: no cover - logging best effort
        logger.warning("[MobileDispatch] Impossible de journaliser l'action %s: %s", kind, exc)
        return None


def _execute_assignment_action(company_id: int, booking_id: int, driver_id: int, action_kind: str) -> Dict[str, Any]:
    tools = AgentTools(company_id)
    assign_result = tools.assign(job_id=booking_id, driver_id=driver_id)

    if not assign_result.get("ok"):
        error_message = assign_result.get("error") or "Assignation impossible"
        if assign_result.get("conflict"):
            company_mobile_dispatch_ns.abort(409, error_message)
            raise AssertionError("Conflict should abort") from None
        company_mobile_dispatch_ns.abort(400, error_message)
        raise AssertionError("Assign failure should abort") from None

    booking = (
        Booking.query.options(
            selectinload(Booking.driver).selectinload(Driver.user),
            selectinload(Booking.assignments).selectinload(Assignment.driver).selectinload(Driver.user),
        )
        .filter(Booking.id == booking_id, Booking.company_id == company_id)
        .first()
    )

    if booking is None:
        company_mobile_dispatch_ns.abort(404, "Course introuvable après assignation")
        raise AssertionError("Booking should exist after assign") from None

    if booking.driver_id != driver_id:
        booking.driver_id = driver_id
        db.session.add(booking)
        db.session.commit()

    event_id = (
        _log_mobile_action(
            tools,
            action_kind,
            payload={
                "booking_id": booking_id,
                "driver_id": driver_id,
                "source": "mobile_enterprise",
            },
            reasoning=f"{action_kind} {booking_id} -> {driver_id}",
        )
        or ""
    )

    scheduled_time = booking.scheduled_time.isoformat() if booking.scheduled_time else None
    diff = assign_result.get("diff", {})
    return {
        "ride_id": str(booking_id),
        "driver_id": str(driver_id),
        "scheduled_time": scheduled_time,
        "fairness_delta": 0.0,
        "audit_event_id": event_id,
        "message": diff.get("action", "assigned"),
    }


@company_mobile_dispatch_ns.route("/v1/status")
class MobileDispatchStatus(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, company_id = _get_company_context()

        # Date cible pour les KPI (défaut: aujourd'hui en heure locale)
        requested_date = request.args.get("date")
        if not requested_date:
            requested_date = now_local().strftime("%Y-%m-%d")

        window_start: datetime
        window_end: datetime
        try:
            window_start, window_end = day_local_bounds(requested_date)
        except Exception as exc:
            company_mobile_dispatch_ns.abort(
                400,
                f"Format de date invalide: {requested_date} (attendu: YYYY-MM-DD)",
            )
            raise AssertionError("Invalid date should abort") from exc

        try:
            # KPI principaux
            bookings_query = Booking.query.filter(
                Booking.company_id == company_id,
                Booking.scheduled_time >= window_start,
                Booking.scheduled_time < window_end,
                Booking.status.in_(
                    [
                        BookingStatus.ACCEPTED,
                        BookingStatus.ASSIGNED,
                        BookingStatus.EN_ROUTE,
                        BookingStatus.IN_PROGRESS,
                        BookingStatus.COMPLETED,
                        BookingStatus.RETURN_COMPLETED,
                    ]
                ),
            )

            total_bookings = bookings_query.count()
            assigned_bookings = bookings_query.filter(Booking.driver_id.isnot(None)).count()
            assignment_rate = assigned_bookings / total_bookings if total_bookings > 0 else 0.0
            at_risk_count = bookings_query.filter(Booking.is_urgent.is_(True)).count()

            kpis = {
                "date": requested_date,
                "total_bookings": total_bookings,
                "assigned_bookings": assigned_bookings,
                "assignment_rate": round(assignment_rate, 4),
                "at_risk": at_risk_count,
            }

            # Santé OSRM via tools agent
            osrm_status_payload = {
                "status": "DOWN",
                "latency_ms": None,
                "last_check": None,
            }
            try:
                tools = AgentTools(company_id)
                with current_app.app_context():
                    osrm_health = tools.osrm_health()
                osrm_state = osrm_health.get("state", "OPEN")
                latency = osrm_health.get("latency_ms")
                test_successful = osrm_health.get("test_successful", False)

                latency_value = latency if isinstance(latency, int) and latency >= 0 else None
                if osrm_state == "CLOSED" and test_successful:
                    osrm_status = "OK"
                elif osrm_state in {"HALF_OPEN", "OPEN"} and test_successful:
                    osrm_status = "WARNING"
                else:
                    osrm_status = "DOWN"

                osrm_status_payload = {
                    "status": osrm_status,
                    "latency_ms": latency_value,
                    "last_check": now_local().isoformat(),
                }
            except Exception as exc:  # pragma: no cover - fallback résilient
                logger.warning(
                    "[MobileDispatch] OSRM health check failed for company %s: %s",
                    company_id,
                    exc,
                )

            # Statut agent orchestrator
            agent_mode_value = getattr(company.dispatch_mode, "value", "manual")
            agent_mode = agent_mode_value.upper()
            agent_active = False
            agent_last_tick = None
            try:
                agent = get_agent_for_company(company_id, app=current_app._get_current_object())
                agent_status = agent.get_status()
                agent_active = bool(agent_status.get("running"))
                agent_last_tick = agent_status.get("last_tick")
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "[MobileDispatch] Impossible de récupérer le statut agent pour company %s: %s",
                    company_id,
                    exc,
                )

            agent_payload = {
                "mode": agent_mode,
                "active": agent_active,
                "last_tick": agent_last_tick,
            }

            # Statut optimizer
            optimizer_payload = {
                "active": False,
                "next_window_start": None,
            }
            try:
                optimizer = get_optimizer_for_company(company_id)
                if optimizer:
                    optimizer_status = optimizer.get_status()
                    optimizer_active = bool(optimizer_status.get("running"))
                    last_check = optimizer_status.get("last_check")
                    interval_seconds = optimizer_status.get("check_interval_seconds")

                    next_window_iso = None
                    if last_check and interval_seconds:
                        try:
                            last_check_dt = datetime.fromisoformat(last_check)
                            next_window_iso = (last_check_dt + timedelta(seconds=int(interval_seconds))).isoformat()
                        except (ValueError, TypeError):
                            next_window_iso = last_check

                    optimizer_payload = {
                        "active": optimizer_active,
                        "next_window_start": next_window_iso,
                    }
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "[MobileDispatch] Impossible de récupérer le statut optimizer pour company %s: %s",
                    company_id,
                    exc,
                )

            response = {
                "osrm": osrm_status_payload,
                "agent": agent_payload,
                "optimizer": optimizer_payload,
                "kpis": kpis,
            }

            return response, 200
        except Exception as exc:
            db.session.rollback()
            logger.exception(
                "[MobileDispatch] Erreur récupération statut pour company %s: %s",
                company_id,
                exc,
            )
            company_mobile_dispatch_ns.abort(500, "Impossible de récupérer le statut dispatch mobile.")


@company_mobile_dispatch_ns.route("/v1/rides")
class MobileDispatchRides(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        _, company_id = _get_company_context()

        requested_date = request.args.get("date") or now_local().strftime("%Y-%m-%d")
        status_filter = (request.args.get("status") or "").strip().lower() or None
        search_query = (request.args.get("q") or "").strip()

        try:
            page = int(request.args.get("page", "1"))
        except ValueError:
            page = 1
        page = max(1, page)

        try:
            page_size = int(request.args.get("page_size", str(DEFAULT_PAGE_SIZE)))
        except ValueError:
            page_size = DEFAULT_PAGE_SIZE
        page_size = max(1, min(page_size, MAX_PAGE_SIZE))

        window_start: datetime
        window_end: datetime
        try:
            window_start, window_end = day_local_bounds(requested_date)
        except Exception as exc:
            company_mobile_dispatch_ns.abort(
                400,
                f"Format de date invalide: {requested_date} (attendu: YYYY-MM-DD)",
            )
            raise AssertionError("Invalid date should abort") from exc

        ACTIVE_BOOKING_STATUSES = [
            BookingStatus.ACCEPTED,
            BookingStatus.ASSIGNED,
            BookingStatus.EN_ROUTE,
            BookingStatus.IN_PROGRESS,
            BookingStatus.COMPLETED,
            BookingStatus.RETURN_COMPLETED,
        ]

        bookings_query = Booking.query.options(
            selectinload(Booking.driver).selectinload(Driver.user),
            selectinload(Booking.assignments).selectinload(Assignment.driver).selectinload(Driver.user),
        ).filter(
            Booking.company_id == company_id,
            or_(
                Booking.scheduled_time.is_(None),
                and_(
                    Booking.scheduled_time >= window_start,
                    Booking.scheduled_time < window_end,
                ),
            ),
            Booking.status.in_(ACTIVE_BOOKING_STATUSES),
        )

        if status_filter == "assigned":
            bookings_query = bookings_query.filter(Booking.driver_id.isnot(None))
        elif status_filter == "unassigned":
            bookings_query = bookings_query.filter(
                Booking.driver_id.is_(None),
                Booking.status.in_(
                    [
                        BookingStatus.ACCEPTED,
                        BookingStatus.ASSIGNED,
                        BookingStatus.EN_ROUTE,
                        BookingStatus.IN_PROGRESS,
                    ]
                ),
            )
        elif status_filter == "urgent":
            bookings_query = bookings_query.filter(Booking.is_urgent.is_(True))
        elif status_filter == "cancelled":
            bookings_query = bookings_query.filter(Booking.status.in_([BookingStatus.CANCELED]))

        if search_query:
            like_value = f"%{search_query}%"
            bookings_query = bookings_query.filter(
                or_(
                    Booking.customer_name.ilike(like_value),
                    Booking.pickup_location.ilike(like_value),
                    Booking.dropoff_location.ilike(like_value),
                )
            )

        total = bookings_query.count()

        bookings = (
            bookings_query.order_by(
                Booking.scheduled_time.is_(None),
                Booking.scheduled_time.asc(),
                Booking.id.asc(),
            )
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        items = [_build_ride_summary(booking) for booking in bookings]

        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "items": items,
        }, 200


@company_mobile_dispatch_ns.route("/v1/rides/<string:ride_id>")
class MobileRideDetail(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, ride_id: str):
        _, company_id = _get_company_context()

        try:
            booking_id = int(ride_id)
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(400, "ride_id invalide (entier attendu)")
            raise AssertionError("Invalid ride_id should abort") from exc

        booking: Booking | None = (
            Booking.query.options(
                selectinload(Booking.driver).selectinload(Driver.user),
                selectinload(Booking.assignments).selectinload(Assignment.driver).selectinload(Driver.user),
            )
            .filter(
                Booking.id == booking_id,
                Booking.company_id == company_id,
            )
            .first()
        )

        if booking is None:
            company_mobile_dispatch_ns.abort(404, "Course introuvable pour cette entreprise")
            raise AssertionError("abort() should have raised an exception")  # Type hint: abort() lève une exception

        active_assignment = _get_active_assignment(booking)
        summary = _build_ride_summary(booking)
        suggestions = _compute_driver_suggestions(company_id, booking)
        history = _build_ride_history(booking, active_assignment)
        conflicts = _build_ride_conflicts(booking)

        notes: List[str] = []
        notes_dispatch = getattr(booking, "notes_medical", None)
        if notes_dispatch:
            notes.append(str(notes_dispatch))

        detail_payload = {
            "summary": summary,
            "suggestions": suggestions,
            "history": history,
            "conflicts": conflicts,
            "notes": notes,
        }

        return detail_payload, 200


@company_mobile_dispatch_ns.route("/v1/rides/<string:ride_id>/assign")
class MobileRideAssign(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, ride_id: str):
        _, company_id = _get_company_context()

        try:
            booking_id = int(ride_id)
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(400, "ride_id invalide (entier attendu)")
            raise AssertionError("Invalid ride_id should abort") from exc

        payload = request.get_json(silent=True) or {}
        driver_id_raw = payload.get("driver_id")
        if driver_id_raw is None:
            company_mobile_dispatch_ns.abort(400, "driver_id manquant dans la requête")
            raise AssertionError("driver_id should not be None after abort") from None

        try:
            driver_id = int(driver_id_raw)
        except (TypeError, ValueError) as exc:
            company_mobile_dispatch_ns.abort(400, "driver_id invalide (entier attendu)")
            raise AssertionError("Invalid driver_id should abort") from exc

        response_payload = _execute_assignment_action(
            company_id=company_id,
            booking_id=booking_id,
            driver_id=driver_id,
            action_kind="mobile_assign",
        )

        return response_payload, 200


@company_mobile_dispatch_ns.route("/v1/rides/<string:ride_id>/reassign")
class MobileRideReassign(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, ride_id: str):
        _, company_id = _get_company_context()

        try:
            booking_id = int(ride_id)
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(400, "ride_id invalide (entier attendu)")
            raise AssertionError("Invalid ride_id should abort") from exc

        payload = request.get_json(silent=True) or {}
        driver_id_raw = payload.get("driver_id")
        if driver_id_raw is None:
            company_mobile_dispatch_ns.abort(400, "driver_id manquant dans la requête")
            raise AssertionError("driver_id should not be None after abort") from None

        try:
            driver_id = int(driver_id_raw)
        except (TypeError, ValueError) as exc:
            company_mobile_dispatch_ns.abort(400, "driver_id invalide (entier attendu)")
            raise AssertionError("Invalid driver_id should abort") from exc

        response_payload = _execute_assignment_action(
            company_id=company_id,
            booking_id=booking_id,
            driver_id=driver_id,
            action_kind="mobile_reassign",
        )

        return response_payload, 200


@company_mobile_dispatch_ns.route("/v1/rides/<string:ride_id>/schedule")
class MobileRideSchedule(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, ride_id: str):
        _, company_id = _get_company_context()

        try:
            booking_id = int(ride_id)
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(400, "ride_id invalide (entier attendu)")
            raise AssertionError("Invalid ride_id should abort") from exc

        payload = request.get_json(silent=True) or {}
        pickup_at_raw = payload.get("pickup_at")
        delta_minutes_raw = payload.get("delta_minutes")

        if pickup_at_raw is None and delta_minutes_raw is None:
            company_mobile_dispatch_ns.abort(400, "Il faut fournir soit pickup_at, soit delta_minutes.")
            raise AssertionError("Schedule payload invalid") from None

        new_datetime: Optional[datetime] = None

        if pickup_at_raw is not None:
            try:
                parsed_dt = parse_local_naive(pickup_at_raw)
            except Exception as exc:
                company_mobile_dispatch_ns.abort(400, "Format pickup_at invalide (ISO attendu)")
                raise AssertionError("pickup_at parse error") from exc
            if parsed_dt is None:
                company_mobile_dispatch_ns.abort(400, "pickup_at ne peut pas être nul")
                raise AssertionError("pickup_at null") from None
            new_datetime = parsed_dt
        elif delta_minutes_raw is not None:
            try:
                delta_minutes = int(delta_minutes_raw)
            except (TypeError, ValueError) as exc:
                company_mobile_dispatch_ns.abort(400, "delta_minutes doit être un entier")
                raise AssertionError("delta parse error") from exc
            booking = Booking.query.filter(Booking.id == booking_id, Booking.company_id == company_id).first()
            if booking is None:
                company_mobile_dispatch_ns.abort(404, "Course introuvable")
                raise AssertionError("Booking not found") from None
            base_dt = booking.scheduled_time or now_local()
            new_datetime = base_dt + timedelta(minutes=delta_minutes)
        else:
            company_mobile_dispatch_ns.abort(400, "Paramètres planning invalides")
            raise AssertionError("Invalid scheduling parameters") from None

        booking = Booking.query.filter(Booking.id == booking_id, Booking.company_id == company_id).first()
        if booking is None:
            company_mobile_dispatch_ns.abort(404, "Course introuvable")
            raise AssertionError("Booking not found") from None

        booking.scheduled_time = new_datetime
        db.session.add(booking)
        db.session.commit()

        tools = AgentTools(company_id)
        event_id = (
            _log_mobile_action(
                tools,
                "mobile_schedule",
                payload={
                    "booking_id": booking_id,
                    "scheduled_time": booking.scheduled_time.isoformat() if booking.scheduled_time else None,
                    "source": "mobile_enterprise",
                },
                reasoning=f"Planification mobile {booking_id} -> {booking.scheduled_time.isoformat() if booking.scheduled_time else 'None'}",
            )
            or ""
        )

        return {
            "ride_id": str(booking_id),
            "scheduled_time": booking.scheduled_time.isoformat() if booking.scheduled_time else None,
            "audit_event_id": event_id,
        }, 200


@company_mobile_dispatch_ns.route("/v1/rides/<string:ride_id>/urgent")
class MobileRideUrgent(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, ride_id: str):
        _, company_id = _get_company_context()

        try:
            booking_id = int(ride_id)
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(400, "ride_id invalide (entier attendu)")
            raise AssertionError("Invalid ride_id should abort") from exc

        payload = request.get_json(silent=True) or {}
        extra_delay_minutes_raw = payload.get("extra_delay_minutes", 15)
        reason = payload.get("reason")

        try:
            extra_delay_minutes = int(extra_delay_minutes_raw)
        except (TypeError, ValueError) as exc:
            company_mobile_dispatch_ns.abort(400, "extra_delay_minutes doit être un entier")
            raise AssertionError("extra delay parse error") from exc

        booking = Booking.query.filter(Booking.id == booking_id, Booking.company_id == company_id).first()
        if booking is None:
            company_mobile_dispatch_ns.abort(404, "Course introuvable")
            raise AssertionError("Booking not found") from None

        booking.is_urgent = True
        if booking.scheduled_time:
            booking.scheduled_time = booking.scheduled_time + timedelta(minutes=extra_delay_minutes)

        db.session.add(booking)
        db.session.commit()

        tools = AgentTools(company_id)
        event_id = (
            _log_mobile_action(
                tools,
                "mobile_mark_urgent",
                payload={
                    "booking_id": booking_id,
                    "reason": reason,
                    "extra_delay_minutes": extra_delay_minutes,
                    "source": "mobile_enterprise",
                },
                reasoning=f"Marquage urgent mobile {booking_id} (+{extra_delay_minutes} min)",
            )
            or ""
        )

        return {
            "ride_id": str(booking_id),
            "is_urgent": True,
            "scheduled_time": booking.scheduled_time.isoformat() if booking.scheduled_time else None,
            "audit_event_id": event_id,
        }, 200


@company_mobile_dispatch_ns.route("/v1/mode")
class MobileDispatchMode(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, company_id = _get_company_context()
        mode_value = getattr(company.dispatch_mode, "value", "manual")
        return {
            "company_id": company_id,
            "dispatch_mode": mode_value,
            "autonomous_config": company.get_autonomous_config(),
        }, 200

    @jwt_required()
    @role_required(UserRole.company)
    def put(self):
        company, company_id = _get_company_context()
        payload = request.get_json(silent=True) or {}
        new_mode = payload.get("dispatch_mode")
        reason = payload.get("reason")

        if not new_mode:
            company_mobile_dispatch_ns.abort(400, "dispatch_mode requis")
            raise AssertionError("dispatch_mode required") from None

        try:
            target_mode = DispatchMode(new_mode)
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(400, "Mode invalide. Utilisez manual, semi_auto ou fully_auto.")
            raise AssertionError("Invalid dispatch mode") from exc

        previous_mode = getattr(company.dispatch_mode, "value", None)
        if previous_mode == target_mode.value:
            return {
                "company_id": company_id,
                "dispatch_mode": previous_mode,
                "previous_mode": previous_mode,
                "effective_at": datetime.now(UTC).isoformat(),
                "message": "Aucun changement (mode identique).",
            }, 200

        cast("Any", company).dispatch_mode = target_mode

        try:
            if target_mode.value == "fully_auto":
                agent = get_agent_for_company(company_id, app=current_app._get_current_object())
                if not agent.state.running:
                    agent.start()
                    logger.info(
                        "[Dispatch-Mobile] Agent démarré automatiquement pour company %s (mode fully_auto)",
                        company_id,
                    )
            elif previous_mode == "fully_auto":
                stop_agent_for_company(company_id)
                logger.info(
                    "[Dispatch-Mobile] Agent arrêté pour company %s (mode %s)",
                    company_id,
                    target_mode.value,
                )
        except Exception as agent_err:
            logger.warning(
                "[Dispatch-Mobile] Erreur lors du contrôle agent (%s): %s",
                company_id,
                agent_err,
            )

        try:
            db.session.add(company)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.exception(
                "[Dispatch-Mobile] Échec mise à jour mode dispatch pour company %s",
                company_id,
            )
            company_mobile_dispatch_ns.abort(500, f"Impossible de mettre à jour le mode dispatch: {exc}")
            raise AssertionError("Commit failed") from exc

        logger.info(
            "[Dispatch-Mobile] Company %s mode %s -> %s (reason=%s)",
            company_id,
            previous_mode,
            target_mode.value,
            reason,
        )

        return {
            "company_id": company_id,
            "dispatch_mode": target_mode.value,
            "previous_mode": previous_mode,
            "effective_at": datetime.now(UTC).isoformat(),
            "message": "Mode mis à jour avec succès.",
        }, 200


@company_mobile_dispatch_ns.route("/v1/settings")
class MobileDispatchSettings(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, _ = _get_company_context()
        return _serialize_dispatch_settings(company), 200

    @jwt_required()
    @role_required(UserRole.company)
    def put(self):
        company, company_id = _get_company_context()
        payload = request.get_json(silent=True) or {}

        if not isinstance(payload, dict):
            company_mobile_dispatch_ns.abort(400, "Payload JSON invalide.")
            raise AssertionError("Invalid payload should abort") from None

        autonomous_config = company.get_autonomous_config() or {}
        dispatch_overrides = autonomous_config.get("dispatch_overrides", {}) or {}

        updated = False

        fairness_update = payload.get("fairness")
        if fairness_update is not None:
            if not isinstance(fairness_update, dict):
                company_mobile_dispatch_ns.abort(400, "fairness doit être un objet JSON.")
                raise AssertionError("Invalid fairness payload should abort") from None
            fairness_overrides = dispatch_overrides.get("fairness", {}) or {}
            if "max_gap" in fairness_update:
                try:
                    max_gap_val = int(fairness_update["max_gap"])
                except (TypeError, ValueError) as exc:
                    company_mobile_dispatch_ns.abort(400, "fairness.max_gap doit être un entier.")
                    raise AssertionError("Invalid fairness value should abort") from exc
                fairness_overrides["max_gap"] = max_gap_val
                updated = True
            if fairness_overrides:
                dispatch_overrides["fairness"] = fairness_overrides

        emergency_update = payload.get("emergency")
        if emergency_update is not None:
            if not isinstance(emergency_update, dict):
                company_mobile_dispatch_ns.abort(400, "emergency doit être un objet JSON.")
                raise AssertionError("Invalid emergency payload should abort") from None
            emergency_overrides = dispatch_overrides.get("emergency", {}) or {}
            if "emergency_penalty" in emergency_update:
                try:
                    penalty_val = float(emergency_update["emergency_penalty"])
                except (TypeError, ValueError) as exc:
                    company_mobile_dispatch_ns.abort(400, "emergency.emergency_penalty doit être numérique.")
                    raise AssertionError("Invalid emergency value should abort") from exc
                emergency_overrides["emergency_penalty"] = penalty_val
                updated = True
            if emergency_overrides:
                dispatch_overrides["emergency"] = emergency_overrides

        service_times_update = payload.get("service_times")
        if service_times_update is not None:
            if not isinstance(service_times_update, dict):
                company_mobile_dispatch_ns.abort(400, "service_times doit être un objet JSON.")
                raise AssertionError("Invalid service_times payload should abort") from None
            service_overrides = dispatch_overrides.get("service_times", {}) or {}
            for key in ("pickup_service_min", "dropoff_service_min", "min_transition_margin_min"):
                if key in service_times_update:
                    try:
                        service_overrides[key] = int(service_times_update[key])
                    except (TypeError, ValueError) as exc:
                        company_mobile_dispatch_ns.abort(400, f"service_times.{key} doit être un entier.")
                        raise AssertionError("Invalid service_times value should abort") from exc
                    updated = True
            if service_overrides:
                dispatch_overrides["service_times"] = service_overrides

        if not updated:
            return {"message": "Aucune modification détectée"}, 200

        autonomous_config["dispatch_overrides"] = dispatch_overrides
        company.set_autonomous_config(autonomous_config)

        try:
            db.session.add(company)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.exception("[MobileDispatch] Échec mise à jour paramètres mobile: %s", exc)
            company_mobile_dispatch_ns.abort(500, "Impossible de sauvegarder les paramètres.")
            raise AssertionError("Settings update failed after abort") from exc

        tools = AgentTools(company_id)
        _log_mobile_action(
            tools,
            "mobile_update_settings",
            payload={
                "fairness": fairness_update,
                "emergency": emergency_update,
                "service_times": service_times_update,
                "source": "mobile_enterprise",
            },
            reasoning="Mise à jour des paramètres dispatch via mobile",
        )

        # Rafraîchir les settings après commit
        db.session.refresh(company)
        return _serialize_dispatch_settings(company), 200


@company_mobile_dispatch_ns.route("/v1/run")
class MobileDispatchRun(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("10/minute")
    def post(self):
        company, company_id = _get_company_context()
        body = request.get_json(silent=True) or {}
        target_date = body.get("date") or now_local().strftime("%Y-%m-%d")

        try:
            datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError as exc:
            company_mobile_dispatch_ns.abort(400, "Format de date invalide (YYYY-MM-DD).")
            raise AssertionError("Invalid dispatch date should abort") from exc

        params: Dict[str, Any] = {
            "company_id": company_id,
            "for_date": target_date,
            "regular_first": True,
            "allow_emergency": True,
            "mode": "auto",
            "dispatch_overrides": {
                "fairness": {
                    "enable_fairness": True,
                    "fairness_window_days": 2,
                    "fairness_weight": 0.7,
                    "reset_daily_load": True,
                },
                "heuristic": {
                    "driver_load_balance": 0.7,
                    "proximity": 0.2,
                    "priority": 0.08,
                    "return_urgency": 0.02,
                },
                "solver": {
                    "max_bookings_per_driver": 999,
                },
                "emergency": {
                    "allow_emergency_drivers": True,
                    "emergency_penalty": 600.0,
                },
            },
        }

        dispatch_overrides = params.get("dispatch_overrides") or {}
        if dispatch_overrides:
            try:
                base_settings = dispatch_settings.for_company(company)
                _, validation = dispatch_settings.merge_overrides(
                    base_settings,
                    dispatch_overrides,
                    return_validation=True,
                )
                logger.info(
                    "[MobileDispatch] Validation overrides: applied=%s ignored=%s errors=%s",
                    validation.get("applied"),
                    validation.get("ignored"),
                    validation.get("errors"),
                )
                critical_errors = validation.get("critical_errors", [])
                if critical_errors:
                    message = "Paramètres critiques ignorés: " + ", ".join(critical_errors)
                    logger.warning(
                        "[MobileDispatch] Overrides rejetés (critique): %s",
                        critical_errors,
                    )
                    company_mobile_dispatch_ns.abort(400, message)
            except ValueError as exc:
                logger.exception("[MobileDispatch] Validation overrides échouée: %s", exc)
                company_mobile_dispatch_ns.abort(400, "Paramètres overrides invalides.")
            params["overrides"] = dispatch_overrides
            params.pop("dispatch_overrides", None)

        job = trigger_job(company_id, params)

        tools = AgentTools(company_id)
        _log_mobile_action(
            tools,
            "mobile_run_dispatch",
            payload={"params": params, "source": "mobile_enterprise"},
            reasoning=f"Lancement dispatch mobile pour {target_date}",
        )

        response = {
            "message": f"Dispatch lancé pour {target_date}",
            "job": job,
            "for_date": target_date,
        }
        return response, 202


@company_mobile_dispatch_ns.route("/v1/optimizer/run")
class MobileOptimizerRun(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("10/minute")
    def post(self):
        _, company_id = _get_company_context()
        body = request.get_json(silent=True) or {}
        target_date = body.get("date")
        if target_date:
            try:
                datetime.strptime(target_date, "%Y-%m-%d")
            except ValueError as exc:
                company_mobile_dispatch_ns.abort(400, "Format de date invalide (YYYY-MM-DD).")
                raise AssertionError("Invalid optimizer date should abort") from exc

        opportunities = check_opportunities_manual(company_id, target_date, app=current_app._get_current_object())
        payload = [opp.to_dict() for opp in opportunities]

        tools = AgentTools(company_id)
        _log_mobile_action(
            tools,
            "mobile_optimizer_manual",
            payload={
                "count": len(payload),
                "for_date": target_date,
                "source": "mobile_enterprise",
            },
            reasoning=f"Relance optimiseur mobile ({len(payload)} opportunités)",
        )

        return {
            "message": "Optimisation recalculée",
            "count": len(payload),
            "opportunities": payload,
            "for_date": target_date,
        }, 200


@company_mobile_dispatch_ns.route("/v1/reset")
class MobileDispatchReset(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        _, company_id = _get_company_context()
        body = request.get_json(silent=True) or {}
        date_str = body.get("date")

        start_datetime: Optional[datetime] = None
        end_datetime: Optional[datetime] = None

        if date_str:
            try:
                target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError as exc:
                company_mobile_dispatch_ns.abort(400, "Format de date invalide. Utilisez YYYY-MM-DD.")
                raise AssertionError("Invalid reset date should abort") from exc
            start_datetime = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=UTC)
            end_datetime = datetime.combine(target_date, datetime.max.time()).replace(tzinfo=UTC)

        try:
            query = Assignment.query.join(Booking).filter(Booking.company_id == company_id)
            if start_datetime and end_datetime:
                query = query.filter(
                    Booking.scheduled_time >= start_datetime,
                    Booking.scheduled_time < end_datetime,
                )

            assignments = query.all()
            booking_ids = [assignment.booking_id for assignment in assignments]

            assignments_deleted = len(assignments)
            for assignment in assignments:
                db.session.delete(assignment)

            bookings_query = Booking.query.filter(Booking.company_id == company_id)
            if booking_ids:
                bookings_query = bookings_query.filter(Booking.id.in_(booking_ids))
            if start_datetime and end_datetime:
                bookings_query = bookings_query.filter(
                    Booking.scheduled_time >= start_datetime,
                    Booking.scheduled_time < end_datetime,
                )

            bookings_reset = 0
            for booking in bookings_query.all():
                if booking.status == BookingStatus.ASSIGNED:
                    booking.status = BookingStatus.ACCEPTED
                    booking.driver_id = None
                    bookings_reset += 1

            db.session.commit()

            tools = AgentTools(company_id)
            _log_mobile_action(
                tools,
                "mobile_reset_assignments",
                payload={
                    "date": date_str,
                    "assignments_deleted": assignments_deleted,
                    "bookings_reset": bookings_reset,
                    "source": "mobile_enterprise",
                },
                reasoning="Réinitialisation des assignations via mobile",
            )

            return {
                "message": "Réinitialisation effectuée",
                "assignments_deleted": assignments_deleted,
                "bookings_reset": bookings_reset,
                "date": date_str or "toutes les dates",
            }, 200
        except Exception as exc:
            db.session.rollback()
            logger.exception("[MobileDispatch] Erreur reset mobile: %s", exc)
            company_mobile_dispatch_ns.abort(500, "Erreur lors de la réinitialisation.")
            raise AssertionError("Reset failed after abort") from exc


@company_mobile_dispatch_ns.route("/v1/incidents")
class MobileDispatchIncidents(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        _, company_id = _get_company_context()
        payload = request.get_json(silent=True) or {}

        if not isinstance(payload, dict):
            company_mobile_dispatch_ns.abort(400, "Payload JSON invalide.")
            raise AssertionError("Invalid incident payload should abort") from None

        incident_type = payload.get("type") or "incident"
        severity = payload.get("severity") or "medium"

        ride_id_raw = payload.get("ride_id")
        driver_id_raw = payload.get("driver_id")
        booking_id = None
        driver_id = None

        if ride_id_raw is not None:
            try:
                booking_id = int(ride_id_raw)
            except (TypeError, ValueError) as exc:
                company_mobile_dispatch_ns.abort(400, "ride_id doit être un entier.")
                raise AssertionError("Invalid ride_id should abort") from exc

        if driver_id_raw is not None:
            try:
                driver_id = int(driver_id_raw)
            except (TypeError, ValueError) as exc:
                company_mobile_dispatch_ns.abort(400, "driver_id doit être un entier.")
                raise AssertionError("Invalid driver_id should abort") from exc

        note = payload.get("note")
        attachments = payload.get("attachments") or []
        if attachments and not isinstance(attachments, list):
            company_mobile_dispatch_ns.abort(400, "attachments doit être une liste.")
            raise AssertionError("Invalid attachments should abort") from None

        action = AutonomousAction()
        action.company_id = company_id
        action.booking_id = booking_id
        action.driver_id = driver_id
        action.action_type = "mobile_incident"
        action.action_description = f"Incident mobile: {incident_type} (severity={severity})"
        action.action_data = json.dumps(
            {
                "type": incident_type,
                "severity": severity,
                "note": note,
                "attachments": attachments,
                "source": "mobile_enterprise",
            }
        )
        action.trigger_source = "mobile_enterprise"
        action.success = True

        try:
            db.session.add(action)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.exception("[MobileDispatch] Échec enregistrement incident mobile: %s", exc)
            company_mobile_dispatch_ns.abort(500, "Impossible d'enregistrer l'incident.")
            raise AssertionError("Incident insert failed after abort") from exc

        tools = AgentTools(company_id)
        _log_mobile_action(
            tools,
            "mobile_incident_report",
            payload={
                "incident_id": action.id,
                "type": incident_type,
                "severity": severity,
                "booking_id": booking_id,
                "driver_id": driver_id,
                "source": "mobile_enterprise",
            },
            reasoning=f"Signalement incident ({incident_type}) via mobile",
        )

        return {
            "message": "Incident enregistré",
            "incident_id": action.id,
        }, 201


@company_mobile_dispatch_ns.route("/v1/chat/messages")
class MobileDispatchChat(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        _, company_id = _get_company_context()

        try:
            limit = max(1, min(100, int(request.args.get("limit", 50))))
        except (TypeError, ValueError):
            company_mobile_dispatch_ns.abort(400, "Paramètre limit invalide.")
            raise AssertionError("Invalid chat limit should abort") from None

        before = request.args.get("before")
        before_dt: Optional[datetime] = None
        if before:
            try:
                before_dt = datetime.fromisoformat(before.rstrip("Z"))
            except ValueError as exc:
                company_mobile_dispatch_ns.abort(400, "Paramètre before invalide (ISO8601 attendu).")
                raise AssertionError("Invalid before timestamp should abort") from exc

        query = Message.query.filter(Message.company_id == company_id).order_by(Message.timestamp.desc())
        if before_dt:
            query = query.filter(Message.timestamp < before_dt)

        messages = list(reversed(query.limit(limit).all()))
        serialized = []
        for message in messages:
            try:
                serialized.append(message.serialize)
            except Exception:
                serialized.append(
                    {
                        "id": message.id,
                        "content": message.content,
                        "timestamp": message.timestamp.isoformat() if message.timestamp else None,
                        "sender_role": getattr(message.sender_role, "value", message.sender_role),
                        "sender_id": message.sender_id,
                        "receiver_id": message.receiver_id,
                    }
                )

        return {
            "messages": serialized,
            "count": len(serialized),
        }, 200

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        _, company_id = _get_company_context()
        user = _get_current_user()
        payload = request.get_json(silent=True) or {}

        content = payload.get("content")
        if not content or not str(content).strip():
            company_mobile_dispatch_ns.abort(400, "Le champ content est requis.")
            raise AssertionError("Chat content invalid should abort") from None

        receiver_id = None
        if "receiver_id" in payload and payload["receiver_id"] is not None:
            try:
                receiver_id = int(payload["receiver_id"])
            except (TypeError, ValueError) as exc:
                company_mobile_dispatch_ns.abort(400, "receiver_id doit être un entier.")
                raise AssertionError("Invalid receiver_id should abort") from exc

        message = Message()
        message.company_id = company_id
        message.sender_id = getattr(user, "id", None)
        message.receiver_id = receiver_id
        message.sender_role = SenderRole.COMPANY
        message.content = str(content)

        try:
            db.session.add(message)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logger.exception("[MobileDispatch] Échec envoi message mobile: %s", exc)
            company_mobile_dispatch_ns.abort(500, "Impossible d'envoyer le message.")
            raise AssertionError("Chat insert failed after abort") from exc

        tools = AgentTools(company_id)
        _log_mobile_action(
            tools,
            "mobile_chat_message",
            payload={
                "message_id": message.id,
                "receiver_id": receiver_id,
                "source": "mobile_enterprise",
            },
            reasoning="Message envoyé via app mobile entreprise",
        )

        return message.serialize, 201
