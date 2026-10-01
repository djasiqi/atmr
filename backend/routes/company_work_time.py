"""API entreprise du temps de travail chauffeur."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import UTC, date, datetime

from flask import current_app, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Resource
from sqlalchemy import text

from domain.work_time.clock import elapsed_minutes, zurich_date_of_instant
from domain.work_time.periods import (
    is_full_calendar_month,
    monthly_closure_refusal,
    parse_ymd,
    zurich_period_bounds,
    zurich_today,
)
from domain.work_time.report import (
    apply_ledger_snapshot,
    closure_parity_errors,
    proposal_for_segment,
)
from ext import db, role_required
from models.booking import Booking
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
from models.enums import BookingStatus, UserRole
from models.user import User
from routes.companies import _get_current_company_via_use_case, companies_ns
from services.work_time.permissions import can_configure, can_manage, can_view
from services.work_time.report_service import (
    WorkTimeReportService,
    _adjustment_dict,
    _estimate_settings,
    _to_segment,
    load_cutover,
    work_date_in_zurich,
)
from shared.audit_helpers import audit_log
from shared.time_utils import LOCAL_TZ

logger = logging.getLogger(__name__)

_WORK_TYPES = {
    "extra_transport",
    "delivery",
    "accompaniment",
    "waiting",
    "administrative",
    "cleaning",
    "training",
    "other",
}
_FILTERS = {"all", "transports", "manual", "adjusted", "anomalies"}
_COMPLETED = {BookingStatus.COMPLETED.value, BookingStatus.RETURN_COMPLETED.value}
_CLOSURE_LOCK_NS = 42021
_CLOSED_PERIOD_ERROR = {
    "error": (
        "Cette date appartient à une période clôturée. "
        "Réouvrez-la explicitement avant de modifier la rémunération."
    )
}


def _company_or_error():
    company, error, status = _get_current_company_via_use_case()
    if error:
        return None, error, status
    company_id = getattr(company, "id", None)
    if company_id is None:
        return None, {"error": "Entreprise introuvable"}, 404
    return company, None, None


def _actor_id() -> int | None:
    public_id = get_jwt_identity()
    user = db.session.query(User).filter(User.public_id == public_id).first()
    if user is None:
        return None
    return int(user.id)


def _service() -> WorkTimeReportService:
    from config import Config

    raw = current_app.config.get(
        "WORK_TIME_ARRIVED_AT_CUTOVER_AT",
        Config.WORK_TIME_ARRIVED_AT_CUTOVER_AT,
    )
    return WorkTimeReportService(load_cutover(str(raw)))


def _parse_dt(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.astimezone(UTC)


def _period_args() -> tuple[str, str] | tuple[dict, int]:
    start = (request.args.get("from") or "").strip()
    end = (request.args.get("to") or "").strip()
    if not start or not end:
        return {"error": "Les paramètres from et to (YYYY-MM-DD) sont requis."}, 400
    try:
        parse_ymd(start)
        parse_ymd(end)
    except ValueError:
        return {"error": "Période invalide. Format attendu : YYYY-MM-DD."}, 400
    if end < start:
        return {"error": "La date de fin précède la date de début."}, 400
    return start, end


def _driver_in_company(driver_id: int, company_id: int) -> Driver | None:
    driver = db.session.get(Driver, driver_id)
    if driver is None or int(driver.company_id) != int(company_id):
        return None
    return driver


def _lock_company_closures(company_id: int) -> None:
    """Sérialise finalisation et réouverture d'une même entreprise."""
    db.session.execute(
        text("SELECT pg_advisory_xact_lock(:ns, :company_id)"),
        {"ns": _CLOSURE_LOCK_NS, "company_id": int(company_id)},
    )


def _active_closures(company_id: int) -> list[DriverWorkTimePeriodClosure]:
    return (
        db.session.query(DriverWorkTimePeriodClosure)
        .filter(
            DriverWorkTimePeriodClosure.company_id == int(company_id),
            DriverWorkTimePeriodClosure.reopened_at.is_(None),
        )
        .all()
    )


def _zurich_day(value: datetime | None) -> date | None:
    raw = zurich_date_of_instant(value)
    if raw is None:
        return None
    return date.fromisoformat(raw)


def _closed_period_response(company_id: int, *days: date | None):
    closures = _active_closures(company_id)
    for day in days:
        if day is None:
            continue
        for row in closures:
            if row.period_from <= day <= row.period_to:
                return _CLOSED_PERIOD_ERROR, 409
    return None


def _filter_days(days: list[dict], mode: str) -> list[dict]:
    if mode == "all":
        return days

    def keep(entry: dict) -> bool:
        if mode == "transports":
            return entry.get("kind") == "transport"
        if mode == "manual":
            return bool(entry.get("is_manual"))
        if mode == "adjusted":
            return bool(entry.get("is_manually_adjusted"))
        if mode == "anomalies":
            return bool(entry.get("anomalies"))
        return True

    filtered = []
    for day in days:
        rows = [entry for entry in day["entries"] if keep(entry)]
        if not rows:
            continue
        filtered.append(
            {
                "date": day["date"],
                "totals": {
                    "worked_minutes": sum(
                        int(row.get("worked_minutes") or 0) for row in rows
                    ),
                    "pending_minutes": sum(
                        int(row.get("proposed_worked_minutes") or 0)
                        for row in rows
                        if row.get("work_time_status") == "pending_validation"
                    ),
                    "compensated_minutes": sum(
                        int(row.get("compensated_minutes") or 0) for row in rows
                    ),
                    "real_minutes": sum(
                        int(row.get("real_minutes") or 0) for row in rows
                    ),
                    "flat_minutes": sum(
                        int(row.get("flat_minutes") or 0)
                        for row in rows
                        if row.get("flat_status") == "calculated"
                    ),
                },
                "entries": rows,
            }
        )
    return filtered


@companies_ns.route("/me/work-time/summary")
class CompanyWorkTimeSummary(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Résumé du temps de travail sur une période Europe/Zurich."""
        if not can_view(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        period = _period_args()
        if isinstance(period[0], dict):
            return period
        report = _service().build(int(company.id), period[0], period[1])
        return {
            "period": report["period"],
            "cutover_at": report["cutover_at"],
            "compensation_source": report["compensation_source"],
            "period_finalized": report.get("period_finalized", False),
            "finalized_at": report.get("finalized_at"),
            "kpis": report["kpis"],
            "contractual_rules": report.get("contractual_rules")
            or {"version_count": 0, "versions": []},
            "drivers": report["drivers"],
            "review_items": report.get("review_items") or [],
        }, 200


@companies_ns.route("/me/work-time/drivers/<int:driver_id>")
class CompanyDriverWorkTime(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, driver_id: int):
        """Détail par jour d'un chauffeur."""
        if not can_view(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        if _driver_in_company(driver_id, int(company.id)) is None:
            return {"error": "Chauffeur introuvable"}, 404
        period = _period_args()
        if isinstance(period[0], dict):
            return period
        mode = (request.args.get("filter") or "all").strip()
        if mode not in _FILTERS:
            return {"error": "Filtre inconnu."}, 400
        try:
            page = max(1, int(request.args.get("page", 1)))
            per_page = min(100, max(1, int(request.args.get("per_page", 31))))
        except ValueError:
            return {"error": "Pagination invalide."}, 400
        report = _service().build(int(company.id), period[0], period[1])
        days = _filter_days(report["days_by_driver"].get(str(driver_id), []), mode)
        total = len(days)
        start = (page - 1) * per_page
        driver_row = next(
            (row for row in report["drivers"] if row["driver_id"] == driver_id),
            None,
        )
        return {
            "driver_id": driver_id,
            "display_name": (driver_row or {}).get("display_name") or "Chauffeur",
            "totals": driver_row,
            "period": report["period"],
            "cutover_at": report["cutover_at"],
            "compensation_source": report["compensation_source"],
            "period_finalized": report.get("period_finalized", False),
            "page": page,
            "per_page": per_page,
            "total_days": total,
            "days": days[start : start + per_page],
        }, 200


@companies_ns.route("/me/work-time/bookings/<int:booking_id>/explain")
class CompanyBookingWorkTimeExplain(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, booking_id: int):
        """Pourquoi cette durée et cette rémunération."""
        if not can_view(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        payload = _service().explain_booking(int(company.id), booking_id)
        if payload is None:
            return {"error": "Course introuvable"}, 404
        return payload, 200


@companies_ns.route("/me/work-time/adjustments")
class CompanyWorkTimeAdjustments(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Historique des corrections d'horaire."""
        if not can_view(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        query = db.session.query(DriverWorkTimeAdjustment).filter(
            DriverWorkTimeAdjustment.company_id == int(company.id)
        )
        booking_id = request.args.get("booking_id")
        driver_id = request.args.get("driver_id")
        if booking_id:
            query = query.filter(DriverWorkTimeAdjustment.booking_id == int(booking_id))
        if driver_id:
            query = query.filter(DriverWorkTimeAdjustment.driver_id == int(driver_id))
        start = (request.args.get("from") or "").strip()
        end = (request.args.get("to") or "").strip()
        if start or end:
            if not start or not end:
                return {"error": "from et to doivent être fournis ensemble."}, 400
            try:
                utc_start, utc_end, _local_start, _local_end = zurich_period_bounds(
                    start, end
                )
            except ValueError:
                return {"error": "Période invalide. Format attendu : YYYY-MM-DD."}, 400
            query = query.filter(
                DriverWorkTimeAdjustment.created_at >= utc_start,
                DriverWorkTimeAdjustment.created_at < utc_end,
            )
        rows = (
            query.order_by(DriverWorkTimeAdjustment.created_at.desc()).limit(200).all()
        )
        return {"adjustments": [_adjustment_dict(row) for row in rows]}, 200

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Enregistre un snapshot complet des deux instants effectifs."""
        if not can_manage(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        body = request.get_json(silent=True) or {}
        reason = str(body.get("reason") or "").strip()
        if not reason:
            return {"error": "Le motif de correction est obligatoire."}, 400
        try:
            booking_id = int(body["booking_id"])
        except (KeyError, TypeError, ValueError):
            return {"error": "booking_id est requis."}, 400
        booking = db.session.get(Booking, booking_id)
        if booking is None or int(booking.company_id or 0) != int(company.id):
            return {"error": "Course introuvable"}, 404
        if booking.driver_id is None:
            return {"error": "Cette course n'a pas de chauffeur."}, 400
        status_value = str(getattr(booking.status, "value", booking.status))
        if status_value not in _COMPLETED:
            return {"error": "Seule une course terminée peut être rectifiée."}, 400

        previous = (
            db.session.query(DriverWorkTimeAdjustment)
            .filter(DriverWorkTimeAdjustment.booking_id == booking_id)
            .order_by(DriverWorkTimeAdjustment.id.desc())
            .first()
        )
        current_arrived = (
            previous.corrected_arrived_at if previous else booking.arrived_at
        )
        current_completed = (
            previous.corrected_completed_at if previous else booking.completed_at
        )
        if "corrected_arrived_at" in body and body.get("corrected_arrived_at"):
            new_arrived = _parse_dt(body.get("corrected_arrived_at"))
        else:
            new_arrived = current_arrived
        if "corrected_completed_at" in body and body.get("corrected_completed_at"):
            new_completed = _parse_dt(body.get("corrected_completed_at"))
        else:
            new_completed = current_completed
        if new_arrived is None or new_completed is None:
            return {
                "error": (
                    "Les deux instants effectifs (arrivée et fin) doivent être "
                    "connus pour enregistrer la correction."
                )
            }, 400
        if new_completed < new_arrived:
            return {"error": "L'heure de fin précède l'heure d'arrivée."}, 400
        closed = _closed_period_response(
            int(company.id),
            _zurich_day(current_completed),
            _zurich_day(new_completed),
        )
        if closed:
            return closed

        row = DriverWorkTimeAdjustment(
            company_id=int(company.id),
            driver_id=int(booking.driver_id),
            booking_id=booking_id,
            original_arrived_at=booking.arrived_at,
            original_completed_at=booking.completed_at,
            original_source="booking",
            corrected_arrived_at=new_arrived,
            corrected_completed_at=new_completed,
            reason=reason,
            comment=(str(body.get("comment")).strip() if body.get("comment") else None),
            created_by_user_id=_actor_id(),
        )
        db.session.add(row)
        db.session.commit()
        audit_log(
            "work_time.adjustment.create",
            "work_time",
            resource_type="booking",
            resource_id=booking_id,
            company=company,
            action_details={"reason": reason, "adjustment_id": row.id},
        )
        return _adjustment_dict(row), 201


@companies_ns.route("/me/work-time/duration-decisions")
class CompanyWorkTimeDurationDecisions(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Valide ou rectifie une durée proposée, sans inventer d'horaires."""
        if not can_manage(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        body = request.get_json(silent=True) or {}
        try:
            booking_id = int(body["booking_id"])
            validated = int(body["validated_worked_minutes"])
        except (KeyError, TypeError, ValueError):
            return {"error": "booking_id et validated_worked_minutes sont requis."}, 400
        if validated < 0:
            return {"error": "La durée retenue doit être positive."}, 400
        booking = db.session.get(Booking, booking_id)
        if booking is None or int(booking.company_id or 0) != int(company.id):
            return {"error": "Course introuvable"}, 404
        if booking.driver_id is None:
            return {"error": "Cette course n'a pas de chauffeur."}, 400
        status_value = str(getattr(booking.status, "value", booking.status))
        if status_value not in _COMPLETED:
            return {"error": "Seule une course terminée peut être validée."}, 400
        estimate = _estimate_settings(int(company.id))
        if not estimate["route_estimate_enabled"]:
            return {"error": "L'estimation de trajet est désactivée."}, 400
        proposal = proposal_for_segment(
            _to_segment(booking, force_route=True),
            margin_minutes=int(estimate["route_margin_minutes"]),
        )
        if proposal is None:
            return {"error": "Aucune durée de trajet n'a pu être calculée."}, 400
        proposed = int(proposal["proposed_worked_minutes"])
        reason = str(body.get("reason") or "").strip()
        if validated == proposed:
            source = "validated_route_estimate"
            reason = reason or "Validation du temps proposé"
        else:
            source = "admin_adjustment"
            if not reason:
                return {
                    "error": "Le motif est obligatoire pour rectifier la durée."
                }, 400
        closed = _closed_period_response(
            int(company.id),
            _zurich_day(booking.completed_at),
        )
        if closed:
            return closed
        row = DriverWorkTimeDurationDecision(
            company_id=int(company.id),
            driver_id=int(booking.driver_id),
            booking_id=booking_id,
            proposed_worked_minutes=proposed,
            validated_worked_minutes=validated,
            route_minutes=proposal.get("route_minutes"),
            margin_minutes=proposal.get("margin_minutes"),
            route_provider=proposal.get("route_provider"),
            source=source,
            reason=reason,
            created_by_user_id=_actor_id(),
        )
        db.session.add(row)
        db.session.commit()
        audit_log(
            "work_time.duration_decision.create",
            "work_time",
            resource_type="booking",
            resource_id=booking_id,
            company=company,
            action_details={
                "source": source,
                "validated_worked_minutes": validated,
                "proposed_worked_minutes": proposed,
            },
        )
        return {
            "id": int(row.id),
            "booking_id": booking_id,
            "source": source,
            "proposed_worked_minutes": proposed,
            "validated_worked_minutes": validated,
            "route_minutes": row.route_minutes,
            "margin_minutes": row.margin_minutes,
            "route_provider": row.route_provider,
            "reason": reason,
        }, 201


@companies_ns.route("/me/work-time/settings")
class CompanyWorkTimeSettingsResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Marge d'estimation du temps travaillé."""
        if not can_view(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        return _estimate_settings(int(company.id)), 200

    @jwt_required()
    @role_required(UserRole.company)
    def put(self):
        """Met à jour la marge, sans toucher à la rémunération."""
        if not can_configure(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        body = request.get_json(silent=True) or {}
        try:
            margin = int(body["route_margin_minutes"])
        except (KeyError, TypeError, ValueError):
            return {"error": "route_margin_minutes est requis."}, 400
        if margin < 0:
            return {"error": "La marge doit être positive."}, 400
        enabled = bool(body.get("route_estimate_enabled", True))
        row = db.session.get(DriverWorkTimeSettings, int(company.id))
        if row is None:
            row = DriverWorkTimeSettings(company_id=int(company.id))
            db.session.add(row)
        row.route_margin_minutes = margin
        row.route_estimate_enabled = enabled
        db.session.commit()
        return {
            "route_margin_minutes": margin,
            "route_estimate_enabled": enabled,
        }, 200


@companies_ns.route("/me/work-time/manual-entries")
class CompanyManualWorkEntries(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Liste des saisies manuelles, y compris annulées."""
        if not can_view(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        query = db.session.query(DriverManualWorkEntry).filter(
            DriverManualWorkEntry.company_id == int(company.id)
        )
        driver_id = request.args.get("driver_id")
        if driver_id:
            query = query.filter(DriverManualWorkEntry.driver_id == int(driver_id))
        start = request.args.get("from")
        end = request.args.get("to")
        if start:
            query = query.filter(DriverManualWorkEntry.work_date >= parse_ymd(start))
        if end:
            query = query.filter(DriverManualWorkEntry.work_date <= parse_ymd(end))
        rows = query.order_by(DriverManualWorkEntry.started_at.desc()).limit(300).all()
        return {"entries": [_manual_public(row) for row in rows]}, 200

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Crée une saisie. La durée est calculée depuis le début et la fin."""
        if not can_manage(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        body = request.get_json(silent=True) or {}
        work_type = str(body.get("work_type") or "").strip()
        if work_type not in _WORK_TYPES:
            return {"error": "Type d'activité inconnu."}, 400
        try:
            driver_id = int(body["driver_id"])
        except (KeyError, TypeError, ValueError):
            return {"error": "driver_id est requis."}, 400
        if _driver_in_company(driver_id, int(company.id)) is None:
            return {"error": "Chauffeur introuvable"}, 404
        try:
            started = _parse_dt(body.get("started_at"))
            ended = _parse_dt(body.get("ended_at"))
        except ValueError:
            return {"error": "Horaires invalides."}, 400
        if started is None or ended is None:
            return {"error": "Le début et la fin sont obligatoires."}, 400
        if ended <= started:
            return {"error": "L'heure de fin doit être après l'heure de début."}, 400
        if "duration_minutes" in body:
            return {
                "error": (
                    "La durée est calculée par le serveur à partir du début et de la fin."
                )
            }, 400
        if body.get("booking_id"):
            try:
                linked_id = int(body["booking_id"])
            except (TypeError, ValueError):
                return {"error": "Course introuvable"}, 404
            linked = db.session.get(Booking, linked_id)
            if linked is None or int(linked.company_id or 0) != int(company.id):
                return {"error": "Course introuvable"}, 404
        minutes = elapsed_minutes(started, ended)
        closed = _closed_period_response(int(company.id), work_date_in_zurich(started))
        if closed:
            return closed
        row = DriverManualWorkEntry(
            company_id=int(company.id),
            driver_id=driver_id,
            work_date=work_date_in_zurich(started),
            started_at=started,
            ended_at=ended,
            duration_minutes=minutes,
            work_type=work_type,
            description=(
                str(body.get("description")).strip()
                if body.get("description")
                else None
            ),
            reference=(
                str(body.get("reference")).strip() if body.get("reference") else None
            ),
            pickup_location=body.get("pickup_location"),
            dropoff_location=body.get("dropoff_location"),
            booking_id=int(body["booking_id"]) if body.get("booking_id") else None,
            created_by_user_id=_actor_id(),
        )
        db.session.add(row)
        db.session.commit()
        audit_log(
            "work_time.manual.create",
            "work_time",
            resource_type="driver_manual_work_entry",
            resource_id=row.id,
            company=company,
            action_details={"work_type": work_type, "duration_minutes": minutes},
        )
        return _manual_public(row), 201


@companies_ns.route("/me/work-time/manual-entries/<int:entry_id>/cancel")
class CompanyManualWorkEntryCancel(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, entry_id: int):
        """Annule une saisie sans la supprimer."""
        if not can_manage(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        row = db.session.get(DriverManualWorkEntry, entry_id)
        if row is None or int(row.company_id) != int(company.id):
            return {"error": "Saisie introuvable"}, 404
        if row.cancelled_at is not None:
            return {"error": "Cette saisie est déjà annulée."}, 409
        body = request.get_json(silent=True) or {}
        reason = str(body.get("reason") or "").strip()
        if not reason:
            return {"error": "Le motif d'annulation est obligatoire."}, 400
        closed = _closed_period_response(int(company.id), row.work_date)
        if closed:
            return closed
        row.cancelled_at = datetime.now(UTC)
        row.cancelled_by_user_id = _actor_id()
        row.cancellation_reason = reason
        db.session.commit()
        audit_log(
            "work_time.manual.cancel",
            "work_time",
            resource_type="driver_manual_work_entry",
            resource_id=row.id,
            company=company,
            action_details={"reason": reason},
        )
        return _manual_public(row), 200


@companies_ns.route("/me/work-time/compensation-policies")
class CompanyCompensationPolicies(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Versions de rémunération de l'entreprise."""
        if not can_view(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        rows = (
            db.session.query(DriverCompensationPolicy)
            .filter(DriverCompensationPolicy.company_id == int(company.id))
            .order_by(DriverCompensationPolicy.effective_from.desc())
            .all()
        )
        return {"policies": [_policy_public(row) for row in rows]}, 200

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Crée une version et clôture la précédente du même périmètre."""
        if not can_configure(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        body = request.get_json(silent=True) or {}
        try:
            effective_from = parse_ymd(str(body["effective_from"]))
            flat_minutes = int(body["transport_flat_minutes"])
        except (KeyError, TypeError, ValueError):
            return {
                "error": "effective_from et transport_flat_minutes sont requis."
            }, 400
        if flat_minutes < 0:
            return {"error": "Les minutes forfaitaires doivent être positives."}, 400
        for closure in _active_closures(int(company.id)):
            if closure.period_to >= effective_from:
                return {
                    "error": (
                        "Cette version couvrirait une période déjà clôturée. "
                        "Réouvrez-la avant de changer la règle."
                    )
                }, 409
        driver_id = body.get("driver_id")
        if driver_id is not None:
            if _driver_in_company(int(driver_id), int(company.id)) is None:
                return {"error": "Chauffeur introuvable"}, 404
            driver_id = int(driver_id)
        scope = db.session.query(DriverCompensationPolicy).filter(
            DriverCompensationPolicy.company_id == int(company.id)
        )
        if driver_id is None:
            scope = scope.filter(DriverCompensationPolicy.driver_id.is_(None))
        else:
            scope = scope.filter(DriverCompensationPolicy.driver_id == driver_id)
        for previous in scope.all():
            if previous.effective_from >= effective_from and (
                previous.effective_until is None
                or previous.effective_until > effective_from
            ):
                return {
                    "error": "Une version existe déjà à partir de cette date ou après."
                }, 409
            if previous.effective_from < effective_from and (
                previous.effective_until is None
                or previous.effective_until > effective_from
            ):
                previous.effective_until = effective_from
        row = DriverCompensationPolicy(
            company_id=int(company.id),
            driver_id=driver_id,
            effective_from=effective_from,
            effective_until=None,
            mode=str(body.get("mode") or "flat_per_trip"),
            transport_flat_minutes=flat_minutes,
            one_way_minutes=flat_minutes,
            round_trip_minutes=flat_minutes,
            intermediate_stop_minutes=None,
            max_reasonable_minutes=(
                int(body["max_reasonable_minutes"])
                if body.get("max_reasonable_minutes") is not None
                else None
            ),
            overlap_threshold_minutes=int(body.get("overlap_threshold_minutes") or 1),
            work_type_rules_json=dict(body.get("work_type_rules") or {}),
            notes=(str(body.get("notes")).strip() if body.get("notes") else None),
            created_by_user_id=_actor_id(),
        )
        db.session.add(row)
        db.session.commit()
        audit_log(
            "work_time.policy.create",
            "work_time",
            resource_type="driver_compensation_policy",
            resource_id=row.id,
            company=company,
            action_details={"effective_from": effective_from.isoformat()},
        )
        return _policy_public(row), 201


@companies_ns.route("/me/work-time/periods/finalize")
class CompanyWorkTimeFinalize(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Fige la rémunération d'un mois civil déjà terminé."""
        if not can_manage(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        body = request.get_json(silent=True) or {}
        try:
            period_from = str(body["from"])
            period_to = str(body["to"])
            parse_ymd(period_from)
            parse_ymd(period_to)
        except (KeyError, ValueError):
            return {"error": "from et to sont requis (YYYY-MM-DD)."}, 400
        _lock_company_closures(int(company.id))
        start = parse_ymd(period_from)
        end = parse_ymd(period_to)
        refusal = monthly_closure_refusal(start, end, zurich_today())
        if refusal:
            return {"error": refusal, "code": "PERIOD_NOT_CLOSABLE"}, 409
        for closure in _active_closures(int(company.id)):
            if closure.period_from == start and closure.period_to == end:
                return {"error": "Cette période est déjà clôturée."}, 409
            if closure.period_from <= end and closure.period_to >= start:
                return {
                    "error": "Une période clôturée chevauche déjà cet intervalle."
                }, 409
        service = _service()
        report = service._dynamic(int(company.id), period_from, period_to)
        simulated = deepcopy(report)
        apply_ledger_snapshot(simulated, list(report["compensation_lines"]))
        divergences = closure_parity_errors(report, simulated)
        if divergences:
            db.session.rollback()
            logger.error(
                "work_time.finalize.parity company=%s from=%s to=%s divergences=%s",
                company.id,
                period_from,
                period_to,
                divergences,
            )
            return {
                "error": "La clôture changerait les heures affichées.",
                "divergences": divergences,
            }, 409
        now = datetime.now(UTC)
        closure = DriverWorkTimePeriodClosure(
            company_id=int(company.id),
            period_from=parse_ymd(period_from),
            period_to=parse_ymd(period_to),
            finalized_at=now,
            finalized_by_user_id=_actor_id(),
        )
        db.session.add(closure)
        db.session.flush()
        for line in report["compensation_lines"]:
            accounting = line.get("accounting_date")
            db.session.add(
                DriverCompensationLedger(
                    company_id=int(company.id),
                    closure_id=int(closure.id),
                    driver_id=line.get("driver_id"),
                    journey_key=str(line["journey_key"]),
                    line_key=str(line.get("line_key") or line["journey_key"]),
                    booking_id=line.get("booking_id")
                    or line.get("attached_to_booking_id"),
                    manual_entry_id=line.get("manual_entry_id"),
                    flat_minutes=line.get("flat_minutes"),
                    accounting_date=parse_ymd(accounting) if accounting else None,
                    compensated_minutes=int(line.get("compensated_minutes") or 0),
                    policy_id=line.get("policy_id"),
                    rule_type=line.get("rule_type"),
                    base_minutes=int(line.get("base_minutes") or 0),
                    intermediate_stop_count=int(
                        line.get("intermediate_stop_count") or 0
                    ),
                    intermediate_stop_minutes=line.get("intermediate_stop_minutes"),
                    classification_source=line.get("classification_source"),
                    journey_status=line.get("journey_status"),
                    compensation_status=line.get("compensation_status"),
                    attached_to_booking_id=line.get("attached_to_booking_id"),
                    generated_at=now,
                    finalized_at=now,
                )
            )
        db.session.commit()
        audit_log(
            "work_time.period.finalize",
            "work_time",
            resource_type="driver_work_time_period_closure",
            resource_id=closure.id,
            company=company,
            action_details={"from": period_from, "to": period_to},
        )
        return {
            "closure_id": int(closure.id),
            "from": period_from,
            "to": period_to,
            "lines": len(report["compensation_lines"]),
        }, 201


@companies_ns.route("/me/work-time/periods/reopen")
class CompanyWorkTimeReopen(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Réouvre une période clôturée. Le ledger reste en historique."""
        if not can_manage(UserRole.company):
            return {"error": "Accès refusé"}, 403
        company, error, status = _company_or_error()
        if error:
            return error, status
        body = request.get_json(silent=True) or {}
        reason = str(body.get("reason") or "").strip()
        if not reason:
            return {"error": "Le motif de réouverture est obligatoire."}, 400
        try:
            period_from = str(body["from"])
            period_to = str(body["to"])
            start = parse_ymd(period_from)
            end = parse_ymd(period_to)
        except (KeyError, ValueError):
            return {"error": "from et to sont requis."}, 400
        if not is_full_calendar_month(start, end):
            return {
                "error": "Seuls les mois civils complets peuvent être réouverts.",
                "code": "PERIOD_NOT_CLOSABLE",
            }, 409
        _lock_company_closures(int(company.id))
        closure = _service()._active_closure(int(company.id), period_from, period_to)
        if closure is None:
            return {"error": "Aucune période clôturée ne correspond."}, 404
        closure.reopened_at = datetime.now(UTC)
        closure.reopened_by_user_id = _actor_id()
        closure.reopen_reason = reason
        db.session.commit()
        audit_log(
            "work_time.period.reopen",
            "work_time",
            resource_type="driver_work_time_period_closure",
            resource_id=closure.id,
            company=company,
            action_details={"reason": reason},
        )
        return {"closure_id": int(closure.id), "reopened": True}, 200


def _manual_public(row: DriverManualWorkEntry) -> dict:
    return {
        "id": int(row.id),
        "driver_id": int(row.driver_id),
        "work_date": row.work_date.isoformat(),
        "started_at": row.started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ended_at": row.ended_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_minutes": int(row.duration_minutes),
        "work_type": row.work_type,
        "description": row.description,
        "reference": row.reference,
        "pickup_location": row.pickup_location,
        "dropoff_location": row.dropoff_location,
        "booking_id": row.booking_id,
        "cancelled_at": (
            row.cancelled_at.strftime("%Y-%m-%dT%H:%M:%SZ")
            if row.cancelled_at
            else None
        ),
        "cancellation_reason": row.cancellation_reason,
        "is_manual": True,
    }


def _policy_public(row: DriverCompensationPolicy) -> dict:
    return {
        "id": int(row.id),
        "driver_id": row.driver_id,
        "effective_from": row.effective_from.isoformat(),
        "effective_until": row.effective_until.isoformat()
        if row.effective_until
        else None,
        "mode": row.mode,
        "transport_flat_minutes": int(row.transport_flat_minutes),
        "max_reasonable_minutes": row.max_reasonable_minutes,
        "overlap_threshold_minutes": int(row.overlap_threshold_minutes or 1),
        "work_type_rules": row.work_type_rules_json or {},
        "notes": row.notes,
        "created_at": row.created_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        if row.created_at
        else None,
    }
