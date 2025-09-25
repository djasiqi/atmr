# backend/routes/dispatch_routes.py
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from flask import request
from flask_restx import Namespace, Resource, fields
from flask_jwt_extended import jwt_required, get_jwt_identity

from ext import role_required, db
from models import (
    User,
    UserRole,
    Company,
    Booking,
    Assignment,
    Driver,
    DispatchRun,
)
from services.unified_dispatch import data
from services.unified_dispatch import settings as ud_settings
from services.unified_dispatch.queue import trigger_job, get_status
from werkzeug.exceptions import UnprocessableEntity
import re
from sqlalchemy import func
from datetime import timezone
from flask_restx.fields import Raw as _Raw
from shared.time_utils import day_local_bounds

dispatch_ns = Namespace("company_dispatch", description="Dispatch par journée (contrat unifié)")
logger = logging.getLogger(__name__)

# ===== Schemas RESTX (simples) =====

preview_response = dispatch_ns.model(
    "DispatchPreviewResponse",
    {
        "bookings": fields.Integer,
        "drivers": fields.Integer,
        "horizon_minutes": fields.Integer,
        "ready": fields.Boolean,
        "reason": fields.String,
    },
)

# ---- Type custom: bool|null uniquement
class NullableBoolean(fields.Raw):
    __schema_type__ = ["boolean", "null"]
    __schema_example__ = None
    def format(self, value):
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("true", "false"):
                return v == "true"
            if v in ("null", "none"):
                return None
        raise UnprocessableEntity("allow_emergency doit être true/false ou null")
    
# ---- Enum -> string (value) pour marshalling propre
class EnumStr(_Raw):
    def format(self, value):
        try:
            return value.value
        except Exception:
            return str(value)


run_model = dispatch_ns.model(
    "RunDispatch",
    {
        "for_date": fields.String(required=True, description="YYYY-MM-DD (local)"),
        "regular_first": fields.Boolean(required=False, default=True),
        # Tri-state: bool ou null -> si absent / null, on laisse le moteur décider via settings
        "allow_emergency": NullableBoolean(required=False, description="bool ou null (hérite si absent)"),
        "run_async": fields.Boolean(required=False, default=True, description="Exécuter en file (async) ou en direct (sync)"),
        "overrides": fields.Raw(required=False, default={}),
        "mode": fields.String(required=False, enum=["auto", "heuristic_only", "solver_only"], default="auto"),
    },
)

assignment_model = dispatch_ns.model(
    "Assignment",
    {
        # Types alignés sur le modèle SQLAlchemy (Integer)
        "id": fields.Integer,
        "booking_id": fields.Integer,
        "driver_id": fields.Integer,
        "status": EnumStr,  # enum rendu en str (value)
        "dispatch_run_id": fields.Integer,
        # Mappe les noms "esthétiques" vers les colonnes réelles (eta_*)
        "estimated_pickup_arrival": fields.DateTime(dt_format="iso8601", attribute="eta_pickup_at"),
        "estimated_dropoff_arrival": fields.DateTime(dt_format="iso8601", attribute="eta_dropoff_at"),
        "created_at": fields.DateTime(dt_format="iso8601"),
        "updated_at": fields.DateTime(dt_format="iso8601"),
    },
)

delay_model = dispatch_ns.model(
    "Delay",
    {
        "assignment_id": fields.String,
        "booking_id": fields.Integer,
        "driver_id": fields.Integer,
        "scheduled_time": fields.String,
        "estimated_arrival": fields.String,
        "delay_minutes": fields.Float,
        "is_pickup": fields.Boolean,
        "booking_status": fields.String,
    },
)

# Add a model for DispatchRun
dispatch_run_model = dispatch_ns.model(
    "DispatchRun",
    {
        "id": fields.Integer,
        "company_id": fields.Integer,
        "day": fields.String(description="YYYY-MM-DD"),
        "status": fields.String,
        "started_at": fields.DateTime(dt_format="iso8601"),
        "completed_at": fields.DateTime(dt_format="iso8601"),
        "created_at": fields.DateTime(dt_format="iso8601"),
        "config": fields.Raw,
        "metrics": fields.Raw,
    },
)

# ===== Helpers =====

dispatch_run_detail_model = dispatch_ns.inherit(
    "DispatchRunDetail",
    dispatch_run_model,
    {"assignments": fields.List(fields.Nested(assignment_model))},
)

def dispatch_run_date(for_date_str):
    """
    Convert a date string (YYYY-MM-DD) to a Python date object
    
    Args:
        for_date_str (str): Date string in YYYY-MM-DD format
        
    Returns:
        date: Python date object
    """
    if not for_date_str:
        return date.today()
        
    if isinstance(for_date_str, date):
        return for_date_str
        
    if isinstance(for_date_str, str):
        try:
            # First try with strptime for consistent format
            return datetime.strptime(for_date_str, "%Y-%m-%d").date()
        except ValueError:
            try:
                # Then try fromisoformat as fallback
                return datetime.fromisoformat(for_date_str).date()
            except ValueError:
                logger.warning(f"[Dispatch] Invalid date format: {for_date_str}, using today's date")
                return date.today()
    
    # If it's not a string or date, return today
    return date.today()


def _get_current_company() -> Company:
    identity = get_jwt_identity()
    if identity is None:
        dispatch_ns.abort(401, "Jeton JWT manquant")

    user: Optional[User] = None
    try:
        user = User.query.get(int(identity))
    except Exception:
        user = None
    if not user:
        user = User.query.filter_by(public_id=str(identity)).first()
    if not user:
        dispatch_ns.abort(403, "Utilisateur introuvable")

    if getattr(user, "company", None):
        return user.company
    if getattr(user, "companies", None):
        return user.companies[0]
    c = Company.query.filter_by(user_id=user.id).first()
    if not c:
        dispatch_ns.abort(403, "Aucune entreprise associée à l'utilisateur.")
    return c


def _parse_date(s: Optional[str]) -> date:
    """
    Parse a date string in YYYY-MM-DD format to a Python date object.
    
    Args:
        s (Optional[str]): Date string in YYYY-MM-DD format
        
    Returns:
        date: Python date object
    """
    if not s:
        return date.today()
    
    try:
        # First try the standard fromisoformat method
        return datetime.fromisoformat(s).date()
    except ValueError:
        # If that fails, try explicit parsing with strptime
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            # If all parsing fails, return today's date
            logger.warning(f"[Dispatch] Invalid date format: {s}, using today's date instead")
            return date.today()


def _to_dict(o: Any) -> Dict[str, Any]:
    return o.to_dict() if hasattr(o, "to_dict") else {}


# ---- Helper: expression temporelle des bookings (fallback multi-colonnes, safe SQLite)
def _booking_time_expr():
    """
    Retourne une clause SQLAlchemy représentant le "temps" d'un booking.
    Ordre de préférence : scheduled_time, pickup_time, date_time, datetime.
    Construction itérative pour éviter COALESCE à 1 seul argument (sqlite).
    """
    expr = None
    for name in ("scheduled_time", "pickup_time", "date_time", "datetime"):
        col = getattr(Booking, name, None)
        if col is None:
            continue
        expr = col if expr is None else func.coalesce(expr, col)
    return expr or getattr(Booking, "scheduled_time")

# ===== Routes =====

@dispatch_ns.route("/run")
class CompanyDispatchRun(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.expect(run_model, validate=True)
    def post(self):
        """
        Lance un dispatch pour une journée donnée.
        - async=true (défaut) : enfile un job via la queue (202)
        - async=false : exécute immédiatement (200)
        """
        body: Dict[str, Any] = request.get_json(force=True) or {}
        logger.info("[Dispatch] /run body: %s", body)

        # --- Validation for_date: doit matcher YYYY-MM-DD
        for_date = body.get("for_date")
        if not for_date:
            dispatch_ns.abort(400, "for_date manquant (YYYY-MM-DD). Utilisez plutôt POST /company_dispatch/run.")
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(for_date)):
            raise UnprocessableEntity("for_date invalide: attendu 'YYYY-MM-DD' (ex: 2025-09-22)")

        regular_first = bool(body.get("regular_first", True))
        # Tri-state: si absent => None (pas d'écrasement des settings)
        allow_emergency = body.get("allow_emergency", None)  # déjà validé par NullableBoolean
        run_async = bool(body.get("run_async", True))
        overrides = body.get("overrides") or {}
        mode = (body.get("mode") or "auto").strip().lower()
        # normalisation des alias pour l'engine
        if mode == "heuristic":
            mode = "heuristic_only"
        elif mode == "solver":
            mode = "solver_only"

        company = _get_current_company()
        logger.info(
            "[Dispatch] Run requested for company=%s date=%s regular_first=%s allow_emergency=%s mode=%s",
            company.id, for_date, regular_first, allow_emergency, mode
        )

        params = {
            "company_id": company.id,
            "for_date": for_date,
            "regular_first": regular_first,
            "allow_emergency": allow_emergency,
            "overrides": overrides,
            "mode": mode,
        }

        if run_async:
            try:
                job = trigger_job(company.id, params)
                job_id = (job or {}).get("id")
                # Certaines implémentations de queue retournent déjà un dispatch_run_id
                dispatch_run_id = (job or {}).get("dispatch_run_id")
                logger.info("[Dispatch] Job queued successfully: job_id=%s dispatch_run_id=%s", job_id, dispatch_run_id)

                # ⚠️ Exigence: toujours renvoyer dispatch_run_id au niveau racine si disponible
                # S'il n'est pas connu au moment de l'enqueue, on renvoie la clé avec valeur None (pour le front).
                resp = {
                    "status": "queued",
                    "job_id": job_id,
                    "dispatch_run_id": dispatch_run_id,
                    "for_date": for_date,  # Add for_date to response for frontend
                }
                return resp, 202
            except Exception as e:
                logger.exception("[Dispatch] trigger_job failed company=%s", company.id)
                dispatch_ns.abort(500, f"Enqueue du run impossible: {e}")

        # Synchrone (debug)
        try:
            from services.unified_dispatch.engine import run as engine_run

            logger.info("[Dispatch] Starting synchronous run for company=%s date=%s", company.id, for_date)
            result = engine_run(
                company_id=company.id,
                mode=mode,
                for_date=for_date,
                regular_first=regular_first,
                allow_emergency=allow_emergency,
                overrides=overrides,
            )
            # Toujours renvoyer un dict structuré (jamais None)
            if not result:
                result = {}
            if not isinstance(result, dict):
                result = {"meta": {"raw": result}}
            result.setdefault("assignments", [])
            result.setdefault("bookings", [])
            result.setdefault("drivers", [])
            result.setdefault("dispatch_run_id", None)
            result.setdefault("meta", {})
            result.setdefault("for_date", for_date)  # Add for_date to response for frontend

            # ⚙️ Promotion de meta.dispatch_run_id -> racine si absent
            try:
                if not result.get("dispatch_run_id"):
                    meta = result.get("meta") or {}
                    candidate = meta.get("dispatch_run_id") or meta.get("run_id")
                    if candidate is not None:
                        result["dispatch_run_id"] = candidate
            except Exception:
                # Ne jamais casser la réponse sync à cause d'une clé manquante
                pass
            
            logger.info(
                "[Dispatch] Sync run completed: company=%s assignments=%s unassigned=%s",
                company.id, len(result.get("assignments", [])), len(result.get("unassigned", []))
            )
            return result, 200
        except UnprocessableEntity as e:
            # 422 pour format/validation invalide (plus sémantique que 400)
            logger.warning("[Dispatch] Validation error: %s", str(e))
            dispatch_ns.abort(422, str(e))
        except Exception as e:
            logger.exception("[Dispatch] run sync failed company=%s for_date=%s", company.id, for_date)
            db.session.rollback()
            dispatch_ns.abort(500, f"Erreur exécution dispatch: {e}")


@dispatch_ns.route("/status")
class CompanyDispatchStatus(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Statut courant du worker de dispatch (coalescing / dernier résultat / dernière erreur)."""
        company = _get_current_company()
        logger.debug("[Dispatch] Status check for company=%s", company.id)
        try:
            return get_status(company.id), 200
        except Exception as e:
            logger.exception("[Dispatch] get_status failed company=%s", company.id)
            dispatch_ns.abort(500, f"Erreur récupération statut: {e}")


@dispatch_ns.route("/preview")
class DispatchPreview(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.marshal_with(preview_response, code=200)
    def get(self):
        """Aperçu de la journée (for_date): nb bookings/drivers et horizon (minutes)."""
        company = _get_current_company()
        for_date = request.args.get("for_date")
        if not for_date:
            dispatch_ns.abort(400, "Paramètre for_date (YYYY-MM-DD) requis pour le preview.")

        # cohérent avec /run
        regular_first = request.args.get("regular_first", "true").lower() != "false"
        ae_q = request.args.get("allow_emergency", None)
        allow_emergency = None if ae_q is None else (ae_q.lower() != "false")

        try:
            problem = data.build_problem_data(
                company_id=company.id,
                for_date=for_date,
                regular_first=regular_first,
                allow_emergency=allow_emergency,
                overrides={},
            ) or {}
        except TypeError:
            dispatch_ns.abort(
                500,
                "build_problem_data ne supporte pas (company_id, for_date, ...). "
                "Aligne data.py sur la nouvelle API.",
            )
        except Exception as e:
            logger.exception("[Dispatch] preview build_problem_data failed company=%s for_date=%s", company.id, for_date)
            dispatch_ns.abort(500, f"Erreur build_problem_data: {e}")

        bookings = len(problem.get("tasks", problem.get("bookings", [])))
        drivers = len(problem.get("vehicles", problem.get("drivers", [])))
        # Fallback propre : utilise la config par défaut du module settings
        horizon_minutes = int(
            problem.get("horizon_minutes")
            or getattr(getattr(ud_settings, "DEFAULT_SETTINGS", None).time, "horizon_minutes", 480)
        )
        ready = bool(bookings and drivers)
        reason = "ok" if ready else ("no_bookings" if not bookings else "no_drivers")
        return {
            "bookings": bookings,
            "drivers": drivers,
            "horizon_minutes": int(horizon_minutes),
            "ready": ready,
            "reason": reason,
        }, 200


@dispatch_ns.route("/trigger")
class DispatchTrigger(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """(Déprécié) Déclenche un run async. Utilisez POST /company_dispatch/run."""
        company = _get_current_company()
        body = request.get_json(silent=True) or {}
        for_date = body.get("for_date")
        if not for_date:
            dispatch_ns.abort(400, "for_date manquant (YYYY-MM-DD). Utilisez plutôt POST /company_dispatch/run.")

        allow_emergency = body.get("allow_emergency", None)
        if allow_emergency is not None:
            allow_emergency = bool(allow_emergency)

        params = {
            "company_id": company.id,
            "for_date": for_date,
            "regular_first": bool(body.get("regular_first", True)),
            "allow_emergency": allow_emergency,
            "overrides": body.get("overrides") or {},
            "mode": body.get("mode", "auto"),
        }
        try:
            job = trigger_job(company.id, params)
            return {"status": "queued", "job_id": job["id"], "dispatch_run_id": job.get("dispatch_run_id")}, 202
        except Exception as e:
            logger.exception("[Dispatch] trigger_job failed (deprecated route) company=%s", company.id)
            dispatch_ns.abort(500, f"Enqueue du run impossible: {e}")


# ===== Endpoints additionnels utiles =====

@dispatch_ns.route("/assignments")
class AssignmentsListResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(params={"date": "YYYY-MM-DD"})
    @dispatch_ns.marshal_list_with(assignment_model)
    def get(self):
        """Liste des assignations pour un jour."""
        
        d = _parse_date(request.args.get("date"))
        from shared.time_utils import day_local_bounds
        d0, d1 = day_local_bounds(d.strftime("%Y-%m-%d"))
        d1 = datetime(d.year, d.month, d.day, 23, 59, 59)

        # 🔒 Filtre multi-colonnes temps (comme le front)
        company = _get_current_company()
        time_expr = _booking_time_expr()

        # Ids des bookings du jour (entreprise courante)
        booking_ids = [
            b.id
            for b in (
                Booking.query.with_entities(Booking.id)
                .filter(
                    Booking.company_id == company.id,
                    time_expr >= d0,
                    time_expr <= d1,
                )
                .all()
            )
        ]

        logger.info(f"[Dispatch] Found {len(booking_ids)} bookings for date {d}")
        if not booking_ids:
            return []

        q = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(Assignment.booking_id.in_(booking_ids), Booking.company_id == company.id)
        )
        # Tri stable
        if hasattr(Assignment, "created_at"):
            q = q.order_by(Assignment.created_at.asc())
        elif hasattr(Assignment, "updated_at"):
            q = q.order_by(Assignment.updated_at.asc())
        assigns = q.all()
        # Retourne les objets ORM : le marshaller appliquera attribute=… et EnumStr
        return assigns


@dispatch_ns.route("/assignments/<int:assignment_id>")
class AssignmentResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.marshal_with(assignment_model)
    def get(self, assignment_id: int):
        """Détail d'une assignation."""
        company = _get_current_company()
        # 🔒 Vérifie la propriété via jointure Booking -> Company
        a: Optional[Assignment] = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(Assignment.id == assignment_id, Booking.company_id == company.id)
            .first()
        )
        if not a:
            dispatch_ns.abort(404, "assignment not found")
        return a

    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.expect(
        dispatch_ns.model("UpdateAssignmentBody", {"driver_id": fields.Integer, "status": fields.String}),
        validate=True,
    )
    @dispatch_ns.marshal_with(assignment_model)
    def patch(self, assignment_id: int):
        """MAJ d'une assignation (driver/status)."""
        company = _get_current_company()
        a: Optional[Assignment] = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(Assignment.id == assignment_id, Booking.company_id == company.id)
            .first()
        )
        if not a:
            dispatch_ns.abort(404, "assignment not found")

        data = request.get_json() or {}
        if "driver_id" in data:
            a.driver_id = data["driver_id"]
        if "status" in data:
            a.status = data["status"]

        a.updated_at = datetime.now(timezone.utc)
        db.session.add(a)
        db.session.commit()
        return a


@dispatch_ns.route("/assignments/<int:assignment_id>/reassign")
class ReassignResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.expect(
        dispatch_ns.model("ReassignBody", {"new_driver_id": fields.Integer(required=True)}),
        validate=True,
    )
    @dispatch_ns.marshal_with(assignment_model)
    def post(self, assignment_id: int):
        """Réassigne une course à un nouveau chauffeur (implémentation simple)."""
        data = request.get_json() or {}
        new_driver_id = data["new_driver_id"]

        company = _get_current_company()
        a: Optional[Assignment] = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(Assignment.id == assignment_id, Booking.company_id == company.id)
            .first()
        )
        if not a:
            dispatch_ns.abort(404, "assignment not found")

        a.driver_id = new_driver_id
        a.updated_at = datetime.now(timezone.utc)
        db.session.add(a)
        db.session.commit()
        return a


@dispatch_ns.route("/runs")
class RunsListResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(params={"limit": "int", "offset": "int"})
    @dispatch_ns.marshal_list_with(dispatch_run_model)
    def get(self):
        """Historique des runs (reverse chrono)."""
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))
        company = _get_current_company()
        q = DispatchRun.query.filter_by(company_id=company.id)
        
        # Fallback de tri: completed_at > started_at > day > created_at > id
        order_cols = []
        if hasattr(DispatchRun, "completed_at"):
            order_cols.append(getattr(DispatchRun, "completed_at").desc())
        if hasattr(DispatchRun, "started_at"):
            order_cols.append(getattr(DispatchRun, "started_at").desc())
        if hasattr(DispatchRun, "day"):
            order_cols.append(getattr(DispatchRun, "day").desc())
        if hasattr(DispatchRun, "created_at"):
            order_cols.append(getattr(DispatchRun, "created_at").desc())
        if hasattr(DispatchRun, "id"):
            order_cols.append(getattr(DispatchRun, "id").desc())
            
        if order_cols:
            q = q.order_by(*order_cols)
            
        runs = q.limit(limit).offset(offset).all()
        # Retourne les objets ORM : marshalling RESTX appliquera le modèle
        return runs

@dispatch_ns.route("/runs/<int:run_id>")
class RunResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.marshal_with(dispatch_run_detail_model)
    def get(self, run_id: int):
        """Détail d'un run + ses assignations."""
        company = _get_current_company()
        r: Optional[DispatchRun] = DispatchRun.query.filter_by(id=run_id, company_id=company.id).first()
        if not r:
            dispatch_ns.abort(404, "dispatch run not found")
        assigns = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(Assignment.dispatch_run_id == run_id, Booking.company_id == company.id)
            .all()
        )
        # Retourne un dict enrichi : le marshaller inclura 'assignments'
        out = r if hasattr(r, "__table__") else r
        return {"id": r.id,
                "company_id": r.company_id,
                "day": str(getattr(r, "day", "")),
                "status": r.status,
                "started_at": getattr(r, "started_at", None),
                "completed_at": getattr(r, "completed_at", None),
                "created_at": getattr(r, "created_at", None),
                "config": getattr(r, "config", None),
                "metrics": getattr(r, "metrics", None),
                "assignments": assigns}


@dispatch_ns.route("/delays")
class DelaysResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(params={"date": "YYYY-MM-DD"})
    @dispatch_ns.marshal_list_with(delay_model)
    def get(self):
        """Retards courants (ETA > horaire + 5 minutes) pour la journée."""
        d = _parse_date(request.args.get("date"))
        from shared.time_utils import day_local_bounds
        d0, d1 = day_local_bounds(d.strftime("%Y-%m-%d"))
        d1 = datetime(d.year, d.month, d.day, 23, 59, 59)

        company = _get_current_company()
        time_expr = _booking_time_expr()
        assigns = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(
                Booking.company_id == company.id,
                time_expr >= d0,
                time_expr <= d1,
                Assignment.status.in_(("assigned", "in_progress")),
            )
            .all()
        )

        out: List[Dict[str, Any]] = []
        for a in assigns:
            b = a.booking
            if not b:
                continue
            buf = 5 * 60  # seconds
            delayed = False
            eta = None
            sched = None
            is_pickup = False
            if a.eta_pickup_at and b.scheduled_time:
                eta = a.eta_pickup_at
                sched = b.scheduled_time
                delayed = (eta - sched).total_seconds() > buf
                is_pickup = True
            if not delayed and a.eta_dropoff_at and getattr(b, "dropoff_time", None):
                eta = a.eta_dropoff_at
                sched = b.dropoff_time
                delayed = (eta - sched).total_seconds() > buf
                is_pickup = False
            if delayed and eta and sched:
                out.append(
                    {
                        "assignment_id": a.id,
                        "booking_id": b.id,
                        "driver_id": a.driver_id,
                        "scheduled_time": sched.isoformat(),
                        "estimated_arrival": eta.isoformat(),
                        "delay_minutes": round((eta - sched).total_seconds() / 60.0, 1),
                        "is_pickup": is_pickup,
                        "booking_status": b.status,
                    }
                )
        return out