# backend/routes/dispatch_routes.py
from __future__ import annotations

import logging
from datetime import date, datetime, timezone, timedelta
from typing import Any, Dict, Optional, cast
import json
import re

from flask import request
from flask_restx import Namespace, Resource, fields
from flask_jwt_extended import jwt_required

from ext import role_required, db
from models import (
    UserRole,
    Company,
    Booking,
    BookingStatus,
    Assignment,
    Driver,
    DispatchRun,
    AssignmentStatus
)
from services.unified_dispatch import data
from services.unified_dispatch.queue import trigger_job, get_status
from services.unified_dispatch.suggestions import generate_suggestions
from services.unified_dispatch.realtime_optimizer import (
    start_optimizer_for_company,
    stop_optimizer_for_company,
    get_optimizer_for_company,
    check_opportunities_manual
)
from werkzeug.exceptions import UnprocessableEntity
from shared.time_utils import day_local_bounds, now_local
from routes.companies import get_company_from_token


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
    def format(self, value):
        if value is None:
            return None
        return bool(value)

# ---- Type custom: dict|null uniquement
class NullableDict(fields.Raw):
    def format(self, value):
        if value is None:
            return None
        return dict(value)

# ---- Type custom: list|null uniquement
class NullableList(fields.Raw):
    def format(self, value):
        if value is None:
            return None
        return list(value)

# ---- Type custom: string|null uniquement
class NullableString(fields.Raw):
    def format(self, value):
        if value is None:
            return None
        return str(value)

# ---- Type custom: int|null uniquement
class NullableInteger(fields.Raw):
    def format(self, value):
        if value is None:
            return None
        return int(value)

# ---- Type custom: float|null uniquement
class NullableFloat(fields.Raw):
    def format(self, value):
        if value is None:
            return None
        return float(value)

# ---- Type custom: date|null uniquement
class NullableDate(fields.Raw):
    def format(self, value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return str(value)

# ---- Type custom: datetime|null uniquement
class NullableDateTime(fields.Raw):
    def format(self, value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

# ---- Type custom: any|null uniquement
class NullableAny(fields.Raw):
    def format(self, value):
        if value is None:
            return None
        return value

# ---- Type custom: enum|null uniquement
class NullableEnum(fields.Raw):
    def __init__(self, enum_class, **kwargs):
        super().__init__(**kwargs)
        self.enum_class = enum_class

    def format(self, value):
        if value is None:
            return None
        return str(value)

# ===== Schemas RESTX (complexes) =====

run_model = dispatch_ns.model(
    "DispatchRunRequest",
    {
        "for_date": fields.String(required=True, description="Date YYYY-MM-DD"),
        "regular_first": fields.Boolean(default=True, description="Priorité aux chauffeurs réguliers"),
        "allow_emergency": NullableBoolean(description="Autoriser les chauffeurs d'urgence"),
        "async": fields.Boolean(default=True, description="Mode asynchrone"),
        "overrides": NullableDict(description="Surcharges de paramètres"),
        "mode": fields.String(description="Mode d'opération (auto|solver_only|heuristic_only)")
    },
)

trigger_model = dispatch_ns.model(
    "DispatchTriggerRequest",
    {
        "for_date": fields.String(required=True, description="Date YYYY-MM-DD"),
        "regular_first": fields.Boolean(default=True, description="Priorité aux chauffeurs réguliers"),
        "allow_emergency": NullableBoolean(description="Autoriser les chauffeurs d'urgence")
    },
)

autorun_model = dispatch_ns.model(
    "DispatchAutorunRequest",
    {
        "enabled": fields.Boolean(required=True, description="Activer/désactiver l'autorun"),
        "interval_sec": fields.Integer(required=False, description="Intervalle en secondes (optionnel)")
    },
)

# ✅ Corrige la sérialisation de booking/driver via Nested plutôt que dict(obj)
booking_model = dispatch_ns.model(
    "BookingBrief",
    {
        "id": fields.Integer,
        "reference": NullableString,
        "company_id": fields.Integer,
        "customer_name": NullableString,
        "pickup_address": NullableString,
        "dropoff_address": NullableString,
        "scheduled_time": NullableDateTime,
        "status": NullableString,
    },
)

driver_user_model = dispatch_ns.model(
    "DriverUserBrief",
    {
        "id": fields.Integer,
        "first_name": NullableString,
        "last_name": NullableString,
        "username": NullableString,
    },
)

driver_model = dispatch_ns.model(
    "DriverBrief",
    {
        "id": fields.Integer,
        "company_id": fields.Integer,
        "user": fields.Nested(driver_user_model, skip_none=True),
        "username": NullableString,  # Champ flat pour faciliter l'accès
        "first_name": NullableString,  # Nom du user
        "last_name": NullableString,   # Prénom du user
        "full_name": NullableString,   # Nom complet calculé
    },
)

assignment_model = dispatch_ns.model(
    "Assignment",
    {
        "id": fields.Integer,
        "booking_id": fields.Integer,
        "driver_id": fields.Integer,
        "dispatch_run_id": fields.Integer,
        "status": NullableString,
        "pickup_eta": NullableString,
        "dropoff_eta": NullableString,
        "created_at": NullableDateTime,
        "updated_at": NullableDateTime,
        "booking": fields.Nested(booking_model, skip_none=True),
        "driver": fields.Nested(driver_model, skip_none=True),
    },
)

assignment_patch_model = dispatch_ns.model(
    "AssignmentPatch",
    {
        "driver_id": fields.Integer,
        "status": fields.String(enum=[s.value for s in AssignmentStatus])
    },
)

reassign_model = dispatch_ns.model(
    "ReassignRequest",
    {
        "new_driver_id": fields.Integer(required=True),
    },
)

dispatch_run_model = dispatch_ns.model(
    "DispatchRun",
    {
        "id": fields.Integer,
        "company_id": fields.Integer,
        "day": NullableDate,
        "created_at": NullableDateTime,
        "started_at": NullableDateTime,
        "completed_at": NullableDateTime,
        "status": NullableString,
        "meta": NullableDict
    },
)

dispatch_run_detail_model = dispatch_ns.model(
    "DispatchRunDetail",
    {
        "id": fields.Integer,
        "company_id": fields.Integer,
        "day": NullableDate,
        "created_at": NullableDateTime,
        "started_at": NullableDateTime,
        "completed_at": NullableDateTime,
        "status": NullableString,
        "meta": NullableDict,
        "assignments": fields.List(fields.Nested(assignment_model))
    },
)

delay_model = dispatch_ns.model(
    "Delay",
    {
        "id": fields.Integer,
        "booking_id": fields.Integer,
        "driver_id": fields.Integer,
        "assignment_id": fields.Integer,
        "pickup_time": NullableDateTime,
        "dropoff_time": NullableDateTime,
        "pickup_eta": NullableDateTime,
        "dropoff_eta": NullableDateTime,
        "pickup_delay_minutes": fields.Integer,
        "dropoff_delay_minutes": fields.Integer,
        "booking": NullableDict,
        "driver": NullableDict
    },
)

# ===== Helpers =====

def _coerce_bool_param(v: Optional[str], default: bool = False) -> bool:
    """Interprète un paramètre bool venant de la query-string."""
    if v is None:
        return default
    v = v.strip().lower()
    return v not in ("0", "false", "no", "off", "")

def _get_current_company() -> Company:
    """Récupère l'entreprise courante en s'alignant sur la logique de routes/companies.py."""
    company, err, code = get_company_from_token()
    if err or company is None:
        # err est typiquement {"error": "..."}
        msg = (err or {}).get("error") if isinstance(err, dict) else "Accès refusé"
        dispatch_ns.abort(code or 403, msg)
        assert False
    return cast(Company, company)

def _current_company_id() -> int:
    c = _get_current_company()
    cid = getattr(c, "id", None)
    return cid if isinstance(cid, int) else int(cast(Any, cid))



def _parse_date(date_str: Optional[str]) -> date:
    """Parse une date YYYY-MM-DD. Si None ou vide, retourne aujourd'hui."""
    if not date_str:
        return date.today()
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        dispatch_ns.abort(400, f"Format de date invalide: {date_str} (attendu: YYYY-MM-DD)")
        assert False

def _booking_time_expr() -> Any:
    B = cast(Any, Booking)
    return B.scheduled_time

def _safe_settings_dict(raw: Any) -> Dict[str, Any]:
    """Décodage JSON tolérant pour company.dispatch_settings."""
    if not isinstance(raw, str):
        return {}
    try:
        return cast(Dict[str, Any], json.loads(raw))
    except Exception:
        return {}
    
def _enrich_assignments(assignments: list[Assignment]) -> list[Assignment]:
    """Charge booking/driver sur chaque assignment pour le marshalling RESTX."""
    for a in assignments:
        a.booking = Booking.query.get(getattr(a, "booking_id", None))
        driver_id_val = getattr(a, "driver_id", None)
        if driver_id_val is not None:
            a.driver = Driver.query.get(driver_id_val)
    return assignments


def _day_bounds(d: date) -> tuple[datetime, datetime]:
    """Bornes locales naïves d'une journée via util partagé."""
    d0, d1 = day_local_bounds(d.strftime("%Y-%m-%d"))
    return d0, d1


# ===== Routes =====

@dispatch_ns.route("/run")
class CompanyDispatchRun(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.expect(run_model, validate=False)
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

        # --- Récupérer l'entreprise courante + id int safe (évite Column[int])
        company = _get_current_company()
        _cid = getattr(company, "id", None)
        company_id: int = _cid if isinstance(_cid, int) else int(cast(Any, _cid))

        # --- Mode async ou sync
        # Accept both 'async' and 'run_async' for compatibility
        is_async = body.get("async")
        if is_async is None:
            is_async = body.get("run_async", True)
        
        mode = body.get("mode")

        # --- Paramètres
        allow_emergency_val = body.get("allow_emergency", None)
        allow_emergency = bool(allow_emergency_val) if allow_emergency_val is not None else None

        params = {
            "company_id": company_id,
            "for_date": for_date,
            "mode": mode,
            "regular_first": bool(body.get("regular_first", True)),
            "allow_emergency": allow_emergency
            
        }

        # --- Surcharges de paramètres
        # Extract mode from root or overrides
        mode = body.get("mode")
        if mode:
            params["mode"] = mode
      
        overrides = body.get("overrides")
        if overrides:
            params["overrides"] = overrides

        # --- Mode async: enfile un job
        if is_async:
            job = trigger_job(company_id, params)
            return job, 202

        # --- Mode sync: exécute immédiatement
        from services.unified_dispatch import engine
        result = engine.run(**params)
        return result, 200

@dispatch_ns.route("/status")
class CompanyDispatchStatus(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Statut courant du worker de dispatch (coalescing / dernier résultat / dernière erreur)."""
        try:
            company_id = _current_company_id()
            logger.debug("[Dispatch] Status check for company=%s", company_id)
            return get_status(company_id), 200
        except Exception as e:
            cid = locals().get("company_id", "?")
            logger.exception("[Dispatch] get_status failed company=%s", cid)
            dispatch_ns.abort(500, f"Erreur récupération statut: {e}")



@dispatch_ns.route("/preview")
class DispatchPreview(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.marshal_with(preview_response)  # <- on supprime `code=...`
    def get(self):
        """Aperçu de la journée (for_date): nb bookings/drivers et horizon (minutes)."""
        company = _get_current_company()
        company_id = cast(int, getattr(company, "id"))  # <- pas de `int(Column[int])`

        for_date = request.args.get("for_date")
        if not for_date:
            dispatch_ns.abort(400, "Paramètre for_date (YYYY-MM-DD) requis pour le preview.")

        # cohérent avec /run
        regular_first = request.args.get("regular_first", "true").lower() != "false"
        allow_emergency_bool = _coerce_bool_param(request.args.get("allow_emergency"), default=False)

        problem = data.build_problem_data(
            company_id=company_id,
            for_date=for_date,
            regular_first=regular_first,
            allow_emergency=allow_emergency_bool,
        )

        # accès tolérant
        if isinstance(problem, dict):
            n_bookings = len(problem.get("bookings", []))
            n_drivers = len(problem.get("drivers", []))
            horizon_minutes = int(problem.get("horizon_minutes", 0))
        else:
            n_bookings = len(getattr(problem, "bookings", []))
            n_drivers = len(getattr(problem, "drivers", []))
            horizon_minutes = int(getattr(problem, "horizon_minutes", 0))

        # On laisse Flask renvoyer 200 par défaut (pas de HTTPStatus dans le return)
        return {
            "bookings": n_bookings,
            "drivers": n_drivers,
            "horizon_minutes": horizon_minutes,
            "ready": n_bookings > 0 and n_drivers > 0,
            "reason": None,
        }


@dispatch_ns.route("/trigger")
class DispatchTrigger(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """(Déprécié) Déclenche un run async. Utilisez POST /company_dispatch/run."""
        company = _get_current_company()
        company_id: int = cast(int, getattr(company, "id"))

        body = request.get_json(silent=True) or {}
        for_date = body.get("for_date")
        if not for_date:
            dispatch_ns.abort(400, "for_date manquant (YYYY-MM-DD). Utilisez plutôt POST /company_dispatch/run.")

        allow_emergency = body.get("allow_emergency", None)
        if allow_emergency is not None:
            allow_emergency = bool(allow_emergency)

        params = {
            "company_id": company_id,
            "for_date": for_date,
            "regular_first": bool(body.get("regular_first", True)),
            "allow_emergency": allow_emergency,
        }

        job = trigger_job(company_id, params)
        return job, 202


@dispatch_ns.route("/autorun/enable")
class DispatchAutorunEnable(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.expect(autorun_model, validate=True)
    def post(self):
        company = _get_current_company()
        company_id: int = int(getattr(company, "id"))

        body = request.get_json(silent=True) or {}
        enabled = bool(body.get("enabled", True))
        interval_sec = body.get("interval_sec")

        try:
            # Lire les réglages existants en toute sécurité (l'attribut peut ne pas exister)
            settings_data: dict[str, Any] = {}
            settings_raw = getattr(company, "dispatch_settings", None)
            if isinstance(settings_raw, str) and settings_raw:
                try:
                    settings_data = json.loads(settings_raw)
                except json.JSONDecodeError:
                    settings_data = {}

            # Mettre à jour
            settings_data["autorun_enabled"] = enabled
            if interval_sec is not None:
                try:
                    settings_data["autorun_interval_sec"] = int(interval_sec)
                except (TypeError, ValueError):
                    # on ignore silencieusement une valeur invalide
                    pass

            # Sauvegarder (évite l'import local de json qui cassait la portée)
            setattr(company, "dispatch_settings", json.dumps(settings_data))
            db.session.add(company)
            db.session.commit()

            return {
                "company_id": company_id,
                "autorun_enabled": enabled,
                "autorun_interval_sec": settings_data.get("autorun_interval_sec", 300),
            }, 200

        except Exception as e:
            db.session.rollback()
            logger.exception("[Dispatch] autorun settings update failed company=%s", company_id)
            dispatch_ns.abort(500, f"Erreur mise à jour autorun: {e}")

@dispatch_ns.route("/assignments")
class AssignmentsListResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(params={"date": "YYYY-MM-DD"})
    @dispatch_ns.marshal_list_with(assignment_model)
    def get(self):
        """Liste des assignations pour un jour."""
        
        d = _parse_date(request.args.get("date"))
        # Utiliser day_local_bounds pour obtenir les bornes locales du jour (naïves)
        # Booking.scheduled_time est naïf local, donc on ne convertit PAS en UTC
        d0_local, d1_local = day_local_bounds(d.strftime("%Y-%m-%d"))
        # Pas de conversion UTC - on utilise directement les bornes locales
        d0, d1 = d0_local, d1_local

        # 🔒 Filtre multi-colonnes temps (comme le front)
        company = _get_current_company()
        time_expr = _booking_time_expr()

        # Ids des bookings du jour (entreprise courante), en excluant les statuts terminés/annulés
        booking_ids = [
            b.id
            for b in (
                Booking.query.filter(
                    Booking.company_id == company.id,
                       time_expr >= d0,     # Comparaison avec bornes locales naïves
                       time_expr < d1,
                       # ✅ Exclure COMPLETED/RETURN_COMPLETED/CANCELLED/CANCELED
                       cast(Any, Booking.status).notin_(
                           [s for s in [
                               getattr(BookingStatus, "COMPLETED", None),
                               getattr(BookingStatus, "RETURN_COMPLETED", None),
                               getattr(BookingStatus, "CANCELLED", None),
                               getattr(BookingStatus, "CANCELED", None),
                           ] if s is not None]
                       ),
                ).all()
            )
        ]

        # Assignations pour ces bookings avec eager loading des relations
        from sqlalchemy.orm import joinedload
        
        assignments = []
        if booking_ids:
            assignments = (
                Assignment.query
                .filter(Assignment.booking_id.in_(booking_ids))
                .options(
                    joinedload(Assignment.booking),  # Charger booking
                    joinedload(Assignment.driver).joinedload(Driver.user)  # Charger driver + user
                )
                .all()
            )
        
        # Enrichir manuellement les champs flat pour Flask-RESTX
        for a in assignments:
            if a.driver and a.driver.user:
                user = a.driver.user
                # Ajouter les champs flat au driver pour le marshalling
                a.driver.username = user.username
                a.driver.first_name = user.first_name
                a.driver.last_name = user.last_name
                full = f"{user.first_name or ''} {user.last_name or ''}".strip()
                a.driver.full_name = full or user.username

        return assignments



@dispatch_ns.route("/assignments/<int:assignment_id>")
class AssignmentResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.marshal_with(assignment_model)
    def get(self, assignment_id: int):
        """Détail d'une assignation."""
        company = _get_current_company()
        a_opt: Optional[Assignment] = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(Assignment.id == assignment_id, Booking.company_id == company.id)
            .first()
        )
        if a_opt is None:
            dispatch_ns.abort(404, "assignment not found")

        a = cast(Assignment, a_opt)
        return a

    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.expect(assignment_patch_model)
    @dispatch_ns.marshal_with(assignment_model)
    def patch(self, assignment_id: int):
        company = _get_current_company()
        a_opt = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(Assignment.id == assignment_id, Booking.company_id == company.id)
            .first()
        )
        if a_opt is None:
            dispatch_ns.abort(404, "assignment not found")

        a = cast(Assignment, a_opt)

        try:
            data = request.get_json() or {}
            if "driver_id" in data:
                a.driver_id = data["driver_id"]
            if "status" in data:
                a.status = data["status"]

            setattr(cast(Any, a), "updated_at", datetime.now(timezone.utc))

            db.session.add(a)
            db.session.commit()
            return a
        except Exception as e:
            db.session.rollback()   # 👈 IMPORTANT
            logger.exception("[Dispatch] patch assignment failed id=%s", assignment_id)
            dispatch_ns.abort(500, f"Erreur MAJ assignation: {e}")



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
        data = request.get_json() or {}
        new_driver_id = int(data["new_driver_id"])
        company = _get_current_company()

        a_opt = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(Assignment.id == assignment_id, Booking.company_id == company.id)
            .first()
        )
        if a_opt is None:
            dispatch_ns.abort(404, "assignment not found")

        try:
            a = cast(Assignment, a_opt)

            driver_opt = Driver.query.filter_by(id=new_driver_id, company_id=company.id).first()
            if driver_opt is None:
                dispatch_ns.abort(404, "driver not found")
            driver = cast(Driver, driver_opt)

            setattr(cast(Any, a), "driver_id", new_driver_id)
            setattr(cast(Any, a), "updated_at", datetime.now(timezone.utc))

            db.session.add(a)
            db.session.commit()

            setattr(cast(Any, a), "booking", Booking.query.get(a.booking_id))
            setattr(cast(Any, a), "driver", driver)

            return a
        except Exception as e:
            db.session.rollback()   # 👈 IMPORTANT
            logger.exception("[Dispatch] reassign failed assignment_id=%s", assignment_id)
            dispatch_ns.abort(500, f"Erreur réassignation: {e}")




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
        order_cols.append(DispatchRun.id.desc())
        
        q = q.order_by(*order_cols)
        return q.limit(limit).offset(offset).all()


@dispatch_ns.route("/runs/<int:run_id>")
class RunResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.marshal_with(dispatch_run_detail_model)
    def get(self, run_id: int):
        # S’assure que la session n’est pas en état “aborted”
        try:
            db.session.rollback()
        except Exception:
            pass
        """Détail d'un run + ses assignations."""
        company = _get_current_company()

        r_opt: Optional[DispatchRun] = DispatchRun.query.filter_by(
            id=run_id, company_id=company.id
        ).first()
        if r_opt is None:
            dispatch_ns.abort(404, "dispatch run not found")

        r = cast(DispatchRun, r_opt)

        assigns = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(Assignment.dispatch_run_id == run_id, Booking.company_id == company.id)
            .all()
        )

        return {
            "id": r.id,
            "company_id": r.company_id,
            "day": str(getattr(r, "day", "")),
            "created_at": getattr(r, "created_at", None),
            "started_at": getattr(r, "started_at", None),
            "completed_at": getattr(r, "completed_at", None),
            "status": getattr(r, "status", None),
            "meta": getattr(r, "meta", {}),
            "assignments": assigns,
        }


@dispatch_ns.route("/delays")
class DelaysResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(params={"date": "YYYY-MM-DD"})
    @dispatch_ns.marshal_list_with(delay_model)
    def get(self):
        """Retards courants (ETA > horaire + 5 minutes) pour la journée."""
        
        # Validation de la date
        date_str = request.args.get("date")
        if not date_str:
            return {"error": "Paramètre 'date' manquant (format: YYYY-MM-DD)"}, 400
        
        try:
            d = _parse_date(date_str)
        except ValueError as e:
            return {"error": f"Format de date invalide: {e}"}, 400

        d0, d1 = day_local_bounds(d.strftime("%Y-%m-%d"))

        company = _get_current_company()
        time_expr = _booking_time_expr()
        assigns = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(
                Booking.company_id == company.id,
                time_expr >= d0,
                time_expr < d1,
            )
            .all()
        )

        # Calculer les retards
        delays = []
        for a in assigns:
            b = Booking.query.get(a.booking_id)
            if not b:
                continue

            # Temps prévus
            pickup_time = getattr(b, "pickup_time", None) or getattr(b, "scheduled_time", None)
            dropoff_time = getattr(b, "dropoff_time", None)

            # Coerce strings -> datetime when needed
            def _to_dt(v):
                if v is None:
                    return None
                if isinstance(v, datetime):
                    return v
                try:
                    # naive ISO string
                    return datetime.fromisoformat(str(v))
                except Exception:
                    return None
            pickup_time = _to_dt(pickup_time)
            dropoff_time = _to_dt(dropoff_time)

            # ETAs (compat: plusieurs noms possibles)
            pickup_eta = (
                getattr(a, "pickup_eta", None)
                or getattr(a, "eta_pickup_at", None)
                or getattr(a, "estimated_pickup_arrival", None)
            )
            dropoff_eta = (
                getattr(a, "dropoff_eta", None)
                or getattr(a, "eta_dropoff_at", None)
                or getattr(a, "estimated_dropoff_arrival", None)
            )

            pickup_eta = _to_dt(pickup_eta)
            dropoff_eta = _to_dt(dropoff_eta)

            # Calcul des retards
            pickup_delay = 0
            if pickup_time and pickup_eta:
                try:
                    pickup_delay = max(0, int((pickup_eta - pickup_time).total_seconds() // 60))
                except Exception:
                    pickup_delay = 0

            dropoff_delay = 0
            if dropoff_time and dropoff_eta:
                try:
                    dropoff_delay = max(0, int((dropoff_eta - dropoff_time).total_seconds() // 60))
                except Exception:
                    dropoff_delay = 0

            # Toujours renvoyer si on a un ETA; le front pourra afficher "À l'heure" (0)
            if pickup_eta or dropoff_eta:
                max_delay = max(v for v in [pickup_delay, dropoff_delay] if v is not None) if (pickup_delay is not None or dropoff_delay is not None) else 0
                
                # ✨ NOUVEAUTÉ: Générer des suggestions intelligentes
                suggestions_list = []
                try:
                    if max_delay != 0:  # Générer suggestions si retard ou avance
                        suggestions_list = generate_suggestions(
                            a, 
                            delay_minutes=max_delay if pickup_delay > 0 else -abs(max_delay),
                            company_id=company.id
                        )
                        suggestions_list = [s.to_dict() for s in suggestions_list]
                except Exception as e:
                    logger.warning("[Delays] Failed to generate suggestions for assignment %s: %s", a.id, e)
                
                delay = {
                    "id": a.id,
                    "booking_id": a.booking_id,
                    "driver_id": a.driver_id,
                    "assignment_id": a.id,
                    "pickup_time": pickup_time,
                    "dropoff_time": dropoff_time,
                    "pickup_eta": pickup_eta,
                    "dropoff_eta": dropoff_eta,
                    "pickup_delay_minutes": pickup_delay,
                    "dropoff_delay_minutes": dropoff_delay,
                    "delay_minutes": max_delay,
                    # infos utiles côté front pour affichage
                    "scheduled_time": getattr(b, "scheduled_time", None),
                    "estimated_arrival": pickup_eta or dropoff_eta,
                    "booking": b,
                    "driver": Driver.query.get(a.driver_id) if a.driver_id else None,
                    # ✨ Suggestions intelligentes
                    "suggestions": suggestions_list,
                }
                delays.append(delay)

        return delays


@dispatch_ns.route("/delays/live")
class LiveDelaysResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(params={"date": "YYYY-MM-DD"})
    def get(self):
        """
        Retards en temps réel avec recalcul des ETAs et suggestions intelligentes.
        Inclut les retards actuels ET prédits, avec suggestions de réassignation
        et impact sur les courses suivantes.
        """
        
        # Validation de la date
        date_str = request.args.get("date")
        if not date_str:
            return {"error": "Paramètre 'date' manquant (format: YYYY-MM-DD)"}, 400
        
        try:
            d = _parse_date(date_str)
        except ValueError as e:
            return {"error": f"Format de date invalide: {e}"}, 400

        d0, d1 = day_local_bounds(d.strftime("%Y-%m-%d"))

        company = _get_current_company()
        time_expr = _booking_time_expr()
        
        # Récupérer toutes les assignations actives (EXCLURE les courses terminées)
        assigns = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(
                Booking.company_id == company.id,
                time_expr >= d0,
                time_expr < d1,
                # ✅ EXCLURE les statuts terminés
                cast(Any, Booking.status).notin_([
                    BookingStatus.COMPLETED,
                    BookingStatus.RETURN_COMPLETED,
                    BookingStatus.CANCELED,
                ]),
            )
            .all()
        )

        delays = []
        for a in assigns:
            b = Booking.query.get(a.booking_id)
            if not b:
                continue
            
            # ✅ Double vérification : skip les courses terminées
            if b.status in [
                BookingStatus.COMPLETED,
                BookingStatus.RETURN_COMPLETED,
                BookingStatus.CANCELED,
            ]:
                continue

            # Récupérer le chauffeur pour position temps réel
            driver = Driver.query.get(a.driver_id) if a.driver_id else None
            
            # Position actuelle du chauffeur
            if driver:
                driver_pos = (
                    getattr(driver, "current_lat", getattr(driver, "latitude", 46.2044)),
                    getattr(driver, "current_lon", getattr(driver, "longitude", 6.1432))
                )
            else:
                driver_pos = None
            
            # Position pickup
            pickup_lat = getattr(b, "pickup_lat", None)
            pickup_lon = getattr(b, "pickup_lon", None)
            pickup_pos = (pickup_lat, pickup_lon) if pickup_lat and pickup_lon else None
            
            # Temps prévus
            pickup_time = getattr(b, "pickup_time", None) or getattr(b, "scheduled_time", None)
            dropoff_time = getattr(b, "dropoff_time", None)

            # Coerce strings -> datetime
            def _to_dt(v):
                if v is None:
                    return None
                if isinstance(v, datetime):
                    return v
                try:
                    return datetime.fromisoformat(str(v))
                except Exception:
                    return None
            
            pickup_time = _to_dt(pickup_time)
            dropoff_time = _to_dt(dropoff_time)

            # Recalcul ETA en temps réel si position chauffeur ET pickup disponibles
            current_eta = None
            if driver_pos and pickup_pos and pickup_time:
                try:
                    # Utiliser calculate_eta pour estimation temps réel
                    eta_seconds = data.calculate_eta(driver_pos, pickup_pos)
                    current_eta = now_local() + timedelta(seconds=eta_seconds)
                except Exception as e:
                    logger.warning("[LiveDelays] Failed to calculate ETA for assignment %s: %s", a.id, e)
            
            # Utiliser ETA planifié en fallback
            if not current_eta:
                current_eta = (
                    getattr(a, "pickup_eta", None)
                    or getattr(a, "eta_pickup_at", None)
                    or getattr(a, "estimated_pickup_arrival", None)
                )
                current_eta = _to_dt(current_eta)

            # Calcul du retard
            delay_minutes = 0
            status = "unknown"
            
            if pickup_time and current_eta:
                try:
                    delay_seconds = (current_eta - pickup_time).total_seconds()
                    delay_minutes = int(delay_seconds / 60)
                    
                    if delay_minutes > 5:
                        status = "late"
                    elif delay_minutes < -5:
                        status = "early"
                    else:
                        status = "on_time"
                except Exception:
                    pass
            elif pickup_time and not current_eta:
                # ⭐ FALLBACK : Si pas d'ETA disponible, comparer heure actuelle vs heure prévue
                # Utile pour détecter les retards même sans GPS
                try:
                    current_time = now_local()
                    time_diff_seconds = (current_time - pickup_time).total_seconds()
                    
                    # Si l'heure actuelle est déjà passée et le chauffeur n'est pas arrivé
                    if time_diff_seconds > 300:  # 5 minutes de buffer
                        delay_minutes = int(time_diff_seconds / 60)
                        status = "late"
                    elif time_diff_seconds < -300:
                        delay_minutes = int(time_diff_seconds / 60)
                        status = "early"
                    else:
                        status = "on_time"
                except Exception as e:
                    logger.warning("[LiveDelays] Failed to calculate time-based delay: %s", e)

            # Générer suggestions intelligentes
            suggestions_list = []
            if delay_minutes != 0:
                try:
                    suggestions = generate_suggestions(a, delay_minutes, company.id)
                    suggestions_list = [s.to_dict() for s in suggestions]
                    logger.info("[LiveDelays] Generated %d suggestions for assignment %s (delay: %d min)", 
                               len(suggestions_list), a.id, delay_minutes)
                except Exception as e:
                    logger.exception("[LiveDelays] Failed to generate suggestions for assignment %s: %s", a.id, e)

            # Vérifier l'impact cascade (courses suivantes du même chauffeur)
            cascade_impact = []
            if driver and delay_minutes > 5:
                try:
                    # Trouver les prochaines courses du chauffeur
                    next_assignments = (
                        Assignment.query
                        .join(Booking, Booking.id == Assignment.booking_id)
                        .filter(
                            Assignment.driver_id == driver.id,
                            Assignment.id != a.id,
                            Booking.scheduled_time > pickup_time,
                            Booking.scheduled_time < pickup_time + timedelta(hours=4)
                        )
                        .order_by(Booking.scheduled_time.asc())
                        .limit(3)
                        .all()
                    )
                    
                    for next_a in next_assignments:
                        next_b = Booking.query.get(next_a.booking_id)
                        if next_b:
                            cascade_impact.append({
                                "booking_id": next_b.id,
                                "scheduled_time": next_b.scheduled_time.isoformat() if next_b.scheduled_time else None,
                                "customer_name": getattr(next_b, "customer_name", None),
                                "potential_delay_minutes": delay_minutes  # Propagation simplifiée
                            })
                except Exception as e:
                    logger.warning("[LiveDelays] Failed to check cascade impact: %s", e)

            # Construire la réponse
            if current_eta or delay_minutes != 0:
                delay = {
                    "id": a.id,
                    "booking_id": a.booking_id,
                    "driver_id": a.driver_id,
                    "assignment_id": a.id,
                    "delay_minutes": delay_minutes,
                    "status": status,
                    "current_eta": current_eta.isoformat() if current_eta else None,
                    "scheduled_time": pickup_time.isoformat() if pickup_time else None,
                    "pickup_time": pickup_time.isoformat() if pickup_time else None,
                    "dropoff_time": dropoff_time.isoformat() if dropoff_time else None,
                    
                    # ✨ Suggestions intelligentes
                    "suggestions": suggestions_list,
                    
                    # ✨ Impact cascade
                    "impacts_next_bookings": cascade_impact,
                    
                    # Infos contextuelles
                    "booking": {
                        "id": b.id,
                        "reference": getattr(b, "reference", None),
                        "customer_name": getattr(b, "customer_name", None),
                        "pickup_address": getattr(b, "pickup_address", None),
                        "dropoff_address": getattr(b, "dropoff_address", None),
                    },
                    "driver": {
                        "id": driver.id,
                        "name": f"{driver.user.first_name} {driver.user.last_name}" if driver and driver.user else None,
                        "current_position": {
                            "lat": driver_pos[0] if driver_pos else None,
                            "lon": driver_pos[1] if driver_pos else None,
                        } if driver_pos else None,
                    } if driver else None,
                }
                delays.append(delay)

        # Statistiques globales
        total = len(delays)
        late = len([d for d in delays if d["status"] == "late"])
        early = len([d for d in delays if d["status"] == "early"])
        on_time = len([d for d in delays if d["status"] == "on_time"])
        
        avg_delay = 0
        if delays:
            delay_values = [d["delay_minutes"] for d in delays]
            avg_delay = sum(delay_values) / len(delay_values) if delay_values else 0

        return {
            "delays": delays,
            "summary": {
                "total": total,
                "late": late,
                "early": early,
                "on_time": on_time,
                "average_delay": round(avg_delay, 2),
            },
            "timestamp": now_local().isoformat(),
        }, 200


@dispatch_ns.route("/optimizer/start")
class OptimizerStartResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """
        Démarre le monitoring en temps réel pour l'entreprise.
        Surveille automatiquement les retards et propose des optimisations.
        """
        company = _get_current_company()
        company_id = int(getattr(company, "id"))
        
        body = request.get_json(silent=True) or {}
        check_interval = int(body.get("check_interval_seconds", 120))  # 2 min par défaut
        
        try:
            optimizer = start_optimizer_for_company(company_id, check_interval)
            status = optimizer.get_status()
            
            return {
                "message": "Monitoring temps réel démarré",
                "status": status
            }, 200
        
        except Exception as e:
            logger.exception("[Optimizer] Failed to start for company %s", company_id)
            return {"error": f"Échec du démarrage: {e}"}, 500


@dispatch_ns.route("/optimizer/stop")
class OptimizerStopResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Arrête le monitoring en temps réel pour l'entreprise."""
        company = _get_current_company()
        company_id = int(getattr(company, "id"))
        
        try:
            stop_optimizer_for_company(company_id)
            
            return {
                "message": "Monitoring temps réel arrêté",
                "company_id": company_id
            }, 200
        
        except Exception as e:
            logger.exception("[Optimizer] Failed to stop for company %s", company_id)
            return {"error": f"Échec de l'arrêt: {e}"}, 500


@dispatch_ns.route("/optimizer/status")
class OptimizerStatusResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère le statut du monitoring temps réel."""
        company = _get_current_company()
        company_id = int(getattr(company, "id"))
        
        try:
            optimizer = get_optimizer_for_company(company_id)
            
            if optimizer is None:
                return {
                    "running": False,
                    "company_id": company_id,
                    "message": "Monitoring non démarré"
                }, 200
            
            status = optimizer.get_status()
            return status, 200
        
        except Exception as e:
            logger.exception("[Optimizer] Failed to get status for company %s", company_id)
            return {"error": f"Échec récupération statut: {e}"}, 500


@dispatch_ns.route("/optimizer/opportunities")
class OptimizerOpportunitiesResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(params={"date": "YYYY-MM-DD (optionnel, défaut: aujourd'hui)"})
    def get(self):
        """
        Récupère les opportunités d'optimisation détectées.
        Mode manuel: lance une vérification à la demande.
        """
        company = _get_current_company()
        company_id = int(getattr(company, "id"))
        
        date_str = request.args.get("date")
        
        try:
            # Vérifier si un optimizer est actif
            optimizer = get_optimizer_for_company(company_id)
            
            if optimizer and optimizer.get_status()["running"]:
                # Utiliser le cache du monitoring actif
                opportunities = optimizer.get_current_opportunities()
            else:
                # Vérification manuelle
                opportunities = check_opportunities_manual(company_id, date_str)
            
            return {
                "opportunities": [o.to_dict() for o in opportunities],
                "count": len(opportunities),
                "critical_count": len([o for o in opportunities if o.severity == "critical"]),
                "high_count": len([o for o in opportunities if o.severity == "high"]),
                "timestamp": now_local().isoformat(),
            }, 200
        
        except Exception as e:
            logger.exception("[Optimizer] Failed to get opportunities for company %s", company_id)
            return {"error": f"Échec récupération opportunités: {e}"}, 500