# backend/routes/dispatch_routes.py
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional, cast
import json
import re

from flask import request
from flask_restx import Namespace, Resource, fields
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy import func

from ext import role_required, db
from models import (
    User,
    UserRole,
    Company,
    Booking,
    Assignment,
    Driver,
    DispatchRun,
    AssignmentStatus
)
from services.unified_dispatch import data
from services.unified_dispatch.queue import trigger_job, get_status
from werkzeug.exceptions import UnprocessableEntity
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
    },
)

trigger_model = dispatch_ns.model(
    "DispatchTriggerRequest",
    {
        "for_date": fields.String(required=True, description="Date YYYY-MM-DD"),
        "regular_first": fields.Boolean(default=True, description="Priorité aux chauffeurs réguliers"),
        "allow_emergency": NullableBoolean(description="Autoriser les chauffeurs d'urgence"),
    },
)

autorun_model = dispatch_ns.model(
    "DispatchAutorunRequest",
    {
        "enabled": fields.Boolean(required=True, description="Activer/désactiver l'autorun"),
        "interval_sec": fields.Integer(required=False, description="Intervalle en secondes (optionnel)"),
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
        "booking": NullableDict,
        "driver": NullableDict,
    },
)

assignment_patch_model = dispatch_ns.model(
    "AssignmentPatch",
    {
        "driver_id": fields.Integer,
        "status": fields.String(enum=[s.value for s in AssignmentStatus]),
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
        "meta": NullableDict,
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
        "assignments": fields.List(fields.Nested(assignment_model)),
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
        "driver": NullableDict,
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
    """Récupère l'entreprise courante depuis le token JWT (garanti non-None)."""
    user_id = get_jwt_identity()
    user = User.query.filter_by(public_id=user_id).first()
    if user is None or not getattr(user, "company_id", None):
        dispatch_ns.abort(403, "Accès refusé: utilisateur sans entreprise")
        assert False  # rassure l'analyseur statique
    if user.role != UserRole.company:
        dispatch_ns.abort(403, "Accès refusé: rôle utilisateur non autorisé")
        assert False
    company = Company.query.get(user.company_id)
    if company is None:
        dispatch_ns.abort(403, "Entreprise introuvable")
        assert False
    return company  # type: ignore[return-value]

def _current_company_id() -> int:
    """Renvoie l'id de l'entreprise courante (évite d'appeler le helper au module-level)."""
    company = _get_current_company()
    cid = getattr(company, "id", None)
    if isinstance(cid, int):
        return cid
    return int(cast(Any, cid))


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
    """Expression SQL: coalesce(pickup_time, scheduled_time)."""
    B = cast(Any, Booking)
    return func.coalesce(B.pickup_time, B.scheduled_time)

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

        # --- Récupérer l'entreprise courante + id int safe (évite Column[int])
        company = _get_current_company()
        _cid = getattr(company, "id", None)
        company_id: int = _cid if isinstance(_cid, int) else int(cast(Any, _cid))

        # --- Mode async ou sync
        is_async = body.get("async", True)

        # --- Paramètres
        allow_emergency_val = body.get("allow_emergency", None)
        allow_emergency = bool(allow_emergency_val) if allow_emergency_val is not None else None

        params = {
            "company_id": company_id,
            "for_date": for_date,
            "regular_first": bool(body.get("regular_first", True)),
            "allow_emergency": allow_emergency,
        }

        # --- Surcharges de paramètres
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
        """Active/désactive l'autorun pour l'entreprise courante."""
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

        # Ids des bookings du jour (entreprise courante)
        booking_ids = [
            b.id
            for b in (
                Booking.query.filter(
                    Booking.company_id == company.id,
                       time_expr >= d0,     # Comparaison avec bornes locales naïves
                       time_expr <= d1, 
                ).all()
            )
        ]

        # Assignations pour ces bookings
        assignments = []
        if booking_ids:
            assignments = Assignment.query.filter(Assignment.booking_id.in_(booking_ids)).all()

        # Enrichir avec booking et driver
        for a in assignments:
            a.booking = Booking.query.get(a.booking_id)
            if a.driver_id:
                a.driver = Driver.query.get(a.driver_id)

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
        """MAJ d'une assignation (driver/status)."""
        company = _get_current_company()
        a_opt: Optional[Assignment] = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(Assignment.id == assignment_id, Booking.company_id == company.id)
            .first()
        )
        if a_opt is None:
            dispatch_ns.abort(404, "assignment not found")

        a = cast(Assignment, a_opt)

        data = request.get_json() or {}
        if "driver_id" in data:
            a.driver_id = data["driver_id"]
        if "status" in data:
            a.status = data["status"]

        setattr(cast(Any, a), "updated_at", datetime.now(timezone.utc))

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
        new_driver_id = int(data["new_driver_id"])

        company = _get_current_company()

        a_opt: Optional[Assignment] = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(Assignment.id == assignment_id, Booking.company_id == company.id)
            .first()
        )
        if a_opt is None:
            dispatch_ns.abort(404, "assignment not found")
        a = cast(Assignment, a_opt)

        # Vérifier que le driver existe et appartient à l'entreprise
        driver_opt = Driver.query.filter_by(id=new_driver_id, company_id=company.id).first()
        if driver_opt is None:
            dispatch_ns.abort(404, "driver not found")
        driver = cast(Driver, driver_opt)

        # Mettre à jour l'assignation
        setattr(cast(Any, a), "driver_id", new_driver_id) 
        setattr(cast(Any, a), "updated_at", datetime.now(timezone.utc))  # évite l'erreur Column[datetime]

        db.session.add(a)
        db.session.commit()

        # Enrichir avec booking et driver (attributs transients pour le marshalling)
        setattr(cast(Any, a), "booking", Booking.query.get(a.booking_id))
        setattr(cast(Any, a), "driver", driver)

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
        order_cols.append(DispatchRun.id.desc())
        
        q = q.order_by(*order_cols)
        return q.limit(limit).offset(offset).all()


@dispatch_ns.route("/runs/<int:run_id>")
class RunResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.marshal_with(dispatch_run_detail_model)
    def get(self, run_id: int):
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
 
        from shared.time_utils import day_local_bounds
        d0, d1 = day_local_bounds(d.strftime("%Y-%m-%d"))

        company = _get_current_company()
        time_expr = _booking_time_expr()
        assigns = (
            Assignment.query.join(Booking, Booking.id == Assignment.booking_id)
            .filter(
                Booking.company_id == company.id,
                time_expr >= d0,
                time_expr <= d1,
            )
            .all()
        )

        # Seuil de retard (minutes)
        delay_threshold = 5

        # Calculer les retards
        delays = []
        for a in assigns:
            b = Booking.query.get(a.booking_id)
            if not b:
                continue

            # Temps prévus
            pickup_time = getattr(b, "pickup_time", None) or getattr(b, "scheduled_time", None)
            dropoff_time = getattr(b, "dropoff_time", None)

            # ETAs
            pickup_eta = getattr(a, "pickup_eta", None)
            dropoff_eta = getattr(a, "dropoff_eta", None)

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

            # Ajouter si retard significatif
            if pickup_delay >= delay_threshold or dropoff_delay >= delay_threshold:
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
                    "booking": b,
                    "driver": Driver.query.get(a.driver_id) if a.driver_id else None,
                }
                delays.append(delay)

        return delays