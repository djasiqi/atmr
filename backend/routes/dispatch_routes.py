# backend/routes/dispatch_routes.py
# pyright: reportAttributeAccessIssue=false
# ruff: noqa: I001  # Imports organisés manuellement pour meilleure lisibilité

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, cast

from flask import current_app, request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import (  # pyright: ignore[reportMissingImports]
    Namespace,
    Resource,
    fields,
)
from marshmallow import (  # pyright: ignore[reportMissingImports]
    Schema,
    fields as ma_fields,
    validate,
)

from ext import db, limiter, role_required

# Enums - à conserver
from models.enums import AssignmentStatus, BookingStatus, UserRole

# Modèles - utilisés pour types/annotations dans ce fichier
# TODO: Vérifier si des requêtes directes existent et les migrer vers repositories
from models import Assignment, Booking, Company, DispatchRun, Driver
from middleware.trace_id import get_trace_id
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from shared.error_handlers import APIErrorHandler

# ✅ REFACTORING: get_company_from_token() n'est plus importé directement
# ✅ DDD: Utilisation de GetCurrentCompanyUseCase via _get_current_company_via_use_case()
from infrastructure.dispatch import data_adapter as data
from infrastructure.dispatch.queue_adapter import get_status, trigger_job
from infrastructure.dispatch.reactive_suggestions_adapter import (
    generate_reactive_suggestions as generate_suggestions,
)
from infrastructure.dispatch.realtime_optimizer_adapter import (
    check_opportunities_manual,
    get_optimizer_for_company,
    start_optimizer_for_company,
    stop_optimizer_for_company,
)
from shared.time_utils import day_local_bounds, now_local

N_BOOKINGS_ZERO = 0
MAX_DELAY_ZERO = 0
PICKUP_DELAY_ZERO = 0
DELAY_MINUTES_THRESHOLD = 5  # Ancien seuil (conservé pour compatibilité)
# ✅ Classification des retards à 3 niveaux
DELAY_MINUTES_REASONABLE_MAX = 5  # 1-5 min : raisonnable
DELAY_MINUTES_MODERATE_MAX = 10  # 5-10 min : modéré
# >10 min : critique
TIME_DIFF_SECONDS_THRESHOLD = 0.300
DELAY_MINUTES_ZERO = 0
PREVIOUS_BOOKING_RELEVANCE_WINDOW_SECONDS = (
    7200  # 2h : fenêtre pour considérer une course précédente comme pertinente
)
TOTAL_FEEDBACKS_ZERO = 0

# RL Dispatch (déploiement production)
try:
    from services.rl.rl_dispatch_manager import RLDispatchManager  # type: ignore

    RL_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    RL_AVAILABLE = False
    RLDispatchManager = None

# Shadow Mode ( Monitoring seulement)
try:
    from services.ml.rl.shadow_mode_manager import ShadowModeManager

    SHADOW_MODE_AVAILABLE = True
    _shadow_manager = None
except ImportError:
    SHADOW_MODE_AVAILABLE = False
    ShadowModeManager = None
    _shadow_manager = None


def get_shadow_manager():
    """Récupère l'instance du shadow manager (singleton)."""
    global _shadow_manager  # noqa: PLW0603
    if not SHADOW_MODE_AVAILABLE or ShadowModeManager is None:
        return None
    if _shadow_manager is None:
        try:
            _shadow_manager = ShadowModeManager()
            logger.info("✅ Shadow Mode Manager initialisé pour dispatch")
        except Exception as e:
            logger.error("❌ Erreur init Shadow Mode: %s", e)
            _shadow_manager = None
    return _shadow_manager


dispatch_ns = Namespace(
    "company_dispatch", description="Dispatch par journée (contrat unifié)"
)
logger = logging.getLogger(__name__)

# ✅ P0: Modèles d'erreur standardisés
api_error_model = create_api_error_model(dispatch_ns)
validation_error_model = create_validation_error_model(dispatch_ns)
not_found_error_model = create_not_found_error_model(dispatch_ns)
permission_error_model = create_permission_error_model(dispatch_ns)

# ===== Schémas de validation Marshmallow =====


class DispatchOverridesSchema(Schema):
    """Schéma de validation pour les overrides de dispatch.

    Valide que seules les clés autorisées sont présentes dans overrides.
    Rejette les clés inconnues pour éviter les erreurs silencieuses.
    """

    heuristic = ma_fields.Dict(required=False, allow_none=True)
    solver = ma_fields.Dict(required=False, allow_none=True)
    service_times = ma_fields.Dict(required=False, allow_none=True)
    pooling = ma_fields.Dict(required=False, allow_none=True)
    time = ma_fields.Dict(required=False, allow_none=True)
    realtime = ma_fields.Dict(required=False, allow_none=True)
    fairness = ma_fields.Dict(required=False, allow_none=True)
    emergency = ma_fields.Dict(required=False, allow_none=True)
    matrix = ma_fields.Dict(required=False, allow_none=True)
    logging = ma_fields.Dict(required=False, allow_none=True)
    features = ma_fields.Dict(required=False, allow_none=True)
    autorun = ma_fields.Dict(required=False, allow_none=True)
    rl = ma_fields.Dict(required=False, allow_none=True)
    clustering = ma_fields.Dict(required=False, allow_none=True)
    multi_objective = ma_fields.Dict(required=False, allow_none=True)
    safety = ma_fields.Dict(required=False, allow_none=True)
    # ⚡ Champs supplémentaires pour fonctionnalités avancées
    reset_existing = ma_fields.Bool(required=False, allow_none=True)
    preferred_driver_id = ma_fields.Int(
        required=False, allow_none=True
    )  # ⚡ Permettre null
    fast_mode = ma_fields.Bool(required=False, allow_none=True)
    driver_load_multipliers = ma_fields.Dict(required=False, allow_none=True)

    class Meta:
        unknown = (
            "EXCLUDE"  # ✅ Rejeter les clés inconnues pour éviter erreurs silencieuses
        )


class DispatchRunSchema(Schema):
    """Schéma de validation pour les paramètres de lancement de dispatch."""

    for_date = ma_fields.Str(
        required=True, validate=validate.Regexp(r"^\d{4}-\d{2}-\d{2}$")
    )
    mode = ma_fields.Str(
        validate=validate.OneOf(["auto", "heuristic_only", "solver_only"])
    )
    regular_first = ma_fields.Bool()
    allow_emergency = ma_fields.Bool()
    overrides = ma_fields.Nested(DispatchOverridesSchema)
    # ✅ UNIFIÉ : Une seule variante pour 'async' avec valeur par défaut
    async_param = ma_fields.Bool(data_key="async", load_default=True)


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


def _make_json_safe(value: Any) -> Any:
    """Convertit récursivement les objets vers des types compatibles JSON."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(v) for v in value]
    return value


# ===== Schemas RESTX (complexes) =====

run_model = dispatch_ns.model(
    "DispatchRunRequest",
    {
        "for_date": fields.String(required=True, description="Date YYYY-MM-DD"),
        "regular_first": fields.Boolean(
            default=True, description="Priorité aux chauffeurs réguliers"
        ),
        "allow_emergency": NullableBoolean(
            description="Autoriser les chauffeurs d'urgence"
        ),
        "async": fields.Boolean(default=True, description="Mode asynchrone"),
        "overrides": NullableDict(description="Surcharges de paramètres"),
        "mode": fields.String(
            description="Mode d'opération (auto|solver_only|heuristic_only)"
        ),
    },
)

trigger_model = dispatch_ns.model(
    "DispatchTriggerRequest",
    {
        "for_date": fields.String(required=True, description="Date YYYY-MM-DD"),
        "regular_first": fields.Boolean(
            default=True, description="Priorité aux chauffeurs réguliers"
        ),
        "allow_emergency": NullableBoolean(
            description="Autoriser les chauffeurs d'urgence"
        ),
    },
)

autorun_model = dispatch_ns.model(
    "DispatchAutorunRequest",
    {
        "enabled": fields.Boolean(
            required=True, description="Activer/désactiver l'autorun"
        ),
        "interval_sec": fields.Integer(
            required=False, description="Intervalle en secondes (optionnel)"
        ),
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
        "last_name": NullableString,  # Prénom du user
        "full_name": NullableString,  # Nom complet calculé
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


def _coerce_bool_param(v: str | None, default: bool = False) -> bool:
    """Interprète un paramètre bool venant de la query-string."""
    if v is None:
        return default
    v = v.strip().lower()
    return v not in ("0", "false", "no", "off", "")


def _get_current_company() -> Company:
    """Récupère l'entreprise courante via use-case (DDD).

    ✅ DDD: Utilise use-case au lieu de service directement.
    """
    from routes.companies import _get_current_company_via_use_case

    company, err, code = _get_current_company_via_use_case()
    if err or company is None:
        # err est typiquement {"error": "..."}
        msg = (err or {}).get("error") if isinstance(err, dict) else "Accès refusé"
        dispatch_ns.abort(code or 403, msg)
        msg = "Company should not be None after abort"
        raise AssertionError(msg)
    return company


def _current_company_id() -> int:
    c = _get_current_company()
    cid = getattr(c, "id", None)
    return cid if isinstance(cid, int) else int(cast("Any", cid))


def _validate_date_format(date_str: str | None) -> str:
    """Valide le format YYYY-MM-DD d'une date.

    Args:
        date_str: Chaîne de date à valider

    Returns:
        La chaîne de date si valide

    Raises:
        Abort 400 si format invalide
    """
    if not date_str:
        dispatch_ns.abort(400, "Paramètre date requis (format: YYYY-MM-DD)")
        msg = "Should not continue after abort"
        raise AssertionError(msg)

    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        dispatch_ns.abort(
            400,
            f"Format de date invalide: {date_str} (attendu: YYYY-MM-DD)",
        )
        msg = "Should not continue after abort"
        raise AssertionError(msg)

    # Vérifier que la date est valide (ex: pas 2025-13-45)
    try:
        date.fromisoformat(date_str)
    except ValueError as err:
        dispatch_ns.abort(
            400,
            f"Date invalide: {date_str} (format correct mais date inexistante)",
        )
        msg = "Should not continue after abort"
        raise AssertionError(msg) from err

    return date_str


def _parse_date(date_str: str | None) -> date:
    """Parse une date YYYY-MM-DD. Si None ou vide, retourne aujourd'hui."""
    if not date_str:
        return datetime.now(UTC).date()
    # Valider le format avant de parser
    _validate_date_format(date_str)
    try:
        # Parse sans timezone (intentionnel car on veut juste la date)
        return date.fromisoformat(date_str)
    except ValueError as err:
        dispatch_ns.abort(
            400, f"Format de date invalide: {date_str} (attendu: YYYY-MM-DD)"
        )
        msg = "Date parsing should not continue after abort"
        raise AssertionError(msg) from err


def _booking_time_expr() -> Any:
    B = cast("Any", Booking)
    return B.scheduled_time


# ===== Routes =====


@dispatch_ns.route("/run")
class CompanyDispatchRun(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("30 per hour")  # ✅ 2.8: Rate limiting lancement dispatch (coûteux)
    @dispatch_ns.expect(run_model, validate=False)
    @dispatch_ns.doc(
        description="""
        Lance un dispatch pour une journée donnée.

        **Mode asynchrone (async=true, par défaut)**:
        - Enfile un job Celery via la queue
        - Retourne 202 avec job_id et dispatch_run_id
        - Utilisez GET /company_dispatch/status pour suivre le statut
        - Recommandé pour >10 bookings

        **Mode synchrone (async=false)**:
        - Exécute le dispatch immédiatement
        - Retourne 200 avec le résultat complet
        - Limité à <10 bookings (sinon erreur 400)
        - Utilisez uniquement pour tests ou petits volumes

        **Overrides**:
        Les overrides permettent de surcharger les paramètres de dispatch:
        - `heuristic`: { "driver_load_balance": 0.5, "proximity": 0.3 }
        - `fairness`: { "fairness_weight": 0.8 }
        - `solver`: { "time_limit_sec": 120 }
        - `preferred_driver_id`: ID du chauffeur préféré
          (ignoré dans Settings mais utilisé par heuristics)
        - `reset_existing`: true pour réinitialiser les assignations existantes
        - `fast_mode`: true pour activer le mode rapide (solver désactivé)

        **Exemples de payload**:

        Dispatch asynchrone simple:
        ```json
        {
          "for_date": "2025-01-15",
          "async": true
        }
        ```

        Dispatch avec overrides:
        ```json
        {
          "for_date": "2025-01-15",
          "async": true,
          "regular_first": true,
          "allow_emergency": false,
          "overrides": {
            "heuristic": {
              "driver_load_balance": 0.7,
              "proximity": 0.2
            },
            "fairness": {
              "fairness_weight": 0.9
            },
            "preferred_driver_id": 123
          }
        }
        ```

        **Validation**:
        Utilisez POST /company_dispatch/settings/validate
        pour valider les overrides avant exécution.
        """,
        responses={
            200: "Dispatch synchrone réussi",
            202: "Dispatch asynchrone enfilé (job_id retourné)",
            400: "Paramètres invalides ou mode sync avec >10 bookings",
            500: "Erreur serveur",
        },
        example={
            "for_date": "2025-01-15",
            "async": True,
            "regular_first": True,
            "allow_emergency": None,
            "overrides": {
                "heuristic": {"driver_load_balance": 0.5},
                "fairness": {"fairness_weight": 0.8},
            },
        },
    )
    def post(self):
        """Lance un dispatch pour une journée donnée.
        - async=true (défaut) : enfile un job via la queue (202)
        - async=false : exécute immédiatement (200).
        """
        body: Dict[str, Any] = request.get_json(force=True) or {}
        logger.info("[Dispatch] /run body: %s", body)

        # ✅ DDD: Utilisation directe de DispatchUseCase
        from application.dispatch.dispatch_use_case import DispatchUseCase
        from domain.dispatch.commands import DispatchRunRequestCommand
        from infrastructure.dispatch.data_adapter import get_bookings_for_day
        from infrastructure.dispatch.engine_runner import run_dispatch_engine
        from infrastructure.dispatch.validation_runner import (
            validate_dispatch_assignments,
        )

        # Créer le use case avec les dépendances
        dispatch_use_case = DispatchUseCase(
            get_bookings_for_day_fn=get_bookings_for_day,
            getenv_fn=os.getenv,
            engine_run_fn=run_dispatch_engine,
            validate_assignments_fn=validate_dispatch_assignments,
        )

        # Validation et normalisation
        validated_data, error_response, status_code = (
            dispatch_use_case.validate_and_normalize_request(
                DispatchRunRequestCommand(company_id=0, body=body)
            )
        )
        if error_response or validated_data is None:
            if error_response:
                return error_response, status_code or 400
            return APIErrorHandler.handle_validation_error(
                "Erreur de validation",
                logger_instance=logger,
            )

        # Normalisation du mode
        effective_mode = dispatch_use_case.normalize_dispatch_mode(validated_data)

        # Récupérer l'entreprise courante
        company = _get_current_company()
        _cid = getattr(company, "id", None)
        company_id: int = _cid if isinstance(_cid, int) else int(cast("Any", _cid))

        # Date
        for_date = validated_data.get("for_date")
        if not for_date or not isinstance(for_date, str):
            return APIErrorHandler.handle_validation_error(
                "for_date is required",
                field="for_date",
                logger_instance=logger,
            )

        # Mode async ou sync
        is_async = validated_data.get("async_mode", True)

        # Vérifier si on doit forcer async
        should_force, force_reason = dispatch_use_case.should_force_async_mode(
            company_id=company_id,
            for_date=for_date,
            is_async=is_async,
            getenv_fn=os.getenv,
        )
        if should_force:
            is_async = True
            validated_data["_force_async_reason"] = force_reason

        # Préparer les paramètres
        params = dispatch_use_case.prepare_dispatch_params(
            validated_data=validated_data,
            company_id=company_id,
            effective_mode=effective_mode,
        )

        # Mode async: enfile un job
        if is_async:
            job = trigger_job(company_id, params)
            # Ajouter avertissement si async forcé automatiquement
            if validated_data.get("_force_async_reason"):
                max_sync_bookings = int(os.getenv("DISPATCH_SYNC_MAX_BOOKINGS", "10"))
                bookings_count = len(get_bookings_for_day(company_id, for_date))
                job["warning"] = validated_data["_force_async_reason"]
                job["bookings_count"] = bookings_count
                job["max_sync_bookings"] = max_sync_bookings
            return job, 202

        # Mode sync: exécute immédiatement
        max_sync_bookings = int(os.getenv("DISPATCH_SYNC_MAX_BOOKINGS", "10"))
        bookings_count = len(get_bookings_for_day(company_id, for_date))
        logger.info(
            "[Dispatch] Mode sync autorisé: %d bookings (limite: %d)",
            bookings_count,
            max_sync_bookings,
        )

        # Exécuter le dispatch
        result, validation_info = dispatch_use_case.execute_dispatch_sync(params)

        # Ajouter les informations de validation au résultat
        if validation_info:
            result["validation"] = validation_info

        safe_result = _make_json_safe(result)
        return safe_result, 200


@dispatch_ns.route("/status")
class CompanyDispatchStatus(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(
        params={
            "date": (
                "Date optionnelle (YYYY-MM-DD) pour obtenir "
                "le statut d'un dispatch spécifique"
            ),
            "run_id": (
                "ID optionnel du DispatchRun pour obtenir le statut d'un run spécifique"
            ),
        },
        description="""
        ✅ P1: Endpoint de polling amélioré pour statut dispatch

        Retourne des informations détaillées sur le statut du dispatch :
        - `is_running`: Si un dispatch est en cours
        - `progress`: Progression (0-100%)
        - `celery_state`: État de la tâche Celery (PENDING, STARTED, SUCCESS, FAILURE)
        - `last_result`: Résultat du dernier dispatch (si disponible)
        - `active_dispatch_run`: Informations sur le DispatchRun actif
        - `counters`: Nombre de bookings, drivers, assignments
        - `estimated_time_remaining`: Temps estimé restant (si disponible)

        **Polling recommandé** :
        - Intervalle initial : 1 seconde
        - Intervalle si running : 2-3 secondes
        - Arrêter le polling si `is_running=false` et `progress=100`
        """,
    )
    def get(self):
        """Statut courant du worker de dispatch
        (coalescing / dernier résultat / dernière erreur).

        ✅ P1: Endpoint amélioré pour polling avec métadonnées détaillées

        Retourne:
        - Le statut du dernier dispatch (si disponible)
        - Le dispatch_run_id actif (si date ou run_id fourni)
        - Le nombre d'assignments créés pour la date
        - Le statut Celery de la tâche en cours
        - Progression et temps estimé restant
        """
        try:
            company_id = _current_company_id()
            for_date = request.args.get("date")  # ✅ Paramètre optionnel pour date
            run_id = request.args.get(
                "run_id"
            )  # ✅ P1: Paramètre optionnel pour run_id

            # ✅ Valider le format YYYY-MM-DD si fourni
            if for_date:
                for_date = _validate_date_format(for_date)

            # ✅ P1: Si run_id fourni, récupérer la date depuis le DispatchRun
            if run_id and not for_date:
                try:
                    run_id_int = int(run_id)
                    # Utiliser le repository pour récupérer le dispatch run
                    # Imports locaux pour éviter dépendances circulaires
                    from repositories.dispatch_run_repository import (
                        DispatchRunRepository,
                    )

                    dispatch_run_repo = DispatchRunRepository()
                    dispatch_run = dispatch_run_repo.find_model_by_id_and_company(
                        run_id_int, company_id
                    )
                    if dispatch_run and dispatch_run.day:
                        for_date = dispatch_run.day.isoformat()
                        logger.debug(
                            "[Dispatch] Resolved date from run_id=%s: %s",
                            run_id,
                            for_date,
                        )
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "[Dispatch] Invalid run_id format: %s (error: %s)", run_id, e
                    )

            logger.debug(
                "[Dispatch] Status check for company=%s date=%s run_id=%s",
                company_id,
                for_date,
                run_id,
            )

            status = get_status(company_id, for_date=for_date)

            # ✅ P1: Enrichir avec temps estimé restant si disponible
            if status.get("is_running") and status.get("celery_state") == "STARTED":
                # Estimation basique : 30-120 secondes selon le nombre de bookings
                bookings_count = status.get("counters", {}).get("bookings", 0)
                if bookings_count > 0:
                    # Estimation : ~2-5 secondes par booking
                    estimated_seconds = min(120, max(30, bookings_count * 3))
                    status["estimated_time_remaining_seconds"] = estimated_seconds
                    status["estimated_completion_time"] = (
                        datetime.now(UTC) + timedelta(seconds=estimated_seconds)
                    ).isoformat()

            return status, 200
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
        company_id = company.id

        for_date = request.args.get("for_date")
        if not for_date:
            dispatch_ns.abort(
                400, "Paramètre for_date (YYYY-MM-DD) requis pour le preview."
            )
        # ✅ Valider le format YYYY-MM-DD
        for_date = _validate_date_format(for_date)

        # cohérent avec /run
        regular_first = request.args.get("regular_first", "true").lower() != "false"
        allow_emergency_bool = _coerce_bool_param(
            request.args.get("allow_emergency"), default=False
        )

        problem = data.build_problem_data(
            company_id=company_id,
            for_date=for_date,
            regular_first=regular_first,
            allow_emergency=allow_emergency_bool,
        )

        # accès tolérant
        n_bookings = len(problem.get("bookings", []))
        n_drivers = len(problem.get("drivers", []))
        horizon_minutes = int(problem.get("horizon_minutes", 0))

        # On laisse Flask renvoyer 200 par défaut (pas de HTTPStatus dans le return)
        return {
            "bookings": n_bookings,
            "drivers": n_drivers,
            "horizon_minutes": horizon_minutes,
            "ready": n_bookings > N_BOOKINGS_ZERO and n_drivers > N_BOOKINGS_ZERO,
            "reason": None,
        }


@dispatch_ns.route("/trigger")
class DispatchTrigger(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("50 per hour")  # ✅ 2.8: Rate limiting trigger dispatch
    @dispatch_ns.doc(
        description="""
        ⚠️ **DÉPRÉCIÉ** - Cet endpoint sera supprimé dans une future version.

        **Migration recommandée**: Utilisez `POST /company_dispatch/run`
        avec `async=true`.

        **Guide de migration**: Voir `/docs/API_MIGRATION_TRIGGER_TO_RUN.md`

        Cet endpoint est maintenu pour compatibilité mais redirige vers `/run`.
        """,
        deprecated=True,
        responses={202: "Job enfilé (via /run)", 400: "Erreur de paramètres"},
    )
    def post(self):
        """(Déprécié) Déclenche un run async. Utilisez POST /company_dispatch/run."""
        company = _get_current_company()
        company_id = company.id

        body = request.get_json(silent=True) or {}
        for_date = body.get("for_date")
        if not for_date:
            dispatch_ns.abort(
                400,
                (
                    "for_date manquant (YYYY-MM-DD). "
                    "Utilisez plutôt POST /company_dispatch/run."
                ),
            )

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


@dispatch_ns.route("/settings/validate")
class DispatchSettingsValidate(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Valide des overrides de settings avant application.

        Accepte un payload avec 'overrides' et retourne :
        - applied: paramètres qui seront appliqués
        - ignored: paramètres ignorés (inconnus ou non applicables)
        - errors: erreurs de validation
        """
        from infrastructure.dispatch import settings_module_adapter as ud_settings

        company = _get_current_company()
        body = request.get_json(silent=True) or {}
        overrides = body.get("overrides", {})

        if not overrides:
            return {
                "valid": True,
                "applied": [],
                "ignored": [],
                "errors": [],
                "message": "Aucun override fourni",
            }, 200

        try:
            # Créer settings de base pour la company
            base_settings = ud_settings.for_company(company)

            # Tenter le merge pour valider
            # Note: on n'utilise pas strict_validation ici pour ne pas bloquer
            # mais on retourne les erreurs dans la réponse
            new_settings = ud_settings.merge_overrides(base_settings, overrides)

            # Récupérer le résultat de validation depuis les logs
            # (On pourrait améliorer merge_overrides pour retourner
            # le résultat de validation)
            # Pour l'instant, on vérifie manuellement les paramètres critiques
            validation_result = {
                "applied": [],
                "ignored": [],
                "errors": [],
            }

            # Vérifier les paramètres appliqués
            if "heuristic" in overrides:
                h_ov = overrides["heuristic"]
                if isinstance(h_ov, dict):
                    if "driver_load_balance" in h_ov:
                        if (
                            new_settings.heuristic.driver_load_balance
                            == h_ov["driver_load_balance"]
                        ):
                            validation_result["applied"].append(
                                "heuristic.driver_load_balance"
                            )
                        else:
                            validation_result["errors"].append(
                                "heuristic.driver_load_balance "
                                + f"demandé={h_ov['driver_load_balance']} "
                                + "mais appliqué="
                                + f"{new_settings.heuristic.driver_load_balance}"
                            )
                    if "proximity" in h_ov:
                        if new_settings.heuristic.proximity == h_ov["proximity"]:
                            validation_result["applied"].append("heuristic.proximity")
                        else:
                            validation_result["errors"].append(
                                f"heuristic.proximity demandé={h_ov['proximity']} "
                                + f"mais appliqué={new_settings.heuristic.proximity}"
                            )

            if "fairness" in overrides:
                f_ov = overrides["fairness"]
                if isinstance(f_ov, dict) and "fairness_weight" in f_ov:
                    if new_settings.fairness.fairness_weight == f_ov["fairness_weight"]:
                        validation_result["applied"].append("fairness.fairness_weight")
                    else:
                        validation_result["errors"].append(
                            "fairness.fairness_weight "
                            + f"demandé={f_ov['fairness_weight']} "
                            + "mais appliqué="
                            + f"{new_settings.fairness.fairness_weight}"
                        )

            # Identifier les clés ignorées (non dans Settings)
            known_ignored_keys = [
                "preferred_driver_id",
                "mode",
                "run_async",
                "reset_existing",
                "fast_mode",
            ]
            for key in overrides:
                if (
                    key
                    not in [
                        "heuristic",
                        "solver",
                        "fairness",
                        "features",
                        "time",
                        "service_times",
                        "pooling",
                        "realtime",
                        "emergency",
                        "matrix",
                        "logging",
                        "autorun",
                        "rl",
                        "clustering",
                        "multi_objective",
                        "safety",
                    ]
                    and key not in known_ignored_keys
                ):
                    validation_result["ignored"].append(key)

            return {
                "valid": len(validation_result["errors"]) == 0,
                "applied": validation_result["applied"],
                "ignored": validation_result["ignored"],
                "errors": validation_result["errors"],
                "message": "Validation complétée"
                if len(validation_result["errors"]) == 0
                else "Erreurs de validation détectées",
            }, 200

        except Exception as e:
            logger.exception("[Dispatch] Erreur validation settings: %s", e)
            return {
                "valid": False,
                "applied": [],
                "ignored": [],
                "errors": [str(e)],
                "message": f"Erreur lors de la validation: {e}",
            }, 400


@dispatch_ns.route("/autorun/enable")
class DispatchAutorunEnable(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.expect(autorun_model, validate=True)
    def post(self):
        company = _get_current_company()
        company_id: int = _current_company_id()

        body = request.get_json(silent=True) or {}
        enabled = bool(body.get("enabled", True))
        interval_sec = body.get("interval_sec")

        try:
            # Lire les réglages existants en toute sécurité
            # (l'attribut peut ne pas exister)
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
                from contextlib import suppress

                with suppress(TypeError, ValueError):
                    settings_data["autorun_interval_sec"] = int(interval_sec)

            # Sauvegarder (évite l'import local de json qui cassait la portée)
            cast("Any", company).dispatch_settings = json.dumps(settings_data)
            db.session.add(company)
            db.session.commit()

            return {
                "company_id": company_id,
                "autorun_enabled": enabled,
                "autorun_interval_sec": settings_data.get("autorun_interval_sec", 300),
            }, 200

        except Exception as e:
            db.session.rollback()
            logger.exception(
                "[Dispatch] autorun settings update failed company=%s", company_id
            )
            dispatch_ns.abort(500, f"Erreur mise à jour autorun: {e}")


@dispatch_ns.route("/assignments/validate")
class ValidateAssignmentsResource(Resource):
    """Valide les assignations existantes pour détecter les conflits temporels."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Vérifie les conflits temporels dans les assignations existantes.

        Query params:
            date: Date au format YYYY-MM-DD (défaut: aujourd'hui)

        Returns:
            {
                "valid": bool,
                "conflicts": List[Dict],
                "summary": Dict
            }
        """
        company_id = _current_company_id()
        date_str = request.args.get("date")

        target_date = (
            datetime.strptime(date_str, "%Y-%m-%d").date()
            if date_str
            else now_local().date()
        )

        try:
            from infrastructure.dispatch.validation_adapter import validate_assignments

            # Récupérer les assignations pour la date
            start_datetime = datetime.combine(target_date, datetime.min.time()).replace(
                tzinfo=UTC
            )
            end_datetime = datetime.combine(target_date, datetime.max.time()).replace(
                tzinfo=UTC
            )

            assignments_data = []
            # ✅ Utiliser le repository pour récupérer les assignments avec eager loading
            # Imports locaux pour éviter dépendances circulaires
            from repositories.assignment_repository import AssignmentRepository

            assignment_repo = AssignmentRepository()
            statuses_list = [
                AssignmentStatus.SCHEDULED,
                AssignmentStatus.EN_ROUTE_PICKUP,
                AssignmentStatus.ARRIVED_PICKUP,
                AssignmentStatus.ONBOARD,
                AssignmentStatus.EN_ROUTE_DROPOFF,
            ]
            assignments = assignment_repo.find_models_with_time_range_and_eager_loading(
                company_id, start_datetime, end_datetime, statuses_list
            )

            for assignment in assignments:
                assignments_data.append(
                    {
                        "booking_id": assignment.booking_id,
                        "driver_id": assignment.driver_id,
                        "scheduled_time": assignment.booking.scheduled_time.isoformat()
                        if assignment.booking.scheduled_time
                        else None,
                    }
                )

            # Valider
            result = validate_assignments(assignments_data, strict=False)

            # ✅ P1: Protéger accès dictionnaires pour éviter KeyError
            warnings = result.get("warnings", [])
            errors = result.get("errors", [])
            return {
                "valid": result.get("valid", False),
                "conflicts": warnings + errors,
                "summary": result.get("stats", {}),
                "date": target_date.isoformat(),
                "total_assignments": len(assignments_data),
            }, 200

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/assignments")
class AssignmentsListResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(params={"date": "YYYY-MM-DD"})
    @dispatch_ns.marshal_list_with(assignment_model)
    def get(self):
        """Liste des assignations pour un jour.

        Retourne toutes les assignations pour la date donnée,
        avec les relations booking et driver chargées.
        """
        try:
            date_str = request.args.get("date")
            logger.info(
                "[Dispatch] /assignments request for date=%s company_id=%s",
                date_str,
                _current_company_id(),
            )

            d = _parse_date(date_str)
            # Utiliser day_local_bounds pour obtenir les bornes locales du jour (naïves)
            # Booking.scheduled_time est naïf local, donc on ne convertit PAS en UTC
            d0local, d1local = day_local_bounds(d.strftime("%Y-%m-%d"))
            # Pas de conversion UTC - on utilise directement les bornes locales
            d0, d1 = d0local, d1local

            logger.debug("[Dispatch] /assignments date bounds: %s to %s", d0, d1)

            # 🔒 Filtre multi-colonnes temps (comme le front)
            company = _get_current_company()
            time_expr = _booking_time_expr()

            # Ids des bookings du jour (entreprise courante),
            # en excluant les statuts terminés/annulés
            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            excluded_statuses = [
                s
                for s in [
                    getattr(BookingStatus, "COMPLETED", None),
                    getattr(BookingStatus, "RETURN_COMPLETED", None),
                    getattr(BookingStatus, "CANCELLED", None),
                    getattr(BookingStatus, "CANCELED", None),
                ]
                if s is not None
            ]

            bookings = booking_repo.find_models_by_company_with_time_expr_and_excluded_statuses(
                company_id=company.id,
                time_expr=time_expr,
                start_datetime=d0,
                end_datetime=d1,
                excluded_statuses=excluded_statuses,
            )
            booking_ids = [b.id for b in bookings]

            logger.info(
                "[Dispatch] /assignments found %d bookings for date=%s company_id=%s",
                len(booking_ids),
                date_str,
                company.id,
            )

            # Assignations pour ces bookings avec eager loading des relations
            assignments = []
            if booking_ids:
                # Utiliser le repository pour récupérer les assignments avec eager loading
                # Imports locaux pour éviter dépendances circulaires
                from repositories.assignment_repository import AssignmentRepository

                assignment_repo = AssignmentRepository()
                assignments = (
                    assignment_repo.find_models_by_booking_ids_with_eager_loading(
                        booking_ids
                    )
                )

                logger.info(
                    (
                        "[Dispatch] /assignments found %d assignments "
                        "for %d bookings date=%s"
                    ),
                    len(assignments),
                    len(booking_ids),
                    date_str,
                )
            else:
                logger.debug(
                    (
                        "[Dispatch] /assignments no bookings found "
                        "for date=%s, returning empty assignments"
                    ),
                    date_str,
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

            logger.debug(
                "[Dispatch] /assignments returning %d assignments for date=%s",
                len(assignments),
                date_str,
            )

            return assignments

        except Exception as e:
            logger.exception(
                "[Dispatch] /assignments error for date=%s company_id=%s: %s",
                request.args.get("date"),
                _current_company_id(),
                e,
            )
            dispatch_ns.abort(500, f"Erreur récupération assignations: {e}")


@dispatch_ns.route("/assignments/<int:assignment_id>")
class AssignmentResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.marshal_with(assignment_model)
    def get(self, assignment_id: int):
        """Détail d'une assignation."""
        company = _get_current_company()
        # Utiliser le repository pour récupérer l'assignment
        # Imports locaux pour éviter dépendances circulaires
        from repositories.assignment_repository import AssignmentRepository

        assignment_repo = AssignmentRepository()
        a_opt: Assignment | None = assignment_repo.find_model_by_id_with_company_check(
            assignment_id, company.id
        )
        if a_opt is None:
            dispatch_ns.abort(404, "assignment not found")

        return cast("Assignment", a_opt)


@dispatch_ns.route("/assignments/<int:assignment_id>")
class AssignmentPatchResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.expect(assignment_patch_model)
    @dispatch_ns.doc(
        responses={
            "200": "Assignation mise à jour avec succès",
            "400": "Erreur de validation",
            "401": "Non authentifié",
            "403": "Non autorisé",
            "404": "Assignation non trouvée",
            "500": "Erreur serveur",
        }
    )
    @dispatch_ns.response(200, "Assignation mise à jour avec succès", assignment_model)
    @dispatch_ns.response(400, "Erreur de validation", validation_error_model)
    @dispatch_ns.response(401, "Non authentifié", permission_error_model)
    @dispatch_ns.response(403, "Non autorisé", permission_error_model)
    @dispatch_ns.response(404, "Assignation non trouvée", not_found_error_model)
    @dispatch_ns.response(500, "Erreur serveur", api_error_model)
    def patch(self, assignment_id: int):
        """Mettre à jour une assignation (driver_id, status).

        ✅ P0: Support trace_id pour le suivi.
        """
        company = _get_current_company()
        # Utiliser le repository pour récupérer l'assignment
        # Imports locaux pour éviter dépendances circulaires
        from repositories.assignment_repository import AssignmentRepository

        assignment_repo = AssignmentRepository()
        a_opt = assignment_repo.find_model_by_id_with_company_check(
            assignment_id, company.id
        )
        if a_opt is None:
            dispatch_ns.abort(404, "assignment not found")

        a = cast("Assignment", a_opt)

        try:
            data = request.get_json() or {}
            # ✅ P1: Protéger accès dictionnaires pour éviter KeyError
            driver_id = data.get("driver_id")
            if driver_id is not None:
                a.driver_id = driver_id
            status = data.get("status")
            if status is not None:
                a.status = status

            cast("Any", a).updated_at = datetime.now(UTC)

            db.session.add(a)
            db.session.commit()

            # ✅ P0: Ajouter trace_id dans la réponse
            trace_id = get_trace_id()
            logger.info(
                "[Dispatch] Assignation mise à jour: assignment_id=%s, company_id=%s",
                assignment_id,
                company.id,
                extra={
                    "trace_id": trace_id,
                    "assignment_id": assignment_id,
                    "company_id": company.id,
                },
            )

            # Ajouter trace_id à l'objet Assignment si possible
            if hasattr(a, "__dict__"):
                a_dict = cast("Any", a).serialize if hasattr(a, "serialize") else {}
                if isinstance(a_dict, dict):
                    a_dict["trace_id"] = trace_id
                    return a_dict, 200

            return a
        except Exception as e:
            db.session.rollback()  # 👈 IMPORTANT
            trace_id = get_trace_id()
            logger.exception(
                "[Dispatch] patch assignment failed id=%s",
                assignment_id,
                extra={"trace_id": trace_id},
            )
            dispatch_ns.abort(500, f"Erreur MAJ assignation: {e}")


@dispatch_ns.route("/assignments/<int:assignment_id>/reassign")
class ReassignResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.expect(
        dispatch_ns.model(
            "ReassignBody", {"new_driver_id": fields.Integer(required=True)}
        ),
        validate=True,
    )
    @dispatch_ns.doc(
        responses={
            "200": "Assignation réassignée avec succès",
            "400": "Erreur de validation",
            "401": "Non authentifié",
            "403": "Non autorisé",
            "404": "Assignation ou chauffeur non trouvé",
            "409": "Conflit temporel détecté",
            "500": "Erreur serveur",
        }
    )
    @dispatch_ns.response(200, "Assignation réassignée avec succès", assignment_model)
    @dispatch_ns.response(400, "Erreur de validation", validation_error_model)
    @dispatch_ns.response(401, "Non authentifié", permission_error_model)
    @dispatch_ns.response(403, "Non autorisé", permission_error_model)
    @dispatch_ns.response(
        404, "Assignation ou chauffeur non trouvé", not_found_error_model
    )
    @dispatch_ns.response(409, "Conflit temporel détecté", api_error_model)
    @dispatch_ns.response(500, "Erreur serveur", api_error_model)
    def post(self, assignment_id: int):
        """Réassigner une assignation à un nouveau chauffeur.

        ✅ P0: Support trace_id pour le suivi.
        """
        data = request.get_json() or {}
        # ✅ P1: Protéger accès dictionnaires pour éviter KeyError
        new_driver_id_raw = data.get("new_driver_id")
        if new_driver_id_raw is None:
            dispatch_ns.abort(400, "new_driver_id est requis")

        # Type narrowing: après la vérification, new_driver_id_raw n'est plus None
        # Utiliser un cast pour aider le type checker
        from typing import cast

        new_driver_id_raw_typed = cast(int | str | float, new_driver_id_raw)

        # Initialiser new_driver_id à None pour aider le type checker
        new_driver_id: int | None = None
        try:
            new_driver_id = int(new_driver_id_raw_typed)
        except (ValueError, TypeError) as e:
            dispatch_ns.abort(400, f"new_driver_id doit être un entier valide: {e}")

        # Vérification explicite pour le type checker (abort() lève une exception, donc ce code ne s'exécute que si new_driver_id est défini)
        if new_driver_id is None:
            dispatch_ns.abort(500, "Erreur interne: new_driver_id non défini")

        # Type narrowing: après la vérification, new_driver_id n'est plus None
        assert new_driver_id is not None, "new_driver_id should not be None here"

        company = _get_current_company()

        # ✅ Utiliser le repository pour récupérer l'assignment avec eager loading
        # Imports locaux pour éviter dépendances circulaires
        from repositories.assignment_repository import AssignmentRepository

        assignment_repo = AssignmentRepository()
        a_opt = assignment_repo.find_model_by_id_with_booking_eager_loading(
            assignment_id, company.id
        )
        if a_opt is None:
            dispatch_ns.abort(404, "assignment not found")

        try:
            a = cast("Assignment", a_opt)
            # ✅ P1: Utiliser la relation eager-loaded au lieu de requête séparée
            booking = a.booking

            # ✅ SHADOW MODE: Prédiction DQN (NON-BLOQUANTE)
            shadow_prediction = None
            if SHADOW_MODE_AVAILABLE and booking:
                try:
                    shadow_mgr = get_shadow_manager()
                    if shadow_mgr:
                        # Utiliser le repository pour récupérer les drivers disponibles
                        # Imports locaux pour éviter dépendances circulaires
                        from repositories.driver_repository import DriverRepository

                        driver_repo = DriverRepository()
                        available_drivers = driver_repo.find_models_by_company_available_with_user_eager_loading(
                            company.id
                        )

                        from collections import defaultdict

                        current_assignments = defaultdict(list)
                        # ✅ P1: Eager loading pour éviter N+1 queries
                        # Utiliser le repository pour récupérer les assignments actifs
                        # Imports locaux pour éviter dépendances circulaires
                        from repositories.assignment_repository import (
                            AssignmentRepository,
                        )

                        assignment_repo = AssignmentRepository()
                        active_statuses = [
                            AssignmentStatus.SCHEDULED,
                            AssignmentStatus.EN_ROUTE_PICKUP,
                        ]
                        active_assignments = assignment_repo.find_models_by_company_with_active_statuses_eager_loading(
                            company.id, active_statuses
                        )
                        for assign in active_assignments:
                            current_assignments[assign.driver_id].append(
                                assign.booking_id
                            )

                        shadow_prediction = shadow_mgr.predict_driver_assignment(
                            booking=booking,
                            available_drivers=available_drivers,
                            current_assignments=dict(current_assignments),
                        )
                        logger.debug(
                            "Shadow prediction for reassign: %s", shadow_prediction
                        )
                except Exception as e:
                    logger.warning("Shadow mode error (non-critique): %s", e)

            # ✅ SYSTÈME ACTUEL: Logique INCHANGÉE
            # Utiliser le repository pour récupérer le driver
            # Imports locaux pour éviter dépendances circulaires
            from repositories.driver_repository import DriverRepository

            driver_repo = DriverRepository()
            driver_opt = driver_repo.find_model_by_id_and_company(
                new_driver_id, company.id
            )
            if driver_opt is None:
                dispatch_ns.abort(404, "driver not found")
            driver = cast("Driver", driver_opt)

            # ✅ VALIDATION : Vérifier conflit temporel AVANT assignation
            if booking and booking.scheduled_time:
                from infrastructure.dispatch.validation_adapter import (
                    check_existing_assignment_conflict,
                )

                has_conflict, conflict_msg = check_existing_assignment_conflict(
                    driver_id=new_driver_id,
                    scheduled_time=booking.scheduled_time,
                    booking_id=booking.id,
                    tolerance_minutes=30,
                )

                if has_conflict:
                    logger.warning(
                        "[Dispatch] Tentative de réassignation créerait un conflit: %s",
                        conflict_msg,
                    )
                    dispatch_ns.abort(
                        409,  # HTTP 409 Conflict
                        f"❌ Impossible d'assigner ce chauffeur : {conflict_msg}",
                    )

            cast("Any", a).driver_id = new_driver_id
            cast("Any", a).updated_at = datetime.now(UTC)

            db.session.add(a)
            db.session.commit()

            # ✅ P0: Ajouter trace_id dans la réponse
            trace_id = get_trace_id()
            logger.info(
                "[Dispatch] Assignation réassignée: assignment_id=%s, new_driver_id=%s, company_id=%s",
                assignment_id,
                new_driver_id,
                company.id,
                extra={
                    "trace_id": trace_id,
                    "assignment_id": assignment_id,
                    "new_driver_id": new_driver_id,
                    "company_id": company.id,
                },
            )

            # ✅ MÉTRIQUES : Marquer suggestion comme appliquée
            try:
                from repositories.rl_suggestion_metric_repository import (
                    RLSuggestionMetricRepository,
                )

                # Trouver la métrique correspondante (la plus récente non appliquée)
                rl_metric_repo = RLSuggestionMetricRepository()
                metric = (
                    rl_metric_repo.find_by_assignment_and_suggested_driver_not_applied(
                        assignment_id=assignment_id, suggested_driver_id=new_driver_id
                    )
                )

                if metric:
                    metric.applied_at = datetime.now(UTC)

                    # Calculer gain réel (approximation basée sur ETA)
                    # Note : Le gain réel précis nécessiterait
                    # de tracker l'ETA avant/après
                    # Pour l'instant, on marque comme "appliqué"
                    # et on calculera le gain plus tard
                    metric.was_successful = True  # Assume succès (à affiner)

                    db.session.add(metric)
                    db.session.commit()
                    logger.info(
                        "[RL] Metric %s marked as applied", metric.suggestion_id
                    )
                else:
                    logger.debug(
                        "[RL] No metric found for assignment %s, driver %s",
                        assignment_id,
                        new_driver_id,
                    )
            except Exception as e:
                db.session.rollback()
                logger.warning("[RL] Failed to update metric (non-critique): %s", e)

            # ✅ P0: Ajouter trace_id dans la réponse
            trace_id = get_trace_id()
            logger.info(
                "[Dispatch] Assignation réassignée: assignment_id=%s, new_driver_id=%s, company_id=%s",
                assignment_id,
                new_driver_id,
                company.id,
                extra={
                    "trace_id": trace_id,
                    "assignment_id": assignment_id,
                    "new_driver_id": new_driver_id,
                    "company_id": company.id,
                },
            )

            # Ajouter trace_id à l'objet Assignment si possible
            if hasattr(a, "__dict__"):
                a_dict = cast("Any", a).serialize if hasattr(a, "serialize") else {}
                if isinstance(a_dict, dict):
                    a_dict["trace_id"] = trace_id
                    return a_dict, 200

            # ✅ CACHE REDIS : Invalider cache suggestions après réassignation
            from ext import redis_client

            if redis_client:
                try:
                    # ✅ P1: Booking déjà chargé via join précédent, pas besoin de requête supplémentaire
                    # booking est déjà disponible depuis la ligne 1461
                    if booking and booking.scheduled_time:
                        for_date_cache = booking.scheduled_time.date().isoformat()

                        # Supprimer toutes les clés de cache pour cette company/date
                        pattern = f"rl_suggestions:{company.id}:{for_date_cache}:*"
                        deleted_count = 0
                        for key in redis_client.scan_iter(match=pattern):
                            redis_client.delete(key)
                            deleted_count += 1

                        logger.info(
                            (
                                "[RL] Cache invalidated: %s keys deleted "
                                "for company %s, date %s"
                            ),
                            deleted_count,
                            company.id,
                            for_date_cache,
                        )
                except Exception as e:
                    logger.warning(
                        "[RL] Cache invalidation error (non-critique): %s", e
                    )

            # ✅ SHADOW MODE: Comparaison (NON-BLOQUANTE)
            if shadow_prediction:
                try:
                    shadow_mgr = get_shadow_manager()
                    if shadow_mgr and booking and driver:
                        from shared.geo_utils import haversine_distance

                        distance = None
                        if booking.pickup_lat and driver.current_lat:
                            distance = haversine_distance(
                                booking.pickup_lat,
                                booking.pickup_lon,
                                driver.current_lat,
                                driver.current_lon,
                            )

                        shadow_mgr.compare_with_actual_decision(
                            prediction=shadow_prediction,
                            actual_driver_id=new_driver_id,
                            outcome_metrics={
                                "distance_km": distance,
                                "reassignment": True,
                            },
                        )
                except Exception as e:
                    logger.warning("Shadow comparison error (non-critique): %s", e)

            # ✅ P1: Booking déjà chargé via join précédent, pas besoin de requête supplémentaire
            cast("Any", a).booking = booking
            cast("Any", a).driver = driver

            # Ajouter trace_id à l'objet Assignment si possible (fallback si pas déjà fait)
            if hasattr(a, "__dict__"):
                a_dict = cast("Any", a).serialize if hasattr(a, "serialize") else {}
                if isinstance(a_dict, dict) and "trace_id" not in a_dict:
                    a_dict["trace_id"] = trace_id
                    return a_dict, 200

            return a
        except Exception as e:
            db.session.rollback()  # 👈 IMPORTANT
            logger.exception(
                "[Dispatch] reassign failed assignment_id=%s", assignment_id
            )
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
        # Utiliser le repository pour obtenir la query
        # Imports locaux pour éviter dépendances circulaires
        from repositories.dispatch_run_repository import DispatchRunRepository

        dispatch_run_repo = DispatchRunRepository()
        q = dispatch_run_repo.find_model_by_company_id_query(company.id)

        # Fallback de tri: completed_at > started_at > day > created_at > id
        order_cols = []
        if hasattr(DispatchRun, "completed_at"):
            order_cols.append(DispatchRun.completed_at.desc())
        if hasattr(DispatchRun, "started_at"):
            order_cols.append(DispatchRun.started_at.desc())
        if hasattr(DispatchRun, "day"):
            order_cols.append(DispatchRun.day.desc())
        if hasattr(DispatchRun, "created_at"):
            order_cols.append(DispatchRun.created_at.desc())
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
        # S'assure que la session n'est pas en état "aborted"
        from contextlib import suppress

        with suppress(Exception):
            db.session.rollback()
        company = _get_current_company()

        # Utiliser le repository pour récupérer le dispatch run
        # Imports locaux pour éviter dépendances circulaires
        from repositories.dispatch_run_repository import DispatchRunRepository

        dispatch_run_repo = DispatchRunRepository()
        r_opt: DispatchRun | None = dispatch_run_repo.find_model_by_id_and_company(
            run_id, company.id
        )
        if r_opt is None:
            dispatch_ns.abort(404, "dispatch run not found")

        r = cast("DispatchRun", r_opt)

        # ✅ Utiliser le repository pour récupérer les assignments avec eager loading
        # Imports locaux pour éviter dépendances circulaires
        from repositories.assignment_repository import AssignmentRepository

        assignment_repo = AssignmentRepository()
        assigns = assignment_repo.find_models_by_dispatch_run_with_eager_loading(
            run_id, company.id
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
            return APIErrorHandler.handle_validation_error(
                "Paramètre 'date' manquant",
                field="date",
                expected_format="YYYY-MM-DD",
                logger_instance=logger,
            )

        try:
            d = _parse_date(date_str)
        except ValueError as e:
            return APIErrorHandler.handle_validation_error(
                f"Format de date invalide: {e}",
                field="date",
                provided_value=date_str,
                expected_format="YYYY-MM-DD",
                logger_instance=logger,
            )

        d0, d1 = day_local_bounds(d.strftime("%Y-%m-%d"))

        company = _get_current_company()
        # ✅ Utiliser le repository pour récupérer les assignments avec eager loading complet
        # Imports locaux pour éviter dépendances circulaires
        from repositories.assignment_repository import AssignmentRepository

        assignment_repo = AssignmentRepository()
        assigns = assignment_repo.find_models_by_company_in_time_window_with_full_eager_loading(
            company.id, d0, d1
        )

        # ✅ P1: Eager loading déjà fait via joinedload dans la requête précédente
        # Les bookings et drivers sont déjà chargés, pas besoin de requêtes supplémentaires
        # Calculer les retards
        delays = []
        for a in assigns:
            b = a.booking  # ✅ Déjà chargé via joinedload
            if not b:
                continue

            # Temps prévus
            pickup_time = getattr(b, "pickup_time", None) or getattr(
                b, "scheduled_time", None
            )
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
            pickup_delay = MAX_DELAY_ZERO
            if pickup_time and pickup_eta:
                try:
                    pickup_delay = max(
                        MAX_DELAY_ZERO,
                        int((pickup_eta - pickup_time).total_seconds() // 60),
                    )
                except Exception:
                    pickup_delay = MAX_DELAY_ZERO

            dropoff_delay = MAX_DELAY_ZERO
            if dropoff_time and dropoff_eta:
                try:
                    dropoff_delay = max(
                        MAX_DELAY_ZERO,
                        int((dropoff_eta - dropoff_time).total_seconds() // 60),
                    )
                except Exception:
                    dropoff_delay = MAX_DELAY_ZERO

            # Toujours renvoyer si on a un ETA; le front pourra afficher "À l'heure" (0)
            if pickup_eta or dropoff_eta:
                max_delay = max(pickup_delay, dropoff_delay)

                # ✨ NOUVEAUTÉ: Générer des suggestions intelligentes
                suggestions_list = []
                try:
                    if (
                        max_delay != MAX_DELAY_ZERO
                    ):  # Générer suggestions si retard ou avance
                        company_id_int = int(cast("Any", company.id))
                        suggestions_list = generate_suggestions(
                            a,
                            delay_minutes=max_delay
                            if pickup_delay > PICKUP_DELAY_ZERO
                            else -abs(max_delay),
                            company_id=company_id_int,
                        )
                        suggestions_list = [s.to_dict() for s in suggestions_list]
                except Exception as e:
                    logger.warning(
                        "[Delays] Failed to generate suggestions for assignment %s: %s",
                        a.id,
                        e,
                    )

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
                    # ✅ P1: Driver déjà chargé via joinedload
                    "driver": a.driver if hasattr(a, "driver") else None,
                    # ✨ Suggestions intelligentes
                    "suggestions": suggestions_list,
                }
                delays.append(delay)

        return delays


# ✅ Fonction helper pour classifier la sévérité d'un retard
def _classify_delay_severity(delay_minutes: int) -> str:
    """Classifie la sévérité d'un retard en 3 niveaux.

    Args:
        delay_minutes: Retard en minutes (positif = retard, négatif = avance)

    Returns:
        - "on_time" : Pas de retard (<= 0 min)
        - "reasonable" : Retard raisonnable (1-5 min)
        - "moderate" : Retard modéré (5-10 min)
        - "critical" : Retard critique (>10 min)
        - "early" : En avance (négatif)
    """
    if delay_minutes <= 0:
        # En avance ou à l'heure
        if delay_minutes < -DELAY_MINUTES_THRESHOLD:
            return "early"
        return "on_time"
    if delay_minutes <= DELAY_MINUTES_REASONABLE_MAX:
        return "reasonable"  # 1-5 min : raisonnable
    if delay_minutes <= DELAY_MINUTES_MODERATE_MAX:
        return "moderate"  # 5-10 min : modéré
    return "critical"  # >10 min : critique


# ✅ Fonction helper pour trouver la course précédente du chauffeur
def _get_driver_previous_booking(
    driver_id: int,
    current_booking: Booking,
    company_id: int,
    current_date_start: datetime,
    current_date_end: datetime,
) -> tuple[Booking, Assignment] | tuple[None, None]:
    """Trouve la course précédente du chauffeur avant la course courante.

    Args:
        driver_id: ID du chauffeur
        current_booking: Course courante pour laquelle on cherche la précédente
        company_id: ID de l'entreprise
        current_date_start: Début de la journée (pour filtrer)
        current_date_end: Fin de la journée (pour filtrer)

    Returns:
        Tuple (Booking, Assignment) de la course précédente, ou (None, None) si aucune
    """
    from shared.time_utils import _booking_time_expr

    try:
        current_scheduled_time = getattr(current_booking, "scheduled_time", None)
        if not current_scheduled_time:
            return None, None

        # Récupérer toutes les assignations du chauffeur pour la même date
        from repositories.assignment_repository import AssignmentRepository

        assignment_repo = AssignmentRepository()
        time_expr = _booking_time_expr()
        excluded_statuses = [
            BookingStatus.COMPLETED,
            BookingStatus.RETURN_COMPLETED,
            BookingStatus.CANCELED,
        ]

        previous_assignments = (
            assignment_repo.find_models_by_driver_with_time_expr_and_excluded_statuses(
                driver_id=driver_id,
                company_id=company_id,
                time_expr=time_expr,
                start_datetime=current_date_start,
                end_datetime=current_date_end,
                before_time=current_scheduled_time,
                excluded_statuses=excluded_statuses,
                limit=1,
            )
        )

        if previous_assignments:
            prev_assignment = previous_assignments[0]
            prev_booking = prev_assignment.booking
            return prev_booking, prev_assignment

        return None, None
    except Exception as e:
        logger.warning(
            "[LiveDelays] Error finding previous booking for driver %d: %s",
            driver_id,
            e,
        )
        return None, None


# ✅ Helper function pour calculer ETA en parallèle (utilisé par LiveDelaysResource)
def _calculate_eta_for_assignment(
    driver_pos: tuple[float, float] | None,
    pickup_pos: tuple[float, float] | None,
    use_haversine_only: bool = False,
) -> int | None:
    """Calcule l'ETA (en secondes) entre driver_pos et pickup_pos.
    Retourne None si les positions ne sont pas disponibles ou en cas d'erreur.
    Cette fonction est thread-safe et peut être appelée en parallèle.

    Args:
        driver_pos: Position actuelle du chauffeur
        pickup_pos: Position de pickup
        use_haversine_only: Si True, utilise uniquement Haversine (bypass OSRM)
    """
    if not driver_pos or not pickup_pos:
        return None

    # ✅ Si OSRM est indisponible (circuit breaker OPEN), utiliser directement Haversine
    if use_haversine_only:
        try:
            from infrastructure.dispatch.settings_adapter import Settings
            from shared.geo_utils import haversine_distance

            # Créer une instance par défaut
            # (DEFAULT_SETTINGS n'existe pas dans settings.py)
            default_settings = Settings()
            distance_km = haversine_distance(
                driver_pos[0], driver_pos[1], pickup_pos[0], pickup_pos[1]
            )
            avg_speed_kmh = float(
                getattr(getattr(default_settings, "matrix", None), "avg_speed_kmh", 25)
            )
            eta_seconds = int((distance_km / max(avg_speed_kmh, 1e-3)) * 3600.0)
            return max(1, eta_seconds)
        except Exception as e:
            logger.warning("[LiveDelays] Haversine fallback failed: %s", e)
            return None

    try:
        return data.calculate_eta(driver_pos, pickup_pos)
    except Exception as e:
        logger.warning("[LiveDelays] Failed to calculate ETA: %s", e)
        return None


@dispatch_ns.route("/delays/live")
class LiveDelaysResource(Resource):
    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])  # ✅ Permettre aux admins d'accéder
    @dispatch_ns.doc(
        params={
            "date": "YYYY-MM-DD",
            "company_id": "ID entreprise (optionnel pour ADMIN)",
        }
    )
    def get(self):  # noqa: PLR0911
        """Retards en temps réel avec recalcul des ETAs et suggestions intelligentes.
        Inclut les retards actuels ET prédits, avec suggestions de réassignation
        et impact sur les courses suivantes.
        ✅ OPTIMISÉ: Parallélise les calculs d'ETA pour améliorer les performances.
        ✅ OPTIMISÉ: Timeout global de 20s pour éviter les timeouts frontend.
        """
        endpoint_start_time = time.time()
        ENDPOINT_TIMEOUT_SECONDS = (
            15  # ✅ Timeout global réduit à 15s pour éviter les timeouts frontend (30s)
        )

        # Validation de la date
        date_str = request.args.get("date")
        if not date_str:
            return APIErrorHandler.handle_validation_error(
                "Paramètre 'date' manquant",
                field="date",
                expected_format="YYYY-MM-DD",
                logger_instance=logger,
            )

        try:
            d = _parse_date(date_str)
        except ValueError as e:
            return APIErrorHandler.handle_validation_error(
                f"Format de date invalide: {e}",
                field="date",
                provided_value=date_str,
                expected_format="YYYY-MM-DD",
                logger_instance=logger,
            )

        d0, d1 = day_local_bounds(d.strftime("%Y-%m-%d"))

        # ✅ Gérer le cas admin : accès total, company_id optionnel
        # ✅ DDD: Utilisation de use-cases au lieu de service directement
        from routes.companies import _get_current_company_via_use_case
        from shared.infrastructure.adapters.auth_adapter import (
            get_current_user_via_use_case,
        )

        company, err, code = _get_current_company_via_use_case()

        # Si erreur et c'est un 404 "No company", vérifier si c'est un admin
        if err and code == HTTPStatus.NOT_FOUND:
            # Utiliser use-case pour récupérer l'utilisateur
            user = get_current_user_via_use_case()
            if user and user.role == UserRole.admin:
                # Admin : accès total, company_id optionnel
                company_id_param = request.args.get("company_id")
                if company_id_param:
                    try:
                        company_id = int(company_id_param)
                    except (ValueError, TypeError):
                        return APIErrorHandler.handle_validation_error(
                            "company_id doit être un nombre entier",
                            field="company_id",
                            provided_value=company_id_param,
                        )

                    # Utiliser le repository pour récupérer l'entreprise
                    from repositories.company_repository import CompanyRepository

                    company_repo = CompanyRepository()
                    company_obj = company_repo.find_model_by_id(company_id)
                    if not company_obj:
                        return APIErrorHandler.handle_not_found(
                            "Entreprise", company_id, logger
                        )
                    company = company_obj
                else:
                    # Admin sans company_id : utiliser la première entreprise trouvée
                    from repositories.company_repository import CompanyRepository

                    company_repo = CompanyRepository()
                    company_obj = company_repo.find_first_model()
                    if not company_obj:
                        return {
                            "error": "Aucune entreprise trouvée dans le système"
                        }, HTTPStatus.NOT_FOUND
                    company = company_obj
            else:
                # Pas admin, retourner l'erreur originale
                return err, code
        elif err:
            # Autre erreur, la retourner
            return err, code

        # Si on arrive ici, company est défini (via get_company_from_token ou logique admin)
        if company is None:
            return APIErrorHandler.handle_not_found("Entreprise", None, logger)

        time_expr = _booking_time_expr()

        # ✅ CRITIQUE: Filtrer les assignations par proximité temporelle
        # (1h avant à 1h après le pickup)
        # On ne doit calculer des ETAs que pour les courses
        # proches de leur heure de pickup
        # Cela évite de calculer des ETAs pour des courses
        # qui sont à H+00 (plusieurs heures avant)
        now = now_local()
        TIME_WINDOW_BEFORE_MINUTES = (
            60  # Commencer à surveiller 1 heure avant (augmenté de 30 à 60 min)
        )
        TIME_WINDOW_AFTER_MINUTES = 60  # Arrêter de surveiller 1 heure après
        time_window_start = now - timedelta(minutes=TIME_WINDOW_BEFORE_MINUTES)
        time_window_end = now + timedelta(minutes=TIME_WINDOW_AFTER_MINUTES)

        # Utiliser le repository pour récupérer les assignments avec filtres complexes
        from repositories.assignment_repository import AssignmentRepository

        assignment_repo = AssignmentRepository()
        excluded_statuses = [
            BookingStatus.COMPLETED,
            BookingStatus.RETURN_COMPLETED,
            BookingStatus.CANCELED,
        ]

        assigns = assignment_repo.find_models_by_company_with_time_expr_and_time_window(
            company_id=company.id,
            time_expr=time_expr,
            day_start=d0,
            day_end=d1,
            window_start=time_window_start,
            window_end=time_window_end,
            excluded_statuses=excluded_statuses,
            limit=50,
        )

        logger.info(
            "[LiveDelays] Found %d assignments in time window [%s, %s] for company %s",
            len(assigns),
            time_window_start.isoformat(),
            time_window_end.isoformat(),
            company.id,
        )

        # ✅ Si aucune assignation dans la fenêtre temporelle, retourner rapidement
        if not assigns:
            logger.debug(
                "[LiveDelays] No assignments in time window, returning empty response"
            )
            return {
                "delays": [],
                "summary": {
                    "total": 0,
                    "late": 0,
                    "early": 0,
                    "on_time": 0,
                    "average_delay": 0,
                },
                "timestamp": now.isoformat(),
            }, 200

        # ✅ P1: Eager loading déjà fait via joinedload dans la requête précédente
        # Les bookings et drivers sont déjà chargés, pas besoin de requêtes supplémentaires
        # ✅ ÉTAPE 1: Préparer les données pour le calcul parallèle des ETAs
        assignment_data = []
        for a in assigns:
            b = a.booking  # ✅ Déjà chargé via joinedload
            if not b:
                continue

            # ✅ Double vérification : skip les courses terminées et IN_PROGRESS
            # IN_PROGRESS = client déjà à bord, donc pas de retard possible
            if b.status in [
                BookingStatus.COMPLETED,
                BookingStatus.RETURN_COMPLETED,
                BookingStatus.CANCELED,
                BookingStatus.IN_PROGRESS,  # ✅ Client déjà à bord, pas de retard
            ]:
                continue

            # Récupérer le chauffeur pour position temps réel
            driver = (
                a.driver if hasattr(a, "driver") else None
            )  # ✅ Déjà chargé via joinedload

            # Position actuelle du chauffeur
            if driver:
                driver_pos = (
                    getattr(
                        driver, "current_lat", getattr(driver, "latitude", 46.2044)
                    ),
                    getattr(
                        driver, "current_lon", getattr(driver, "longitude", 6.1432)
                    ),
                )
            else:
                driver_pos = None

            # Position pickup
            pickup_lat = getattr(b, "pickup_lat", None)
            pickup_lon = getattr(b, "pickup_lon", None)
            pickup_pos = (pickup_lat, pickup_lon) if pickup_lat and pickup_lon else None

            assignment_data.append(
                {
                    "assignment": a,
                    "booking": b,
                    "driver": driver,
                    "driver_pos": driver_pos,
                    "pickup_pos": pickup_pos,
                }
            )

        # ✅ ÉTAPE 2: Vérifier le circuit breaker OSRM AVANT de lancer les calculs
        # Si OSRM est indisponible, utiliser Haversine directement pour tous les calculs
        use_haversine_only = False
        try:
            from services.osrm_client import _osrm_circuit_breaker

            if _osrm_circuit_breaker.state == "OPEN":
                logger.info(
                    "[LiveDelays] OSRM circuit breaker is OPEN, using Haversine only for all ETA calculations"
                )
                use_haversine_only = True
        except Exception as e:
            logger.warning(
                (
                    "[LiveDelays] Could not check OSRM circuit breaker: %s, "
                    "will try OSRM first"
                ),
                e,
            )
            # Si on ne peut pas vérifier le circuit breaker, continuer normalement

        # ✅ ÉTAPE 3: Calculer tous les ETAs en parallèle avec timeout global
        eta_results = {}  # {assignment_id: eta_seconds}
        assignments_needing_eta = [
            (i, data_item)
            for i, data_item in enumerate(assignment_data)
            if data_item["driver_pos"] and data_item["pickup_pos"]
        ]

        if assignments_needing_eta:
            start_time = time.time()
            # ✅ Timeout global réduit : 3s si Haversine (rapide),
            # 5s si OSRM (peut être lent)
            GLOBAL_TIMEOUT_SECONDS = 3 if use_haversine_only else 5

            def _calculate_with_index(index_data):
                _, data_item = index_data  # Ignore l'index, on n'en a pas besoin
                eta_sec = _calculate_eta_for_assignment(
                    data_item["driver_pos"],
                    data_item["pickup_pos"],
                    use_haversine_only=use_haversine_only,
                )
                return (data_item["assignment"].id, eta_sec)

            # ✅ Utiliser ThreadPoolExecutor avec max_workers réduit
            # pour éviter la surcharge
            max_workers = min(
                5, len(assignments_needing_eta)
            )  # Max 5 workers en parallèle (réduit de 10 à 5)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                try:
                    futures = {
                        executor.submit(_calculate_with_index, item): item[1][
                            "assignment"
                        ].id
                        for item in assignments_needing_eta
                    }
                    # ✅ Utiliser as_completed avec timeout global
                    # pour éviter d'attendre trop longtemps
                    completed = 0
                    for future in as_completed(futures, timeout=GLOBAL_TIMEOUT_SECONDS):
                        try:
                            assignment_id, eta_sec = future.result()
                            if eta_sec is not None:
                                eta_results[assignment_id] = eta_sec
                            completed += 1
                            # Si on dépasse le timeout global, arrêter d'attendre
                            if time.time() - start_time >= GLOBAL_TIMEOUT_SECONDS:
                                logger.warning(
                                    (
                                        "[LiveDelays] Global timeout (%ds) reached, "
                                        "stopping ETA calculations (%d/%d completed)"
                                    ),
                                    GLOBAL_TIMEOUT_SECONDS,
                                    completed,
                                    len(futures),
                                )
                                break
                        except Exception as e:
                            assignment_id_key = futures.get(future, "unknown")
                            logger.warning(
                                (
                                    "[LiveDelays] ETA calculation failed "
                                    "for assignment %s: %s"
                                ),
                                assignment_id_key,
                                e,
                            )
                except FutureTimeoutError:
                    logger.warning(
                        (
                            "[LiveDelays] Global timeout (%ds) reached "
                            "for ETA calculations, using partial results "
                            "(%d/%d completed)"
                        ),
                        GLOBAL_TIMEOUT_SECONDS,
                        len(eta_results),
                        len(assignments_needing_eta),
                    )
                except Exception as e:
                    logger.warning(
                        "[LiveDelays] Error in parallel ETA calculation: %s", e
                    )

            elapsed_time = time.time() - start_time
            logger.info(
                (
                    "[LiveDelays] Calculated %d ETAs in parallel in %.2fs "
                    "(timeout: %ds, total: %d)"
                ),
                len(eta_results),
                elapsed_time,
                GLOBAL_TIMEOUT_SECONDS,
                len(assignments_needing_eta),
            )

        # ✅ ÉTAPE 4: Construire les delays avec les ETAs calculés
        delays = []
        for data_item in assignment_data:
            # ✅ Vérifier le timeout global pour éviter les timeouts frontend
            if time.time() - endpoint_start_time >= ENDPOINT_TIMEOUT_SECONDS:
                logger.warning(
                    (
                        "[LiveDelays] Endpoint timeout (%ds) reached, "
                        "returning partial results (%d/%d delays)"
                    ),
                    ENDPOINT_TIMEOUT_SECONDS,
                    len(delays),
                    len(assignment_data),
                )
                break
            a = data_item["assignment"]
            b = data_item["booking"]
            driver = data_item["driver"]
            driver_pos = data_item["driver_pos"]
            pickup_pos = data_item["pickup_pos"]

            # Temps prévus
            pickup_time = getattr(b, "pickup_time", None) or getattr(
                b, "scheduled_time", None
            )
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

            # ✅ ÉTAPE 4.1: Prendre en compte les courses précédentes du chauffeur
            # Si le chauffeur a une course précédente en cours ou récente,
            # calculer l'ETA depuis la destination de cette course plutôt que depuis la position GPS actuelle
            effective_driver_pos = driver_pos  # Position à utiliser pour le calcul ETA

            if driver and driver.id and pickup_time:
                try:
                    prev_b, prev_a = _get_driver_previous_booking(
                        driver.id, b, company.id, d0, d1
                    )
                    if prev_b and prev_a:
                        # Vérifier si la course précédente est en cours ou récente
                        prev_status = getattr(prev_b, "status", None)
                        prev_dropoff_time = getattr(
                            prev_b, "dropoff_time", None
                        ) or getattr(prev_b, "scheduled_time", None)

                        if prev_status in [
                            BookingStatus.ASSIGNED,
                            BookingStatus.EN_ROUTE,
                            BookingStatus.IN_PROGRESS,
                        ] or (
                            prev_dropoff_time
                            and pickup_time
                            and (pickup_time - prev_dropoff_time).total_seconds()
                            < PREVIOUS_BOOKING_RELEVANCE_WINDOW_SECONDS
                        ):
                            # ✅ Utiliser la destination de la course précédente comme point de départ
                            prev_dropoff_lat = getattr(prev_b, "dropoff_lat", None)
                            prev_dropoff_lon = getattr(prev_b, "dropoff_lon", None)

                            if prev_dropoff_lat and prev_dropoff_lon:
                                effective_driver_pos = (
                                    prev_dropoff_lat,
                                    prev_dropoff_lon,
                                )
                                logger.debug(
                                    (
                                        "[LiveDelays] Using previous booking dropoff "
                                        "as starting point for booking %d "
                                        "(previous: %d, status: %s)"
                                    ),
                                    b.id,
                                    prev_b.id,
                                    prev_status,
                                )
                except Exception as e:
                    logger.warning(
                        "[LiveDelays] Error processing previous booking for assignment %s: %s",
                        a.id,
                        e,
                    )

            # ✅ Utiliser l'ETA calculé en parallèle (basé sur position GPS actuelle)
            # OU recalculer avec la position effective (course précédente si applicable)
            current_eta = None
            if a.id in eta_results and pickup_time:
                # ✅ ETA calculé en parallèle disponible (basé sur position GPS actuelle)
                eta_seconds = eta_results[a.id]

                # ✅ Si on a une course précédente pertinente, recalculer depuis sa destination
                if effective_driver_pos != driver_pos and pickup_pos:
                    try:
                        recalc_eta_seconds = _calculate_eta_for_assignment(
                            effective_driver_pos, pickup_pos
                        )
                        if recalc_eta_seconds:
                            current_eta = now_local() + timedelta(
                                seconds=recalc_eta_seconds
                            )
                            logger.debug(
                                (
                                    "[LiveDelays] Recalculated ETA from previous booking "
                                    "dropoff for assignment %d: %d min"
                                ),
                                a.id,
                                recalc_eta_seconds // 60,
                            )
                        else:
                            # Fallback sur ETA parallèle si recalcul échoue
                            current_eta = now_local() + timedelta(seconds=eta_seconds)
                    except Exception as e:
                        logger.warning(
                            "[LiveDelays] Failed to recalculate ETA from previous booking for assignment %s: %s",
                            a.id,
                            e,
                        )
                        # Fallback sur ETA parallèle
                        current_eta = now_local() + timedelta(seconds=eta_seconds)
                else:
                    # Pas de course précédente pertinente, utiliser ETA parallèle
                    current_eta = now_local() + timedelta(seconds=eta_seconds)
            elif effective_driver_pos and pickup_pos and pickup_time:
                # Fallback: calculer maintenant avec position effective
                try:
                    eta_seconds = _calculate_eta_for_assignment(
                        effective_driver_pos, pickup_pos
                    )
                    if eta_seconds:
                        current_eta = now_local() + timedelta(seconds=eta_seconds)
                except Exception as e:
                    logger.warning(
                        "[LiveDelays] Failed to calculate ETA for assignment %s: %s",
                        a.id,
                        e,
                    )

            # Utiliser ETA planifié en fallback
            if not current_eta:
                current_eta = (
                    getattr(a, "pickup_eta", None)
                    or getattr(a, "eta_pickup_at", None)
                    or getattr(a, "estimated_pickup_arrival", None)
                )
                current_eta = _to_dt(current_eta)

            # ✅ LOGIQUE INTELLIGENTE DE DÉTECTION DE RETARD
            # Basée sur l'ETA GPS vs temps restant et le statut de la course
            delay_minutes = 0
            status = "unknown"
            booking_status = getattr(b, "status", None)

            if pickup_time and current_eta:
                try:
                    current_time = now_local()
                    time_remaining_until_pickup = (
                        pickup_time - current_time
                    ).total_seconds() / 60.0  # en minutes

                    # Calculer l'ETA en minutes depuis maintenant
                    eta_from_now_seconds = (current_eta - current_time).total_seconds()
                    eta_from_now_minutes = eta_from_now_seconds / 60.0

                    if pickup_time > current_time:
                        # ✅ Course dans le futur : logique intelligente
                        # Exemple : Chauffeur à 27 min du pickup,
                        # il reste 40 min → PAS de retard (il a le temps)
                        # Exemple : Chauffeur à 27 min du pickup,
                        # il reste 25 min → Retard de 2 min si EN_ROUTE,
                        # problème si ASSIGNED

                        if eta_from_now_minutes <= time_remaining_until_pickup:
                            # ✅ Le chauffeur arrivera à temps
                            # (ETA GPS <= temps restant)
                            status = "on_time"
                            delay_minutes = 0
                        else:
                            # ⚠️ Le chauffeur sera en retard
                            # (ETA GPS > temps restant)
                            potential_delay_minutes = int(
                                eta_from_now_minutes - time_remaining_until_pickup
                            )

                            # Vérifier le statut de la course
                            # ✅ IN_PROGRESS exclu : client déjà à bord, pas de retard possible
                            is_en_route = booking_status == BookingStatus.EN_ROUTE

                            if is_en_route:
                                # ✅ Chauffeur en mouvement :
                                # le retard se propagera à l'arrivée
                                # Exemple : Part avec 2 min de retard
                                # → 2 min de retard à l'arrivée
                                delay_minutes = potential_delay_minutes
                                # ✅ Utiliser la classification à 3 niveaux
                                severity = _classify_delay_severity(delay_minutes)
                                status = severity if severity != "early" else "on_time"
                                logger.debug(
                                    (
                                        "[LiveDelays] Driver EN_ROUTE "
                                        "but will be %d min late "
                                        "(ETA: %.1f min, remaining: %.1f min)"
                                    ),
                                    delay_minutes,
                                    eta_from_now_minutes,
                                    time_remaining_until_pickup,
                                )
                            else:
                                # ❌ Chauffeur pas encore en route :
                                # signaler un problème
                                # Il devrait être en route mais ne l'est pas
                                delay_minutes = potential_delay_minutes
                                # ✅ Utiliser la classification à 3 niveaux
                                severity = _classify_delay_severity(delay_minutes)
                                status = severity if severity != "early" else "on_time"
                                logger.warning(
                                    (
                                        "[LiveDelays] Driver should be EN_ROUTE "
                                        "but status is %s. Will be %d min late "
                                        "(ETA: %.1f min, remaining: %.1f min)"
                                    ),
                                    booking_status,
                                    delay_minutes,
                                    eta_from_now_minutes,
                                    time_remaining_until_pickup,
                                )
                    else:
                        # ✅ Course passée ou en cours : calculer le retard normalement
                        delay_seconds = (current_eta - pickup_time).total_seconds()
                        delay_minutes = int(delay_seconds / 60)

                        # ✅ Utiliser la classification à 3 niveaux
                        status = _classify_delay_severity(delay_minutes)
                except Exception as e:
                    logger.warning(
                        "[LiveDelays] Error calculating intelligent delay: %s", e
                    )
            elif pickup_time and not current_eta:
                # ⭐ FALLBACK : Si pas d'ETA disponible,
                # comparer heure actuelle vs heure prévue
                # Utile pour détecter les retards même sans GPS
                try:
                    current_time = now_local()
                    time_diff_seconds = (current_time - pickup_time).total_seconds()

                    # Si l'heure actuelle est déjà passée
                    # et le chauffeur n'est pas arrivé
                    if (
                        time_diff_seconds > TIME_DIFF_SECONDS_THRESHOLD
                    ):  # 5 minutes de buffer
                        delay_minutes = int(time_diff_seconds / 60)
                        # ✅ Utiliser la classification à 3 niveaux
                        status = _classify_delay_severity(delay_minutes)
                    elif time_diff_seconds < -TIME_DIFF_SECONDS_THRESHOLD:
                        delay_minutes = int(time_diff_seconds / 60)
                        status = "early"
                    else:
                        status = "on_time"
                except Exception as e:
                    logger.warning(
                        "[LiveDelays] Failed to calculate time-based delay: %s", e
                    )

            # ✅ OPTIMISATION CRITIQUE: Désactiver suggestions et cascade
            # pour améliorer les performances
            # Ces fonctionnalités sont coûteuses et peuvent être calculées
            # de manière asynchrone si nécessaire
            suggestions_list = []
            cascade_impact = []

            # ✅ DÉSACTIVÉ TEMPORAIREMENT pour améliorer les performances
            # de l'endpoint /delays/live
            # Les suggestions et l'impact cascade peuvent être calculés
            # de manière asynchrone si nécessaire
            # if delay_minutes > DELAY_MINUTES_THRESHOLD:
            # Seulement pour retards > 15 min
            #     try:
            #         company_id_int = int(cast("Any", company.id))
            #         suggestions = generate_suggestions(
            #             a, delay_minutes, company_id_int
            #         )
            #         suggestions_list = [s.to_dict() for s in suggestions]
            #     except Exception as e:
            #         logger.warning(
            #             "[LiveDelays] Failed to generate suggestions "
            #             "for assignment %s: %s", a.id, e
            #         )

            # ✅ DÉSACTIVÉ TEMPORAIREMENT pour améliorer les performances
            # ✅ DÉSACTIVÉ TEMPORAIREMENT pour améliorer les performances
            # if driver and delay_minutes > DELAY_MINUTES_THRESHOLD
            # and pickup_time is not None:
            #     try:
            #         # Trouver les prochaines courses du chauffeur
            #         next_assignments = (
            #             Assignment.query
            #             .join(Booking, Booking.id == Assignment.booking_id)
            #             .filter(
            #                 Assignment.driver_id == driver.id,
            #                 Assignment.id != a.id,
            #                 Booking.scheduled_time > pickup_time,
            #                 Booking.scheduled_time < pickup_time + timedelta(hours=4)
            #             )
            #             .order_by(Booking.scheduled_time.asc())
            #             .limit(3)
            #             .all()
            #         )
            #
            #         for next_a in next_assignments:
            #             next_b = Booking.query.get(next_a.booking_id)
            #             if next_b:
            #                 cascade_impact.append({
            #                     "booking_id": next_b.id,
            #                     "scheduled_time": (
            #                         next_b.scheduled_time.isoformat()
            #                         if next_b.scheduled_time else None
            #                     ),
            #                     "customer_name": getattr(
            #                         next_b, "customer_name", None
            #                     ),
            #                     "potential_delay_minutes": delay_minutes
            #                     # Propagation simplifiée
            #                 })
            #     except Exception as e:
            #         logger.warning(
            #             "[LiveDelays] Failed to check cascade impact: %s", e
            #         )

            # Construire la réponse
            if current_eta or delay_minutes != DELAY_MINUTES_ZERO:
                # ✅ Classifier la sévérité du retard
                delay_severity = _classify_delay_severity(delay_minutes)

                delay = {
                    "id": a.id,
                    "booking_id": a.booking_id,
                    "driver_id": a.driver_id,
                    "assignment_id": a.id,
                    "delay_minutes": delay_minutes,
                    "status": status,
                    "delay_severity": delay_severity,  # ✅ Niveau de sévérité (reasonable/moderate/critical)
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
                        "name": f"{driver.user.first_name} {driver.user.last_name}"
                        if driver and driver.user
                        else None,
                        "current_position": {
                            "lat": driver_pos[0] if driver_pos else None,
                            "lon": driver_pos[1] if driver_pos else None,
                        }
                        if driver_pos
                        else None,
                    }
                    if driver
                    else None,
                }
                delays.append(delay)

        # Statistiques globales
        total = len(delays)
        # ✅ Compter les retards selon la nouvelle classification
        # "late" était l'ancien statut, maintenant on a reasonable/moderate/critical
        late = len(
            [
                d
                for d in delays
                if d.get("status") in ["late", "reasonable", "moderate", "critical"]
                or d.get("delay_severity") in ["reasonable", "moderate", "critical"]
            ]
        )
        early = len([d for d in delays if d["status"] == "early"])
        on_time = len([d for d in delays if d["status"] == "on_time"])

        # ✅ Statistiques détaillées par sévérité
        reasonable_count = len(
            [d for d in delays if d.get("delay_severity") == "reasonable"]
        )
        moderate_count = len(
            [d for d in delays if d.get("delay_severity") == "moderate"]
        )
        critical_count = len(
            [d for d in delays if d.get("delay_severity") == "critical"]
        )

        avg_delay = 0
        if delays:
            delay_values = [d["delay_minutes"] for d in delays]
            avg_delay = sum(delay_values) / len(delay_values) if delay_values else 0

        return {
            "delays": delays,
            "summary": {
                "total": total,
                "late": late,  # Tous les retards (reasonable + moderate + critical)
                "early": early,
                "on_time": on_time,
                "average_delay": round(avg_delay, 2),
                # ✅ Détails par sévérité
                "reasonable": reasonable_count,  # 1-5 min
                "moderate": moderate_count,  # 5-10 min
                "critical": critical_count,  # >10 min
            },
            "timestamp": now_local().isoformat(),
        }, 200


@dispatch_ns.route("/optimizer/start")
class OptimizerStartResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Démarre le monitoring en temps réel pour l'entreprise.
        Surveille automatiquement les retards et propose des optimisations.
        """
        company_id = _current_company_id()

        body = request.get_json(silent=True) or {}
        check_interval = int(
            body.get("check_interval_seconds", 120)
        )  # 2 min par défaut

        try:
            optimizer = start_optimizer_for_company(company_id, check_interval)
            status = optimizer.get_status()

            return {"message": "Monitoring temps réel démarré", "status": status}, 200

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/optimizer/stop")
class OptimizerStopResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Arrête le monitoring en temps réel pour l'entreprise.

        ⚠️ En mode fully_auto, l'optimiseur ne peut pas être arrêté manuellement.
        Changez le mode de dispatch pour arrêter l'optimiseur.
        """
        company_id = _current_company_id()
        company = _get_current_company()

        try:
            # ✅ Empêcher l'arrêt si l'entreprise est en mode fully_auto
            current_mode = (
                getattr(company.dispatch_mode, "value", None)
                if hasattr(company, "dispatch_mode")
                else None
            )

            if current_mode == "fully_auto":
                logger.warning(
                    (
                        "[Optimizer] Tentative d'arrêt refusée pour company %s "
                        "(mode fully_auto actif)"
                    ),
                    company_id,
                )
                return {
                    "success": False,
                    "error": (
                        "Impossible d'arrêter l'optimiseur en mode fully_auto. "
                        "Changez le mode de dispatch pour arrêter l'optimiseur."
                    ),
                    "current_mode": current_mode,
                }, 403  # Forbidden

            stop_optimizer_for_company(company_id)

            return {
                "message": "Monitoring temps réel arrêté",
                "company_id": company_id,
            }, 200

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/optimizer/status")
class OptimizerStatusResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère le statut du monitoring temps réel."""
        company_id = _current_company_id()

        try:
            # #region agent log
            try:
                import json

                with Path(r"c:\Users\jasiq\atmr\.cursor\debug.log").open(
                    "a", encoding="utf-8"
                ) as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "C",
                                "location": "dispatch_routes.py:2886",
                                "message": "OptimizerStatusResource.get entry",
                                "data": {"company_id": company_id},
                                "timestamp": int(__import__("time").time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            optimizer = get_optimizer_for_company(company_id)

            if optimizer is None:
                return {
                    "running": False,
                    "company_id": company_id,
                    "message": "Monitoring non démarré",
                }, 200

            status = optimizer.get_status()
            # #region agent log
            try:
                import json

                with Path(r"c:\Users\jasiq\atmr\.cursor\debug.log").open(
                    "a", encoding="utf-8"
                ) as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "C",
                                "location": "dispatch_routes.py:2895",
                                "message": "OptimizerStatusResource.get before return",
                                "data": {"status_type": str(type(status))},
                                "timestamp": int(__import__("time").time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            return status, 200

        except Exception as e:
            # #region agent log
            try:
                import json
                import traceback

                with Path(r"c:\Users\jasiq\atmr\.cursor\debug.log").open(
                    "a", encoding="utf-8"
                ) as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "C",
                                "location": "dispatch_routes.py:2898",
                                "message": "OptimizerStatusResource.get exception",
                                "data": {
                                    "error": str(e),
                                    "error_type": str(type(e).__name__),
                                    "traceback": traceback.format_exc(),
                                },
                                "timestamp": int(__import__("time").time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            return APIErrorHandler.handle_exception(e, logger)


# ===== Routes Agent Dispatch Intelligent =====


@dispatch_ns.route("/agent/start")
class AgentStartResource(Resource):
    """Démarre l'agent dispatch intelligent."""

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Démarre l'agent dispatch pour l'entreprise."""
        company_id = _current_company_id()

        try:
            from services.agent_dispatch.orchestrator import get_agent_for_company

            agent = get_agent_for_company(
                company_id, app=current_app._get_current_object()
            )
            agent.start()

            status = agent.get_status()
            return {
                "success": True,
                "message": "Agent démarré",
                "status": status,
            }, 200

        except Exception as e:
            logger.exception("[Agent] Failed to start for company %s", company_id)
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/agent/stop")
class AgentStopResource(Resource):
    """Arrête l'agent dispatch intelligent."""

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Arrête l'agent dispatch pour l'entreprise.

        ⚠️ En mode fully_auto, l'agent ne peut pas être arrêté manuellement.
        Changez le mode de dispatch pour arrêter l'agent.
        """
        company_id = _current_company_id()
        company = _get_current_company()

        try:
            # ✅ Empêcher l'arrêt si l'entreprise est en mode fully_auto
            current_mode = (
                getattr(company.dispatch_mode, "value", None)
                if hasattr(company, "dispatch_mode")
                else None
            )

            if current_mode == "fully_auto":
                logger.warning(
                    (
                        "[Agent] Tentative d'arrêt refusée pour company %s "
                        "(mode fully_auto actif)"
                    ),
                    company_id,
                )
                return {
                    "success": False,
                    "error": (
                        "Impossible d'arrêter l'agent en mode fully_auto. "
                        "Changez le mode de dispatch pour arrêter l'agent."
                    ),
                    "current_mode": current_mode,
                }, 403  # Forbidden

            from services.agent_dispatch.orchestrator import stop_agent_for_company

            stop_agent_for_company(company_id)

            return {
                "success": True,
                "message": "Agent arrêté",
            }, 200

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/agent/status")
class AgentStatusResource(Resource):
    """Récupère le statut de l'agent dispatch intelligent."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère le statut de l'agent."""
        company_id = _current_company_id()

        try:
            from services.agent_dispatch.orchestrator import get_agent_for_company

            agent = get_agent_for_company(company_id)
            status = agent.get_status()
            return status, 200

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/optimizer/opportunities")
class OptimizerOpportunitiesResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(params={"date": "YYYY-MM-DD (optionnel, défaut: aujourd'hui)"})
    def get(self):
        """Récupère les opportunités d'optimisation détectées.
        Mode manuel: lance une vérification à la demande.
        """
        company_id = _current_company_id()

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
                "critical_count": len(
                    [o for o in opportunities if o.severity == "critical"]
                ),
                "high_count": len([o for o in opportunities if o.severity == "high"]),
                "timestamp": now_local().isoformat(),
            }, 200

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/dashboard/realtime")
class RealtimeDashboardResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(params={"date": "YYYY-MM-DD (optionnel, défaut: aujourd'hui)"})
    def get(self):
        """Dashboard temps réel pour les dispatchers.
        Combine métriques de qualité, retards, opportunités et charge chauffeurs.
        """
        company_id = _current_company_id()

        date_str = request.args.get("date")
        if not date_str:
            date_str = datetime.now(UTC).date().strftime("%Y-%m-%d")

        try:
            # 1. Métriques de qualité du dernier dispatch
            quality_metrics = None
            try:
                from infrastructure.dispatch.dispatch_metrics_adapter import (
                    DispatchMetricsCollector,
                )

                collector = DispatchMetricsCollector(company_id)
                metrics = collector.collect_for_date(date_str)
                quality_metrics = metrics.to_summary()
            except Exception as e:
                logger.warning("[Dashboard] Failed to get quality metrics: %s", e)
                quality_metrics = {
                    "quality_score": 0,
                    "assignment_rate": 0,
                    "on_time_rate": 0,
                    "pooling_rate": 0,
                    "fairness": 0,
                    "avg_delay": 0,
                }

            # 2. Retards en cours (live)
            assigns = []
            current_delays = []
            try:
                d0, d1 = day_local_bounds(date_str)
                # Utiliser le repository pour récupérer les assignments avec eager loading
                from repositories.assignment_repository import AssignmentRepository

                assignment_repo = AssignmentRepository()
                excluded_statuses = [
                    BookingStatus.COMPLETED,
                    BookingStatus.RETURN_COMPLETED,
                    BookingStatus.CANCELED,
                ]

                assigns = assignment_repo.find_models_by_company_with_time_range_and_excluded_statuses_eager_loading(
                    company_id=company_id,
                    start_datetime=d0,
                    end_datetime=d1,
                    excluded_statuses=excluded_statuses,
                )

                for a in assigns:
                    b = a.booking  # ✅ Déjà chargé via joinedload
                    if not b or not b.scheduled_time:
                        continue

                    # Calculer retard simplifié
                    current_time = now_local()
                    if a.eta_pickup_at and b.scheduled_time:
                        delay_minutes = int(
                            (a.eta_pickup_at - b.scheduled_time).total_seconds() / 60
                        )
                    else:
                        # Fallback: comparer heure actuelle vs scheduled_time
                        delay_minutes = int(
                            (current_time - b.scheduled_time).total_seconds() / 60
                        )

                    if abs(delay_minutes) >= DELAY_MINUTES_THRESHOLD:
                        current_delays.append(
                            {
                                "assignment_id": a.id,
                                "booking_id": b.id,
                                "driver_id": a.driver_id,
                                "delay_minutes": delay_minutes,
                                "status": "late"
                                if delay_minutes > DELAY_MINUTES_ZERO
                                else "early",
                                "customer_name": b.customer_name,
                                "scheduled_time": b.scheduled_time.isoformat()
                                if b.scheduled_time
                                else None,
                            }
                        )

                # Trier par retard décroissant
                current_delays.sort(key=lambda x: -abs(x["delay_minutes"]))

            except Exception as e:
                logger.warning("[Dashboard] Failed to get current delays: %s", e)
                current_delays = []

            # 3. Opportunités d'optimisation
            opportunities = []
            try:
                optimizer = get_optimizer_for_company(company_id)
                if optimizer and optimizer.get_status()["running"]:
                    opportunities = [
                        o.to_dict() for o in optimizer.get_current_opportunities()
                    ]
                else:
                    opportunities = [
                        o.to_dict()
                        for o in check_opportunities_manual(company_id, date_str)
                    ]
            except Exception as e:
                logger.warning("[Dashboard] Failed to get opportunities: %s", e)

            # 4. Charge par chauffeur
            driver_load = {}
            try:
                for a in assigns:
                    if bool(a.driver_id):
                        driver_load[a.driver_id] = driver_load.get(a.driver_id, 0) + 1

                # ✅ P1: Regrouper les requêtes pour éviter N+1
                # Enrichir avec infos chauffeur
                driver_load_details = []
                if driver_load:
                    driver_ids_list = list(driver_load.keys())
                    # Utiliser le repository pour récupérer les drivers avec eager loading
                    # Imports locaux pour éviter dépendances circulaires
                    from repositories.driver_repository import DriverRepository

                    driver_repo = DriverRepository()
                    drivers_list = (
                        driver_repo.find_models_by_ids_with_user_eager_loading(
                            driver_ids_list
                        )
                    )
                    drivers_map = {d.id: d for d in drivers_list}
                    for driver_id, count in driver_load.items():
                        driver = drivers_map.get(driver_id)
                        if driver and driver.user:
                            driver_load_details.append(
                                {
                                    "driver_id": driver_id,
                                    "name": (
                                        f"{driver.user.first_name} {driver.user.last_name}"
                                    ),
                                    "bookings_count": count,
                                    "is_emergency": getattr(
                                        driver, "is_emergency", False
                                    ),
                                }
                            )

                # Trier par charge décroissante
                driver_load_details.sort(key=lambda x: -x["bookings_count"])

            except Exception as e:
                logger.warning("[Dashboard] Failed to get driver load: %s", e)
                driver_load_details = []

            # 5. Statistiques rapides
            stats = {
                "total_bookings": len(assigns),
                "delayed_bookings": len(
                    [d for d in current_delays if d["status"] == "late"]
                ),
                "early_bookings": len(
                    [d for d in current_delays if d["status"] == "early"]
                ),
                "on_time_bookings": len(assigns) - len(current_delays),
                "critical_opportunities": len(
                    [o for o in opportunities if o.get("severity") == "critical"]
                ),
                "drivers_active": len(driver_load),
            }

            return (
                {
                    "date": date_str,
                    "timestamp": now_local().isoformat(),
                    "quality_metrics": quality_metrics,
                    "current_delays": current_delays[:20],  # Top 20
                    "opportunities": opportunities[:10],  # Top 10
                    "driver_load": driver_load_details[:15],  # Top 15
                    "stats": stats,
                },
                200,
            )

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


# ===== GESTION DES MODES AUTONOMES =====


@dispatch_ns.route("/mode")
class DispatchModeResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère le mode de dispatch actuel et la configuration autonome.

        Returns:
            - dispatch_mode: Mode actuel (manual, semi_auto, fully_auto)
            - autonomous_config: Configuration détaillée
            - description: Explication des modes

        """
        company = _get_current_company()
        company_id = _current_company_id()

        dispatch_mode_val = (
            company.dispatch_mode.value
            if hasattr(company.dispatch_mode, "value")
            else str(company.dispatch_mode)
        )
        return {
            "company_id": company_id,
            "dispatch_mode": dispatch_mode_val,
            "autonomous_config": company.get_autonomous_config(),
            "modes_available": {
                "manual": {
                    "label": "Manuel",
                    "description": "Assignations 100% manuelles, aucune automatisation",
                    "features": [
                        "Contrôle total sur chaque assignation",
                        "Suggestions affichées uniquement",
                        "Aucun dispatch automatique",
                    ],
                },
                "semi_auto": {
                    "label": "Semi-Automatique",
                    "description": (
                        "Dispatch sur demande ou périodique, validation manuelle"
                    ),
                    "features": [
                        "Dispatch optimisé avec OR-Tools",
                        "Monitoring temps réel",
                        "Suggestions affichées (non appliquées)",
                        "Déclenchement manuel ou périodique",
                    ],
                },
                "fully_auto": {
                    "label": "Totalement Automatique",
                    "description": "Système 100% autonome avec application automatique",
                    "features": [
                        "Dispatch automatique périodique",
                        "Monitoring temps réel actif",
                        "Application automatique des suggestions 'safe'",
                        "Ré-optimisation automatique si problème",
                        "Intervention humaine pour cas critiques uniquement",
                    ],
                },
            },
        }, 200

    @jwt_required()
    @role_required(UserRole.company)
    def put(self):
        """Change le mode de dispatch et/ou met à jour la configuration autonome.
        Body:
        {
            "dispatch_mode": "fully_auto",  // optionnel
            "autonomous_config": { ... }     // optionnel
        }.

        Returns:
            Configuration mise à jour

        """
        company = _get_current_company()
        company_id = _current_company_id()
        body = request.get_json() or {}

        # Changer le mode
        new_mode = body.get("dispatch_mode")
        # Récupérer l'ancien mode de manière sécurisée (SQLAlchemy Column)
        old_mode = (
            getattr(company.dispatch_mode, "value", None)
            if hasattr(company, "dispatch_mode")
            else None
        )
        if new_mode:
            from models.enums import DispatchMode

            try:
                cast("Any", company).dispatch_mode = DispatchMode(new_mode)
                logger.info(
                    "[Dispatch] Company %s changed mode to: %s (from: %s)",
                    company_id,
                    new_mode,
                    old_mode,
                )

                # ✅ Démarrer/arrêter l'agent automatiquement selon le mode
                try:
                    from services.agent_dispatch.orchestrator import (
                        get_agent_for_company,
                        stop_agent_for_company,
                    )

                    if new_mode == "fully_auto":
                        # Démarrer l'agent automatiquement en mode fully_auto
                        agent = get_agent_for_company(
                            company_id, app=current_app._get_current_object()
                        )
                        if not agent.state.running:
                            agent.start()
                            logger.info(
                                (
                                    "[Dispatch] 🤖 Agent démarré automatiquement "
                                    "pour company %s (mode fully_auto)"
                                ),
                                company_id,
                            )
                    elif old_mode == "fully_auto" and new_mode != "fully_auto":
                        # Arrêter l'agent si on sort du mode fully_auto
                        stop_agent_for_company(company_id)
                        logger.info(
                            (
                                "[Dispatch] ⏸️ Agent arrêté automatiquement "
                                "pour company %s (mode changé vers %s)"
                            ),
                            company_id,
                            new_mode,
                        )
                except Exception as agent_err:
                    # Ne pas faire échouer le changement de mode
                    # si l'agent a un problème
                    logger.warning(
                        "[Dispatch] ⚠️ Erreur gestion agent lors changement mode: %s",
                        agent_err,
                    )
            except ValueError:
                return {
                    "error": (
                        f"Mode invalide: {new_mode}. "
                        "Valeurs possibles: manual, semi_auto, fully_auto"
                    )
                }, 400

        # Mettre à jour la config
        new_config = body.get("autonomous_config")
        if new_config:
            # Valider et merger avec config par défaut
            current_config = company.get_autonomous_config()

            # Deep merge de la nouvelle config
            def deep_merge(
                base: Dict[str, Any], override: Dict[str, Any]
            ) -> Dict[str, Any]:
                result = base.copy()
                for key, value in override.items():
                    # ✅ P1: Protéger accès dictionnaires pour éviter KeyError
                    existing_value = result.get(key)
                    if (
                        existing_value is not None
                        and isinstance(existing_value, dict)
                        and isinstance(value, dict)
                    ):
                        result[key] = deep_merge(existing_value, value)
                    else:
                        result[key] = value
                return result

            merged_config = deep_merge(current_config, new_config)
            company.set_autonomous_config(merged_config)

            logger.info(
                "[Dispatch] Company %s updated autonomous config: %s",
                company_id,
                list(new_config.keys()),
            )

        try:
            db.session.add(company)
            db.session.commit()

            dispatch_mode_val = (
                company.dispatch_mode.value
                if hasattr(company.dispatch_mode, "value")
                else str(company.dispatch_mode)
            )
            return {
                "company_id": company_id,
                "dispatch_mode": dispatch_mode_val,
                "autonomous_config": company.get_autonomous_config(),
                "message": "Configuration mise à jour avec succès",
            }, 200

        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/autonomous/status")
class AutonomousStatusResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère le statut du système autonome pour l'entreprise.

        Returns:
            - Mode actuel
            - État des automatisations (autorun, realtime optimizer)
            - Configuration active
            - Statistiques récentes

        """
        company_id = _current_company_id()

        from infrastructure.dispatch.autonomous_manager_adapter import (
            get_manager_for_company,
        )

        try:
            manager = get_manager_for_company(company_id)

            # Vérifier si le RealtimeOptimizer tourne actuellement
            from infrastructure.dispatch.realtime_optimizer_adapter import (
                get_optimizer_for_company,
            )

            optimizer = get_optimizer_for_company(company_id)
            optimizer_running = (
                optimizer.get_status() if optimizer else {"running": False}
            )

            dispatch_mode_val = (
                manager.mode.value
                if hasattr(manager.mode, "value")
                else str(manager.mode)
            )
            return {
                "company_id": company_id,
                "dispatch_mode": dispatch_mode_val,
                "autorun_enabled": manager.should_run_autorun(),
                "realtime_optimizer_enabled": manager.should_run_realtime_optimizer(),
                "config": manager.config,
                "celery_status": {
                    "autorun_tick": "running via Celery Beat (every 5 min)",
                    "realtime_monitoring": "running via Celery Beat (every 2 min)",
                },
                "optimizer_thread_status": optimizer_running,
                "features_active": {
                    "auto_dispatch": manager.should_run_autorun(),
                    "realtime_monitoring": manager.should_run_realtime_optimizer(),
                    "auto_apply_suggestions": manager.mode == "fully_auto",
                    "auto_reoptimization": manager.mode == "fully_auto",
                },
            }, 200

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/autonomous/test")
class AutonomousTestResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Teste le système autonome en mode dry-run (simulation).
        Permet de voir ce que le système ferait sans réellement appliquer les actions.
        Body:
        {
            "date": "2025-0.1-17"  // optionnel, défaut: aujourd'hui
        }.

        Returns:
            Simulation des actions qui seraient effectuées

        """
        company_id = _current_company_id()
        body = request.get_json() or {}

        date_str = body.get("date")
        if not date_str:
            date_str = datetime.now(UTC).date().strftime("%Y-%m-%d")

        try:
            # Récupérer les opportunités actuelles
            from infrastructure.dispatch.realtime_optimizer_adapter import (
                check_opportunities_manual,
            )

            opportunities = check_opportunities_manual(
                company_id=company_id, for_date=date_str, app=None
            )

            # Construire le résultat détaillé
            simulated_actions = []
            for opp in opportunities:
                for suggestion in opp.suggestions:
                    simulated_actions.append(
                        {
                            "action": suggestion.action,
                            "message": suggestion.message,
                            "priority": suggestion.priority,
                            "booking_id": suggestion.booking_id,
                            "driver_id": suggestion.driver_id,
                            "would_auto_apply": False,
                            "reason": "requires manual approval",
                        }
                    )

            return {
                "company_id": company_id,
                "dispatch_mode": "manual",
                "date": date_str,
                "test_results": {
                    "opportunities_found": len(opportunities),
                    "would_auto_apply": 0,
                    "would_require_manual": len(simulated_actions),
                    "blocked_by_limits": 0,
                },
                "simulated_actions": simulated_actions,
                "recommendation": (
                    "ℹ️ Aucune action automatique détectée (normal si pas de retard)"
                ),
            }, 200

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


# ===== RL DISPATCH (PRODUCTION) =====


@dispatch_ns.route("/rl/status")
class RLDispatchStatus(Resource):
    """Statut de l'agent RL en production."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère le statut de l'agent RL.

        Returns:
            - available: Agent RL disponible
            - loaded: Modèle chargé
            - statistics: Statistiques d'utilisation

        """
        if not RL_AVAILABLE:
            return {
                "available": False,
                "message": "Module RL non disponible (dépendances manquantes)",
            }, 200

        try:
            # Initialiser manager RL
            if RLDispatchManager is None:
                return {
                    "available": False,
                    "message": "RLDispatchManager non disponible",
                }, 200

            rl_manager = RLDispatchManager()

            stats = rl_manager.get_statistics()

            return {
                "available": True,
                "loaded": stats["is_loaded"],
                "model_path": stats["model_path"],
                "statistics": {
                    "suggestions_total": stats["suggestions_count"],
                    "errors": stats["errors_count"],
                    "fallbacks": stats["fallback_count"],
                    "success_rate": f"{stats['success_rate'] * 100:.1f}%",
                    "fallback_rate": f"{stats['fallback_rate'] * 100:.1f}%",
                },
            }, 200

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/rl/suggestions")
class RLDispatchSuggestions(Resource):
    """Obtenir toutes les suggestions RL pour une date."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Obtient toutes les suggestions RL pour une date donnée.

        Query params:
            for_date: Date au format YYYY-MM-DD
            min_confidence: Confiance minimale (0-1, défaut: 0)
            limit: Nombre max de suggestions (défaut: 20)

        Returns:
            Liste de suggestions triées par confiance décroissante

        ⚠️ NON-DÉTERMINISME PAR DESIGN :
        Les suggestions RL sont intentionnellement non déterministes :
        - L'agent DQN utilise epsilon-greedy (exploration vs exploitation)
        - L'échantillonnage prioritaire du replay buffer introduit de la variabilité
        - L'ordre de traitement des assignments peut varier
        - Les Q-values peuvent avoir des égalités résolues de manière non déterministe

        Ce comportement est voulu pour :
        - Explorer différentes stratégies d'optimisation
        - Éviter la stagnation dans des solutions sous-optimales
        - Améliorer la robustesse du système

        Note : Les suggestions sont mises en cache (TTL 30s) pour réduire la variabilité
        entre appels successifs, mais deux appels avec cache vidé peuvent produire
        des résultats différents.

        """
        # Variables pour stocker le résultat
        result = None
        status_code = 200

        if not RL_AVAILABLE:
            result = {"suggestions": [], "message": "Module RL non disponible"}
        else:
            try:
                company = _get_current_company()
                for_date_str = request.args.get("for_date")
                min_confidence = float(request.args.get("min_confidence", 0))
                limit = int(request.args.get("limit", 20))

                if not for_date_str:
                    result = {"error": "for_date requis (YYYY-MM-DD)"}
                    status_code = 400
                else:
                    # ✅ Valider le format YYYY-MM-DD
                    # _validate_date_format fait abort(400) si format invalide
                    for_date_str = _validate_date_format(for_date_str)

                    # ✅ CACHE REDIS : Clé unique par company/date/params
                    cache_key = (
                        f"rl_suggestions:{company.id}:{for_date_str}:"
                        f"{min_confidence}:{limit}"
                    )

                    # Check cache
                    from ext import redis_client

                    if redis_client:
                        try:
                            cached_bytes = redis_client.get(cache_key)
                            if cached_bytes:
                                logger.info("[RL] Cache hit for %s", cache_key)
                                # Décoder bytes → str avant json.loads
                                cached_str = cached_bytes.decode("utf-8")
                                suggestions_data = json.loads(cached_str)
                                result = {
                                    "suggestions": suggestions_data,
                                    "total": len(suggestions_data),
                                    "date": for_date_str,
                                    "cached": True,
                                }
                        except Exception as e:
                            logger.warning("[RL] Cache read error: %s", e)

                    if result is None:  # Pas de cache hit
                        # Parse date (DTZ007: OK car on compare juste la date,
                        # pas de timezone nécessaire)
                        for_date = datetime.strptime(for_date_str, "%Y-%m-%d").date()

                        # Récupérer tous les assignments actifs pour cette date
                        # Utiliser le repository pour récupérer les assignments
                        # Imports locaux pour éviter dépendances circulaires
                        from repositories.assignment_repository import (
                            AssignmentRepository,
                        )

                        from models.enums import AssignmentStatus

                        assignment_repo = AssignmentRepository()
                        assignment_statuses = [
                            AssignmentStatus.SCHEDULED,
                            AssignmentStatus.EN_ROUTE_PICKUP,
                            AssignmentStatus.ARRIVED_PICKUP,
                            AssignmentStatus.ONBOARD,
                            AssignmentStatus.EN_ROUTE_DROPOFF,
                        ]
                        assignments = assignment_repo.find_models_by_company_and_date_with_status_eager_loading(
                            company.id, for_date, assignment_statuses
                        )

                        if not assignments:
                            result = {
                                "suggestions": [],
                                "message": "Aucun assignment actif pour cette date",
                            }
                        else:
                            # Récupérer tous les conducteurs disponibles
                            # avec leur relation user
                            # Utiliser le repository pour récupérer les drivers
                            # Imports locaux pour éviter dépendances circulaires
                            from repositories.driver_repository import DriverRepository

                            driver_repo = DriverRepository()
                            drivers = driver_repo.find_models_by_company_available_with_user_eager_loading_limited(
                                company.id, 10
                            )

                            if not drivers:
                                result = {
                                    "suggestions": [],
                                    "message": "Aucun conducteur disponible",
                                }
                            else:
                                # Utiliser le générateur RL pour créer des suggestions
                                from services.ml.rl.suggestion_generator import (
                                    get_suggestion_generator,
                                )

                                generator = get_suggestion_generator()
                                all_suggestions = generator.generate_suggestions(
                                    company_id=int(company.id),
                                    assignments=assignments,
                                    drivers=drivers,
                                    for_date=for_date_str,
                                    min_confidence=min_confidence,
                                    max_suggestions=limit,
                                )

                                # ✅ MÉTRIQUES : Logger les suggestions générées
                                try:
                                    from datetime import datetime as dt

                                    from models import RLSuggestionMetric

                                    for suggestion in all_suggestions:
                                        # Créer ID unique pour la suggestion
                                        suggestion_id = (
                                            f"{suggestion['assignment_id']}_"
                                            f"{int(dt.now(UTC).timestamp() * 1000)}"
                                        )

                                        metric = RLSuggestionMetric()
                                        metric.company_id = int(company.id)
                                        metric.suggestion_id = suggestion_id
                                        metric.booking_id = suggestion["booking_id"]
                                        metric.assignment_id = suggestion[
                                            "assignment_id"
                                        ]
                                        metric.current_driver_id = suggestion[
                                            "current_driver_id"
                                        ]
                                        metric.suggested_driver_id = suggestion[
                                            "suggested_driver_id"
                                        ]
                                        metric.confidence = suggestion["confidence"]
                                        metric.expected_gain_minutes = suggestion.get(
                                            "expected_gain_minutes", 0
                                        )
                                        metric.q_value = suggestion.get("q_value")
                                        metric.source = suggestion["source"]
                                        metric.generated_at = dt.now(UTC)
                                        metric.additional_data = {
                                            "message": suggestion.get("message"),
                                            "for_date": for_date_str,
                                            "min_confidence": min_confidence,
                                        }
                                        db.session.add(metric)

                                        # Ajouter l'ID à la suggestion
                                        # pour tracking frontend
                                        suggestion["metric_id"] = suggestion_id

                                    db.session.commit()
                                    logger.info(
                                        "[RL] Logged %s suggestion metrics",
                                        len(all_suggestions),
                                    )
                                except Exception as e:
                                    db.session.rollback()
                                    logger.warning(
                                        "[RL] Failed to log metrics (non-critique): %s",
                                        e,
                                    )

                                # ✅ CACHE REDIS : Stocker en cache (TTL 30s)
                                from ext import redis_client

                                if redis_client and all_suggestions:
                                    try:
                                        redis_client.setex(
                                            cache_key,
                                            30,  # TTL 30 secondes
                                            # (sync avec auto-refresh frontend)
                                            json.dumps(all_suggestions),
                                        )
                                        logger.info(
                                            "[RL] Cached %s suggestions for %s",
                                            len(all_suggestions),
                                            cache_key,
                                        )
                                    except Exception as e:
                                        logger.warning("[RL] Cache write error: %s", e)

                                result = {
                                    "suggestions": all_suggestions,
                                    "total": len(all_suggestions),
                                    "date": for_date_str,
                                    "cached": False,
                                }

            except ValueError:
                result = {"error": "Format date invalide (attendu: YYYY-MM-DD)"}
                status_code = 400
            except Exception as e:
                logger.exception("[RL] Failed to get RL suggestions")
                result = {"error": f"Échec récupération suggestions RL: {e}"}
                status_code = 500

        return result, status_code


@dispatch_ns.route("/rl/metrics")
class RLMetricsResource(Resource):
    """Récupérer les métriques de performance des suggestions RL."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère les métriques de performance des suggestions RL.

        Query params:
            days: Nombre de jours d'historique (défaut: 30)

        Returns:
            Statistiques agrégées et détails des suggestions

        """
        try:
            company_id = _current_company_id()
            days = int(request.args.get("days", 30))

            # Calculer date de début
            cutoff = datetime.now(UTC) - timedelta(days=days)

            # Récupérer toutes les métriques pour cette entreprise
            from repositories.rl_suggestion_metric_repository import (
                RLSuggestionMetricRepository,
            )

            rl_metric_repo = RLSuggestionMetricRepository()
            metrics = rl_metric_repo.find_models_by_company_and_cutoff(
                company_id=company_id, cutoff=cutoff
            )

            if not metrics:
                return {
                    "period_days": days,
                    "total_suggestions": 0,
                    "message": "Aucune métrique disponible pour cette période",
                }, 200

            # Calculer statistiques agrégées
            total = len(metrics)
            applied = len([m for m in metrics if m.applied_at])
            rejected = len([m for m in metrics if m.rejected_at])
            pending = total - applied - rejected

            # Confiance moyenne
            avg_confidence = sum(m.confidence for m in metrics) / total if total else 0

            # Précision gain (seulement pour suggestions appliquées)
            applied_metrics = [m for m in metrics if m.actual_gain_minutes is not None]
            if applied_metrics:
                accuracies = [
                    float(acc)
                    for m in applied_metrics
                    if (acc := m.calculate_gain_accuracy()) is not None
                ]
                avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            else:
                avg_accuracy = None

            # Répartition par source
            dqn_count = len([m for m in metrics if m.source == "dqn_model"])
            heuristic_count = len([m for m in metrics if m.source == "basic_heuristic"])
            fallback_rate = heuristic_count / total if total else 0

            # Gain total estimé et réel
            total_expected_gain = sum(m.expected_gain_minutes or 0 for m in metrics)
            total_actual_gain = sum(m.actual_gain_minutes or 0 for m in applied_metrics)

            # Top suggestions (meilleures performances)
            top_suggestions = sorted(
                [m.to_dict() for m in applied_metrics if m.was_successful],
                key=lambda x: x.get("actual_gain", 0),
                reverse=True,
            )[:10]

            # Évolution par jour (derniers 7 jours)
            from collections import defaultdict

            daily_stats: dict[str, dict[str, Any]] = defaultdict(
                lambda: {"generated": 0, "applied": 0, "avg_confidence": []}
            )

            for m in metrics:
                day_key = (
                    m.generated_at.date().isoformat()
                    if bool(m.generated_at)
                    else "unknown"
                )
                daily_stats[day_key]["generated"] += 1
                conf_list = cast("list[float]", daily_stats[day_key]["avg_confidence"])
                conf_list.append(m.confidence)
                if m.applied_at:
                    daily_stats[day_key]["applied"] += 1

            # Formater daily_stats
            confidence_history = []
            for day, stats in sorted(daily_stats.items(), reverse=True)[:7]:
                conf_values = cast("list[float]", stats["avg_confidence"])
                avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0
                confidence_history.append(
                    {
                        "date": day,
                        "generated": stats["generated"],
                        "applied": stats["applied"],
                        "avg_confidence": round(avg_conf, 2),
                    }
                )

            confidence_history.reverse()  # Ordre chronologique

            return {
                "period_days": days,
                "total_suggestions": total,
                "applied_count": applied,
                "rejected_count": rejected,
                "pending_count": pending,
                "application_rate": round(applied / total, 2) if total else 0,
                "rejection_rate": round(rejected / total, 2) if total else 0,
                "avg_confidence": round(avg_confidence, 2),
                "avg_gain_accuracy": round(avg_accuracy, 2)
                if avg_accuracy is not None
                else None,
                "fallback_rate": round(fallback_rate, 2),
                "total_expected_gain_minutes": total_expected_gain,
                "total_actual_gain_minutes": total_actual_gain,
                "by_source": {
                    "dqn_model": dqn_count,
                    "basic_heuristic": heuristic_count,
                },
                "top_suggestions": top_suggestions,
                "confidence_history": confidence_history,
                "timestamp": datetime.now(UTC).isoformat(),
            }, 200

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/rl/feedback")
class RLFeedbackResource(Resource):
    """Enregistrer feedback utilisateur sur suggestion RL."""

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Enregistre le feedback utilisateur sur une suggestion RL.

        Body:
        {
            "suggestion_id": "1231234567890",
            "action": "applied" | "rejected" | "ignored",
            "feedback_reason": "Optionnel: Pourquoi rejeté",
            "actual_outcome": {  # Optionnel, si appliqué
                "gain_minutes": 12,
                "was_better": true,
                "satisfaction": 4
            }
        }

        Returns:
            Feedback enregistré + métriques mises à jour

        """
        try:
            company_id = _current_company_id()
            body = request.get_json() or {}

            # Validation
            suggestion_id = body.get("suggestion_id")
            action = body.get("action")

            if not suggestion_id:
                return APIErrorHandler.handle_validation_error(
                    "suggestion_id requis",
                    field="suggestion_id",
                    logger_instance=logger,
                )

            if action not in ["applied", "rejected", "ignored"]:
                return {
                    "error": "action doit être 'applied', 'rejected' ou 'ignored'"
                }, 400

            # Importer repositories
            from repositories.rl_feedback_repository import RLFeedbackRepository
            from repositories.rl_suggestion_metric_repository import (
                RLSuggestionMetricRepository,
            )

            # Récupérer la métrique de suggestion associée
            rl_metric_repo = RLSuggestionMetricRepository()
            metric = rl_metric_repo.find_by_suggestion_id_and_company(
                suggestion_id=suggestion_id, company_id=company_id
            )

            if not metric:
                return APIErrorHandler.handle_not_found(
                    "Suggestion", suggestion_id, logger
                )

            # Vérifier si feedback déjà existant
            rl_feedback_repo = RLFeedbackRepository()
            existing_feedback = rl_feedback_repo.find_by_suggestion_id_and_company(
                suggestion_id=suggestion_id, company_id=company_id
            )

            if existing_feedback:
                return APIErrorHandler.handle_conflict_error(
                    "Feedback déjà enregistré pour cette suggestion",
                    resource_type="Feedback",
                    resource_id=suggestion_id,
                    logger_instance=logger,
                )

            # Récupérer user_id depuis JWT
            from flask_jwt_extended import get_jwt_identity  # pyright: ignore[reportMissingImports]

            user_id_from_jwt = get_jwt_identity()

            # Créer feedback
            from models import RLFeedback

            feedback = RLFeedback()
            feedback.company_id = company_id
            feedback.suggestion_id = suggestion_id
            feedback.booking_id = metric.booking_id
            feedback.assignment_id = metric.assignment_id
            feedback.current_driver_id = metric.current_driver_id
            feedback.suggested_driver_id = metric.suggested_driver_id
            feedback.action = action
            feedback.feedback_reason = body.get("feedback_reason")
            feedback.user_id = user_id_from_jwt
            feedback.suggestion_generated_at = metric.generated_at
            feedback.suggestion_confidence = metric.confidence
            feedback.additional_data = body.get("additional_data")

            # Si appliqué avec résultat, extraire infos
            if action == "applied" and body.get("actual_outcome"):
                outcome = body["actual_outcome"]
                feedback.was_successful = outcome.get("was_better", True)
                feedback.actual_gain_minutes = outcome.get("gain_minutes", 0)

                # Mettre à jour la métrique aussi
                metric.applied_at = datetime.now(UTC)
                metric.actual_gain_minutes = feedback.actual_gain_minutes
                metric.was_successful = feedback.was_successful
                db.session.add(metric)

            elif action == "rejected":
                # Marquer la métrique comme rejetée
                metric.rejected_at = datetime.now(UTC)
                metric.was_successful = False
                db.session.add(metric)

            # Sauvegarder feedback
            db.session.add(feedback)
            db.session.commit()

            logger.info(
                "[RL] Feedback enregistré: %s action=%s company=%s",
                suggestion_id,
                action,
                company_id,
            )

            # Calculer reward pour le ré-entraînement
            reward = feedback.calculate_reward()

            # Statistiques après ce feedback
            rl_feedback_repo = RLFeedbackRepository()
            total_feedbacks = rl_feedback_repo.count_by_company(company_id=company_id)
            applied_count = rl_feedback_repo.count_by_company_and_action(
                company_id=company_id, action="applied"
            )

            return {
                "message": "Feedback enregistré avec succès",
                "feedback_id": feedback.id,
                "suggestion_id": suggestion_id,
                "action": action,
                "reward": reward,
                "stats": {
                    "total_feedbacks": total_feedbacks,
                    "applied_count": applied_count,
                    "application_rate": applied_count / total_feedbacks
                    if total_feedbacks > TOTAL_FEEDBACKS_ZERO
                    else TOTAL_FEEDBACKS_ZERO,
                },
            }, 201

        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/rl/toggle")
class RLDispatchToggle(Resource):
    """Activer/désactiver dispatch RL."""

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Active ou désactive le dispatch RL pour l'entreprise.

        Body:
        {
            "enabled": true/false
        }

        Returns:
            Configuration mise à jour

        """
        company = _get_current_company()
        body = request.get_json() or {}

        enabled = body.get("enabled")
        if enabled is None:
            return APIErrorHandler.handle_validation_error(
                "enabled requis (true/false)",
                field="enabled",
                logger_instance=logger,
            )

        try:
            # Mettre à jour config
            config = company.get_autonomous_config()

            if "rl_dispatch" not in config:
                config["rl_dispatch"] = {}

            config["rl_dispatch"]["enabled"] = bool(enabled)
            config["rl_dispatch"]["model_path"] = config["rl_dispatch"].get(
                "model_path", "data/rl/models/dqn_best.pth"
            )
            config["rl_dispatch"]["fallback_to_heuristic"] = config["rl_dispatch"].get(
                "fallback_to_heuristic", True
            )

            company.set_autonomous_config(config)
            db.session.add(company)
            db.session.commit()

            logger.info(
                "[RL] Company %s %s RL dispatch",
                company.id,
                "enabled" if enabled else "disabled",
            )

            return {
                "company_id": company.id,
                "rl_dispatch_enabled": enabled,
                "config": config["rl_dispatch"],
                "message": (
                    f"Dispatch RL {'activé' if enabled else 'désactivé'} avec succès"
                ),
            }, 200

        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


# ===== PARAMÈTRES AVANCÉS DE DISPATCH =====


@dispatch_ns.route("/advanced_settings")
class DispatchAdvancedSettingsResource(Resource):
    """Gestion des paramètres avancés de dispatch
    (heuristic, solver, fairness, emergency, etc.)
    Stockés dans company.autonomous_config sous la clé 'dispatch_overrides'.
    """

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère les paramètres avancés de dispatch sauvegardés.

        Returns:
            {
                "dispatch_overrides": { ... } ou null si non configuré
            }

        """
        company = _get_current_company()
        company_id = _current_company_id()

        # Récupérer la config autonome complète
        autonomous_config = company.get_autonomous_config()

        # Extraire les dispatch_overrides
        dispatch_overrides = autonomous_config.get("dispatch_overrides", None)

        logger.info(
            "[Dispatch] Company %s fetched advanced settings: %s",
            company_id,
            "configured" if dispatch_overrides else "not configured",
        )

        return {"company_id": company_id, "dispatch_overrides": dispatch_overrides}, 200

    @jwt_required()
    @role_required(UserRole.company)
    def put(self):
        """Sauvegarde les paramètres avancés de dispatch.
        Body:
        {
            "dispatch_overrides": {
                "heuristic": { "proximity_weight": 0.3, ... },
                "solver": { "time_limit": 60, ... },
                "emergency": { "allow_emergency": false, ... },
                ...
            }
        }.

        Returns:
            Paramètres sauvegardés

        """
        company = _get_current_company()
        company_id = _current_company_id()
        body = request.get_json() or {}

        dispatch_overrides = body.get("dispatch_overrides")

        if dispatch_overrides is None:
            return APIErrorHandler.handle_validation_error(
                "Le champ 'dispatch_overrides' est requis",
                field="dispatch_overrides",
                logger_instance=logger,
            )

        # Valider que c'est un dict
        if not isinstance(dispatch_overrides, dict):
            return APIErrorHandler.handle_validation_error(
                "dispatch_overrides doit être un objet JSON",
                field="dispatch_overrides",
                provided_value=type(dispatch_overrides).__name__,
                logger_instance=logger,
            )

        # Récupérer la config actuelle
        current_config = company.get_autonomous_config()

        # Mettre à jour uniquement la clé dispatch_overrides
        current_config["dispatch_overrides"] = dispatch_overrides

        # Sauvegarder
        company.set_autonomous_config(current_config)

        try:
            db.session.add(company)
            db.session.commit()

            logger.info(
                "[Dispatch] Company %s saved advanced settings: %s",
                company_id,
                list(dispatch_overrides.keys()),
            )

            return {
                "company_id": company_id,
                "dispatch_overrides": dispatch_overrides,
                "message": "Paramètres avancés sauvegardés avec succès",
            }, 200

        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.company)
    def delete(self):
        """Supprime les paramètres avancés (reset aux valeurs par défaut)."""
        company = _get_current_company()
        company_id = _current_company_id()

        # Récupérer la config actuelle
        current_config = company.get_autonomous_config()

        # Supprimer la clé dispatch_overrides
        if "dispatch_overrides" in current_config:
            del current_config["dispatch_overrides"]

        # Sauvegarder
        company.set_autonomous_config(current_config)

        try:
            db.session.add(company)
            db.session.commit()

            logger.info(
                "[Dispatch] Company %s deleted advanced settings (reset to defaults)",
                company_id,
            )

            return {
                "company_id": company_id,
                "message": "Paramètres avancés réinitialisés aux valeurs par défaut",
            }, 200

        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


# ===== FEATURE FLAGS & PERFORMANCE METRICS (Phase 0) =====


@dispatch_ns.route("/features/flags")
class FeatureFlagsResource(Resource):
    """Gestion des feature flags pour le dispatch."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère les feature flags actifs pour l'entreprise.

        Returns:
            Dict des feature flags avec leur état actuel
        """
        company = _get_current_company()

        # Récupérer la config depuis company.autonomous_config
        config = {}
        try:
            settings_raw = getattr(company, "autonomous_config", None)
            if isinstance(settings_raw, str) and settings_raw:
                config = json.loads(settings_raw)
        except (json.JSONDecodeError, TypeError):
            config = {}

        # Extraire les feature flags (avec valeurs par défaut)
        features = config.get("features", {})

        return {
            "company_id": company.id,
            "features": {
                "enable_solver": features.get("enable_solver", True),
                "enable_heuristics": features.get("enable_heuristics", True),
                "enable_events": features.get("enable_events", True),
                "enable_db_bulk_ops": features.get("enable_db_bulk_ops", True),
                "enable_rl": features.get("enable_rl", False),
                "enable_rl_apply": features.get("enable_rl_apply", False),
                "enable_clustering": features.get("enable_clustering", False),
                "enable_parallel_heuristics": features.get(
                    "enable_parallel_heuristics", False
                ),
            },
        }, 200

    @jwt_required()
    @role_required(UserRole.company)
    def put(self):
        """Met à jour les feature flags.

        Body:
        {
            "features": {
                "enable_rl": true,
                "enable_rl_apply": false,
                ...
            }
        }
        """
        company = _get_current_company()
        company_id: int = _current_company_id()

        body = request.get_json() or {}
        new_features = body.get("features")

        if not new_features or not isinstance(new_features, dict):
            return {
                "error": "Le champ 'features' est requis et doit être un objet"
            }, 400

        # Charger la config existante
        config = {}
        try:
            settings_raw = getattr(company, "autonomous_config", None)
            if isinstance(settings_raw, str) and settings_raw:
                config = json.loads(settings_raw)
        except (json.JSONDecodeError, TypeError):
            config = {}

        # Merger les nouveaux features
        if "features" not in config:
            config["features"] = {}
        config["features"].update(new_features)

        # Sauvegarder
        try:
            cast("Any", company).autonomous_config = json.dumps(config)
            db.session.add(company)
            db.session.commit()

            logger.info(
                "[Dispatch] Company %s updated feature flags: %s",
                company_id,
                list(new_features.keys()),
            )

            return {
                "company_id": company_id,
                "features": config["features"],
                "message": "Feature flags mis à jour avec succès",
            }, 200

        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/metrics/performance")
class PerformanceMetricsResource(Resource):
    """Métriques de performance pour le dispatch."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère les métriques de performance pour une date.

        Query params:
            date: Date au format YYYY-MM-DD (optionnel, défaut: aujourd'hui)
            dispatch_run_id: ID du dispatch run (optionnel, prioritaire sur date)

        Returns:
            Métriques de performance détaillées
        """
        company_id = _current_company_id()
        date_str = request.args.get("date")
        dispatch_run_id_str = request.args.get("dispatch_run_id")
        error_response = None
        result_response = None

        # Si dispatch_run_id fourni, l'utiliser en priorité
        if dispatch_run_id_str:
            try:
                dispatch_run_id = int(dispatch_run_id_str)
            except ValueError:
                error_response = ({"error": "dispatch_run_id invalide"}, 400)
            else:
                # Utiliser le repository pour récupérer le dispatch run
                # Imports locaux pour éviter dépendances circulaires
                from repositories.dispatch_run_repository import DispatchRunRepository

                dispatch_run_repo = DispatchRunRepository()
                dispatch_run = dispatch_run_repo.find_model_by_id_and_company(
                    dispatch_run_id, company_id
                )
                if not dispatch_run:
                    error_response = ({"error": "Dispatch run non trouvé"}, 404)
                else:
                    try:
                        meta = dispatch_run.meta or {}
                        perf_metrics = meta.get("performance_metrics")
                        if perf_metrics:
                            result_response = (
                                {
                                    "dispatch_run_id": dispatch_run_id,
                                    "company_id": company_id,
                                    "date": dispatch_run.day.isoformat()
                                    if dispatch_run.day
                                    else None,
                                    "status": dispatch_run.status.value
                                    if hasattr(dispatch_run.status, "value")
                                    else str(dispatch_run.status),
                                    "metrics": perf_metrics,
                                },
                                200,
                            )
                        else:
                            result_response = (
                                {
                                    "dispatch_run_id": dispatch_run_id,
                                    "message": (
                                        "Aucune métrique disponible "
                                        "pour ce dispatch run"
                                    ),
                                },
                                200,
                            )
                    except Exception as e:
                        logger.exception(
                            "[Dispatch] Failed to extract performance metrics"
                        )
                        error_response = (
                            {"error": f"Erreur lors de l'extraction: {e}"},
                            500,
                        )

        # Sinon, utiliser la date
        if not error_response and not result_response and dispatch_run_id_str is None:
            if not date_str:
                date_str = datetime.now(UTC).date().isoformat()
            try:
                date_obj = date.fromisoformat(date_str)
            except ValueError:
                error_response = (
                    {"error": "Format de date invalide (attendu: YYYY-MM-DD)"},
                    400,
                )
            else:
                # Utiliser le repository pour récupérer le dispatch run
                # Imports locaux pour éviter dépendances circulaires
                from repositories.dispatch_run_repository import DispatchRunRepository

                dispatch_run_repo = DispatchRunRepository()
                dispatch_run = dispatch_run_repo.find_model_by_company_and_day_ordered(
                    company_id, date_obj
                )
                if not dispatch_run:
                    result_response = (
                        {
                            "date": date_str,
                            "message": "Aucun dispatch run trouvé pour cette date",
                        },
                        200,
                    )
                else:
                    try:
                        meta = dispatch_run.meta or {}
                        perf_metrics = meta.get("performance_metrics")
                        if perf_metrics:
                            result_response = (
                                {
                                    "dispatch_run_id": dispatch_run.id,
                                    "date": date_str,
                                    "company_id": company_id,
                                    "status": dispatch_run.status.value
                                    if hasattr(dispatch_run.status, "value")
                                    else str(dispatch_run.status),
                                    "metrics": perf_metrics,
                                },
                                200,
                            )
                        else:
                            result_response = (
                                {
                                    "dispatch_run_id": dispatch_run.id,
                                    "date": date_str,
                                    "message": "Aucune métrique disponible",
                                },
                                200,
                            )
                    except Exception as e:
                        error_response = APIErrorHandler.handle_exception(e, logger)

        if error_response:
            return error_response
        if result_response:
            return result_response
        return APIErrorHandler.handle_exception(
            Exception("Erreur inconnue lors de la récupération des métriques"),
            logger,
        )


@dispatch_ns.route("/metrics/prometheus")
class PrometheusMetricsResource(Resource):
    """Export des métriques au format Prometheus."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Exporte les métriques au format Prometheus.

        Query params:
            date: Date au format YYYY-MM-DD (optionnel, défaut: aujourd'hui)
            dispatch_run_id: ID du dispatch run (optionnel, prioritaire sur date)

        Returns:
            Métriques au format Prometheus (text/plain)
        """
        from flask import Response  # pyright: ignore[reportMissingImports]

        company_id = _current_company_id()
        date_str = request.args.get("date")
        dispatch_run_id_str = request.args.get("dispatch_run_id")
        error_response = None
        response = None

        # Si dispatch_run_id fourni, l'utiliser en priorité
        if dispatch_run_id_str:
            try:
                dispatch_run_id = int(dispatch_run_id_str)
            except ValueError:
                error_response = ({"error": "dispatch_run_id invalide"}, 400)
            else:
                # Utiliser le repository pour récupérer le dispatch run
                # Imports locaux pour éviter dépendances circulaires
                from repositories.dispatch_run_repository import DispatchRunRepository

                dispatch_run_repo = DispatchRunRepository()
                dispatch_run = dispatch_run_repo.find_model_by_id_and_company(
                    dispatch_run_id, company_id
                )
                if not dispatch_run:
                    error_response = ({"error": "Dispatch run non trouvé"}, 404)
                else:
                    meta = dispatch_run.meta or {}
                    perf_metrics = meta.get("performance_metrics")
                    if not perf_metrics:
                        response = (
                            Response("# No metrics available", mimetype="text/plain"),
                            200,
                        )
        else:
            # Sinon, utiliser la date
            if not date_str:
                date_str = datetime.now(UTC).date().isoformat()
            try:
                date_obj = date.fromisoformat(date_str)
            except ValueError:
                error_response = (
                    {"error": "Format de date invalide (attendu: YYYY-MM-DD)"},
                    400,
                )
            else:
                # Utiliser le repository pour récupérer le dispatch run
                # Imports locaux pour éviter dépendances circulaires
                from repositories.dispatch_run_repository import DispatchRunRepository

                dispatch_run_repo = DispatchRunRepository()
                dispatch_run = dispatch_run_repo.find_model_by_company_and_day_ordered(
                    company_id, date_obj
                )
                if not dispatch_run:
                    response = (
                        Response(
                            f"# No dispatch run found for date {date_str}",
                            mimetype="text/plain",
                        ),
                        200,
                    )
                else:
                    meta = dispatch_run.meta or {}
                    perf_metrics = meta.get("performance_metrics")
                    if not perf_metrics:
                        response = (
                            Response("# No metrics available", mimetype="text/plain"),
                            200,
                        )

        # Convertir en format Prometheus si métriques disponibles
        local_vars = locals()
        if (
            not error_response
            and not response
            and "perf_metrics" in local_vars
            and local_vars.get("perf_metrics")
            and "dispatch_run" in local_vars
        ):
            try:
                from infrastructure.dispatch.performance_metrics_adapter import (
                    DispatchPerformanceMetrics,
                )

                perf_metrics = local_vars["perf_metrics"]
                dispatch_run = local_vars["dispatch_run"]

                metrics = DispatchPerformanceMetrics(
                    dispatch_run_id=dispatch_run.id,
                    company_id=company_id,
                    timestamp=datetime.now(UTC),
                    data_collection_time=perf_metrics.get("timing", {}).get(
                        "data_collection", 0.0
                    ),
                    heuristics_time=perf_metrics.get("timing", {}).get(
                        "heuristics", 0.0
                    ),
                    solver_time=perf_metrics.get("timing", {}).get("solver", 0.0),
                    persistence_time=perf_metrics.get("timing", {}).get(
                        "persistence", 0.0
                    ),
                    total_time=perf_metrics.get("timing", {}).get("total", 0.0),
                    sql_queries_count=perf_metrics.get("counters", {}).get(
                        "sql_queries", 0
                    ),
                    cache_hits=perf_metrics.get("counters", {}).get("cache_hits", 0),
                    cache_misses=perf_metrics.get("counters", {}).get(
                        "cache_misses", 0
                    ),
                    bookings_processed=perf_metrics.get("counters", {}).get(
                        "bookings_processed", 0
                    ),
                    drivers_available=perf_metrics.get("counters", {}).get(
                        "drivers_available", 0
                    ),
                    quality_score=perf_metrics.get("quality", {}).get(
                        "quality_score", 0.0
                    ),
                    assignment_rate=perf_metrics.get("quality", {}).get(
                        "assignment_rate", 0.0
                    ),
                    algorithm_used=perf_metrics.get("algorithm", "unknown"),
                    feature_flags=perf_metrics.get("feature_flags", {}),
                )
                prometheus_text = metrics.to_prometheus_format()
                response = (Response(prometheus_text, mimetype="text/plain"), 200)
            except Exception as e:
                logger.exception("[Dispatch] Failed to convert to Prometheus format")
                error_response = ({"error": f"Erreur lors de la conversion: {e}"}, 500)

        if error_response:
            return error_response
        if response:
            return response
        return (Response("# No metrics available", mimetype="text/plain"), 200)


@dispatch_ns.route("/metrics/a1-compliance")
class A1ComplianceResource(Resource):
    """Métriques de conformité A1 (prévention conflits temporels)."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère les métriques de conformité A1 sur N jours.

        Query params:
            days: Nombre de jours à analyser (défaut: 7)

        Returns:
            Métriques de conformité A1 avec violation_rate
        """
        company_id = _current_company_id()
        days = request.args.get("days", 7, type=int)

        try:
            # Récupérer les dispatch runs des N derniers jours
            start_date = datetime.now(UTC).date() - timedelta(days=days)
            # Utiliser le repository pour récupérer les dispatch runs
            # Imports locaux pour éviter dépendances circulaires
            from repositories.dispatch_run_repository import DispatchRunRepository

            dispatch_run_repo = DispatchRunRepository()
            runs = dispatch_run_repo.find_models_by_company_and_date_range(
                company_id, start_date
            )

            # Calculer les statistiques
            total_conflicts = sum(
                r.meta.get("performance_metrics", {}).get("temporal_conflicts", 0)
                for r in runs
                if r.meta
            )
            total_bookings = sum(
                r.meta.get("counters", {}).get("bookings_processed", 0)
                for r in runs
                if r.meta
            )

            violation_rate = (
                total_conflicts / total_bookings if total_bookings > 0 else 0
            )
            threshold = 0.001  # 0.1%

            return {
                "temporal_conflicts": total_conflicts,
                "total_bookings": total_bookings,
                "violation_rate": violation_rate,
                "threshold": threshold,
                "compliant": violation_rate < threshold,
                "days": days,
                "runs_analyzed": len(runs),
            }, 200

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/metrics/a1-rejects")
class A1RejectsResource(Resource):
    """Détails des rejets pour conflits temporels."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère les détails des rejets pour conflits temporels.

        Query params:
            days: Nombre de jours à analyser (défaut: 1)
            limit: Nombre max de rejets (défaut: 100)

        Returns:
            Liste des rejets avec conflict_penalty
        """
        company_id = _current_company_id()
        days = request.args.get("days", 1, type=int)
        limit = request.args.get("limit", 100, type=int)

        try:
            # Récupérer les dispatch runs
            from repositories.dispatch_run_repository import DispatchRunRepository

            dispatch_run_repo = DispatchRunRepository()
            start_date = datetime.now(UTC).date() - timedelta(days=days)
            runs = dispatch_run_repo.find_models_by_company_and_date_from(
                company_id=company_id, start_date=start_date, limit=100
            )

            all_rejects = []
            for run in runs:
                if not run.meta:
                    continue

                debug = run.meta.get("debug", {})
                temporal_rejects = debug.get("temporal_conflict_rejects", [])

                for reject in temporal_rejects:
                    reject["dispatch_run_id"] = run.id
                    reject["created_at"] = (
                        run.created_at.isoformat() if run.created_at else None
                    )
                    all_rejects.append(reject)

            # Limiter et retourner
            all_rejects = all_rejects[:limit]

            return {
                "rejects": all_rejects,
                "count": len(all_rejects),
                "days": days,
            }, 200

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/metrics/a1-backout")
class A1BackoutResource(Resource):
    """Vérifie conformité A1 et active backout si nécessaire."""

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Vérifie conformité A1 et active backout si nécessaire.

        Body (JSON):
            days: Nombre de jours à analyser (défaut: 7)

        Returns:
            Décision de backout avec violation_rate
        """
        company_id = _current_company_id()
        days = request.json.get("days", 7) if request.json else 7

        try:
            # Récupérer statistiques
            from repositories.dispatch_run_repository import DispatchRunRepository

            dispatch_run_repo = DispatchRunRepository()
            start_date = datetime.now(UTC).date() - timedelta(days=days)
            runs = dispatch_run_repo.find_models_by_company_and_date_from(
                company_id=company_id, start_date=start_date, limit=None
            )

            total_conflicts = sum(
                r.meta.get("performance_metrics", {}).get("temporal_conflicts", 0)
                for r in runs
                if r.meta
            )
            total_bookings = sum(
                r.meta.get("counters", {}).get("bookings_processed", 0)
                for r in runs
                if r.meta
            )

            violation_rate = (
                total_conflicts / total_bookings if total_bookings > 0 else 0
            )
            threshold = 0.001

            backout_needed = violation_rate >= threshold

            if backout_needed:
                logger.error(
                    (
                        "[A1] ❌ Backout recommandé: violation_rate=%.4f >= "
                        "threshold=%.4f (company_id=%s)"
                    ),
                    violation_rate,
                    threshold,
                    company_id,
                )
                # TODO: Désactiver automatiquement le feature flag via DB

            return {
                "compliant": violation_rate < threshold,
                "violation_rate": violation_rate,
                "threshold": threshold,
                "backout_needed": backout_needed,
                "total_conflicts": total_conflicts,
                "total_bookings": total_bookings,
                "message": "Backout recommandé" if backout_needed else "Conforme",
            }, 200

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/reset")
class ResetAssignmentsResource(Resource):
    """Réinitialise toutes les assignations pour permettre un redémarrage propre."""

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Réinitialise toutes les assignations et remet les courses au statut ACCEPTED.

        Body (JSON, optionnel):
            {
                "date": "2025-11-06"  // optionnel, défaut: toutes les dates
            }

        Returns:
            {
                "message": "Réinitialisation effectuée",
                "assignments_deleted": int,
                "bookings_reset": int
            }
        """
        company_id = _current_company_id()
        body = request.get_json() or {}
        date_str = body.get("date")
        start_datetime = None
        end_datetime = None

        try:
            # Récupérer toutes les assignations de la company
            # Utiliser le repository pour obtenir la query
            # Imports locaux pour éviter dépendances circulaires
            from repositories.assignment_repository import AssignmentRepository
            from sqlalchemy.orm import joinedload

            assignment_repo = AssignmentRepository()
            start_datetime = None
            end_datetime = None

            # Filtrer par date si fournie
            if date_str:
                try:
                    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    start_datetime = datetime.combine(
                        target_date, datetime.min.time()
                    ).replace(tzinfo=UTC)
                    end_datetime = datetime.combine(
                        target_date, datetime.max.time()
                    ).replace(tzinfo=UTC)
                except ValueError:
                    return {
                        "error": "Format de date invalide. Utilisez YYYY-MM-DD"
                    }, 400

            query = assignment_repo.find_models_by_company_with_date_filter_query(
                company_id, start_datetime, end_datetime
            )
            # ✅ P1: Eager loading pour éviter N+1 queries si booking est accédé
            assignments = query.options(joinedload(Assignment.booking)).all()
            booking_ids = [a.booking_id for a in assignments]

            # Supprimer toutes les assignations
            assignments_count = len(assignments)
            for assignment in assignments:
                db.session.delete(assignment)

            # Remettre les bookings au statut ACCEPTED et nettoyer driver_id
            # Utiliser le repository pour obtenir la query
            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            bookings_query = booking_repo.find_models_by_company_with_filters_query(
                company_id, booking_ids, start_datetime, end_datetime
            )

            bookings = bookings_query.all()
            bookings_count = 0
            for booking in bookings:
                # Remettre au statut ACCEPTED si actuellement ASSIGNED
                if booking.status == BookingStatus.ASSIGNED:
                    booking.status = BookingStatus.ACCEPTED
                    booking.driver_id = None
                    bookings_count += 1

            db.session.commit()

            logger.info(
                (
                    "[RESET] ✅ Réinitialisation effectuée pour company_id=%s: "
                    "%d assignations supprimées, %d bookings réinitialisés"
                ),
                company_id,
                assignments_count,
                bookings_count,
            )

            return {
                "message": "Réinitialisation effectuée avec succès",
                "assignments_deleted": assignments_count,
                "bookings_reset": bookings_count,
                "date": date_str or "toutes les dates",
            }, 200

        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)
