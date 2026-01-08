# backend/routes/dispatch/dispatch_run.py
"""Endpoints pour le lancement et le suivi des dispatches."""

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, cast

from flask import request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import Resource  # pyright: ignore[reportMissingImports]

from ext import limiter, role_required
from infrastructure.dispatch import data_adapter as data
from infrastructure.dispatch.queue_adapter import get_status, trigger_job
from models.enums import UserRole
from repositories.dispatch_run_repository import DispatchRunRepository
from routes.dispatch import dispatch_ns
from routes.dispatch.dispatch_helpers import (
    _coerce_bool_param,
    _current_company_id,
    _get_current_company,
    _make_json_safe,
    _validate_date_format,
)
from routes.dispatch.dispatch_schemas import (
    preview_response,
    run_model,
)
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)

# Initialisation des repositories
dispatch_run_repo = DispatchRunRepository()

# Constantes
N_BOOKINGS_ZERO = 0


@dispatch_ns.route("/run")
class CompanyDispatchRun(Resource):
    """Endpoint pour lancer un dispatch."""

    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("10000 per hour")  # ⚠️ C2: Augmenté temporairement pour load testing (normalement 30/h)
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
                from infrastructure.dispatch.data_adapter import get_bookings_for_day

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
    """Endpoint pour obtenir le statut d'un dispatch."""

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
                    dispatch_run = dispatch_run_repo.find_model_by_id_and_company(
                        dispatch_run_id=run_id_int, company_id=company_id
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
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/preview")
class DispatchPreview(Resource):
    """Endpoint pour obtenir un aperçu de la journée."""

    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.marshal_with(preview_response)
    def get(self):
        """Aperçu de la journée (for_date): nb bookings/drivers et horizon (minutes)."""
        company = _get_current_company()
        company_id = company.id

        for_date = request.args.get("for_date")
        if not for_date:
            return APIErrorHandler.handle_validation_error(
                "Paramètre for_date (YYYY-MM-DD) requis pour le preview.",
                field="for_date",
                expected_format="YYYY-MM-DD",
                logger_instance=logger,
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
    """Endpoint déprécié pour déclencher un dispatch (redirige vers /run)."""

    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("10000 per hour")  # ⚠️ C2: Augmenté temporairement pour load testing (normalement 50/h)
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
            return APIErrorHandler.handle_validation_error(
                (
                    "for_date manquant (YYYY-MM-DD). "
                    "Utilisez plutôt POST /company_dispatch/run."
                ),
                field="for_date",
                expected_format="YYYY-MM-DD",
                logger_instance=logger,
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
