# backend/routes/dispatch/dispatch_advanced.py
"""Endpoints avancés pour le dispatch (settings, features, mode, autonomous, dashboard, reset)."""

# ruff: noqa: I001  # Imports organisés manuellement pour meilleure lisibilité
import json
import logging
from datetime import UTC, datetime
from typing import Any, Dict, cast

from flask import current_app, request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import Resource  # pyright: ignore[reportMissingImports]
from http import HTTPStatus

from ext import db, role_required
from models import Assignment, DispatchMode
from models.enums import BookingStatus, UserRole
from repositories.assignment_repository import AssignmentRepository
from repositories.booking_repository import BookingRepository
from repositories.driver_repository import DriverRepository
from routes.dispatch import dispatch_ns
from routes.dispatch.dispatch_helpers import (
    _current_company_id,
    _get_current_company,
)
from shared.error_handlers import APIErrorHandler
from shared.time_utils import day_local_bounds, now_local

logger = logging.getLogger(__name__)

# Initialisation des repositories
assignment_repo = AssignmentRepository()
booking_repo = BookingRepository()
driver_repo = DriverRepository()

# Constantes
DELAY_MINUTES_THRESHOLD = 5
DELAY_MINUTES_ZERO = 0


@dispatch_ns.route("/dashboard/realtime")
class RealtimeDashboardResource(Resource):
    """Dashboard temps réel pour les dispatchers."""

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
            try:
                d0, d1 = day_local_bounds(date_str)
                # ✅ P1: Eager loading pour éviter N+1 queries
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

                current_delays = []
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
                from infrastructure.dispatch.realtime_optimizer_adapter import (
                    check_opportunities_manual,
                    get_optimizer_for_company,
                )

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
                driver_load_details = []
                if driver_load:
                    driver_ids_list = list(driver_load.keys())
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
                HTTPStatus.OK,
            )

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


# ===== GESTION DES MODES AUTONOMES =====


@dispatch_ns.route("/mode")
class DispatchModeResource(Resource):
    """Gestion du mode de dispatch."""

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

        return {
            "company_id": company_id,
            "dispatch_mode": company.dispatch_mode.value,
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
        }, HTTPStatus.OK

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
        # Récupérer l'ancien mode de manière sécurisée
        old_mode = (
            getattr(company.dispatch_mode, "value", None)
            if hasattr(company, "dispatch_mode")
            else None
        )
        if new_mode:
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
                    from services.dispatch.agent.orchestrator import (
                        get_agent_for_company,
                        stop_agent_for_company,
                    )

                    if new_mode == "fully_auto":
                        # Démarrer l'agent automatiquement en mode fully_auto
                        agent = get_agent_for_company(
                            company_id,
                            app=current_app._get_current_object(),
                        )
                        if not agent.state.running:
                            agent.start()
                            logger.info(
                                "[Dispatch] 🤖 Agent démarré automatiquement pour company %s (mode fully_auto)",
                                company_id,
                            )
                    elif old_mode == "fully_auto" and new_mode != "fully_auto":
                        # Arrêter l'agent si on sort du mode fully_auto
                        stop_agent_for_company(company_id)
                        logger.info(
                            "[Dispatch] ⏸️ Agent arrêté automatiquement pour company %s (mode changé vers %s)",
                            company_id,
                            new_mode,
                        )
                except Exception as agent_err:
                    # Ne pas faire échouer le changement de mode
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
                }, HTTPStatus.BAD_REQUEST

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

            return {
                "company_id": company_id,
                "dispatch_mode": company.dispatch_mode.value,
                "autonomous_config": company.get_autonomous_config(),
                "message": "Configuration mise à jour avec succès",
            }, HTTPStatus.OK

        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/autonomous/status")
class AutonomousStatusResource(Resource):
    """Statut du système autonome."""

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

            return {
                "company_id": company_id,
                "dispatch_mode": manager.mode.value,
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
            }, HTTPStatus.OK

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/autonomous/test")
class AutonomousTestResource(Resource):
    """Test du système autonome en mode dry-run."""

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Teste le système autonome en mode dry-run (simulation).
        Permet de voir ce que le système ferait sans réellement appliquer les actions.
        Body:
        {
            "date": "2025-01-17"  // optionnel, défaut: aujourd'hui
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
            }, HTTPStatus.OK

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


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

        return {
            "company_id": company_id,
            "dispatch_overrides": dispatch_overrides,
        }, HTTPStatus.OK

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
            }, HTTPStatus.OK

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
            }, HTTPStatus.OK

        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


# ===== FEATURE FLAGS & PERFORMANCE METRICS =====


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
        }, HTTPStatus.OK

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
            }, HTTPStatus.BAD_REQUEST

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
            }, HTTPStatus.OK

        except Exception as e:
            db.session.rollback()
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
            # Filtrer par date si fournie
            start_datetime = None
            end_datetime = None
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
                    }, HTTPStatus.BAD_REQUEST

            # Récupérer toutes les assignations de la company avec eager loading
            query = assignment_repo.find_models_by_company_with_date_filter_query(
                company_id=company_id,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
            )
            # ✅ P1: Eager loading pour éviter N+1 queries
            from sqlalchemy.orm import joinedload

            assignments = query.options(joinedload(Assignment.booking)).all()
            booking_ids = [a.booking_id for a in assignments]

            # Supprimer toutes les assignations
            assignments_count = len(assignments)
            for assignment in assignments:
                db.session.delete(assignment)

            # Remettre les bookings au statut ACCEPTED et nettoyer driver_id
            bookings_query = booking_repo.find_models_by_company_with_filters_query(
                company_id=company_id,
                booking_ids=booking_ids if booking_ids else None,
                start_datetime=start_datetime if date_str else None,
                end_datetime=end_datetime if date_str else None,
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
                "[RESET] ✅ Réinitialisation effectuée pour company_id=%s: %d assignations supprimées, %d bookings réinitialisés",
                company_id,
                assignments_count,
                bookings_count,
            )

            return {
                "message": "Réinitialisation effectuée avec succès",
                "assignments_deleted": assignments_count,
                "bookings_reset": bookings_count,
                "date": date_str or "toutes les dates",
            }, HTTPStatus.OK

        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)
