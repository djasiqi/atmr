# backend/routes/dispatch/dispatch_rl.py
"""Endpoints pour le dispatch par Reinforcement Learning (RL)."""

# ruff: noqa: I001  # Imports organisés manuellement pour meilleure lisibilité
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from flask import request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import (  # pyright: ignore[reportMissingImports]
    get_jwt_identity,
    jwt_required,
)
from flask_restx import Resource  # pyright: ignore[reportMissingImports]
from http import HTTPStatus

from ext import db, redis_client, role_required
from models import RLFeedback, RLSuggestionMetric
from models.enums import AssignmentStatus, UserRole
from repositories.assignment_repository import AssignmentRepository
from repositories.driver_repository import DriverRepository
from repositories.rl_feedback_repository import RLFeedbackRepository
from repositories.rl_suggestion_metric_repository import (
    RLSuggestionMetricRepository,
)
from routes.dispatch import dispatch_ns
from routes.dispatch.dispatch_helpers import (
    _current_company_id,
    _get_current_company,
    _validate_date_format,
)
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)

# Initialisation des repositories
assignment_repo = AssignmentRepository()
driver_repo = DriverRepository()
rl_metric_repo = RLSuggestionMetricRepository()
rl_feedback_repo = RLFeedbackRepository()

# RL Dispatch (déploiement production)
try:
    from services.rl.rl_dispatch_manager import RLDispatchManager  # type: ignore[reportMissingImports]

    RL_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    RL_AVAILABLE = False
    RLDispatchManager = None

# Constante
TOTAL_FEEDBACKS_ZERO = 0


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
            }, HTTPStatus.OK

        try:
            # Initialiser manager RL
            if RLDispatchManager is None:
                return {
                    "available": False,
                    "message": "RLDispatchManager non disponible",
                }, HTTPStatus.OK

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
            }, HTTPStatus.OK

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
        status_code = HTTPStatus.OK

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
                    status_code = HTTPStatus.BAD_REQUEST
                else:
                    # ✅ Valider le format YYYY-MM-DD
                    for_date_str = _validate_date_format(for_date_str)

                    # ✅ CACHE REDIS : Clé unique par company/date/params
                    cache_key = (
                        f"rl_suggestions:{company.id}:{for_date_str}:"
                        f"{min_confidence}:{limit}"
                    )

                    # Check cache
                    if redis_client:
                        try:
                            cached_bytes = redis_client.get(cache_key)
                            if cached_bytes:
                                logger.info("[RL] Cache hit for %s", cache_key)
                                # Décoder bytes → str avant json.loads
                                cached_str = cast(bytes, cached_bytes).decode("utf-8")
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
                        # Parse date
                        for_date = datetime.strptime(for_date_str, "%Y-%m-%d").date()

                        # Récupérer tous les assignments actifs pour cette date
                        assignments = assignment_repo.find_models_by_company_and_date_with_status_eager_loading(
                            company_id=company.id,
                            for_date=for_date,
                            statuses=[
                                AssignmentStatus.SCHEDULED,
                                AssignmentStatus.EN_ROUTE_PICKUP,
                                AssignmentStatus.ARRIVED_PICKUP,
                                AssignmentStatus.ONBOARD,
                                AssignmentStatus.EN_ROUTE_DROPOFF,
                            ],
                        )

                        if not assignments:
                            result = {
                                "suggestions": [],
                                "message": "Aucun assignment actif pour cette date",
                            }
                        else:
                            # Récupérer tous les conducteurs disponibles
                            drivers = driver_repo.find_models_by_company_available_with_user_eager_loading_limited(
                                company_id=company.id, limit=10
                            )

                            if not drivers:
                                result = {
                                    "suggestions": [],
                                    "message": "Aucun conducteur disponible",
                                }
                            else:
                                # Utiliser le générateur RL pour créer des suggestions
                                from services.rl.suggestion_generator import (
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
                                    for suggestion in all_suggestions:
                                        # Créer ID unique pour la suggestion
                                        suggestion_id = (
                                            f"{suggestion['assignment_id']}_"
                                            f"{int(datetime.now(UTC).timestamp() * 1000)}"
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
                                        metric.generated_at = datetime.now(UTC)
                                        metric.additional_data = {
                                            "message": suggestion.get("message"),
                                            "for_date": for_date_str,
                                            "min_confidence": min_confidence,
                                        }
                                        db.session.add(metric)

                                        # Ajouter l'ID à la suggestion pour tracking frontend
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
                                if redis_client and all_suggestions:
                                    try:
                                        redis_client.setex(
                                            cache_key,
                                            30,  # TTL 30 secondes
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
                status_code = HTTPStatus.BAD_REQUEST
            except Exception as e:
                logger.exception("[RL] Failed to get RL suggestions")
                result = {"error": f"Échec récupération suggestions RL: {e}"}
                status_code = HTTPStatus.INTERNAL_SERVER_ERROR

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
            metrics = rl_metric_repo.find_models_by_company_and_cutoff(
                company_id=company_id, cutoff=cutoff
            )

            if not metrics:
                return {
                    "period_days": days,
                    "total_suggestions": 0,
                    "message": "Aucune métrique disponible pour cette période",
                }, HTTPStatus.OK

            # Calculer statistiques agrégées
            total = len(metrics)
            applied = len([m for m in metrics if m.applied_at])
            rejected = len([m for m in metrics if m.rejected_at])
            pending = total - applied - rejected

            # Confiance moyenne
            confidences = [
                conf
                for m in metrics
                if (conf := getattr(m, "confidence", None)) is not None
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Précision gain (seulement pour suggestions appliquées)
            applied_metrics = [m for m in metrics if m.actual_gain_minutes is not None]
            if applied_metrics:
                accuracies = [
                    acc
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
            }, HTTPStatus.OK

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
                }, HTTPStatus.BAD_REQUEST

            # Récupérer la métrique de suggestion associée
            metric = rl_metric_repo.find_by_suggestion_id_and_company(
                suggestion_id=suggestion_id, company_id=company_id
            )

            if not metric:
                return APIErrorHandler.handle_not_found(
                    "Suggestion", suggestion_id, logger
                )

            # Vérifier si feedback déjà existant
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
            user_id_from_jwt = get_jwt_identity()

            # Créer feedback
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
            }, HTTPStatus.CREATED

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
            }, HTTPStatus.OK

        except Exception as e:
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)
