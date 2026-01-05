# backend/routes/dispatch/dispatch_metrics.py
"""Endpoints pour les métriques de dispatch."""

# ruff: noqa: I001  # Imports organisés manuellement pour meilleure lisibilité
import logging
from datetime import UTC, date, datetime, timedelta

from flask import Response, request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import Resource  # pyright: ignore[reportMissingImports]
from http import HTTPStatus

from ext import role_required
from models.enums import UserRole
from repositories.dispatch_run_repository import DispatchRunRepository
from routes.dispatch import dispatch_ns
from routes.dispatch.dispatch_helpers import _current_company_id
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)

# Initialisation des repositories
dispatch_run_repo = DispatchRunRepository()


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
                error_response = (
                    {"error": "dispatch_run_id invalide"},
                    HTTPStatus.BAD_REQUEST,
                )
            else:
                dispatch_run = dispatch_run_repo.find_model_by_id_and_company(
                    dispatch_run_id, company_id
                )
                if not dispatch_run:
                    error_response = (
                        {"error": "Dispatch run non trouvé"},
                        HTTPStatus.NOT_FOUND,
                    )
                else:
                    try:
                        meta = getattr(dispatch_run, "meta", None) or {}
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
                                HTTPStatus.OK,
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
                                HTTPStatus.OK,
                            )
                    except Exception as e:
                        logger.exception(
                            "[Dispatch] Failed to extract performance metrics"
                        )
                        error_response = (
                            {"error": f"Erreur lors de l'extraction: {e}"},
                            HTTPStatus.INTERNAL_SERVER_ERROR,
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
                    HTTPStatus.BAD_REQUEST,
                )
            else:
                dispatch_run = dispatch_run_repo.find_model_by_company_and_day_ordered(
                    company_id, date_obj
                )
                if not dispatch_run:
                    result_response = (
                        {
                            "date": date_str,
                            "message": "Aucun dispatch run trouvé pour cette date",
                        },
                        HTTPStatus.OK,
                    )
                else:
                    try:
                        meta = getattr(dispatch_run, "meta", None) or {}
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
                                HTTPStatus.OK,
                            )
                        else:
                            result_response = (
                                {
                                    "dispatch_run_id": dispatch_run.id,
                                    "date": date_str,
                                    "message": "Aucune métrique disponible",
                                },
                                HTTPStatus.OK,
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
                error_response = (
                    {"error": "dispatch_run_id invalide"},
                    HTTPStatus.BAD_REQUEST,
                )
            else:
                dispatch_run = dispatch_run_repo.find_model_by_id_and_company(
                    dispatch_run_id, company_id
                )
                if not dispatch_run:
                    error_response = (
                        {"error": "Dispatch run non trouvé"},
                        HTTPStatus.NOT_FOUND,
                    )
                else:
                    meta = getattr(dispatch_run, "meta", None) or {}
                    perf_metrics = meta.get("performance_metrics")
                    if not perf_metrics:
                        response = (
                            Response("# No metrics available", mimetype="text/plain"),
                            HTTPStatus.OK,
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
                    HTTPStatus.BAD_REQUEST,
                )
            else:
                dispatch_run = dispatch_run_repo.find_model_by_company_and_day_ordered(
                    company_id, date_obj
                )
                if not dispatch_run:
                    response = (
                        Response(
                            f"# No dispatch run found for date {date_str}",
                            mimetype="text/plain",
                        ),
                        HTTPStatus.OK,
                    )
                else:
                    meta = getattr(dispatch_run, "meta", None) or {}
                    perf_metrics = meta.get("performance_metrics")
                    if not perf_metrics:
                        response = (
                            Response("# No metrics available", mimetype="text/plain"),
                            HTTPStatus.OK,
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
                response = (
                    Response(prometheus_text, mimetype="text/plain"),
                    HTTPStatus.OK,
                )
            except Exception as e:
                logger.exception("[Dispatch] Failed to convert to Prometheus format")
                error_response = (
                    {"error": f"Erreur lors de la conversion: {e}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        if error_response:
            return error_response
        if response:
            return response
        return (
            Response("# No metrics available", mimetype="text/plain"),
            HTTPStatus.OK,
        )


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
            runs = dispatch_run_repo.find_models_by_company_and_date_range(
                company_id, start_date
            )

            # Calculer les statistiques
            total_conflicts = sum(
                getattr(r, "meta", {})
                .get("performance_metrics", {})
                .get("temporal_conflicts", 0)
                for r in runs
                if getattr(r, "meta", None)
            )
            total_bookings = sum(
                getattr(r, "meta", {}).get("counters", {}).get("bookings_processed", 0)
                for r in runs
                if getattr(r, "meta", None)
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
            }, HTTPStatus.OK

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
            start_date = datetime.now(UTC).date() - timedelta(days=days)
            runs = dispatch_run_repo.find_models_by_company_and_date_range(
                company_id, start_date, limit=100
            )

            all_rejects = []
            for run in runs:
                if not getattr(run, "meta", None):
                    continue

                debug = getattr(run, "meta", {}).get("debug", {})
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
            }, HTTPStatus.OK

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
            start_date = datetime.now(UTC).date() - timedelta(days=days)
            runs = dispatch_run_repo.find_models_by_company_and_date_range(
                company_id, start_date
            )

            total_conflicts = sum(
                getattr(r, "meta", {})
                .get("performance_metrics", {})
                .get("temporal_conflicts", 0)
                for r in runs
                if getattr(r, "meta", None)
            )
            total_bookings = sum(
                getattr(r, "meta", {}).get("counters", {}).get("bookings_processed", 0)
                for r in runs
                if getattr(r, "meta", None)
            )

            violation_rate = (
                total_conflicts / total_bookings if total_bookings > 0 else 0
            )
            threshold = 0.001

            backout_needed = violation_rate >= threshold

            if backout_needed:
                logger.error(
                    "[A1] ❌ Backout recommandé: violation_rate=%.4f >= threshold=%.4f (company_id=%s)",
                    violation_rate,
                    threshold,
                    company_id,
                )
            else:
                logger.info(
                    "[A1] ✅ Conformité OK: violation_rate=%.4f < threshold=%.4f (company_id=%s)",
                    violation_rate,
                    threshold,
                    company_id,
                )

            return {
                "backout_needed": backout_needed,
                "violation_rate": violation_rate,
                "threshold": threshold,
                "temporal_conflicts": total_conflicts,
                "total_bookings": total_bookings,
                "days": days,
                "runs_analyzed": len(runs),
                "message": (
                    "Backout recommandé" if backout_needed else "Conformité OK"
                ),
            }, HTTPStatus.OK

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)
