# backend/routes/analytics.py
"""Routes API pour les analytics et métriques de dispatch."""

import csv
import io
import logging
from datetime import date, datetime, timedelta
from typing import Any, cast

from flask import make_response, request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import Namespace, Resource  # pyright: ignore[reportMissingImports]
from sqlalchemy import and_

from ext import db, role_required
from models import Booking, EtaAccuracyLog
from models.enums import UserRole
from repositories.assignment_repository import AssignmentRepository
from routes.companies import get_company_from_token
from services.analytics.aggregator import get_period_analytics, get_weekly_summary
from services.analytics.insights import detect_patterns, generate_insights
from shared.error_handlers import APIErrorHandler

# Note: Modèles (Booking, EtaAccuracyLog) utilisés pour requêtes complexes
# TODO: Migrer vers repositories quand les méthodes nécessaires seront disponibles

logger = logging.getLogger(__name__)

analytics_ns = Namespace("analytics", description="Analytics et métriques de dispatch")

# Initialisation des repositories
assignment_repo = AssignmentRepository()


@analytics_ns.route("/dashboard")
class AnalyticsDashboard(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @analytics_ns.param(
        "period",
        "Période d'analyse (7d|30d|90d|1y, défaut: 30d)",
        type="string",
        enum=["7d", "30d", "90d", "1y"],
        default="30d",
    )
    @analytics_ns.param(
        "start_date",
        "Date de début (YYYY-MM-DD, optionnel)",
        type="string",
        pattern="^\\d{4}-\\d{2}-\\d{2}$",
    )
    @analytics_ns.param(
        "end_date",
        "Date de fin (YYYY-MM-DD, optionnel)",
        type="string",
        pattern="^\\d{4}-\\d{2}-\\d{2}$",
    )
    def get(self):
        """Récupère les analytics pour le dashboard.

        Query params:
            - period: Période prédéfinie (7d|30d|90d|1y), défaut: 30d
            - start_date: Date de début personnalisée (YYYY-MM-DD), optionnel
            - end_date: Date de fin personnalisée (YYYY-MM-DD), optionnel
        """
        logger.info("[Analytics] Dashboard endpoint called")

        try:
            # Récupérer la company depuis le token JWT
            logger.debug("[Analytics] Calling get_company_from_token()")
            company, err, code = get_company_from_token()
            logger.debug(
                "[Analytics] Company: %s, err: %s, code: %s", company, err, code
            )

            if err or company is None:
                msg = (
                    (err or {}).get("error")
                    if isinstance(err, dict)
                    else "Company not found"
                ) or "Company not found"
                logger.warning("[Analytics] Company not found: %s", msg)
                return APIErrorHandler.handle_not_found(
                    msg,
                    None,
                    logger,
                )

            # ✅ 2.4: Validation Marshmallow pour query params
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

            from schemas.analytics_schemas import AnalyticsDashboardQuerySchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            args_dict = dict(request.args)
            try:
                validated_args = validate_request(
                    AnalyticsDashboardQuerySchema(), args_dict, strict=False
                )
                period = validated_args.get("period", "30d")
                start_str = validated_args.get("start_date")
                end_str = validated_args.get("end_date")
            except ValidationError as e:
                return handle_validation_error(e)

            if start_str and end_str:
                try:
                    start_date = date.fromisoformat(start_str)
                    end_date = date.fromisoformat(end_str)
                except ValueError:
                    return {
                        "success": False,
                        "error": "Invalid date format. Use YYYY-MM-DD",
                    }, 400
            else:
                # Inclure jusqu'à demain pour capturer les dispatches futurs
                end_date = date.today() + timedelta(days=1)
                if period == "7d":
                    start_date = end_date - timedelta(days=7)
                elif period == "90d":
                    start_date = end_date - timedelta(days=90)
                elif period == "1y":
                    start_date = end_date - timedelta(days=0.365)
                else:
                    start_date = end_date - timedelta(days=30)

            logger.info(
                "[Analytics] Fetching analytics for company %s, period %s to %s",
                company.id,
                start_date,
                end_date,
            )
            analytics = get_period_analytics(company.id, start_date, end_date)

            logger.debug("[Analytics] Generating insights for company %s", company.id)
            insights = generate_insights(company.id, analytics)
            analytics["insights"] = insights

            logger.info(
                "[Analytics] Returning analytics data: %s days",
                len(analytics.get("trends", [])),
            )
            return {"success": True, "data": analytics}

        except Exception as e:
            logger.error("[Analytics] Error in dashboard endpoint: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@analytics_ns.route("/insights")
class AnalyticsInsights(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @analytics_ns.param(
        "lookback_days",
        "Nombre de jours à analyser (1-365, défaut: 30)",
        type="integer",
        minimum=1,
        maximum=365,
        default=30,
    )
    def get(self):
        """Génère des insights intelligents.

        Query params:
            - lookback_days: Nombre de jours à analyser en arrière (1-365), défaut: 30
        """
        try:
            company, err, _ = get_company_from_token()
            if err or company is None:
                msg = (
                    (err or {}).get("error")
                    if isinstance(err, dict)
                    else "Company not found"
                ) or "Company not found"
                return APIErrorHandler.handle_not_found(
                    msg,
                    None,
                    logger,
                )

            # ✅ 2.4: Validation Marshmallow pour query params
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

            from schemas.analytics_schemas import AnalyticsInsightsQuerySchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            args_dict = dict(request.args)
            try:
                validated_args = validate_request(
                    AnalyticsInsightsQuerySchema(), args_dict, strict=False
                )
                lookback_days = validated_args.get("lookback_days", 30)
            except ValidationError as e:
                return handle_validation_error(e)

            patterns = detect_patterns(company.id, lookback_days)

            return {"success": True, "data": patterns}

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate insights: {e!s}",
            }, 500


@analytics_ns.route("/weekly-summary")
class WeeklySummary(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @analytics_ns.param(
        "week_start",
        "Date de début de semaine (YYYY-MM-DD, optionnel)",
        type="string",
        pattern="^\\d{4}-\\d{2}-\\d{2}$",
    )
    def get(self):
        """Récupère un résumé hebdomadaire.

        Query params:
            - week_start: Date de début de semaine (YYYY-MM-DD), optionnel
        """
        try:
            company, err, _ = get_company_from_token()
            if err or company is None:
                msg = (
                    (err or {}).get("error")
                    if isinstance(err, dict)
                    else "Company not found"
                ) or "Company not found"
                return APIErrorHandler.handle_not_found(
                    msg,
                    None,
                    logger,
                )

            # ✅ 2.4: Validation Marshmallow pour query params
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

            from schemas.analytics_schemas import AnalyticsWeeklySummaryQuerySchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            args_dict = dict(request.args)
            try:
                validated_args = validate_request(
                    AnalyticsWeeklySummaryQuerySchema(), args_dict, strict=False
                )
                week_start_str = validated_args.get("week_start")
            except ValidationError as e:
                return handle_validation_error(e)

            if week_start_str:
                try:
                    week_start = date.fromisoformat(week_start_str)
                except ValueError:
                    return {
                        "success": False,
                        "error": "Invalid date format. Use YYYY-MM-DD",
                    }, 400
            else:
                today = date.today()
                week_start = today - timedelta(days=today.weekday())

            summary = get_weekly_summary(company.id, week_start)

            return {"success": True, "data": summary}

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to fetch weekly summary: {e!s}",
            }, 500


@analytics_ns.route("/export")
class ExportAnalytics(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @analytics_ns.param(
        "start_date",
        "Date de début (YYYY-MM-DD, requis)",
        type="string",
        required=True,
        pattern="^\\d{4}-\\d{2}-\\d{2}$",
    )
    @analytics_ns.param(
        "end_date",
        "Date de fin (YYYY-MM-DD, requis)",
        type="string",
        required=True,
        pattern="^\\d{4}-\\d{2}-\\d{2}$",
    )
    @analytics_ns.param(
        "format",
        "Format d'export (csv|json, défaut: csv)",
        type="string",
        enum=["csv", "json"],
        default="csv",
    )
    def get(self):
        """Exporte les analytics dans un format donné.

        Query params:
            - start_date: Date de début (YYYY-MM-DD), requis
            - end_date: Date de fin (YYYY-MM-DD), requis
            - format: Format d'export (csv|json), défaut: csv
        """
        try:
            company, err, _ = get_company_from_token()
            if err or company is None:
                msg = (
                    (err or {}).get("error")
                    if isinstance(err, dict)
                    else "Company not found"
                ) or "Company not found"
                return APIErrorHandler.handle_not_found(
                    msg,
                    None,
                    logger,
                )

            # ✅ 2.4: Validation Marshmallow pour query params
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

            from schemas.analytics_schemas import AnalyticsExportQuerySchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            args_dict = dict(request.args)
            try:
                validated_args = validate_request(
                    AnalyticsExportQuerySchema(), args_dict
                )
                start_str = validated_args["start_date"]
                end_str = validated_args["end_date"]
                export_format = validated_args.get("format", "csv")
            except ValidationError as e:
                return handle_validation_error(e)

            try:
                start_date = date.fromisoformat(start_str)
                end_date = date.fromisoformat(end_str)
            except ValueError:
                return {
                    "success": False,
                    "error": "Invalid date format. Use YYYY-MM-DD",
                }, 400

            analytics = get_period_analytics(company.id, start_date, end_date)

            if export_format == "csv":
                output = io.StringIO()
                writer = csv.writer(output)

                writer.writerow(
                    [
                        "Date",
                        "Bookings",
                        "On-Time Rate (%)",
                        "Avg Delay (min)",
                        "Quality Score",
                    ]
                )

                for trend in analytics.get("trends", []):
                    writer.writerow(
                        [
                            trend["date"],
                            trend["bookings"],
                            trend["on_time_rate"],
                            trend["avg_delay"],
                            trend["quality_score"],
                        ]
                    )

                response = make_response(output.getvalue())
                response.headers["Content-Disposition"] = (
                    f"attachment; filename=analytics_{start_date}_{end_date}.csv"
                )
                response.headers["Content-Type"] = "text/csv"

                return response
            return {"success": True, "data": analytics}

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to export analytics: {e!s}",
            }, 500


@analytics_ns.route("/eta/accuracy")
class EtaAccuracy(Resource):
    """Métriques précision ETA (prédit vs réel)."""

    @jwt_required()
    @role_required(UserRole.company)
    @analytics_ns.param(
        "start_date",
        "Date de début (YYYY-MM-DD, optionnel)",
        type="string",
        pattern="^\\d{4}-\\d{2}-\\d{2}$",
    )
    @analytics_ns.param(
        "end_date",
        "Date de fin (YYYY-MM-DD, optionnel)",
        type="string",
        pattern="^\\d{4}-\\d{2}-\\d{2}$",
    )
    @analytics_ns.param(
        "source",
        "Source ETA (osrm, osrm_ml, haversine, etc.)",
        type="string",
    )
    @analytics_ns.param(
        "group_by",
        "Grouper par (hour, day, source, zone)",
        type="string",
        enum=["hour", "day", "source", "zone"],
    )
    def get(self):
        """Récupère les métriques de précision ETA.

        Query params:
            - start_date: Date de début (YYYY-MM-DD), optionnel
            - end_date: Date de fin (YYYY-MM-DD), optionnel
            - source: Filtrer par source ETA (osrm, osrm_ml, etc.)
            - group_by: Grouper résultats (hour, day, source, zone)

        Returns:
            Dict avec métriques précision (MAE, RMSE, précision moyenne, etc.)
        """
        try:
            # Récupérer la company depuis le token JWT
            company, err, _ = get_company_from_token()
            if err or company is None:
                msg = (
                    (err or {}).get("error")
                    if isinstance(err, dict)
                    else "Company not found"
                ) or "Company not found"
                return APIErrorHandler.handle_not_found(
                    msg,
                    None,
                    logger,
                )

            # Paramètres de requête
            start_str = request.args.get("start_date")
            end_str = request.args.get("end_date")
            source_filter = request.args.get("source")
            group_by = request.args.get("group_by", "day")

            # Dates par défaut (30 derniers jours)
            if start_str and end_str:
                try:
                    start_date = datetime.fromisoformat(start_str).date()
                    end_date = datetime.fromisoformat(end_str).date()
                except ValueError:
                    return {
                        "success": False,
                        "error": "Invalid date format. Use YYYY-MM-DD",
                    }, 400
            else:
                end_date = date.today()
                start_date = end_date - timedelta(days=30)

            # Requête de base
            query = db.session.query(EtaAccuracyLog).filter(
                and_(
                    EtaAccuracyLog.created_at
                    >= datetime.combine(start_date, datetime.min.time()),
                    EtaAccuracyLog.created_at
                    <= datetime.combine(end_date, datetime.max.time()),
                    # Filtrer par company via bookings
                    EtaAccuracyLog.booking_id.in_(
                        db.session.query(Booking.id).filter(
                            Booking.company_id == company.id
                        )
                    ),
                )
            )

            # Filtrer par source si fourni
            if source_filter:
                query = query.filter(EtaAccuracyLog.source == source_filter)

            # Filtrer seulement les logs avec actual_duration (trajets terminés)
            query = query.filter(EtaAccuracyLog.actual_duration_seconds.isnot(None))

            # Calculer métriques globales
            logs = query.all()

            if not logs:
                return {
                    "success": True,
                    "data": {
                        "total_samples": 0,
                        "mae_seconds": 0,
                        "rmse_seconds": 0,
                        "mean_error_seconds": 0,
                        "accuracy_percent": 0,
                        "by_source": {},
                        "by_hour": {},
                    },
                }

            # Métriques globales
            total_samples = len(logs)
            errors = [abs(log.error_seconds or 0) for log in logs]
            mean_error = sum(errors) / total_samples if total_samples > 0 else 0
            mae_seconds = mean_error  # Mean Absolute Error
            rmse_seconds = (
                (sum(e**2 for e in errors) / total_samples) ** 0.5
                if total_samples > 0
                else 0
            )

            # Précision (erreur < 10% de l'ETA prédit)
            accurate_count = sum(
                1
                for log in logs
                if log.predicted_eta_seconds > 0
                and abs(log.error_seconds or 0) < (log.predicted_eta_seconds * 0.1)
            )
            accuracy_percent = (
                (accurate_count / total_samples * 100) if total_samples > 0 else 0
            )

            # Grouper par source
            by_source: dict[str, dict[str, Any]] = {}
            sources = {log.source for log in logs}
            for src in sources:
                src_logs = [log for log in logs if log.source == src]
                src_errors = [abs(log.error_seconds or 0) for log in src_logs]
                by_source[src] = {
                    "count": len(src_logs),
                    "mae_seconds": sum(src_errors) / len(src_errors)
                    if src_errors
                    else 0,
                    "accuracy_percent": (
                        sum(
                            1
                            for log in src_logs
                            if log.predicted_eta_seconds > 0
                            and abs(log.error_seconds or 0)
                            < (log.predicted_eta_seconds * 0.1)
                        )
                        / len(src_logs)
                        * 100
                        if src_logs
                        else 0
                    ),
                }

            # Grouper par heure (si group_by=hour)
            by_hour: dict[int, dict[str, Any]] = {}
            if group_by == "hour":
                for log in logs:
                    hour = log.created_at.hour if log.created_at else 0
                    if hour not in by_hour:
                        by_hour[hour] = {"count": 0, "errors": []}
                    by_hour[hour]["count"] += 1
                    by_hour[hour]["errors"].append(abs(log.error_seconds or 0))

                # Calculer MAE par heure
                for hour, data in by_hour.items():
                    errors_list = data["errors"]
                    by_hour[hour] = {
                        "count": data["count"],
                        "mae_seconds": sum(errors_list) / len(errors_list)
                        if errors_list
                        else 0,
                    }

            return {
                "success": True,
                "data": {
                    "total_samples": total_samples,
                    "mae_seconds": round(mae_seconds, 2),
                    "rmse_seconds": round(rmse_seconds, 2),
                    "mean_error_seconds": round(mean_error, 2),
                    "accuracy_percent": round(accuracy_percent, 2),
                    "by_source": by_source,
                    "by_hour": by_hour,
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                    },
                },
            }

        except Exception as e:
            logger.exception("[Analytics] Erreur ETA accuracy: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@analytics_ns.route("/dispatch")
class DispatchAnalytics(Resource):
    """✅ 3.4.3: Analytics spécifiques au dispatch."""

    @jwt_required()
    @role_required(UserRole.company)
    @analytics_ns.param(
        "start_date",
        "Date de début (YYYY-MM-DD, optionnel, défaut: 30 jours)",
        type="string",
        pattern="^\\d{4}-\\d{2}-\\d{2}$",
    )
    @analytics_ns.param(
        "end_date",
        "Date de fin (YYYY-MM-DD, optionnel, défaut: aujourd'hui)",
        type="string",
        pattern="^\\d{4}-\\d{2}-\\d{2}$",
    )
    def get(self):
        """Récupère les métriques de dispatch.

        Métriques retournées:
        - Coût moyen par trajet
        - Satisfaction client (retards, annulations)
        - Équité charge entre chauffeurs
        - Taux réassignation
        """
        try:
            company, err, _ = get_company_from_token()
            if err or company is None:
                error_message = (
                    err.get("error", "Company not found")
                    if isinstance(err, dict)
                    else str(err)
                    if err
                    else "Company not found"
                )
                return APIErrorHandler.handle_not_found(
                    error_message,
                    None,
                    logger,
                )

            # Période par défaut: 30 derniers jours
            end_date = date.today()
            start_date_str = request.args.get("start_date")
            end_date_str = request.args.get("end_date")

            if start_date_str:
                try:
                    start_date = date.fromisoformat(start_date_str)
                except ValueError:
                    return APIErrorHandler.handle_validation_error(
                        "Invalid start_date format. Use YYYY-MM-DD",
                        field="start_date",
                        logger_instance=logger,
                    )
            else:
                start_date = end_date - timedelta(days=30)

            if end_date_str:
                try:
                    end_date = date.fromisoformat(end_date_str)
                except ValueError:
                    return APIErrorHandler.handle_validation_error(
                        "Invalid end_date format. Use YYYY-MM-DD",
                        field="end_date",
                        logger_instance=logger,
                    )

            # Récupérer les assignments de la période
            from datetime import UTC as UTC_TZ

            from models import BookingStatus

            start_datetime = datetime.combine(start_date, datetime.min.time()).replace(
                tzinfo=UTC_TZ
            )
            end_datetime = datetime.combine(
                end_date + timedelta(days=1), datetime.min.time()
            ).replace(tzinfo=UTC_TZ)

            assignments = assignment_repo.find_models_by_company_with_time_range_and_excluded_statuses(
                company_id=company.id,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                excluded_statuses=[],  # Pas d'exclusion de statuts
            )

            if not assignments:
                return {
                    "company_id": company.id,
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                    },
                    "metrics": {
                        "total_bookings": 0,
                        "avg_cost_per_trip": 0,
                        "customer_satisfaction": {
                            "on_time_rate": 0,
                            "delay_rate": 0,
                            "cancellation_rate": 0,
                            "avg_delay_minutes": 0,
                        },
                        "driver_fairness": {
                            "load_std_dev": 0,
                            "min_load": 0,
                            "max_load": 0,
                            "avg_load": 0,
                        },
                        "reassignment_rate": 0,
                        "total_reassignments": 0,
                    },
                }, 200

            # Calculer métriques
            total_bookings = len(assignments)
            completed_bookings = [
                a
                for a in assignments
                if a.booking and a.booking.status == BookingStatus.COMPLETED
            ]

            # Coût moyen par trajet (approximation basée sur distance)
            total_cost = 0.0
            for a in completed_bookings:
                if a.booking and a.booking.pickup_lat and a.booking.dropoff_lat:
                    from shared.geo_utils import haversine_distance

                    distance_km = haversine_distance(
                        a.booking.pickup_lat,
                        a.booking.pickup_lon,
                        a.booking.dropoff_lat,
                        a.booking.dropoff_lon,
                    )
                    # Estimation: 2€/km (à ajuster selon modèle tarifaire)
                    total_cost += distance_km * 2.0

            avg_cost_per_trip = (
                total_cost / len(completed_bookings) if completed_bookings else 0.0
            )

            # Satisfaction client (retards, annulations)
            on_time_count = 0
            delayed_count = 0
            total_delay_minutes = 0
            for a in completed_bookings:
                if a.booking and a.booking.scheduled_time and a.eta_pickup_at:
                    delay_seconds = (
                        a.eta_pickup_at - a.booking.scheduled_time
                    ).total_seconds()
                    delay_minutes = int(delay_seconds / 60)
                    ON_TIME_THRESHOLD_MINUTES = 5  # ±5 min = à l'heure
                    if abs(delay_minutes) <= ON_TIME_THRESHOLD_MINUTES:
                        on_time_count += 1
                    else:
                        delayed_count += 1
                        total_delay_minutes += max(0, delay_minutes)

            on_time_rate = (
                on_time_count / len(completed_bookings) if completed_bookings else 0.0
            )
            delay_rate = (
                delayed_count / len(completed_bookings) if completed_bookings else 0.0
            )
            avg_delay_minutes = (
                total_delay_minutes / delayed_count if delayed_count > 0 else 0.0
            )

            # Taux d'annulation
            cancelled_bookings = [
                a
                for a in assignments
                if a.booking and a.booking.status == BookingStatus.CANCELED
            ]
            cancellation_rate = (
                len(cancelled_bookings) / total_bookings if total_bookings > 0 else 0.0
            )

            # Équité charge entre chauffeurs
            from collections import defaultdict

            driver_loads: dict[int, int] = defaultdict(int)
            for a in assignments:
                if bool(a.driver_id):
                    driver_loads[cast(int, a.driver_id)] += 1

            if driver_loads:
                loads = list(driver_loads.values())
                avg_load = sum(loads) / len(loads)
                variance = sum((x - avg_load) ** 2 for x in loads) / len(loads)
                load_std_dev = variance**0.5
                min_load = min(loads)
                max_load = max(loads)
            else:
                avg_load = 0.0
                load_std_dev = 0.0
                min_load = 0
                max_load = 0

            # Taux réassignation (approximation: assignments avec updated_at >
            # created_at)
            reassigned_count = 0
            for a in assignments:
                updated_at = getattr(a, "updated_at", None)
                created_at = getattr(a, "created_at", None)
                if (
                    updated_at is not None
                    and created_at is not None
                    and updated_at > created_at
                ):
                    # Vérifier si c'est une vraie réassignation (changement driver)
                    # Note: Cette heuristique est approximative
                    reassigned_count += 1

            reassignment_rate = (
                reassigned_count / total_bookings if total_bookings > 0 else 0.0
            )

            return {
                "company_id": company.id,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
                "metrics": {
                    "total_bookings": total_bookings,
                    "avg_cost_per_trip": round(avg_cost_per_trip, 2),
                    "customer_satisfaction": {
                        "on_time_rate": round(on_time_rate, 3),
                        "delay_rate": round(delay_rate, 3),
                        "cancellation_rate": round(cancellation_rate, 3),
                        "avg_delay_minutes": round(avg_delay_minutes, 1),
                    },
                    "driver_fairness": {
                        "load_std_dev": round(load_std_dev, 2),
                        "min_load": min_load,
                        "max_load": max_load,
                        "avg_load": round(avg_load, 2),
                    },
                    "reassignment_rate": round(reassignment_rate, 3),
                    "total_reassignments": reassigned_count,
                },
            }, 200

        except Exception as e:
            logger.exception("[Analytics] Error in dispatch analytics: %s", e)
            return APIErrorHandler.handle_exception(e, logger)
