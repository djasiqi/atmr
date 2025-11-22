# backend/routes/analytics.py
"""Routes API pour les analytics et métriques de dispatch."""

import csv
import io
import logging
from datetime import date, timedelta

from flask import make_response, request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource

from ext import role_required
from models import UserRole
from routes.companies import get_company_from_token
from services.analytics.aggregator import get_period_analytics, get_weekly_summary
from services.analytics.insights import detect_patterns, generate_insights

logger = logging.getLogger(__name__)

analytics_ns = Namespace("analytics", description="Analytics et métriques de dispatch")


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
                )
                logger.warning("[Analytics] Company not found: %s", msg)
                return {"success": False, "error": msg}, code or 404

            # ✅ 2.4: Validation Marshmallow pour query params
            from marshmallow import ValidationError

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
            return {"success": False, "error": f"Failed to fetch analytics: {e!s}"}, 500


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
            company, err, code = get_company_from_token()
            if err or company is None:
                msg = (
                    (err or {}).get("error")
                    if isinstance(err, dict)
                    else "Company not found"
                )
                return {"success": False, "error": msg}, code or 404

            # ✅ 2.4: Validation Marshmallow pour query params
            from marshmallow import ValidationError

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
            company, err, code = get_company_from_token()
            if err or company is None:
                msg = (
                    (err or {}).get("error")
                    if isinstance(err, dict)
                    else "Company not found"
                )
                return {"success": False, "error": msg}, code or 404

            # ✅ 2.4: Validation Marshmallow pour query params
            from marshmallow import ValidationError

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
            company, err, code = get_company_from_token()
            if err or company is None:
                msg = (
                    (err or {}).get("error")
                    if isinstance(err, dict)
                    else "Company not found"
                )
                return {"success": False, "error": msg}, code or 404

            # ✅ 2.4: Validation Marshmallow pour query params
            from marshmallow import ValidationError

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
