# backend/routes/analytics.py
"""
Routes API pour les analytics et métriques de dispatch.
"""

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

analytics_ns = Namespace('analytics', description='Analytics et métriques de dispatch')


@analytics_ns.route('/dashboard')
class AnalyticsDashboard(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère les analytics pour le dashboard"""

        logger.info("[Analytics] Dashboard endpoint called")

        try:
            # Récupérer la company depuis le token JWT
            logger.debug("[Analytics] Calling get_company_from_token()")
            company, err, code = get_company_from_token()
            logger.debug(f"[Analytics] Company: {company}, err: {err}, code: {code}")

            if err or company is None:
                msg = (err or {}).get("error") if isinstance(err, dict) else "Company not found"
                logger.warning(f"[Analytics] Company not found: {msg}")
                return {"success": False, "error": msg}, code or 404

            period = request.args.get('period', '30d')
            start_str = request.args.get('start_date')
            end_str = request.args.get('end_date')

            if start_str and end_str:
                try:
                    start_date = date.fromisoformat(start_str)
                    end_date = date.fromisoformat(end_str)
                except ValueError:
                    return {"success": False, "error": "Invalid date format. Use YYYY-MM-DD"}, 400
            else:
                # Inclure jusqu'à demain pour capturer les dispatches futurs
                end_date = date.today() + timedelta(days=1)
                if period == '7d':
                    start_date = end_date - timedelta(days=7)
                elif period == '90d':
                    start_date = end_date - timedelta(days=90)
                elif period == '1y':
                    start_date = end_date - timedelta(days=365)
                else:
                    start_date = end_date - timedelta(days=30)

            logger.info(f"[Analytics] Fetching analytics for company {company.id}, period {start_date} to {end_date}")
            analytics = get_period_analytics(company.id, start_date, end_date)

            logger.debug(f"[Analytics] Generating insights for company {company.id}")
            insights = generate_insights(company.id, analytics)
            analytics["insights"] = insights

            logger.info(f"[Analytics] Returning analytics data: {len(analytics.get('trends', []))} days")
            return {"success": True, "data": analytics}

        except Exception as e:
            logger.error(f"[Analytics] Error in dashboard endpoint: {e}", exc_info=True)
            return {"success": False, "error": f"Failed to fetch analytics: {str(e)}"}, 500


@analytics_ns.route('/insights')
class AnalyticsInsights(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Génère des insights intelligents"""

        try:
            company, err, code = get_company_from_token()
            if err or company is None:
                msg = (err or {}).get("error") if isinstance(err, dict) else "Company not found"
                return {"success": False, "error": msg}, code or 404

            lookback_days = int(request.args.get('lookback_days', 30))
            patterns = detect_patterns(company.id, lookback_days)

            return {"success": True, "data": patterns}

        except Exception as e:
            return {"success": False, "error": f"Failed to generate insights: {str(e)}"}, 500


@analytics_ns.route('/weekly-summary')
class WeeklySummary(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère le résumé hebdomadaire"""

        try:
            company, err, code = get_company_from_token()
            if err or company is None:
                msg = (err or {}).get("error") if isinstance(err, dict) else "Company not found"
                return {"success": False, "error": msg}, code or 404

            week_start_str = request.args.get('week_start')
            if week_start_str:
                try:
                    week_start = date.fromisoformat(week_start_str)
                except ValueError:
                    return {"success": False, "error": "Invalid date format. Use YYYY-MM-DD"}, 400
            else:
                today = date.today()
                week_start = today - timedelta(days=today.weekday())

            summary = get_weekly_summary(company.id, week_start)

            return {"success": True, "data": summary}

        except Exception as e:
            return {"success": False, "error": f"Failed to fetch weekly summary: {str(e)}"}, 500


@analytics_ns.route('/export')
class ExportAnalytics(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Exporte les analytics en CSV ou JSON"""

        try:
            company, err, code = get_company_from_token()
            if err or company is None:
                msg = (err or {}).get("error") if isinstance(err, dict) else "Company not found"
                return {"success": False, "error": msg}, code or 404

            start_str = request.args.get('start_date')
            end_str = request.args.get('end_date')
            export_format = request.args.get('format', 'csv')

            if not start_str or not end_str:
                return {"success": False, "error": "start_date and end_date are required"}, 400

            try:
                start_date = date.fromisoformat(start_str)
                end_date = date.fromisoformat(end_str)
            except ValueError:
                return {"success": False, "error": "Invalid date format. Use YYYY-MM-DD"}, 400

            analytics = get_period_analytics(company.id, start_date, end_date)

            if export_format == 'csv':
                output = io.StringIO()
                writer = csv.writer(output)

                writer.writerow(['Date', 'Bookings', 'On-Time Rate (%)', 'Avg Delay (min)', 'Quality Score'])

                for trend in analytics.get('trends', []):
                    writer.writerow([
                        trend['date'],
                        trend['bookings'],
                        trend['on_time_rate'],
                        trend['avg_delay'],
                        trend['quality_score']
                    ])

                response = make_response(output.getvalue())
                response.headers["Content-Disposition"] = f"attachment; filename=analytics_{start_date}_{end_date}.csv"
                response.headers["Content-Type"] = "text/csv"

                return response
            else:
                return {"success": True, "data": analytics}

        except Exception as e:
            return {"success": False, "error": f"Failed to export analytics: {str(e)}"}, 500
