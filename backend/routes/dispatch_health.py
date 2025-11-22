"""Routes pour le monitoring de santé du système de dispatch."""

import logging
from datetime import UTC, datetime, timedelta

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource, fields

from ext import db, role_required
from models import DispatchRun, UserRole
from routes.companies import get_company_from_token

logger = logging.getLogger(__name__)

# Constantes pour éviter les valeurs magiques
MIN_RATES_FOR_TREND = 2
RECENT_RATES_COUNT = 3
MIN_COMPARISON_RATES = 3

dispatch_health_ns = Namespace(
    "company_dispatch_health", description="Monitoring de santé du dispatch"
)

# ===== Schemas RESTX =====

health_response = dispatch_health_ns.model(
    "DispatchHealthResponse",
    {
        "last_24h_runs": fields.Integer(
            description="Nombre de runs dans les dernières 24h"
        ),
        "avg_assignment_rate": fields.Float(description="Taux d'assignation moyen"),
        "avg_run_time_sec": fields.Float(
            description="Temps d'exécution moyen en secondes"
        ),
        "osrm_availability": fields.Float(description="Disponibilité OSRM (0-1)"),
        "failed_runs": fields.Integer(description="Nombre de runs échoués"),
        "unassigned_reasons": fields.Raw(description="Raisons de non-assignation"),
        "performance_trend": fields.String(description="Tendance de performance"),
    },
)

trends_response = dispatch_health_ns.model(
    "DispatchTrendsResponse",
    {
        "days": fields.List(fields.String, description="Dates"),
        "assignment_rates": fields.List(
            fields.Float, description="Taux d'assignation par jour"
        ),
        "run_times": fields.List(
            fields.Float, description="Temps d'exécution par jour"
        ),
        "osrm_availability": fields.List(
            fields.Float, description="Disponibilité OSRM par jour"
        ),
        "failed_runs": fields.List(fields.Integer, description="Runs échoués par jour"),
    },
)

# ===== Routes =====


@dispatch_health_ns.route("/dlq")
class DLQResource(Resource):
    """✅ A3: Métriques DLQ (Dead Letter Queue) pour les tâches Celery échouées."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère le backlog DLQ et l'âge max des messages.

        Returns:
            JSON avec backlog, âge max, détails des tâches échouées, alertes
        """
        from models import TaskFailure

        try:
            # Récupérer toutes les tâches échouées
            failures = (
                TaskFailure.query.order_by(TaskFailure.last_seen.desc())
                .limit(100)
                .all()
            )

            # Calculer backlog et âge max
            backlog = len(failures)
            now = datetime.now(UTC)

            max_age_seconds = 0
            oldest_task = None
            for failure in failures:
                age = (now - failure.last_seen).total_seconds()
                if age > max_age_seconds:
                    max_age_seconds = age
                    oldest_task = failure

            max_age_minutes = int(max_age_seconds / 60)

            # ✅ A3: Seuils d'alerte
            DLQ_BACKLOG_THRESHOLD = 10
            DLQ_AGE_THRESHOLD_MINUTES = 5
            MAX_EXAMPLES_PER_TASK = 3

            # Grouper par task_name
            by_task = {}
            for failure in failures:
                task_name = failure.task_name or "unknown"
                if task_name not in by_task:
                    by_task[task_name] = {
                        "count": 0,
                        "last_seen": failure.last_seen.isoformat()
                        if failure.last_seen
                        else None,
                        "examples": [],
                    }
                by_task[task_name]["count"] += 1
                if len(by_task[task_name]["examples"]) < MAX_EXAMPLES_PER_TASK:
                    by_task[task_name]["examples"].append(
                        {
                            "task_id": failure.task_id,
                            "exception": failure.exception[:200],
                            "first_seen": failure.first_seen.isoformat()
                            if failure.first_seen
                            else None,
                            "last_seen": failure.last_seen.isoformat()
                            if failure.last_seen
                            else None,
                            "failure_count": failure.failure_count,
                        }
                    )

            # ✅ A3: Alertes
            alerts = []
            if backlog > DLQ_BACKLOG_THRESHOLD:
                alerts.append(
                    {
                        "severity": "critical",
                        "message": f"DLQ backlog élevé: {backlog} tâches en échec",
                        "threshold": DLQ_BACKLOG_THRESHOLD,
                        "value": backlog,
                    }
                )
            if max_age_minutes > DLQ_AGE_THRESHOLD_MINUTES:
                alerts.append(
                    {
                        "severity": "warning",
                        "message": f"Âge max DLQ: {max_age_minutes} minutes (> {DLQ_AGE_THRESHOLD_MINUTES} min)",
                        "threshold": DLQ_AGE_THRESHOLD_MINUTES,
                        "value": max_age_minutes,
                    }
                )

            return {
                "backlog": backlog,
                "max_age_minutes": max_age_minutes,
                "max_age_task": {
                    "task_id": oldest_task.task_id,
                    "task_name": oldest_task.task_name,
                    "last_seen": oldest_task.last_seen.isoformat()
                    if oldest_task and oldest_task.last_seen
                    else None,
                }
                if oldest_task
                else None,
                "by_task_name": by_task,
                "alerts": alerts,
                "timestamp": now.isoformat(),
            }, 200

        except Exception as e:
            logger.exception("[DLQ] Error retrieving DLQ metrics: %s", e)
            return {"error": str(e)}, 500


@dispatch_health_ns.route("/health")
class DispatchHealth(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_health_ns.marshal_with(health_response)
    def get(self):
        """Récupère les métriques de santé du dispatch pour l'entreprise courante."""
        company, err, code = get_company_from_token()
        if err:
            return {"error": err}, code

        company_id = getattr(company, "id", None)

        # Calculer la période des dernières 24h
        now = datetime.now(UTC)
        last_24h = now - timedelta(hours=24)

        # Récupérer les runs des dernières 24h
        runs = (
            db.session.query(DispatchRun)
            .filter(
                DispatchRun.company_id == company_id, DispatchRun.created_at >= last_24h
            )
            .all()
        )

        if not runs:
            return {
                "last_24h_runs": 0,
                "avg_assignment_rate": 0.0,
                "avg_run_time_sec": 0.0,
                "osrm_availability": 1.0,
                "failed_runs": 0,
                "unassigned_reasons": {},
                "performance_trend": "no_data",
            }

        # Calculer les métriques
        total_runs = len(runs)
        successful_runs = [r for r in runs if r.status == "COMPLETED"]
        failed_runs = [r for r in runs if r.status == "FAILED"]

        # Taux d'assignation moyen
        assignment_rates = []
        run_times = []
        osrm_availability_scores = []
        unassigned_reasons = {}

        for run in successful_runs:
            summary = getattr(run, "result_summary", None)
            if summary:
                # Taux d'assignation
                if "assignment_rate" in summary:
                    assignment_rates.append(summary["assignment_rate"])

                # Temps d'exécution
                if "run_time_sec" in summary:
                    run_times.append(summary["run_time_sec"])

                # Disponibilité OSRM
                if "osrm_availability" in summary:
                    osrm_availability_scores.append(summary["osrm_availability"])

                # Raisons de non-assignation
                if "unassigned_reasons" in summary:
                    for _booking_id, reasons in summary["unassigned_reasons"].items():
                        for reason in reasons:
                            unassigned_reasons[reason] = (
                                unassigned_reasons.get(reason, 0) + 1
                            )

        # Calculer les moyennes
        avg_assignment_rate = (
            sum(assignment_rates) / len(assignment_rates) if assignment_rates else 0.0
        )
        avg_run_time_sec = sum(run_times) / len(run_times) if run_times else 0.0
        osrm_availability = (
            sum(osrm_availability_scores) / len(osrm_availability_scores)
            if osrm_availability_scores
            else 1.0
        )

        # Déterminer la tendance de performance
        if len(assignment_rates) >= MIN_RATES_FOR_TREND:
            recent_avg = sum(assignment_rates[-RECENT_RATES_COUNT:]) / min(
                RECENT_RATES_COUNT, len(assignment_rates)
            )
            older_avg = (
                sum(assignment_rates[:-RECENT_RATES_COUNT])
                / max(1, len(assignment_rates) - RECENT_RATES_COUNT)
                if len(assignment_rates) > MIN_COMPARISON_RATES
                else recent_avg
            )

            if recent_avg > older_avg * 1.05:
                performance_trend = "improving"
            elif recent_avg < older_avg * 0.95:
                performance_trend = "declining"
            else:
                performance_trend = "stable"
        else:
            performance_trend = "insufficient_data"

        return {
            "last_24h_runs": total_runs,
            "avg_assignment_rate": round(avg_assignment_rate, 3),
            "avg_run_time_sec": round(avg_run_time_sec, 2),
            "osrm_availability": round(osrm_availability, 3),
            "failed_runs": len(failed_runs),
            "unassigned_reasons": unassigned_reasons,
            "performance_trend": performance_trend,
        }


@dispatch_health_ns.route("/health/trends")
class DispatchTrends(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_health_ns.marshal_with(trends_response)
    def get(self):
        """Récupère les tendances de performance du dispatch sur plusieurs jours."""
        company, err, code = get_company_from_token()
        if err:
            return {"error": err}, code

        company_id = getattr(company, "id", None)

        # Récupérer le paramètre days (défaut: 7)
        days = request.args.get("days", 7, type=int)
        days = min(days, 30)  # Limiter à 30 jours max

        # Calculer la période
        now = datetime.now(UTC)
        start_date = now - timedelta(days=days)

        # Récupérer les runs de la période
        runs = (
            db.session.query(DispatchRun)
            .filter(
                DispatchRun.company_id == company_id,
                DispatchRun.created_at >= start_date,
            )
            .order_by(DispatchRun.created_at)
            .all()
        )

        # Grouper par jour
        daily_data = {}
        for run in runs:
            day_key = run.created_at.date().isoformat()
            if day_key not in daily_data:
                daily_data[day_key] = {
                    "runs": [],
                    "assignment_rates": [],
                    "run_times": [],
                    "osrm_scores": [],
                    "failed_count": 0,
                }

            daily_data[day_key]["runs"].append(run)

            if run.status == "FAILED":
                daily_data[day_key]["failed_count"] += 1
            elif run.status == "COMPLETED":
                summary = getattr(run, "result_summary", None)
                if isinstance(summary, dict):
                    if "assignment_rate" in summary:
                        daily_data[day_key]["assignment_rates"].append(
                            summary["assignment_rate"]
                        )
                    if "run_time_sec" in summary:
                        daily_data[day_key]["run_times"].append(summary["run_time_sec"])
                    if "osrm_availability" in summary:
                        daily_data[day_key]["osrm_scores"].append(
                            summary["osrm_availability"]
                        )

        # Construire les listes de résultats
        days_list = []
        assignment_rates_list = []
        run_times_list = []
        osrm_availability_list = []
        failed_runs_list = []

        # Générer les données pour chaque jour de la période
        current_date = start_date.date()
        end_date = now.date()

        while current_date <= end_date:
            day_key = current_date.isoformat()
            days_list.append(day_key)

            if day_key in daily_data:
                data = daily_data[day_key]

                # Moyennes pour le jour
                avg_assignment_rate = (
                    sum(data["assignment_rates"]) / len(data["assignment_rates"])
                    if data["assignment_rates"]
                    else 0.0
                )
                avg_run_time = (
                    sum(data["run_times"]) / len(data["run_times"])
                    if data["run_times"]
                    else 0.0
                )
                avg_osrm = (
                    sum(data["osrm_scores"]) / len(data["osrm_scores"])
                    if data["osrm_scores"]
                    else 1.0
                )

                assignment_rates_list.append(round(avg_assignment_rate, 3))
                run_times_list.append(round(avg_run_time, 2))
                osrm_availability_list.append(round(avg_osrm, 3))
                failed_runs_list.append(data["failed_count"])
            else:
                # Pas de données pour ce jour
                assignment_rates_list.append(0.0)
                run_times_list.append(0.0)
                osrm_availability_list.append(1.0)
                failed_runs_list.append(0)

            current_date += timedelta(days=1)

        return {
            "days": days_list,
            "assignment_rates": assignment_rates_list,
            "run_times": run_times_list,
            "osrm_availability": osrm_availability_list,
            "failed_runs": failed_runs_list,
        }
