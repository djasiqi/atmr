#!/usr/bin/env python3
"""Routes pour l'affichage et la gestion du Shadow Mode (F-05).

Endpoints protégés JWT / tenant / IP — sans Swagger public.
"""

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, cast

from flask import Blueprint, jsonify, request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]

from ext import redis_client, role_required
from models import UserRole
from security.ip_whitelist import ip_whitelist_required
from services.ml.rl.shadow_mode_manager import ShadowModeManager
from shared.error_handlers import APIErrorHandler
from shared.tenant_guard import assert_company_access

logger = logging.getLogger(__name__)

shadow_mode_bp = Blueprint("shadow_mode", __name__, url_prefix="/api/shadow-mode")

_shadow_manager_cache: dict[str, ShadowModeManager] = {}


def get_shadow_manager() -> ShadowModeManager:
    """Récupère le gestionnaire shadow mode (lazy initialization)."""
    if "manager" not in _shadow_manager_cache:
        data_dir = os.getenv("RL_SHADOW_MODE_DIR", "/app/data/rl/shadow_mode")
        _shadow_manager_cache["manager"] = ShadowModeManager(data_dir=data_dir)
    return _shadow_manager_cache["manager"]


_STATE_KEY = "shadow_mode:active"
_ACTIVE_COUNT_KEY = "shadow_mode:admin_count"
_FALLBACK_STATE = {"active": False, "count": 0}


def _get_state_from_store() -> bool:
    if redis_client:
        try:
            value = redis_client.get(_STATE_KEY)
            if value is None:
                return False
            return cast(bytes, value).decode("utf-8") == "1"
        except Exception:
            return bool(_FALLBACK_STATE["active"])
    return bool(_FALLBACK_STATE["active"])


def _set_state_in_store(active: bool) -> None:
    if redis_client:
        try:
            redis_client.set(_STATE_KEY, "1" if active else "0")
            _FALLBACK_STATE["active"] = active
            return
        except Exception:
            _FALLBACK_STATE["active"] = active
            return
    _FALLBACK_STATE["active"] = active


def _get_count_from_store() -> int:
    if redis_client:
        try:
            value = cast(Any, redis_client.get(_ACTIVE_COUNT_KEY))
            if value is None:
                return 0
            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8", "ignore") or "0"
            return int(value)
        except Exception:
            return int(_FALLBACK_STATE["count"])
    return int(_FALLBACK_STATE["count"])


def _set_count_in_store(count: int) -> None:
    count = max(count, 0)
    if redis_client:
        try:
            redis_client.set(_ACTIVE_COUNT_KEY, str(count))
            _FALLBACK_STATE["count"] = count
            return
        except Exception:
            _FALLBACK_STATE["count"] = count
            return
    _FALLBACK_STATE["count"] = count


def _shadow_mode_enabled() -> bool:
    env_override = os.getenv("SHADOW_MODE_ENABLED")
    if env_override is not None:
        return env_override.lower() in {"1", "true", "yes", "on"}
    return _get_state_from_store()


def _session_placeholder() -> Dict[str, Any]:
    return {
        "agreement_rate": 0.0,
        "comparisons_count": 0,
        "predictions_count": 0,
        "disagreements_count": 0,
        "high_confidence_disagreements": 0,
        "last_event_at": None,
    }


def _prepare_csv_data(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    csv_data = []
    for report in reports:
        csv_data.append(
            {
                "company_id": report["company_id"],
                "date": report["date"],
                "total_decisions": report["total_decisions"],
                "agreement_rate": report.get("statistics", {}).get("agreement_rate", 0),
                "avg_eta_delta": report.get("statistics", {})
                .get("eta_delta", {})
                .get("mean", 0),
                "avg_delay_delta": report.get("statistics", {})
                .get("delay_delta", {})
                .get("mean", 0),
                "rl_confidence": report.get("statistics", {})
                .get("rl_confidence", {})
                .get("mean", 0),
                "eta_improvement_rate": report.get("kpis_summary", {}).get(
                    "eta_improvement_rate", 0
                ),
                "violation_rate": report.get("kpis_summary", {}).get(
                    "violation_rate", 0
                ),
            }
        )
    return csv_data


@shadow_mode_bp.route("/status", methods=["GET"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def get_shadow_mode_status():
    enabled = _shadow_mode_enabled()
    session_stats = _session_placeholder()
    return jsonify(
        {
            "status": "active" if enabled else "inactive",
            "message": (
                "Shadow Mode actif – données disponibles"
                if enabled
                else "Shadow Mode non activé dans l'environnement courant"
            ),
            "last_updated": datetime.now(UTC).isoformat(),
            "comparisons_count": session_stats["comparisons_count"],
            "predictions_count": session_stats["predictions_count"],
        }
    ), 200


@shadow_mode_bp.route("/stats", methods=["GET"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def get_shadow_mode_stats():
    enabled = _shadow_mode_enabled()
    return jsonify(
        {
            "session_stats": _session_placeholder(),
            "status": "active" if enabled else "inactive",
            "last_updated": datetime.now(UTC).isoformat(),
        }
    ), 200


@shadow_mode_bp.route("/predictions", methods=["GET"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def get_shadow_mode_predictions():
    return jsonify({"predictions": [], "count": 0}), 200


@shadow_mode_bp.route("/comparisons", methods=["GET"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def get_shadow_mode_comparisons():
    return jsonify({"comparisons": [], "count": 0}), 200


@shadow_mode_bp.route("/session", methods=["POST", "DELETE"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def toggle_shadow_mode_session():
    if request.method == "POST":
        current = _get_count_from_store()
        new_count = current + 1
        _set_count_in_store(new_count)
        _set_state_in_store(new_count > 0)
        return jsonify({"status": "activated", "active": True, "count": new_count}), 200

    current = _get_count_from_store()
    new_count = max(current - 1, 0)
    _set_count_in_store(new_count)
    _set_state_in_store(new_count > 0)
    return jsonify(
        {"status": "deactivated", "active": new_count > 0, "count": new_count}
    ), 200


@shadow_mode_bp.route("/reports/daily/<int:company_id>", methods=["GET"])
@jwt_required()
@role_required(["ADMIN", "COMPANY"])
def get_daily_report(company_id: int):
    """Rapport quotidien — lecture seule (build, sans persist)."""
    try:
        _user, access_err = assert_company_access(
            company_id, resource="shadow_daily_report"
        )
        if access_err:
            return access_err

        company_key = str(company_id)
        date_str = request.args.get("date")
        date = (
            datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC).date()
            if date_str
            else datetime.now(UTC).date()
        )
        report = get_shadow_manager().build_daily_report(company_key, date)
        return jsonify(report), 200
    except ValueError as e:
        return APIErrorHandler.handle_validation_error(
            f"Format de date invalide: {e}",
            field="date",
            logger_instance=logger,
        )
    except Exception as e:
        return APIErrorHandler.handle_exception(e, logger)


@shadow_mode_bp.route("/reports/daily/<int:company_id>", methods=["POST"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def post_daily_decision(company_id: int):
    """Enregistre une décision humain/RL (log_decision_comparison)."""
    try:
        _user, access_err = assert_company_access(
            company_id, resource="shadow_log_decision"
        )
        if access_err:
            return access_err

        company_key = str(company_id)
        data = request.get_json()
        required_fields = ["booking_id", "human_decision", "rl_decision"]
        for field in required_fields:
            if not data or field not in data:
                return APIErrorHandler.handle_validation_error(
                    f"Champ requis manquant: {field}",
                    field=field,
                    logger_instance=logger,
                )

        kpis = get_shadow_manager().log_decision_comparison(
            company_id=company_key,
            booking_id=data["booking_id"],
            human_decision=data["human_decision"],
            rl_decision=data["rl_decision"],
            context=data.get("context", {}),
        )
        return jsonify(
            {"message": "Décision enregistrée avec succès", "kpis": kpis}
        ), 201
    except Exception as e:
        return APIErrorHandler.handle_exception(e, logger)


@shadow_mode_bp.route("/reports/summary/<int:company_id>", methods=["GET"])
@jwt_required()
@role_required(["ADMIN", "COMPANY"])
def get_company_summary_route(company_id: int):
    try:
        _user, access_err = assert_company_access(
            company_id, resource="shadow_company_summary"
        )
        if access_err:
            return access_err

        company_key = str(company_id)
        days = int(request.args.get("days", 7))
        summary = get_shadow_manager().get_company_summary(company_key, days)
        return jsonify(summary), 200
    except ValueError as e:
        return APIErrorHandler.handle_validation_error(
            f"Paramètre invalide: {e}",
            logger_instance=logger,
        )
    except Exception as e:
        return APIErrorHandler.handle_exception(e, logger)


@shadow_mode_bp.route("/kpis/metrics/<int:company_id>", methods=["GET"])
@jwt_required()
@role_required(["ADMIN", "COMPANY"])
def get_kpi_metrics(company_id: int):
    try:
        _user, access_err = assert_company_access(
            company_id, resource="shadow_kpi_metrics"
        )
        if access_err:
            return access_err

        company_key = str(company_id)
        days = int(request.args.get("days", 7))
        metric = request.args.get("metric")
        summary = get_shadow_manager().get_company_summary(company_key, days)

        if summary.get("total_decisions", 0) == 0:
            return jsonify(
                {
                    "company_id": company_key,
                    "message": "Aucune donnée disponible pour cette période",
                }
            ), 200

        if metric:
            end_date = datetime.now(UTC).date()
            start_date = end_date - timedelta(days=days - 1)
            metric_data = []
            for i in range(days):
                date = start_date + timedelta(days=i)
                company_data = get_shadow_manager()._filter_data_by_company_and_date(
                    company_key, date
                )
                for kpi in company_data["kpis"]:
                    if metric in kpi:
                        metric_data.append(
                            {"date": date.isoformat(), "value": kpi[metric]}
                        )
            return jsonify(
                {
                    "company_id": company_key,
                    "metric": metric,
                    "period_days": days,
                    "data": metric_data,
                }
            ), 200

        return jsonify(
            {
                "company_id": company_key,
                "period_days": days,
                "summary": summary,
                "available_metrics": list(get_shadow_manager().kpi_metrics.keys()),
            }
        ), 200
    except ValueError as e:
        return APIErrorHandler.handle_validation_error(
            f"Paramètre invalide: {e}",
            logger_instance=logger,
        )
    except Exception as e:
        return APIErrorHandler.handle_exception(e, logger)


@shadow_mode_bp.route("/kpis/export/<int:company_id>", methods=["GET"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def export_company_data(company_id: int):
    try:
        _user, access_err = assert_company_access(
            company_id, resource="shadow_kpi_export"
        )
        if access_err:
            return access_err

        company_key = str(company_id)
        export_format = request.args.get("format", "json")
        days = int(request.args.get("days", 7))
        end_date = datetime.now(UTC).date()
        start_date = end_date - timedelta(days=days - 1)

        reports = []
        for i in range(days):
            date = start_date + timedelta(days=i)
            report = get_shadow_manager().build_daily_report(company_key, date)
            if report.get("total_decisions", 0) > 0:
                reports.append(report)

        if not reports:
            return jsonify(
                {"message": "Aucune donnée à exporter pour cette période"}
            ), 200

        if export_format == "csv":
            csv_data = []
            for report in reports:
                csv_data.append(
                    {
                        "company_id": report["company_id"],
                        "date": report["date"],
                        "total_decisions": report["total_decisions"],
                        "agreement_rate": report.get("statistics", {}).get(
                            "agreement_rate", 0
                        ),
                        "avg_eta_delta": report.get("statistics", {})
                        .get("eta_delta", {})
                        .get("mean", 0),
                        "avg_delay_delta": report.get("statistics", {})
                        .get("delay_delta", {})
                        .get("mean", 0),
                        "rl_confidence": report.get("statistics", {})
                        .get("rl_confidence", {})
                        .get("mean", 0),
                    }
                )
            return jsonify(
                {
                    "format": "csv",
                    "data": csv_data,
                    "message": "Données prêtes pour conversion CSV",
                }
            ), 200

        if export_format == "both":
            return jsonify(
                {
                    "format": "both",
                    "reports": reports,
                    "csv_data": _prepare_csv_data(reports),
                    "message": "Données exportées en JSON et CSV",
                }
            ), 200

        return jsonify(
            {
                "format": "json",
                "reports": reports,
                "total_reports": len(reports),
                "message": "Données exportées en JSON",
            }
        ), 200
    except ValueError as e:
        return APIErrorHandler.handle_validation_error(
            f"Paramètre invalide: {e}",
            logger_instance=logger,
        )
    except Exception as e:
        return APIErrorHandler.handle_exception(e, logger)


@shadow_mode_bp.route("/health")
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "service": "shadow_mode",
            "timestamp": datetime.now(UTC).isoformat(),
            "data_dir": str(get_shadow_manager().data_dir),
            "total_decisions": len(get_shadow_manager().decision_metadata["timestamp"]),
        }
    )


@shadow_mode_bp.route("/companies")
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def list_companies():
    try:
        companies = list(set(get_shadow_manager().decision_metadata["company_id"]))
        company_stats = []
        for company_id in companies:
            summary = get_shadow_manager().get_company_summary(company_id, 7)
            company_stats.append(
                {
                    "company_id": company_id,
                    "total_decisions_7d": summary.get("total_decisions", 0),
                    "avg_agreement_rate": summary.get("avg_agreement_rate", 0),
                    "avg_eta_improvement": summary.get("avg_eta_improvement", 0),
                }
            )
        return jsonify(
            {"companies": company_stats, "total_companies": len(companies)}
        ), 200
    except Exception as e:
        return jsonify(
            {"error": f"Erreur lors de la récupération des entreprises: {e}"}
        ), 500


def register_shadow_mode_routes(app):
    """Enregistre les routes shadow mode avec l'application Flask."""
    app.register_blueprint(shadow_mode_bp)
    return shadow_mode_bp
