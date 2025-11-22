#!/usr/bin/env python3
"""Routes pour l'affichage et la gestion du Shadow Mode.

Fournit des endpoints REST pour consulter les rapports,
KPIs et métriques du mode shadow.
"""

import os
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, cast

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt, jwt_required
from flask_restx import Api, Namespace, Resource, fields

from ext import redis_client
from services.rl.shadow_mode_manager import ShadowModeManager

# Créer le blueprint
shadow_mode_bp = Blueprint("shadow_mode", __name__, url_prefix="/api/shadow-mode")

# Créer l'API RESTX
api = Api(shadow_mode_bp, doc="/docs/", title="Shadow Mode API")

# Namespace pour les rapports
reports_ns = Namespace("reports", description="Rapports Shadow Mode")
api.add_namespace(reports_ns)

# Namespace pour les KPIs
kpis_ns = Namespace("kpis", description="KPIs Shadow Mode")
api.add_namespace(kpis_ns)

# Modèles de données pour la documentation API
decision_model = api.model(
    "Decision",
    {
        "company_id": fields.String(required=True, description="ID de l'entreprise"),
        "booking_id": fields.String(required=True, description="ID de la réservation"),
        "human_decision": fields.Raw(required=True, description="Décision humaine"),
        "rl_decision": fields.Raw(required=True, description="Décision RL"),
        "context": fields.Raw(description="Contexte de la décision"),
    },
)

daily_report_model = api.model(
    "DailyReport",
    {
        "company_id": fields.String(description="ID de l'entreprise"),
        "date": fields.String(description="Date du rapport"),
        "total_decisions": fields.Integer(description="Nombre total de décisions"),
        "statistics": fields.Raw(description="Statistiques quotidiennes"),
        "kpis_summary": fields.Raw(description="Résumé des KPIs"),
        "top_insights": fields.List(fields.String, description="Insights principaux"),
        "recommendations": fields.List(fields.String, description="Recommandations"),
    },
)

company_summary_model = api.model(
    "CompanySummary",
    {
        "company_id": fields.String(description="ID de l'entreprise"),
        "period_days": fields.Integer(description="Période analysée (jours)"),
        "total_decisions": fields.Integer(description="Total des décisions"),
        "avg_decisions_per_day": fields.Float(description="Moyenne décisions/jour"),
        "avg_agreement_rate": fields.Float(description="Taux d'accord moyen"),
        "avg_eta_improvement": fields.Float(description="Amélioration ETA moyenne"),
        "trend_analysis": fields.Raw(description="Analyse des tendances"),
    },
)

# Initialiser le gestionnaire shadow mode
shadow_manager = ShadowModeManager()


_STATE_KEY = "shadow_mode:active"
_ACTIVE_COUNT_KEY = "shadow_mode:admin_count"
_FALLBACK_STATE = {"active": False, "count": 0}


def _get_state_from_store() -> bool:
    """Lit l'état courant (Redis si dispo, sinon fallback mémoire)."""
    if redis_client:
        try:
            value = redis_client.get(_STATE_KEY)
            if value is None:
                return False
            return value.decode("utf-8") == "1"
        except Exception:
            return bool(_FALLBACK_STATE["active"])
    return bool(_FALLBACK_STATE["active"])


def _set_state_in_store(active: bool) -> None:
    """Persiste l'état (Redis si dispo, sinon fallback mémoire)."""
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
    """Structure par défaut renvoyée lorsque le Shadow Mode est inactif."""
    return {
        "agreement_rate": 0.0,
        "comparisons_count": 0,
        "predictions_count": 0,
        "disagreements_count": 0,
        "high_confidence_disagreements": 0,
        "last_event_at": None,
    }


@shadow_mode_bp.route("/status", methods=["GET"])
def get_shadow_mode_status():
    """Retourne l'état global du Shadow Mode.

    Cette route est consommée par le dashboard admin pour déterminer si des
    statistiques doivent être affichées. Tant que le backend n'a pas encore
    branché le Shadow Mode, on renvoie un état « inactive » mais avec un code 200.
    """

    enabled = _shadow_mode_enabled()
    session_stats = _session_placeholder()

    payload: Dict[str, Any] = {
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

    return jsonify(payload), 200


@shadow_mode_bp.route("/stats", methods=["GET"])
def get_shadow_mode_stats():
    """Retourne les statistiques de session Shadow Mode.

    Tant que le moteur RL n'alimente pas ces métriques, on renvoie une structure
    vide mais cohérente afin que le frontend reste fonctionnel.
    """

    enabled = _shadow_mode_enabled()
    session_stats = _session_placeholder()

    payload: Dict[str, Any] = {
        "session_stats": session_stats,
        "status": "active" if enabled else "inactive",
        "last_updated": datetime.now(UTC).isoformat(),
    }

    return jsonify(payload), 200


@shadow_mode_bp.route("/predictions", methods=["GET"])
def get_shadow_mode_predictions():
    """Retourne la liste des dernières prédictions RL.

    Placeholder pour intégration future : renvoie une liste vide mais la route
    existe pour éviter les 404 côté front.
    """

    return jsonify({"predictions": [], "count": 0}), 200


@shadow_mode_bp.route("/comparisons", methods=["GET"])
def get_shadow_mode_comparisons():
    """Retourne les comparaisons humain vs RL les plus récentes.

    Placeholder renvoyant un tableau vide afin d'éviter les erreurs front.
    """

    return jsonify({"comparisons": [], "count": 0}), 200


@shadow_mode_bp.route("/session", methods=["POST", "DELETE"])
@jwt_required()
def toggle_shadow_mode_session():
    """Active/désactive le Shadow Mode selon la session admin."""

    claims = get_jwt() or {}
    role = str(claims.get("role", "")).upper()

    if role != "ADMIN":
        return jsonify({"error": "Accès réservé aux administrateurs."}), 403

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


@reports_ns.route("/daily/<string:company_id>")
class DailyReport(Resource):
    """Endpoint pour les rapports quotidiens."""

    @reports_ns.doc("get_daily_report")
    @reports_ns.marshal_with(daily_report_model)
    def get(self, company_id: str):
        """Récupère le rapport quotidien pour une entreprise.

        Query Parameters:
            date: Date du rapport (format YYYY-MM-DD, par défaut aujourd'hui)
        """
        try:
            # Récupérer la date depuis les paramètres
            date_str = request.args.get("date")
            date = (
                datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC).date()
                if date_str
                else datetime.now(UTC).date()
            )

            # Générer le rapport
            report = shadow_manager.generate_daily_report(company_id, date)

            return report, 200

        except ValueError as e:
            return {"error": f"Format de date invalide: {e}"}, 400
        except Exception as e:
            return {"error": f"Erreur lors de la génération du rapport: {e}"}, 500

    @reports_ns.doc("log_decision")
    @reports_ns.expect(decision_model)
    def post(self, company_id: str):
        """Enregistre une nouvelle décision pour comparaison."""
        try:
            data = request.get_json()

            # Valider les données requises
            required_fields = ["booking_id", "human_decision", "rl_decision"]
            for field in required_fields:
                if field not in data:
                    return {"error": f"Champ requis manquant: {field}"}, 400

            # Enregistrer la décision
            kpis = shadow_manager.log_decision_comparison(
                company_id=company_id,
                booking_id=data["booking_id"],
                human_decision=data["human_decision"],
                rl_decision=data["rl_decision"],
                context=data.get("context", {}),
            )

            return {"message": "Décision enregistrée avec succès", "kpis": kpis}, 201

        except Exception as e:
            return {"error": f"Erreur lors de l'enregistrement: {e}"}, 500


@reports_ns.route("/summary/<string:company_id>")
class CompanySummary(Resource):
    """Endpoint pour les résumés d'entreprise."""

    @reports_ns.doc("get_company_summary")
    @reports_ns.marshal_with(company_summary_model)
    def get(self, company_id: str):
        """Récupère un résumé multi-jours pour une entreprise.

        Query Parameters:
            days: Nombre de jours à analyser (défaut: 7)
        """
        try:
            # Récupérer le nombre de jours
            days = int(request.args.get("days", 7))

            # Générer le résumé
            summary = shadow_manager.get_company_summary(company_id, days)

            return summary, 200

        except ValueError as e:
            return {"error": f"Paramètre invalide: {e}"}, 400
        except Exception as e:
            return {"error": f"Erreur lors de la génération du résumé: {e}"}, 500


@kpis_ns.route("/metrics/<string:company_id>")
class KPIMetrics(Resource):
    """Endpoint pour les métriques KPIs."""

    @kpis_ns.doc("get_kpi_metrics")
    def get(self, company_id: str):
        """Récupère les métriques KPIs détaillées pour une entreprise.

        Query Parameters:
            days: Nombre de jours à analyser (défaut: 7)
            metric: Métrique spécifique à récupérer
        """
        try:
            days = int(request.args.get("days", 7))
            metric = request.args.get("metric")

            # Générer le résumé pour obtenir les données
            summary = shadow_manager.get_company_summary(company_id, days)

            if summary.get("total_decisions", 0) == 0:
                return {
                    "company_id": company_id,
                    "message": "Aucune donnée disponible pour cette période",
                }, 200

            # Filtrer par métrique si spécifiée
            if metric:
                # Récupérer les données brutes pour cette métrique
                end_date = datetime.now(UTC).date()
                start_date = end_date - timedelta(days=days - 1)

                metric_data = []
                for i in range(days):
                    date = start_date + timedelta(days=i)
                    company_data = shadow_manager._filter_data_by_company_and_date(
                        company_id, date
                    )

                    for kpi in company_data["kpis"]:
                        if metric in kpi:
                            metric_data.append(
                                {"date": date.isoformat(), "value": kpi[metric]}
                            )

                return {
                    "company_id": company_id,
                    "metric": metric,
                    "period_days": days,
                    "data": metric_data,
                }, 200

            # Retourner toutes les métriques
            return {
                "company_id": company_id,
                "period_days": days,
                "summary": summary,
                "available_metrics": list(shadow_manager.kpi_metrics.keys()),
            }, 200

        except ValueError as e:
            return {"error": f"Paramètre invalide: {e}"}, 400
        except Exception as e:
            return {"error": f"Erreur lors de la récupération des métriques: {e}"}, 500


@kpis_ns.route("/export/<string:company_id>")
class ExportData(Resource):
    """Endpoint pour l'export des données."""

    @kpis_ns.doc("export_company_data")
    def get(self, company_id: str):
        """Exporte les données d'une entreprise en CSV/JSON.

        Query Parameters:
            format: Format d'export (csv, json, both) - défaut: json
            days: Nombre de jours à exporter (défaut: 7)
        """
        try:
            export_format = request.args.get("format", "json")
            days = int(request.args.get("days", 7))

            # Générer les rapports pour la période
            end_date = datetime.now(UTC).date()
            start_date = end_date - timedelta(days=days - 1)

            reports = []
            for i in range(days):
                date = start_date + timedelta(days=i)
                report = shadow_manager.generate_daily_report(company_id, date)
                if report.get("total_decisions", 0) > 0:
                    reports.append(report)

            if not reports:
                return {"message": "Aucune donnée à exporter pour cette période"}, 200

            # Préparer la réponse selon le format
            if export_format == "csv":
                # Convertir en format CSV-friendly
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

                return {
                    "format": "csv",
                    "data": csv_data,
                    "message": "Données prêtes pour conversion CSV",
                }, 200

            if export_format == "both":
                return {
                    "format": "both",
                    "reports": reports,
                    "csv_data": self._prepare_csv_data(reports),
                    "message": "Données exportées en JSON et CSV",
                }, 200

            # json
            return {
                "format": "json",
                "reports": reports,
                "total_reports": len(reports),
                "message": "Données exportées en JSON",
            }, 200

        except ValueError as e:
            return {"error": f"Paramètre invalide: {e}"}, 400
        except Exception as e:
            return {"error": f"Erreur lors de l'export: {e}"}, 500

    def _prepare_csv_data(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prépare les données pour l'export CSV."""
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
                    "eta_improvement_rate": report.get("kpis_summary", {}).get(
                        "eta_improvement_rate", 0
                    ),
                    "violation_rate": report.get("kpis_summary", {}).get(
                        "violation_rate", 0
                    ),
                }
            )
        return csv_data


@shadow_mode_bp.route("/health")
def health_check():
    """Endpoint de santé pour le shadow mode."""
    return jsonify(
        {
            "status": "healthy",
            "service": "shadow_mode",
            "timestamp": datetime.now(UTC).isoformat(),
            "data_dir": str(shadow_manager.data_dir),
            "total_decisions": len(shadow_manager.decision_metadata["timestamp"]),
        }
    )


@shadow_mode_bp.route("/companies")
def list_companies():
    """Liste toutes les entreprises avec des données shadow mode."""
    try:
        # Récupérer toutes les entreprises depuis les métadonnées
        companies = list(set(shadow_manager.decision_metadata["company_id"]))

        # Ajouter des statistiques pour chaque entreprise
        company_stats = []
        for company_id in companies:
            summary = shadow_manager.get_company_summary(company_id, 7)
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
