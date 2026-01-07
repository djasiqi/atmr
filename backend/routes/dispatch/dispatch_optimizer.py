# backend/routes/dispatch/dispatch_optimizer.py
"""Endpoints pour l'optimiseur et l'agent de dispatch."""

# ruff: noqa: I001  # Imports organisés manuellement pour meilleure lisibilité
import logging

from flask import current_app, request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import Resource  # pyright: ignore[reportMissingImports]
from http import HTTPStatus

from ext import role_required
from models.enums import UserRole
from routes.dispatch import dispatch_ns
from routes.dispatch.dispatch_helpers import _current_company_id, _get_current_company
from shared.error_handlers import APIErrorHandler
from shared.time_utils import now_local

logger = logging.getLogger(__name__)

# ===== Routes Optimizer =====


@dispatch_ns.route("/optimizer/start")
class OptimizerStartResource(Resource):
    """Démarre le monitoring en temps réel pour l'entreprise."""

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Démarre le monitoring en temps réel pour l'entreprise.
        Surveille automatiquement les retards et propose des optimisations.
        """
        company_id = _current_company_id()

        body = request.get_json(silent=True) or {}
        check_interval = int(
            body.get("check_interval_seconds", 120)
        )  # 2 min par défaut

        try:
            from infrastructure.dispatch.realtime_optimizer_adapter import (
                start_optimizer_for_company,
            )

            optimizer = start_optimizer_for_company(company_id, check_interval)
            status = optimizer.get_status()

            return {
                "message": "Monitoring temps réel démarré",
                "status": status,
            }, HTTPStatus.OK

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/optimizer/stop")
class OptimizerStopResource(Resource):
    """Arrête le monitoring en temps réel pour l'entreprise."""

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Arrête le monitoring en temps réel pour l'entreprise.

        ⚠️ En mode fully_auto, l'optimiseur ne peut pas être arrêté manuellement.
        Changez le mode de dispatch pour arrêter l'optimiseur.
        """
        company_id = _current_company_id()
        company = _get_current_company()

        try:
            # ✅ Empêcher l'arrêt si l'entreprise est en mode fully_auto
            current_mode = (
                getattr(company.dispatch_mode, "value", None)
                if hasattr(company, "dispatch_mode")
                else None
            )

            if current_mode == "fully_auto":
                logger.warning(
                    "[Optimizer] Tentative d'arrêt refusée pour company %s (mode fully_auto actif)",
                    company_id,
                )
                return {
                    "success": False,
                    "error": (
                        "Impossible d'arrêter l'optimiseur en mode fully_auto. "
                        "Changez le mode de dispatch pour arrêter l'optimiseur."
                    ),
                    "current_mode": current_mode,
                }, HTTPStatus.FORBIDDEN

            from infrastructure.dispatch.realtime_optimizer_adapter import (
                stop_optimizer_for_company,
            )

            stop_optimizer_for_company(company_id)

            return {
                "message": "Monitoring temps réel arrêté",
                "company_id": company_id,
            }, HTTPStatus.OK

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/optimizer/status")
class OptimizerStatusResource(Resource):
    """Récupère le statut du monitoring temps réel."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère le statut du monitoring temps réel."""
        company_id = _current_company_id()

        try:
            from infrastructure.dispatch.realtime_optimizer_adapter import (
                get_optimizer_for_company,
            )

            optimizer = get_optimizer_for_company(company_id)

            if optimizer is None:
                return {
                    "running": False,
                    "company_id": company_id,
                    "message": "Monitoring non démarré",
                }, HTTPStatus.OK

            status = optimizer.get_status()
            return status, HTTPStatus.OK

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/optimizer/opportunities")
class OptimizerOpportunitiesResource(Resource):
    """Récupère les opportunités d'optimisation détectées."""

    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(params={"date": "YYYY-MM-DD (optionnel, défaut: aujourd'hui)"})
    def get(self):
        """Récupère les opportunités d'optimisation détectées.
        Mode manuel: lance une vérification à la demande.
        """
        company_id = _current_company_id()

        date_str = request.args.get("date")

        try:
            from infrastructure.dispatch.realtime_optimizer_adapter import (
                check_opportunities_manual,
                get_optimizer_for_company,
            )

            # Vérifier si un optimizer est actif
            optimizer = get_optimizer_for_company(company_id)

            if optimizer and optimizer.get_status()["running"]:
                # Utiliser le cache du monitoring actif
                opportunities = optimizer.get_current_opportunities()
            else:
                # Vérification manuelle
                opportunities = check_opportunities_manual(company_id, date_str)

            return {
                "opportunities": [o.to_dict() for o in opportunities],
                "count": len(opportunities),
                "critical_count": len(
                    [o for o in opportunities if o.severity == "critical"]
                ),
                "high_count": len([o for o in opportunities if o.severity == "high"]),
                "timestamp": now_local().isoformat(),
            }, HTTPStatus.OK

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


# ===== Routes Agent Dispatch Intelligent =====


@dispatch_ns.route("/agent/start")
class AgentStartResource(Resource):
    """Démarre l'agent dispatch intelligent."""

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Démarre l'agent dispatch pour l'entreprise."""
        company_id = _current_company_id()

        try:
            from services.dispatch.agent.orchestrator import get_agent_for_company

            agent = get_agent_for_company(
                company_id,
                app=current_app._get_current_object(),
            )
            agent.start()

            status = agent.get_status()
            return {
                "success": True,
                "message": "Agent démarré",
                "status": status,
            }, HTTPStatus.OK

        except Exception as e:
            logger.exception("[Agent] Failed to start for company %s", company_id)
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/agent/stop")
class AgentStopResource(Resource):
    """Arrête l'agent dispatch intelligent."""

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Arrête l'agent dispatch pour l'entreprise.

        ⚠️ En mode fully_auto, l'agent ne peut pas être arrêté manuellement.
        Changez le mode de dispatch pour arrêter l'agent.
        """
        company_id = _current_company_id()
        company = _get_current_company()

        try:
            # ✅ Empêcher l'arrêt si l'entreprise est en mode fully_auto
            current_mode = (
                getattr(company.dispatch_mode, "value", None)
                if hasattr(company, "dispatch_mode")
                else None
            )

            if current_mode == "fully_auto":
                logger.warning(
                    "[Agent] Tentative d'arrêt refusée pour company %s (mode fully_auto actif)",
                    company_id,
                )
                return {
                    "success": False,
                    "error": (
                        "Impossible d'arrêter l'agent en mode fully_auto. "
                        "Changez le mode de dispatch pour arrêter l'agent."
                    ),
                    "current_mode": current_mode,
                }, HTTPStatus.FORBIDDEN

            from services.dispatch.agent.orchestrator import stop_agent_for_company

            stop_agent_for_company(company_id)

            return {
                "success": True,
                "message": "Agent arrêté",
            }, HTTPStatus.OK

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/agent/status")
class AgentStatusResource(Resource):
    """Récupère le statut de l'agent dispatch intelligent."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère le statut de l'agent."""
        company_id = _current_company_id()

        try:
            from services.dispatch.agent.orchestrator import get_agent_for_company

            agent = get_agent_for_company(company_id)
            status = agent.get_status()
            return status, HTTPStatus.OK

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)
