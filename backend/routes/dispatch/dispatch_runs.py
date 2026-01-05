# backend/routes/dispatch/dispatch_runs.py
"""Endpoints pour la gestion des runs de dispatch."""

import logging
from contextlib import suppress

from flask import request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import Resource  # pyright: ignore[reportMissingImports]

from ext import db, role_required
from models import DispatchRun, UserRole
from repositories.assignment_repository import AssignmentRepository
from repositories.dispatch_run_repository import DispatchRunRepository
from routes.dispatch import dispatch_ns
from routes.dispatch.dispatch_helpers import _get_current_company
from routes.dispatch.dispatch_schemas import (
    dispatch_run_detail_model,
    dispatch_run_model,
)
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)

# Initialisation des repositories
dispatch_run_repo = DispatchRunRepository()
assignment_repo = AssignmentRepository()


@dispatch_ns.route("/runs")
class RunsListResource(Resource):
    """Historique des runs de dispatch."""

    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.doc(params={"limit": "int", "offset": "int"})
    @dispatch_ns.marshal_list_with(dispatch_run_model)
    def get(self):
        """Historique des runs (reverse chrono)."""
        try:
            limit = int(request.args.get("limit", 50))
            offset = int(request.args.get("offset", 0))
            company = _get_current_company()
            return dispatch_run_repo.find_models_by_company_with_custom_order(
                company_id=company.id, limit=limit, offset=offset
            )

        except Exception as e:
            logger.exception("Erreur récupération liste runs: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@dispatch_ns.route("/runs/<int:run_id>")
class RunResource(Resource):
    """Détail d'un run de dispatch avec ses assignations."""

    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.marshal_with(dispatch_run_detail_model)
    def get(self, run_id: int):
        """Détail d'un run + ses assignations."""
        try:
            # S'assure que la session n'est pas en état "aborted"
            with suppress(Exception):
                db.session.rollback()

            company = _get_current_company()

            r_opt: DispatchRun | None = dispatch_run_repo.find_model_by_id_and_company(
                run_id, company.id
            )
            if r_opt is None:
                raise APIErrorHandler.not_found(message="Dispatch run not found")

            r = r_opt

            # ✅ P1: Eager loading pour éviter N+1 queries
            assigns = assignment_repo.find_models_by_dispatch_run_with_eager_loading(
                dispatch_run_id=run_id, company_id=company.id
            )

            return {
                "id": r.id,
                "company_id": r.company_id,
                "day": str(getattr(r, "day", "")),
                "created_at": getattr(r, "created_at", None),
                "started_at": getattr(r, "started_at", None),
                "completed_at": getattr(r, "completed_at", None),
                "status": getattr(r, "status", None),
                "meta": getattr(r, "meta", {}),
                "assignments": assigns,
            }

        except Exception as e:
            logger.exception("Erreur récupération run id=%s: %s", run_id, e)
            return APIErrorHandler.handle_exception(e, logger)
