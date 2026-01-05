# backend/routes/dispatch/dispatch_settings.py
"""Endpoints pour la gestion des paramètres de dispatch."""

# ruff: noqa: I001  # Imports organisés manuellement pour meilleure lisibilité
import json
import logging
from contextlib import suppress
from typing import Any, cast

from flask import request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import Resource  # pyright: ignore[reportMissingImports]
from http import HTTPStatus

from ext import db, role_required
from models.enums import UserRole
from routes.dispatch import dispatch_ns
from routes.dispatch.dispatch_helpers import _current_company_id, _get_current_company
from routes.dispatch.dispatch_schemas import autorun_model
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)


@dispatch_ns.route("/settings/validate")
class DispatchSettingsValidate(Resource):
    """Valide des overrides de settings avant application."""

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Valide des overrides de settings avant application.

        Accepte un payload avec 'overrides' et retourne :
        - applied: paramètres qui seront appliqués
        - ignored: paramètres ignorés (inconnus ou non applicables)
        - errors: erreurs de validation
        """
        from infrastructure.dispatch import settings_module_adapter as ud_settings

        try:
            company = _get_current_company()
            body = request.get_json(silent=True) or {}
            overrides = body.get("overrides", {})

            if not overrides:
                return {
                    "valid": True,
                    "applied": [],
                    "ignored": [],
                    "errors": [],
                    "message": "Aucun override fourni",
                }, HTTPStatus.OK

            # Créer settings de base pour la company
            base_settings = ud_settings.for_company(company)

            # Tenter le merge pour valider
            # Note: on n'utilise pas strict_validation ici pour ne pas bloquer
            # mais on retourne les erreurs dans la réponse
            new_settings = ud_settings.merge_overrides(base_settings, overrides)

            # Récupérer le résultat de validation depuis les logs
            # (On pourrait améliorer merge_overrides pour retourner
            # le résultat de validation)
            # Pour l'instant, on vérifie manuellement les paramètres critiques
            validation_result = {
                "applied": [],
                "ignored": [],
                "errors": [],
            }

            # Vérifier les paramètres appliqués
            if "heuristic" in overrides:
                h_ov = overrides["heuristic"]
                if isinstance(h_ov, dict):
                    if "driver_load_balance" in h_ov:
                        if (
                            new_settings.heuristic.driver_load_balance
                            == h_ov["driver_load_balance"]
                        ):
                            validation_result["applied"].append(
                                "heuristic.driver_load_balance"
                            )
                        else:
                            validation_result["errors"].append(
                                "heuristic.driver_load_balance "
                                + f"demandé={h_ov['driver_load_balance']} "
                                + "mais appliqué="
                                + f"{new_settings.heuristic.driver_load_balance}"
                            )
                    if "proximity" in h_ov:
                        if new_settings.heuristic.proximity == h_ov["proximity"]:
                            validation_result["applied"].append("heuristic.proximity")
                        else:
                            validation_result["errors"].append(
                                f"heuristic.proximity demandé={h_ov['proximity']} "
                                + f"mais appliqué={new_settings.heuristic.proximity}"
                            )

            if "fairness" in overrides:
                f_ov = overrides["fairness"]
                if isinstance(f_ov, dict) and "fairness_weight" in f_ov:
                    if new_settings.fairness.fairness_weight == f_ov["fairness_weight"]:
                        validation_result["applied"].append("fairness.fairness_weight")
                    else:
                        validation_result["errors"].append(
                            "fairness.fairness_weight "
                            + f"demandé={f_ov['fairness_weight']} "
                            + "mais appliqué="
                            + f"{new_settings.fairness.fairness_weight}"
                        )

            # Identifier les clés ignorées (non dans Settings)
            known_ignored_keys = [
                "preferred_driver_id",
                "mode",
                "run_async",
                "reset_existing",
                "fast_mode",
            ]
            for key in overrides:
                if (
                    key
                    not in [
                        "heuristic",
                        "solver",
                        "fairness",
                        "features",
                        "time",
                        "service_times",
                        "pooling",
                        "realtime",
                        "emergency",
                        "matrix",
                        "logging",
                        "autorun",
                        "rl",
                        "clustering",
                        "multi_objective",
                        "safety",
                    ]
                    and key not in known_ignored_keys
                ):
                    validation_result["ignored"].append(key)

            return {
                "valid": len(validation_result["errors"]) == 0,
                "applied": validation_result["applied"],
                "ignored": validation_result["ignored"],
                "errors": validation_result["errors"],
                "message": "Validation complétée"
                if len(validation_result["errors"]) == 0
                else "Erreurs de validation détectées",
            }, HTTPStatus.OK

        except Exception as e:
            logger.exception("Erreur validation settings: %s", e)
            raise APIErrorHandler.bad_request(
                message="Erreur lors de la validation",
                details={
                    "valid": False,
                    "applied": [],
                    "ignored": [],
                    "errors": [str(e)],
                },
            ) from e


@dispatch_ns.route("/autorun/enable")
class DispatchAutorunEnable(Resource):
    """Active ou désactive l'autorun pour une company."""

    @jwt_required()
    @role_required(UserRole.company)
    @dispatch_ns.expect(autorun_model, validate=True)
    def post(self):
        """Active ou désactive l'autorun pour une company."""
        company_id: int | None = None
        try:
            company = _get_current_company()
            company_id = _current_company_id()

            body = request.get_json(silent=True) or {}
            enabled = bool(body.get("enabled", True))
            interval_sec = body.get("interval_sec")

            # Lire les réglages existants en toute sécurité
            # (l'attribut peut ne pas exister)
            settings_data: dict[str, Any] = {}
            settings_raw = getattr(company, "dispatch_settings", None)
            if isinstance(settings_raw, str) and settings_raw:
                try:
                    settings_data = json.loads(settings_raw)
                except json.JSONDecodeError:
                    settings_data = {}

            # Mettre à jour
            settings_data["autorun_enabled"] = enabled
            if interval_sec is not None:
                with suppress(TypeError, ValueError):
                    settings_data["autorun_interval_sec"] = int(interval_sec)

            # Sauvegarder
            cast("Any", company).dispatch_settings = json.dumps(settings_data)
            db.session.add(company)
            db.session.commit()

            return {
                "company_id": company_id,
                "autorun_enabled": enabled,
                "autorun_interval_sec": settings_data.get("autorun_interval_sec", 300),
            }, HTTPStatus.OK

        except Exception as e:
            db.session.rollback()
            logger.exception(
                "autorun settings update failed company=%s", company_id or "unknown"
            )
            raise APIErrorHandler.internal_error(
                message="Erreur mise à jour autorun",
                details={"error": str(e)},
            ) from e
