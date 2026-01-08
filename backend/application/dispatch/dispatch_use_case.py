"""Use-cases du module Dispatch (Clean Architecture - couche Application).

Migration progressive:
- `services/dispatch_service.py` devient une façade legacy
- la logique métier est portée ici
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from marshmallow import ValidationError  # pyright: ignore[reportMissingImports]

from domain.dispatch.commands import DispatchRunRequestCommand
from schemas.dispatch_overrides_schema import DispatchOverridesSchema
from schemas.dispatch_schemas import DispatchRunRequestSchema
from schemas.validation_utils import handle_validation_error, validate_request

logger = logging.getLogger(__name__)


class DispatchUseCase:
    """Use-case Application: orchestration d'un run de dispatch.

    Responsabilités:
        - Validation/normalisation du payload
        - Calcul du mode effectif
        - Décision async vs sync (heuristique)
        - Préparation des paramètres d'exécution
        - Exécution sync (appel engine + validation)

    Important:
        Pour préserver l'isolation Clean Architecture, ce use-case **nécessite**
        l'injection de `get_bookings_for_day_fn`, `engine_run_fn` et
        `validate_assignments_fn` (pas d'import direct de `services/*`).
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        *,
        get_bookings_for_day_fn: Callable[[int, str], list[Any]] | None = None,
        getenv_fn: Callable[[str, str], str] | None = None,
        engine_run_fn: Callable[..., dict[str, Any]] | None = None,
        validate_assignments_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        """Initialise le use-case.

        Args:
            get_bookings_for_day_fn: Fonction de lecture des bookings pour
                le dimensionnement.
            getenv_fn: Fonction getenv (injection pour tests).
            engine_run_fn: Fonction d'exécution (adapter Infrastructure).
            validate_assignments_fn: Fonction de validation post-run
                (adapter Infrastructure).
        """

        def _missing_get_bookings_for_day(
            _company_id: int, _for_date: str
        ) -> list[Any]:
            raise RuntimeError(
                "DispatchUseCase nécessite une dépendance injectée "
                "`get_bookings_for_day_fn`. "
                + "Utiliser la façade DispatchService (ou une factory) pour "
                + "le wiring production."
            )

        self.get_bookings_for_day = (
            get_bookings_for_day_fn or _missing_get_bookings_for_day
        )
        self.getenv = getenv_fn
        self._engine_run = engine_run_fn
        self._validate_assignments = validate_assignments_fn

    def validate_and_normalize_request(
        self, cmd: DispatchRunRequestCommand
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int | None]:
        """Valide et normalise une requête de dispatch."""
        try:
            validated_data = validate_request(
                DispatchRunRequestSchema(), cmd.body, strict=False
            )
        except ValidationError as e:
            error_response, status_code = handle_validation_error(e)
            return None, error_response, status_code

        overrides_raw = validated_data.get("overrides")
        if overrides_raw and isinstance(overrides_raw, dict):
            try:
                original_keys = set(overrides_raw.keys())
                validated_overrides = DispatchOverridesSchema().load(
                    overrides_raw, unknown="exclude"
                )
                if validated_overrides is not None and isinstance(
                    validated_overrides, dict
                ):
                    validated_keys = set(validated_overrides.keys())
                    invalid_keys = original_keys - validated_keys

                    if invalid_keys:
                        logger.warning(
                            "[DispatchService] Overrides invalides "
                            "(clés non autorisées): %s",
                            invalid_keys,
                        )
                        return (
                            None,
                            {
                                "error": "validation_error",
                                "message": "Clés non autorisées dans overrides",
                                "errors": {
                                    "overrides": {
                                        "invalid_keys": list(invalid_keys),
                                        "message": (
                                            f"Les clés suivantes ne sont pas "
                                            f"autorisées: {', '.join(invalid_keys)}"
                                        ),
                                    }
                                },
                            },
                            400,
                        )

                    validated_data["overrides"] = validated_overrides
            except ValidationError as e:
                logger.warning(
                    "[DispatchService] Overrides invalides (erreur de validation): %s",
                    e.messages,
                )
                return (
                    None,
                    {
                        "error": "validation_error",
                        "message": "Clés non autorisées dans overrides",
                        "errors": {"overrides": e.messages},
                    },
                    400,
                )

        return validated_data, None, None

    def normalize_dispatch_mode(self, validated_data: dict[str, Any]) -> str | None:
        requested_mode = (
            (validated_data.get("mode") or "").strip().lower()
            if validated_data.get("mode")
            else None
        )
        final_mode = (
            (validated_data.get("finalMode") or validated_data.get("final_mode") or "")
            .strip()
            .lower()
            if (validated_data.get("finalMode") or validated_data.get("final_mode"))
            else None
        )

        if requested_mode not in {None, "auto", "heuristic_only", "solver_only"}:
            requested_mode = None
        if final_mode and final_mode not in {"auto", "heuristic_only", "solver_only"}:
            final_mode = None

        effective_mode = requested_mode or final_mode
        if effective_mode == "semi_auto":
            effective_mode = "heuristic_only"

        return effective_mode or validated_data.get("mode")

    def should_force_async_mode(
        self,
        *,
        company_id: int,
        for_date: str,
        is_async: bool,
        getenv_fn: Callable[[str, str], str],
    ) -> tuple[bool, str | None]:
        if is_async:
            return False, None

        max_sync_bookings = int(getenv_fn("DISPATCH_SYNC_MAX_BOOKINGS", "10"))
        bookings_count = len(self.get_bookings_for_day(company_id, for_date))

        if bookings_count > max_sync_bookings:
            logger.info(
                "[DispatchService] ⚠️ Forcing async mode: %d bookings > %d "
                "(max sync). Mode sync désactivé pour éviter timeout HTTP.",
                bookings_count,
                max_sync_bookings,
            )
            reason = (
                f"Mode async forcé automatiquement: {bookings_count} bookings "
                f"> {max_sync_bookings} (limite mode sync)"
            )
            return True, reason

        return False, None

    def prepare_dispatch_params(
        self,
        *,
        validated_data: dict[str, Any],
        company_id: int,
        effective_mode: str | None,
    ) -> dict[str, Any]:
        for_date = validated_data.get("for_date")
        allow_emergency_val = validated_data.get("allow_emergency")
        allow_emergency: bool | None = (
            bool(allow_emergency_val) if allow_emergency_val is not None else None
        )
        regular_first = validated_data.get("regular_first", True)
        overrides = validated_data.get("overrides")

        return {
            "company_id": company_id,
            "for_date": for_date,
            "mode": effective_mode,
            "regular_first": regular_first,
            "allow_emergency": allow_emergency,
            "overrides": overrides,
        }

    def execute_dispatch_sync(
        self, params: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Exécute un dispatch en mode synchrone.

        Args:
            params: Paramètres d'exécution (company_id, for_date, overrides, etc.).

        Returns:
            (result, validation_info) où validation_info contient erreurs/warnings.

        Raises:
            RuntimeError: si les dépendances `engine_run_fn`/
                `validate_assignments_fn` ne sont pas injectées.
        """
        engine_run = self._engine_run
        validate_assignments = self._validate_assignments
        if engine_run is None or validate_assignments is None:
            raise RuntimeError(
                "DispatchUseCase.execute_dispatch_sync nécessite des "
                "dépendances injectées (engine_run_fn et "
                "validate_assignments_fn). Utiliser la façade DispatchService "
                "(ou une factory) pour le wiring production."
            )

        result = engine_run(**params)

        assignments_list = result.get("assignments", [])
        validation_info: dict[str, Any] | None = None
        if assignments_list:
            validation_result = validate_assignments(assignments_list, strict=False)
            if not validation_result["valid"]:
                logger.warning(
                    "[DispatchService] Conflits temporels détectés pour "
                    "company %s, date %s",
                    params["company_id"],
                    params.get("for_date"),
                )
                for error in validation_result["errors"]:
                    logger.error("  %s", error)

                validation_info = {
                    "has_errors": True,
                    "errors": validation_result["errors"],
                    "warnings": validation_result.get("warnings", []),
                }
            elif validation_result.get("warnings"):
                validation_info = {
                    "has_errors": False,
                    "warnings": validation_result["warnings"],
                }

        return result, validation_info
