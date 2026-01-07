# backend/services/unified_dispatch/shadow_mode_orchestrator.py
"""Orchestrateur pour AB Router et Shadow Mode.

✅ REFACTORING: Extraction de la logique AB Router / Shadow Mode depuis engine.py
pour améliorer la modularité et la testabilité.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List

from services.unified_dispatch.ml.ab_router import ABRouter
from services.unified_dispatch.metrics.dispatch import (
    QUALITY_THRESHOLD,
    DispatchMetricsCollector,
)

logger = logging.getLogger(__name__)


class ShadowModeOrchestrator:
    """Orchestrateur pour AB Router et Shadow Mode.

    Gère:
    - La décision d'activer RL via AB Router avec quality check
    - La génération et stockage des suggestions shadow mode
    """

    def __init__(self, settings: Any) -> None:  # pyright: ignore[reportMissingSuperCall]
        """Initialise l'orchestrateur.

        Args:
            settings: Configuration settings avec RLSettings
        """
        self.settings = settings
        self.ab_router = ABRouter(settings)

    def should_apply_rl_with_guards(
        self,
        company_id: int,
        dispatch_run_id: int | None,
        final_assignments: List[Any],
        problem: Dict[str, Any],
        company: Any | None,
    ) -> tuple[bool, float | None]:
        """Détermine si RL doit être appliqué avec tous les garde-fous.

        Args:
            company_id: ID de l'entreprise
            dispatch_run_id: ID du dispatch run (optionnel)
            final_assignments: Liste des assignations finales
            problem: Problème complet
            company: Objet Company (optionnel)

        Returns:
            Tuple (should_apply, quality_score_pre_apply)
        """
        if not dispatch_run_id:
            return False, None

        try:
            should_apply_rl = self.ab_router.should_apply_rl(company_id)

            # ✅ B1: Calculer quality_score pré-apply pour garde-fou
            quality_score_pre_apply: float | None = None
            if should_apply_rl and len(final_assignments) > 0 and company:
                try:
                    collector = DispatchMetricsCollector(company_id)
                    # Calcul rapide du quality score avant RL
                    # Convertir run_date en date si nécessaire
                    run_date_value = problem.get("for_date")
                    if isinstance(run_date_value, str):
                        from datetime import datetime

                        run_date_obj = datetime.fromisoformat(run_date_value).date()
                    elif isinstance(run_date_value, date):
                        run_date_obj = run_date_value
                    else:
                        # Fallback: utiliser aujourd'hui
                        from datetime import UTC, datetime

                        run_date_obj = datetime.now(UTC).date()

                    temp_metrics = collector._calculate_metrics(
                        dispatch_run_id=dispatch_run_id,
                        run_date=run_date_obj,
                        assignments=final_assignments,
                        all_bookings=problem.get("bookings", []),
                        run_metadata={},
                    )
                    quality_score_pre_apply = temp_metrics.quality_score

                    logger.info(
                        "[B1] Quality score pré-apply: %.1f/100 (dominants: %s)",
                        quality_score_pre_apply,
                        ", ".join(
                            f"{k}={v:.1f}"
                            for k, v in list(temp_metrics.dominant_factors.items())[:2]
                        ),
                    )

                    # ✅ B1: Garde-fou - Désactiver RL apply si quality_score < 70
                    if quality_score_pre_apply < QUALITY_THRESHOLD:
                        logger.warning(
                            (
                                "[B1] ⚠️ Auto-apply RL désactivé: "
                                "quality_score=%.1f < seuil=%d"
                            ),
                            quality_score_pre_apply,
                            QUALITY_THRESHOLD,
                        )
                        should_apply_rl = False
                    else:
                        logger.info(
                            (
                                "[B1] ✅ Auto-apply RL autorisé: "
                                "quality_score=%.1f >= seuil=%d"
                            ),
                            quality_score_pre_apply,
                            QUALITY_THRESHOLD,
                        )
                except Exception as e:
                    logger.warning(
                        "[B1] Failed to calculate pre-apply quality score: %s", e
                    )

            if should_apply_rl:
                logger.info(
                    "[ABRouter] Company %d: RL apply-mode enabled (bucket routing)",
                    company_id,
                )
                # Activer enable_rl_apply pour cette entreprise
                self.settings.features.enable_rl_apply = True
            else:
                logger.debug(
                    (
                        "[ABRouter] Company %d: RL apply-mode disabled "
                        "(not in bucket or quality guard triggered)"
                    ),
                    company_id,
                )
                self.settings.features.enable_rl_apply = False

            return should_apply_rl, quality_score_pre_apply

        except Exception as e:
            logger.warning("[ABRouter] Failed to check routing: %s", e)
            self.settings.features.enable_rl_apply = False
            return False, None

    def generate_and_store_shadow_suggestions(
        self,
        dispatch_run_id: int | None,
        problem: Dict[str, Any],
        final_assignments: List[Any],
        used_heuristic: bool,
        used_solver: bool,
    ) -> int:
        """Génère et stocke les suggestions shadow mode.

        Args:
            dispatch_run_id: ID du dispatch run
            problem: Problème complet
            final_assignments: Liste des assignations finales
            used_heuristic: Si les heuristiques ont été utilisées
            used_solver: Si le solveur a été utilisé

        Returns:
            Nombre de suggestions stockées
        """
        if not dispatch_run_id or not getattr(
            self.settings.features, "enable_rl", False
        ):
            return 0

        try:
            from services.rl.shadow_mode_manager import ShadowModeManager

            shadow_manager = ShadowModeManager()

            # Construire les assignations courantes
            current_assignments = {
                a.booking_id: a.driver_id for a in final_assignments if a.booking_id
            }

            # Générer suggestions shadow
            shadow_suggestions = shadow_manager.generate_shadow_suggestions(
                bookings=problem.get("bookings", []),
                drivers=problem.get("drivers", []),
                current_assignments=current_assignments,
            )

            # Stocker les suggestions
            if shadow_suggestions:
                # Helper pour obtenir les IDs restants
                def remaining_ids_from(prob: Dict[str, Any]) -> List[int]:
                    """Retourne les IDs de bookings non assignés."""
                    assigned_ids = {
                        a.booking_id for a in final_assignments if a.booking_id
                    }
                    all_booking_ids = {
                        b.id for b in prob.get("bookings", []) if hasattr(b, "id")
                    }
                    return list(all_booking_ids - assigned_ids)

                rem_before_apply = remaining_ids_from(problem)
                kpi_snapshot = {
                    "assignments_count": len(final_assignments),
                    "unassigned_count": len(rem_before_apply)
                    if rem_before_apply
                    else 0,
                    "heuristic_used": used_heuristic,
                    "solver_used": used_solver,
                }
                shadow_suggestions_stored = shadow_manager.store_shadow_suggestions(
                    dispatch_run_id=dispatch_run_id,
                    suggestions=shadow_suggestions,
                    kpi_snapshot=kpi_snapshot,
                )
                logger.info(
                    "[ShadowMode] Stored %s shadow suggestions for dispatch_run %s",
                    shadow_suggestions_stored,
                    dispatch_run_id,
                )
                return shadow_suggestions_stored

        except Exception as e:
            logger.warning(
                "[ShadowMode] Failed to generate/store shadow suggestions: %s", e
            )

        return 0
