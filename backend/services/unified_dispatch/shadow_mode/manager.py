# backend/services/unified_dispatch/orchestration/shadow_mode_manager.py
"""Gestionnaire du mode shadow.

Ce module gère le mode shadow pour le dispatch. Il est responsable de :
- La décision d'appliquer les suggestions RL (Reinforcement Learning)
- La génération et le stockage des suggestions shadow mode
- L'intégration avec ShadowModeOrchestrator

Le mode shadow permet de :
- Tester de nouvelles stratégies (RL) sans les appliquer réellement
- Collecter des données pour l'entraînement des modèles RL
- Comparer les performances entre stratégies (AB testing)

Side-effects:
    - Accès DB (écriture suggestions shadow mode)
    - Métriques: Tracking des décisions RL
"""

from __future__ import annotations

import logging
from typing import Any

from services.unified_dispatch.shadow_mode.orchestrator import ShadowModeOrchestrator

logger = logging.getLogger(__name__)


class ShadowModeManager:
    """Gestionnaire du mode shadow pour le dispatch.

    Cette classe centralise la logique du mode shadow :
    - Décision d'appliquer les suggestions RL
    - Génération et stockage des suggestions shadow mode
    - Intégration avec ShadowModeOrchestrator

    Le mode shadow permet de tester de nouvelles stratégies sans les
    appliquer réellement, collectant des données pour l'entraînement
    et la comparaison de performances.

    Exemple:
        >>> manager = ShadowModeManager(settings)
        >>> should_apply, quality_score = manager.should_apply_rl(
        ...     company_id=1,
        ...     dispatch_run_id=42,
        ...     final_assignments=assignments,
        ...     problem=problem,
        ...     company=company
        ... )
        >>> if should_apply:
        ...     stored = manager.generate_and_store_suggestions(
        ...         dispatch_run_id=42,
        ...         problem=problem,
        ...         final_assignments=assignments,
        ...         used_heuristic=True,
        ...         used_solver=False
        ...     )
    """

    def __init__(self, settings: Any) -> None:  # pyright: ignore[reportMissingSuperCall]
        """Initialise le gestionnaire du mode shadow.

        Args:
            settings: Settings de dispatch avec configuration shadow mode
        """
        self._shadow_orchestrator = ShadowModeOrchestrator(settings)

    def should_apply_rl(
        self,
        company_id: int,
        dispatch_run_id: int | None,
        final_assignments: list[Any],
        problem: dict[str, Any],
        company: Any,
    ) -> tuple[bool, float | None]:
        """Détermine si les suggestions RL doivent être appliquées.

        Utilise ShadowModeOrchestrator pour décider si les suggestions RL
        doivent être appliquées pour ce dispatch. Prend en compte :
        - Les feature flags dans les settings
        - La qualité de la solution actuelle
        - Les critères d'éligibilité de l'entreprise

        Args:
            company_id: ID de l'entreprise
            dispatch_run_id: ID du DispatchRun (peut être None)
            final_assignments: List des assignations finales
            problem: Dict contenant les données du problème
            company: Objet Company

        Returns:
            Tuple (should_apply, quality_score) où :
            - should_apply: True si les suggestions RL doivent être appliquées
            - quality_score: Score de qualité de la solution actuelle (peut être None)

        Side-effects:
            - Métriques: Tracking des décisions RL
        """
        return self._shadow_orchestrator.should_apply_rl_with_guards(
            company_id=company_id,
            dispatch_run_id=dispatch_run_id,
            final_assignments=final_assignments,
            problem=problem,
            company=company,
        )

    def generate_and_store_suggestions(
        self,
        dispatch_run_id: int | None,
        problem: dict[str, Any],
        final_assignments: list[Any],
        used_heuristic: bool,
        used_solver: bool,
    ) -> bool:
        """Génère et stocke les suggestions shadow mode.

        Génère des suggestions RL pour ce dispatch et les stocke en base
        de données pour analyse ultérieure. Les suggestions sont générées
        même si elles ne sont pas appliquées (mode shadow).

        Args:
            dispatch_run_id: ID du DispatchRun (peut être None)
            problem: Dict contenant les données du problème
            final_assignments: List des assignations finales
            used_heuristic: Bool indiquant si heuristique utilisée
            used_solver: Bool indiquant si solver utilisé

        Returns:
            True si les suggestions ont été générées et stockées avec succès,
            False sinon

        Side-effects:
            - Accès DB (écriture suggestions shadow mode)
        """
        count = self._shadow_orchestrator.generate_and_store_shadow_suggestions(
            dispatch_run_id=dispatch_run_id,
            problem=problem,
            final_assignments=final_assignments,
            used_heuristic=used_heuristic,
            used_solver=used_solver,
        )
        return count > 0
