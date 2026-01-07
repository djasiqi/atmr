# backend/services/unified_dispatch/orchestration/dispatch_orchestrator.py
"""Orchestrateur principal pour le dispatch.

✅ REFACTORING: Extraction de la logique principale de engine.run()
pour améliorer la modularité et réduire engine.py à < 1000 lignes.

Cet orchestrateur coordonne les différentes étapes du dispatch :
1. Initialisation (Company lookup, configuration)
2. Locking (verrou d'idempotence)
3. DispatchRun management (création/réutilisation)
4. Problem building (construction du problème VRPTW)
5. Clustering (dispatch géographique si nécessaire)
6. Pipeline dispatch (heuristique → solver → fallback)
7. Shadow Mode (AB Router, suggestions RL)
8. Apply (application des assignations)
9. Métriques finalization
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from services.unified_dispatch.orchestration.assignment_applier_wrapper import (
    AssignmentApplierWrapper,
)
from services.unified_dispatch.orchestration.clustering_manager import (
    ClusteringManager,
)
from services.unified_dispatch.orchestration.dispatch_run_manager import (
    DispatchRunManager,
)
from services.unified_dispatch.orchestration.initializer import DispatchInitializer
from services.unified_dispatch.orchestration.metrics_finalizer import (
    MetricsFinalizer,
)
from services.unified_dispatch.orchestration.pipeline_executor import (
    PipelineExecutor,
)
from services.unified_dispatch.orchestration.problem_builder import ProblemBuilder
from services.unified_dispatch.orchestration.utils import safe_int

logger = logging.getLogger(__name__)


class DispatchOrchestrator:
    """Orchestrateur principal pour le dispatch.

    Coordonne toutes les étapes du processus de dispatch de manière modulaire :
    1. Initialisation (Company lookup, configuration)
    2. Locking (verrou d'idempotence Redis)
    3. DispatchRun management (création/réutilisation)
    4. Problem building (construction du problème VRPTW)
    5. Clustering (dispatch géographique si nécessaire)
    6. Pipeline dispatch (heuristique → solver → fallback)
    7. Shadow Mode (AB Router, suggestions RL)
    8. Apply (application des assignations)
    9. Métriques finalization

    Side-effects:
        - Accès DB (lecture/écriture Company, DispatchRun, Assignment, Booking)
        - Redis: Verrous distribués pour éviter runs concurrents
        - Socket.IO: Émissions d'événements temps réel
        - Métriques: Prometheus, logging

    Exemple:
        >>> orchestrator = DispatchOrchestrator()
        >>> result = orchestrator.run_dispatch(
        ...     company_id=1,
        ...     for_date="2025-01-13",
        ...     mode="auto"
        ... )
        >>> result["assignments_count"]  # doctest: +SKIP
        42
    """

    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        """Initialise l'orchestrateur."""
        self._initializer = DispatchInitializer()
        self._dispatch_run_manager = DispatchRunManager()
        self._problem_builder = ProblemBuilder()
        self._clustering_manager = ClusteringManager()
        self._pipeline_executor = PipelineExecutor()
        self._assignment_applier_wrapper = AssignmentApplierWrapper()
        self._metrics_finalizer = MetricsFinalizer()

    def execute(
        self,
        company_id: int,
        mode: str = "auto",
        custom_settings: Any = None,
        *,
        for_date: str | None = None,
        regular_first: bool = True,
        allow_emergency: bool | None = None,
        overrides: dict[str, Any] | None = None,
        existing_dispatch_run_id: int | None = None,
        raise_on_company_not_found: bool = False,
    ) -> Dict[str, Any]:
        """Exécute le dispatch complet.

        Cette méthode orchestre toutes les étapes du dispatch :
        1. Initialisation et validation
        2. Configuration
        3. Locking
        4. DispatchRun management
        5. Problem building
        6. Clustering (si activé)
        7. Pipeline dispatch
        8. Shadow Mode
        9. Apply
        10. Métriques finalization

        Args:
            company_id: ID de l'entreprise
            mode: Mode de dispatch ("auto", "heuristic_only", "solver_only")
            custom_settings: Settings personnalisés (optionnel)
            for_date: Date au format YYYY-MM-DD (None = aujourd'hui)
            regular_first: Prioriser les courses régulières
            allow_emergency: Autoriser les courses d'urgence
            overrides: Overrides de configuration
            existing_dispatch_run_id: ID d'un DispatchRun existant (pour reprise)
            raise_on_company_not_found: Si True, lève une exception si Company introuvable

        Returns:
            Dict contenant assignments, unassigned, meta, dispatch_run_id, etc.
        """
        from datetime import UTC, datetime

        from services.unified_dispatch import performance_metrics
        from services.unified_dispatch.locking import RedisLockManager
        from services.unified_dispatch.ml.rl_kpi_monitor import RLKPIMonitor

        # 1. Initialisation et validation
        company, error_result = self._initializer.find_and_validate_company(
            company_id, for_date, mode, raise_on_company_not_found
        )
        if error_result:
            return error_result

        # 2. Configuration
        if not company:
            return {
                "assignments": [],
                "unassigned": [],
                "bookings": [],
                "drivers": [],
                "meta": {"reason": "company_not_found"},
                "debug": {"reason": "company_not_found", "company_id": company_id},
            }

        settings, mode, allow_emg, is_fast_mode = self._initializer.configure_settings(
            company=company,
            mode=mode,
            custom_settings=custom_settings,
            allow_emergency=allow_emergency,
            overrides=overrides,
        )

        day_str = for_date or datetime.now(UTC).strftime("%Y-%m-%d")

        # 3. Locking (verrou d'idempotence par (entreprise, jour))
        lock_manager = RedisLockManager()
        lock_key = f"dispatch:{company_id}:{day_str}"
        if not lock_manager.acquire_lock(lock_key, timeout_seconds=300):
            logger.warning(
                "[Orchestrator] Failed to acquire lock for company=%s day=%s",
                company_id,
                day_str,
            )
            return {
                "assignments": [],
                "unassigned": [],
                "bookings": [],
                "drivers": [],
                "meta": {"reason": "lock_failed", "for_date": day_str},
                "debug": {"reason": "lock_failed", "for_date": day_str},
            }

        # Variable pour stocker le résultat final
        result: Dict[str, Any] | None = None

        try:
            # 4. DispatchRun management (création/réutilisation)
            dispatch_run, error_result = self._dispatch_run_manager.create_or_reuse(
                company=company,
                company_id=company_id,
                day_str=day_str,
                mode=mode,
                regular_first=regular_first,
                allow_emg=allow_emg,
                for_date=for_date,
                existing_id=existing_dispatch_run_id,
            )
            if error_result:
                result = error_result
            else:
                # 5. Problem building (construction du problème VRPTW)
                perf_collector = None
                if dispatch_run:
                    perf_collector = performance_metrics.DispatchMetricsCollector(
                        company_id=company_id,
                        dispatch_run_id=dispatch_run.id if dispatch_run else None,
                    )
                    performance_metrics.reset_sql_counter()
                    perf_collector.start_timer("data_collection")

                problem, error_result = self._problem_builder.build(
                    _company=company,
                    company_id=company_id,
                    dispatch_run=dispatch_run,
                    settings=settings,
                    for_date=for_date,
                    day_str=day_str,
                    regular_first=regular_first,
                    allow_emg=allow_emg,
                    overrides=overrides,
                    perf_collector=perf_collector,
                )
                if error_result:
                    result = error_result
                elif not problem:
                    result = {
                        "assignments": [],
                        "unassigned": [],
                        "bookings": [],
                        "drivers": [],
                        "meta": {"reason": "problem_build_failed"},
                        "debug": {"reason": "problem_build_failed"},
                    }
                else:
                    # 6. Pipeline dispatch (heuristique → solver → fallback)
                    (
                        final_assignments,
                        unassigned_ids,
                        used_heuristic,
                        used_solver,
                        used_fallback,
                        used_emergency_pass,
                        should_apply_rl,
                        _quality_score_pre_apply,
                        error_result,
                    ) = self._pipeline_executor.execute(
                        company=company,
                        company_id=company_id,
                        problem=problem,
                        dispatch_run=dispatch_run,
                        settings=settings,
                        mode=mode,
                        regular_first=regular_first,
                        allow_emg=allow_emg,
                        is_fast_mode=is_fast_mode,
                        perf_collector=perf_collector,
                    )
                    if error_result:
                        result = error_result
                    else:
                        # 7. Initialiser KPI Monitor (optionnel)
                        kpi_monitor = None
                        try:
                            kpi_monitor = RLKPIMonitor(settings)
                        except (
                            ValueError,
                            TypeError,
                            AttributeError,
                            ImportError,
                        ) as e:
                            # Erreurs attendues : validation, imports manquants (KPI Monitor optionnel)
                            logger.debug(
                                "[Orchestrator] Failed to initialize KPI Monitor (expected error: %s)",
                                type(e).__name__,
                            )
                        except Exception:
                            # Erreur inattendue : logger et continuer sans KPI Monitor
                            logger.warning(
                                "[Orchestrator] Failed to initialize KPI Monitor (unexpected error)"
                            )

                        # 8. Apply (application des assignations en DB)
                        self._assignment_applier_wrapper.apply(
                            company=company,
                            final_assignments=final_assignments,
                            dispatch_run_id=safe_int(getattr(dispatch_run, "id", None)),
                            perf_collector=perf_collector,
                        )

                        # 9. Métriques finalization
                        # Note: h_res et s_res ne sont pas disponibles depuis le pipeline
                        # car ils sont encapsulés dans _execute_dispatch_pipeline
                        # On passe None et la finalisation s'adaptera
                        if not problem:
                            problem = {}

                        result = self._metrics_finalizer.finalize(
                            company_id=company_id,
                            problem=problem,
                            final_assignments=final_assignments,
                            unassigned_ids=unassigned_ids,
                            dispatch_run=dispatch_run,
                            settings=settings,
                            mode=mode,
                            regular_first=regular_first,
                            allow_emg=allow_emg,
                            for_date=for_date,
                            day_str=day_str,
                            used_heuristic=used_heuristic,
                            used_solver=used_solver,
                            used_fallback=used_fallback,
                            used_emergency_pass=used_emergency_pass,
                            h_res=None,  # h_res non disponible depuis le pipeline
                            s_res=None,  # s_res non disponible depuis le pipeline
                            perf_collector=perf_collector,
                            should_apply_rl=should_apply_rl,
                            kpi_monitor=kpi_monitor,
                        )

        finally:
            # Libérer le verrou
            lock_manager.release_lock(lock_key)

        return result
