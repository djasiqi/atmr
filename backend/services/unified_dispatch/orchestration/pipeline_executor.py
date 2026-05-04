# backend/services/unified_dispatch/orchestration/pipeline_executor.py
"""Exécuteur du pipeline de dispatch.

Ce module gère l'exécution complète du pipeline de dispatch. Il orchestre :
- Le clustering géographique (si activé)
- La séparation des drivers réguliers/urgences
- L'exécution du pipeline (heuristique → solver → fallback)
- La gestion des passes multiples (urgents, réguliers Pass 1, urgences Pass 2)
- Le mode shadow (AB Router, suggestions RL)

Le pipeline suit cette logique :
1. Si clustering activé et > threshold : utiliser clustering
2. Sinon : pipeline direct avec séparation réguliers/urgences
3. Pipeline : heuristique → solver (si activé) → fallback (si nécessaire)
4. Shadow mode : génération de suggestions RL si activé

Side-effects:
    - Accès DB (lecture bookings, drivers)
    - Appels aux heuristiques, solver, fallback
    - Appels récursifs au dispatch pour clustering
    - Métriques: Performance tracking via perf_collector
"""

from __future__ import annotations  # noqa: I001

import logging
from collections.abc import Iterable
from typing import Any, Dict, List, cast

from models import Company, Driver, DriverType
from services.unified_dispatch.data import loader as data
from services.unified_dispatch.optimization import heuristics, solver
from services.unified_dispatch.orchestration.clustering_manager import (
    ClusteringManager,
)
from services.unified_dispatch.shadow_mode.manager import (
    ShadowModeManager,
)
from sqlalchemy.exc import DBAPIError, OperationalError

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Exécuteur du pipeline de dispatch.

    Cette classe orchestre l'exécution complète du pipeline de dispatch :
    - Décision d'utiliser le clustering
    - Séparation des drivers réguliers/urgences
    - Exécution séquentielle : heuristique → solver → fallback
    - Gestion des passes multiples pour optimiser les assignations
    - Intégration avec le shadow mode

    Exemple:
        >>> executor = PipelineExecutor()
        >>> assignments, unassigned, used_h, used_s, used_f, used_emg, error = (
        ...     executor.execute(
        ...         company=company,
        ...         company_id=1,
        ...         problem=problem,
        ...         dispatch_run=dispatch_run,
        ...         settings=settings,
        ...         mode="auto",
        ...         regular_first=True,
        ...         allow_emg=True,
        ...         is_fast_mode=False,
        ...         perf_collector=perf_collector
        ...     )
        ... )
    """

    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        """Initialise l'exécuteur de pipeline."""
        self._clustering_manager = ClusteringManager()

    def _extend_unique(
        self,
        assigns: Iterable[Any],
        final_assignments: List[Any],
        assigned_set: set[int],
    ) -> None:
        """Étend les assignations de manière unique.

        Ajoute les assignations à final_assignments uniquement si leur booking_id
        n'est pas déjà présent dans assigned_set.

        Args:
            assigns: Itérable d'assignations à ajouter
            final_assignments: Liste des assignations finales (modifiée en place)
            assigned_set: Set des IDs de bookings déjà assignés (modifié en place)
        """
        for a in assigns:
            bid_raw = getattr(a, "booking_id", None)
            try:
                bid = int(bid_raw) if bid_raw is not None else None
            except (ValueError, TypeError, OverflowError):
                # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
                bid = None
            except Exception:
                # Erreur inattendue : logger et utiliser None
                logger.debug(
                    "[PipelineExecutor] Unexpected error converting booking_id to int: %s",
                    bid_raw,
                )
                bid = None
            if bid is None or bid in assigned_set:
                continue
            final_assignments.append(a)
            assigned_set.add(bid)

    def _remaining_ids_from(
        self, problem: Dict[str, Any], assigned_set: set[int]
    ) -> List[int]:
        """Obtient les IDs de bookings non assignés.

        Args:
            problem: Dict contenant les données du problème avec clé "bookings"
            assigned_set: Set des IDs de bookings déjà assignés

        Returns:
            List des IDs de bookings non assignés
        """
        res: List[int] = []
        for b in problem.get("bookings", []):
            try:
                bid = int(cast("Any", getattr(b, "id", None)))
            except (ValueError, TypeError, OverflowError):
                # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
                bid = None
            except Exception:
                # Erreur inattendue : logger et utiliser None
                logger.debug(
                    "[PipelineExecutor] Unexpected error converting booking id to int"
                )
                bid = None
            if bid is not None and bid not in assigned_set:
                res.append(bid)
        return res

    def _filter_problem(
        self,
        problem_dict: Dict[str, Any],
        booking_ids: List[int],
        company: Company,
        settings: Any,
    ) -> Dict[str, Any]:
        """Filtre le problème pour ne garder que certains bookings.

        Reconstruit un sous-problème avec les mêmes settings mais seulement
        les bookings spécifiés. Utile pour les passes multiples ou le fallback.

        Args:
            problem_dict: Dict contenant les données du problème original
            booking_ids: List des IDs de bookings à inclure
            company: Objet Company
            settings: Settings de dispatch

        Returns:
            Dict avec nouveau problème filtré
        """
        bookings_map = {b.id: b for b in problem_dict.get("bookings", [])}
        new_bookings = [bookings_map[bid] for bid in booking_ids if bid in bookings_map]
        drivers = problem_dict.get("drivers", [])

        # Propager les métadonnées importantes
        result = data.build_vrptw_problem(
            company,
            new_bookings,
            drivers,
            settings=settings,
            base_time=problem_dict.get("base_time"),
            for_date=problem_dict.get("for_date"),
        )

        # Propager les états et métadonnées
        for key in [
            "preferred_driver_id",
            "company_coords",
            "driver_load_multipliers",
            "busy_until",
            "driver_scheduled_times",
            "proposed_load",
            "for_date",
            "dispatch_run_id",
        ]:
            if key in problem_dict:
                result[key] = problem_dict[key]

        return result

    def execute(
        self,
        company: Company,
        company_id: int,
        problem: Dict[str, Any],
        dispatch_run: Any | None,
        settings: Any,
        mode: str,
        regular_first: bool,
        allow_emg: bool,
        is_fast_mode: bool,
        perf_collector: Any | None,
    ) -> tuple[
        List[Any],  # final_assignments
        List[int],  # unassigned_ids
        bool,  # used_heuristic
        bool,  # used_solver
        bool,  # used_fallback
        bool,  # used_emergency_pass
        bool,  # should_apply_rl
        float | None,  # quality_score_pre_apply
        Dict[str, Any] | None,  # error_result
    ]:
        """Exécute le pipeline de dispatch complet.

        Orchestre toutes les étapes du pipeline :
        1. Clustering géographique (si activé et > threshold)
        2. Séparation réguliers/urgences
        3. Pipeline commun (urgents, Pass 1 réguliers, Pass 2 urgences, pipeline direct)
        4. Shadow Mode (génération suggestions RL)

        Args:
            company: Objet Company
            company_id: ID de l'entreprise
            problem: Problème VRPTW construit
            dispatch_run: DispatchRun (peut être None)
            settings: Settings de dispatch
            mode: Mode de dispatch ("auto", "heuristic_only", "solver_only")
            regular_first: Prioriser les courses régulières
            allow_emg: Autoriser les courses d'urgence
            is_fast_mode: Mode rapide activé
            perf_collector: Collecteur de métriques de performance

        Returns:
            Tuple (final_assignments, unassigned_ids, used_heuristic, used_solver,
            used_fallback, used_emergency_pass, error_result) où :
            - final_assignments: List des assignations finales
            - unassigned_ids: List des IDs de bookings non assignés
            - used_heuristic: Bool indiquant si heuristique utilisée
            - used_solver: Bool indiquant si solver utilisé
            - used_fallback: Bool indiquant si fallback utilisé
            - used_emergency_pass: Bool indiquant si pass d'urgence utilisé
            - error_result: Dict avec résultat d'erreur structuré si échec, None sinon

        Side-effects:
            - Accès DB (lecture bookings, drivers)
            - Appels aux heuristiques, solver, fallback
            - Appels récursifs au dispatch pour clustering
            - Métriques: Performance tracking via perf_collector
        """
        # Initialisation
        final_assignments: List[Any] = []
        assigned_set: set[int] = set()
        used_heuristic = False
        used_solver = False
        used_fallback = False
        used_emergency_pass = False

        # 5.5) Clustering géographique (si activé et > threshold)
        zones: List[Any] = []
        clustering_used = False

        if self._clustering_manager.should_use_clustering(problem, settings):
            try:
                zones = self._clustering_manager.create_zones(problem, settings)
                if len(zones) > 1:
                    clustering_result = self._clustering_manager.dispatch_zones(
                        zones=zones,
                        company=company,
                        problem=problem,
                        mode=mode,
                        settings=settings,
                    )
                    clustering_final_assignments = clustering_result["assignments"]
                    _clustering_unassigned_ids = clustering_result["unassigned"]

                    final_assignments = clustering_final_assignments
                    assigned_set = {
                        a.booking_id
                        for a in clustering_final_assignments
                        if a.booking_id
                    }
                    used_heuristic = True
                    used_solver = True
                    used_fallback = True
                    clustering_used = True

            except (ValueError, TypeError, AttributeError) as e:
                # Erreurs de validation : paramètres invalides, attributs manquants
                logger.warning(
                    "[PipelineExecutor] Clustering failed (validation error: %s), falling back to normal pipeline: %s",
                    type(e).__name__,
                    e,
                )
            except Exception as e:
                # Erreur inattendue : logger et continuer avec fallback
                logger.warning(
                    "[PipelineExecutor] Clustering failed (unexpected error), falling back to normal pipeline: %s",
                    e,
                )

        # 5) Séparation réguliers/urgences
        regs: List[Driver] = []
        emgs: List[Driver] = []
        try:
            regs, emgs = data.get_available_drivers_split(company_id)
        except (OperationalError, DBAPIError) as e:
            # Erreurs DB attendues : connexion, timeout
            logger.warning(
                "[PipelineExecutor] get_available_drivers_split failed (DB error: %s), using fallback",
                type(e).__name__,
            )
            for d in problem.get("drivers", []):
                d_type = getattr(d, "driver_type", None)
                (
                    emgs
                    if (
                        d_type == DriverType.EMERGENCY
                        or str(d_type).endswith("EMERGENCY")
                    )
                    else regs
                ).append(d)
        except Exception as e:
            # Erreur inattendue : logger et utiliser fallback
            logger.warning(
                "[PipelineExecutor] get_available_drivers_split failed (unexpected error: %s), using fallback",
                e,
            )
            for d in problem.get("drivers", []):
                d_type = getattr(d, "driver_type", None)
                (
                    emgs
                    if (
                        d_type == DriverType.EMERGENCY
                        or str(d_type).endswith("EMERGENCY")
                    )
                    else regs
                ).append(d)

        # 6) Pipeline commun
        if not clustering_used:
            final_assignments.clear()
            assigned_set.clear()

        # 6.a Urgents
        if not clustering_used:
            try:
                urgent_ids = data.pick_urgent_returns(problem, settings=settings) or []
            except (ValueError, TypeError, AttributeError) as e:
                # Erreurs de validation : paramètres invalides, attributs manquants
                logger.debug(
                    "[PipelineExecutor] pick_urgent_returns failed (validation error: %s), using empty list",
                    type(e).__name__,
                )
                urgent_ids = []
            except Exception as e:
                # Erreur inattendue : logger et utiliser liste vide
                logger.warning(
                    "[PipelineExecutor] pick_urgent_returns failed (unexpected error: %s), using empty list",
                    e,
                )
                urgent_ids = []
            if urgent_ids:
                try:
                    urg_res = heuristics.assign_urgent(
                        problem, urgent_ids, settings=settings
                    )
                    self._extend_unique(
                        urg_res.assignments, final_assignments, assigned_set
                    )
                except (ValueError, TypeError, AttributeError) as e:
                    # Erreurs de validation : paramètres invalides, attributs manquants
                    logger.warning(
                        "[PipelineExecutor] assign_urgent failed (validation error: %s)",
                        type(e).__name__,
                    )
                except Exception:
                    # Erreur inattendue : logger avec trace complète
                    logger.exception(
                        "[PipelineExecutor] assign_urgent failed (unexpected error)"
                    )

        # Pour méta/debug (non utilisé pour l'instant mais conservé pour compatibilité)
        _phase = "regular_only" if regular_first else "direct"

        # Variables pour le pipeline
        h_res = None
        s_res = None
        fb = None

        # 6.b Pass 1 - réguliers (si regular_first)
        if regular_first and regs and company and not clustering_used:
            logger.info(
                "[PipelineExecutor] === Pass 1: Regular drivers only (%d drivers) ===",
                len(regs),
            )

            bookings_list = problem.get("bookings", [])
            prob_regs = data.build_vrptw_problem(
                company,
                bookings_list,
                regs,
                settings=settings,
                base_time=problem.get("base_time"),
                for_date=problem.get("for_date"),
            )

            # Propager les métadonnées importantes
            for key in [
                "preferred_driver_id",
                "company_coords",
                "driver_load_multipliers",
            ]:
                if key in problem:
                    prob_regs[key] = problem[key]

            remaining_ids = self._remaining_ids_from(prob_regs, assigned_set)

            # Heuristique Pass 1
            if (
                remaining_ids
                and mode in ("auto", "heuristic_only")
                and getattr(settings.features, "enable_heuristics", True)
            ):
                try:
                    h_sub = self._filter_problem(
                        prob_regs, remaining_ids, company, settings
                    )
                    if perf_collector:
                        with perf_collector.time_step("heuristics"):
                            h_res = heuristics.assign(h_sub, settings=settings)
                    else:
                        h_res = heuristics.assign(h_sub, settings=settings)
                    used_heuristic = True
                    self._extend_unique(
                        h_res.assignments, final_assignments, assigned_set
                    )
                    logger.info(
                        "[PipelineExecutor] Heuristic P1: %d assignés, %d restants",
                        len(h_res.assignments),
                        len(h_res.unassigned_booking_ids),
                    )

                    # Post-opt RL : logs skips explicites (agrégation / alerting)
                    if mode == "auto" and len(final_assignments) > 0:
                        if is_fast_mode:
                            logger.info(
                                "[PipelineExecutor] RL_POSTOPT_SKIPPED reason=fast_mode"
                            )
                        elif not getattr(settings.features, "enable_rl_postopt", False):
                            logger.info(
                                "[PipelineExecutor] RL_POSTOPT_SKIPPED reason=feature_disabled (enable_rl_postopt=false)"
                            )

                    # Optimisation RL post-assignation (gelé par défaut — enable_rl_postopt)
                    if (
                        mode == "auto"
                        and not is_fast_mode
                        and len(final_assignments) > 0
                    ):
                        try:
                            if getattr(settings.features, "enable_rl_postopt", False):
                                from services.unified_dispatch.ml.rl_optimizer import (
                                    RLDispatchOptimizer,
                                )

                                logger.info(
                                    "[PipelineExecutor] 🧠 Tentative d'optimisation RL des assignations..."
                                )

                                optimizer = RLDispatchOptimizer(
                                    model_path="data/rl/models/dispatch_optimized_v2.pth",
                                    max_swaps=15,
                                    min_improvement=0.3,
                                    config_context="production",
                                )

                                if optimizer.is_available():
                                    initial = [
                                        {
                                            "booking_id": a.booking_id,
                                            "driver_id": a.driver_id,
                                        }
                                        for a in final_assignments
                                    ]

                                    optimized = optimizer.optimize_assignments(
                                        initial_assignments=initial,
                                        bookings=bookings_list,
                                        drivers=regs,
                                        matrix_quality=prob_regs.get("matrix_quality"),
                                        coord_quality=prob_regs.get("coord_quality"),
                                    )

                                    # Appliquer les changements
                                    for i, a in enumerate(final_assignments):
                                        if i < len(optimized):
                                            new_driver_id = optimized[i]["driver_id"]
                                            if a.driver_id != new_driver_id:
                                                logger.info(
                                                    "[PipelineExecutor] RL swap: Booking %d → Driver %d (was %d)",
                                                    a.booking_id,
                                                    new_driver_id,
                                                    a.driver_id,
                                                )
                                                a.driver_id = new_driver_id
                                else:
                                    logger.info(
                                        "[PipelineExecutor] RL_POSTOPT_SKIPPED reason=model_unavailable"
                                    )

                        except ImportError as e:
                            logger.info(
                                "[PipelineExecutor] RL_POSTOPT_SKIPPED reason=import_error detail=%s",
                                e,
                            )
                            logger.debug(
                                "[PipelineExecutor] RL optimization import failed: %s",
                                type(e).__name__,
                            )
                        except (
                            ValueError,
                            TypeError,
                            AttributeError,
                        ) as e:
                            # Erreurs attendues : validation, imports manquants
                            logger.debug(
                                "[PipelineExecutor] RL optimization failed (expected error: %s)",
                                type(e).__name__,
                            )
                        except Exception:
                            # Erreur inattendue : logger avec trace complète
                            logger.exception(
                                "[PipelineExecutor] RL optimization failed (unexpected error)"
                            )

                except (ValueError, TypeError, AttributeError) as e:
                    # Erreurs de validation : paramètres invalides, attributs manquants
                    logger.warning(
                        "[PipelineExecutor] Heuristic P1 failed (validation error: %s)",
                        type(e).__name__,
                    )
                except Exception:
                    # Erreur inattendue : logger avec trace complète
                    logger.exception(
                        "[PipelineExecutor] Heuristic P1 failed (unexpected error)"
                    )

            # Solver Pass 1
            remaining_ids = self._remaining_ids_from(prob_regs, assigned_set)
            if (
                remaining_ids
                and mode in ("auto", "solver_only")
                and getattr(settings.features, "enable_solver", True)
            ):
                try:
                    s_sub = self._filter_problem(
                        prob_regs, remaining_ids, company, settings
                    )
                    if h_res and h_res.assignments:
                        s_sub["heuristic_assignments"] = h_res.assignments
                    if perf_collector:
                        with perf_collector.time_step("solver"):
                            s_res = solver.solve(s_sub, settings=settings)
                    else:
                        s_res = solver.solve(s_sub, settings=settings)
                    used_solver = True
                    self._extend_unique(
                        s_res.assignments, final_assignments, assigned_set
                    )
                    logger.info(
                        "[PipelineExecutor] Solver P1: %d assignés, %d non assignés",
                        len(s_res.assignments),
                        len(s_res.unassigned_booking_ids),
                    )
                except (ValueError, TypeError, AttributeError) as e:
                    # Erreurs de validation : paramètres invalides, attributs manquants
                    logger.warning(
                        "[PipelineExecutor] Solver P1 failed (validation error: %s)",
                        type(e).__name__,
                    )
                except Exception:
                    # Erreur inattendue : logger avec trace complète
                    logger.exception(
                        "[PipelineExecutor] Solver P1 failed (unexpected error)"
                    )

            # Fallback Pass 1
            remaining_ids = self._remaining_ids_from(prob_regs, assigned_set)
            if remaining_ids:
                try:
                    fb = heuristics.closest_feasible(
                        prob_regs, remaining_ids, settings=settings
                    )
                    used_fallback = True
                    self._extend_unique(fb.assignments, final_assignments, assigned_set)
                    logger.info(
                        "[PipelineExecutor] Fallback P1: +%d, reste=%d",
                        len(fb.assignments),
                        len(fb.unassigned_booking_ids),
                    )
                except (ValueError, TypeError, AttributeError) as e:
                    # Erreurs de validation : paramètres invalides, attributs manquants
                    logger.warning(
                        "[PipelineExecutor] Fallback P1 failed (validation error: %s)",
                        type(e).__name__,
                    )
                except Exception:
                    # Erreur inattendue : logger avec trace complète
                    logger.exception(
                        "[PipelineExecutor] Fallback P1 failed (unexpected error)"
                    )

        # 6.c Pass 2 - urgences si nécessaire
        remaining_all = self._remaining_ids_from(problem, assigned_set)
        if remaining_all and allow_emg and emgs and company and not clustering_used:
            try:
                used_emergency_pass = True
                logger.info(
                    "[PipelineExecutor] === Pass 2: Adding emergency drivers (%d total) ===",
                    len(regs) + len(emgs),
                )

                bookings_list = problem.get("bookings", [])
                prob_full = data.build_vrptw_problem(
                    company,
                    bookings_list,
                    regs + emgs,
                    settings=settings,
                    base_time=problem.get("base_time"),
                    for_date=problem.get("for_date"),
                )

                # Propager les métadonnées
                for key in [
                    "preferred_driver_id",
                    "company_coords",
                    "driver_load_multipliers",
                ]:
                    if key in problem:
                        prob_full[key] = problem[key]

                # Injecter les états du Pass 1
                latest_result = (
                    fb if (fb and hasattr(fb, "debug") and fb.debug) else h_res
                )
                if (
                    latest_result
                    and hasattr(latest_result, "debug")
                    and latest_result.debug
                ):
                    prob_full["busy_until"] = latest_result.debug.get("busy_until", {})
                    prob_full["driver_scheduled_times"] = latest_result.debug.get(
                        "driver_scheduled_times", {}
                    )
                    prob_full["proposed_load"] = latest_result.debug.get(
                        "proposed_load", {}
                    )

                rem = self._remaining_ids_from(prob_full, assigned_set)

                # Heuristique Pass 2
                h2 = None
                if (
                    rem
                    and mode in ("auto", "heuristic_only")
                    and getattr(settings.features, "enable_heuristics", True)
                ):
                    h_sub2 = self._filter_problem(prob_full, rem, company, settings)
                    h2 = heuristics.assign(h_sub2, settings=settings)
                    used_heuristic = True
                    self._extend_unique(h2.assignments, final_assignments, assigned_set)
                    logger.info(
                        "[PipelineExecutor] Heuristic P2: %d assignés, %d restants",
                        len(h2.assignments),
                        len(h2.unassigned_booking_ids),
                    )

                # Solver Pass 2
                rem = self._remaining_ids_from(prob_full, assigned_set)
                if (
                    rem
                    and mode in ("auto", "solver_only")
                    and getattr(settings.features, "enable_solver", True)
                ):
                    s_sub2 = self._filter_problem(prob_full, rem, company, settings)
                    if h2 and h2.assignments:
                        s_sub2["heuristic_assignments"] = h2.assignments
                    s2 = solver.solve(s_sub2, settings=settings)
                    used_solver = True
                    self._extend_unique(s2.assignments, final_assignments, assigned_set)
                    logger.info(
                        "[PipelineExecutor] Solver P2: %d assignés, %d non assignés",
                        len(s2.assignments),
                        len(s2.unassigned_booking_ids),
                    )

                # Fallback Pass 2
                rem = self._remaining_ids_from(prob_full, assigned_set)
                if rem:
                    if h2 and hasattr(h2, "debug") and h2.debug:
                        prob_full["busy_until"] = h2.debug.get("busy_until", {})
                        prob_full["driver_scheduled_times"] = h2.debug.get(
                            "driver_scheduled_times", {}
                        )
                        prob_full["proposed_load"] = h2.debug.get("proposed_load", {})

                    fb2 = heuristics.closest_feasible(prob_full, rem, settings=settings)
                    used_fallback = True
                    self._extend_unique(
                        fb2.assignments, final_assignments, assigned_set
                    )
                    logger.info(
                        "[PipelineExecutor] Fallback P2: +%d, reste=%d",
                        len(fb2.assignments),
                        len(fb2.unassigned_booking_ids),
                    )

            except (ValueError, TypeError, AttributeError) as e:
                # Erreurs de validation : paramètres invalides, attributs manquants
                logger.warning(
                    "[PipelineExecutor] Emergency pass failed (validation error: %s)",
                    type(e).__name__,
                )
            except Exception:
                # Erreur inattendue : logger avec trace complète
                logger.exception(
                    "[PipelineExecutor] Emergency pass failed (unexpected error)"
                )

        # 6.d Pas de regular_first → pipeline direct
        if not regular_first and company and not clustering_used:
            _phase = "direct"
            rem = self._remaining_ids_from(problem, assigned_set)
            if (
                rem
                and mode in ("auto", "heuristic_only")
                and getattr(settings.features, "enable_heuristics", True)
            ):
                h_sub = self._filter_problem(problem, rem, company, settings)
                h_res = heuristics.assign(h_sub, settings=settings)
                self._extend_unique(h_res.assignments, final_assignments, assigned_set)
            rem = self._remaining_ids_from(problem, assigned_set)
            if (
                rem
                and mode in ("auto", "solver_only")
                and getattr(settings.features, "enable_solver", True)
            ):
                used_solver = True
                s_sub = self._filter_problem(problem, rem, company, settings)
                if h_res and h_res.assignments:
                    s_sub["heuristic_assignments"] = h_res.assignments
                if perf_collector:
                    with perf_collector.time_step("solver"):
                        s_res = solver.solve(s_sub, settings=settings)
                else:
                    s_res = solver.solve(s_sub, settings=settings)
                self._extend_unique(s_res.assignments, final_assignments, assigned_set)
            rem = self._remaining_ids_from(problem, assigned_set)
            if rem:
                fb = heuristics.closest_feasible(problem, rem, settings=settings)
                used_fallback = True
                self._extend_unique(fb.assignments, final_assignments, assigned_set)

        # 6.5) Shadow Mode
        shadow_mode_manager = ShadowModeManager(settings)
        _should_apply_rl, _quality_score_pre_apply = (
            shadow_mode_manager.should_apply_rl(
                company_id=company_id,
                dispatch_run_id=dispatch_run.id if dispatch_run else None,
                final_assignments=final_assignments,
                problem=problem,
                company=company,
            )
        )

        _shadow_suggestions_stored = shadow_mode_manager.generate_and_store_suggestions(
            dispatch_run_id=dispatch_run.id if dispatch_run else None,
            problem=problem,
            final_assignments=final_assignments,
            used_heuristic=used_heuristic,
            used_solver=used_solver,
        )

        # Calculer les IDs non assignés
        unassigned_ids = self._remaining_ids_from(problem, assigned_set)

        return (
            final_assignments,
            unassigned_ids,
            used_heuristic,
            used_solver,
            used_fallback,
            used_emergency_pass,
            _should_apply_rl,
            _quality_score_pre_apply,
            None,
        )

    def _execute_heuristic(
        self, problem: dict[str, Any], settings: Any, mode: str
    ) -> Any:
        """Exécute l'heuristique de dispatch.

        Args:
            problem: Dict contenant les données du problème
            settings: Settings de dispatch
            mode: Mode de dispatch

        Returns:
            Résultat de l'heuristique (HeuristicResult)
        """
        # TODO: Implémenter lors de l'extraction (Phase 7)
        # 1. Appeler heuristics.run() avec problem, settings, mode
        # 2. Retourner résultat
        pass

    def _execute_solver(self, problem: dict[str, Any], settings: Any, mode: str) -> Any:
        """Exécute le solver de dispatch.

        Args:
            problem: Dict contenant les données du problème
            settings: Settings de dispatch
            mode: Mode de dispatch

        Returns:
            Résultat du solver (SolverResult) ou None si désactivé/échec
        """
        # TODO: Implémenter lors de l'extraction (Phase 7)
        # 1. Vérifier si solver activé dans settings.features.enable_solver
        # 2. Si mode == "heuristic_only": retourner None
        # 3. Appeler solver.run() avec problem, settings, mode
        # 4. Retourner résultat ou None
        pass

    def _execute_fallback(self, problem: dict[str, Any], settings: Any) -> Any:
        """Exécute le fallback de dispatch.

        Le fallback est utilisé quand heuristique et solver n'ont pas
        assigné tous les bookings.

        Args:
            problem: Dict contenant les données du problème
            settings: Settings de dispatch

        Returns:
            Résultat du fallback (FallbackResult)
        """
        # TODO: Implémenter lors de l'extraction (Phase 7)
        # 1. Filtrer problem pour ne garder que bookings non assignés
        # 2. Appeler heuristics.run() avec problème filtré et mode fallback
        # 3. Retourner résultat
        pass
