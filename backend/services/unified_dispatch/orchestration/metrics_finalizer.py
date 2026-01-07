# backend/services/unified_dispatch/orchestration/metrics_finalizer.py
"""Finaliseur de métriques pour le dispatch.

Ce module gère la finalisation et l'enregistrement des métriques du dispatch.
Il est responsable de :
- Le calcul des métriques agrégées (assignations, non-assignés, temps)
- L'analyse des raisons de non-assignation
- L'enregistrement des métriques Prometheus (optionnel)
- La construction du résultat final avec métadonnées

Side-effects:
    - Métriques: Enregistrement Prometheus (optionnel)
    - Accès DB (lecture DispatchRun pour métadonnées)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError

from services.unified_dispatch.orchestration.result_builder import ResultBuilder
from services.unified_dispatch.orchestration.utils import safe_int, to_date_ymd
from services.unified_dispatch.transaction_helpers import _begin_tx

logger = logging.getLogger(__name__)


class MetricsFinalizer:
    """Finaliseur de métriques pour le dispatch.

    Cette classe centralise la logique de finalisation des métriques :
    - Calcul des métriques agrégées
    - Analyse des raisons de non-assignation
    - Enregistrement Prometheus (optionnel)
    - Construction du résultat final

    Exemple:
        >>> finalizer = MetricsFinalizer()
        >>> result = finalizer.finalize(
        ...     company_id=1,
        ...     problem=problem,
        ...     final_assignments=assignments,
        ...     unassigned_ids=[5, 10],
        ...     dispatch_run=dispatch_run,
        ...     settings=settings,
        ...     mode="auto",
        ...     regular_first=True,
        ...     allow_emg=True,
        ...     for_date="2025-01-14",
        ...     day_str="2025-01-14",
        ...     used_heuristic=True,
        ...     used_solver=False,
        ...     used_fallback=False,
        ...     used_emergency_pass=False,
        ...     h_res=h_result,
        ...     s_res=None,
        ...     perf_collector=perf_collector,
        ...     should_apply_rl=False,
        ...     kpi_monitor=None
        ... )
    """

    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        """Initialise le finaliseur de métriques."""
        self._result_builder = ResultBuilder()

    def finalize(
        self,
        company_id: int,
        problem: dict[str, Any],
        final_assignments: list[Any],
        unassigned_ids: list[int],
        dispatch_run: Any | None,
        settings: Any,
        mode: str,
        regular_first: bool,
        allow_emg: bool,
        for_date: str | None,
        day_str: str,
        used_heuristic: bool,
        used_solver: bool,
        used_fallback: bool,
        used_emergency_pass: bool,
        h_res: Any | None,
        s_res: Any | None,
        perf_collector: Any | None,
        should_apply_rl: bool,
        kpi_monitor: Any | None,
    ) -> dict[str, Any]:
        """Finalise les métriques et construit le résultat final.

        Calcule toutes les métriques agrégées, analyse les raisons de
        non-assignation, enregistre les métriques Prometheus (si activé),
        et construit le résultat final avec toutes les métadonnées.

        Args:
            company_id: ID de l'entreprise
            problem: Dict contenant les données du problème VRPTW
            final_assignments: List des assignations finales
            unassigned_ids: List des IDs de bookings non assignés
            dispatch_run: DispatchRun (peut être None)
            settings: Settings de dispatch
            mode: Mode de dispatch
            regular_first: Prioriser les courses régulières
            allow_emg: Autoriser les courses d'urgence
            for_date: Date du dispatch
            day_str: Date au format YYYY-MM-DD
            used_heuristic: Bool indiquant si heuristique utilisée
            used_solver: Bool indiquant si solver utilisé
            used_fallback: Bool indiquant si fallback utilisé
            used_emergency_pass: Bool indiquant si pass d'urgence utilisé
            h_res: Résultat heuristique (peut être None)
            s_res: Résultat solver (peut être None)
            perf_collector: Collecteur de métriques de performance (optionnel)
            should_apply_rl: Bool indiquant si RL doit être appliqué
            kpi_monitor: KPI Monitor pour RL (optionnel)

        Returns:
            Dict contenant le résultat final du dispatch avec :
            - dispatch_run_id: ID du DispatchRun
            - assignments: List des assignations sérialisées
            - unassigned: List des IDs non assignés
            - bookings: List des bookings sérialisés
            - drivers: List des drivers sérialisés
            - meta: Dict avec métriques agrégées
            - debug: Dict avec informations de debug

        Side-effects:
            - Métriques: Enregistrement Prometheus (optionnel)
            - Accès DB (lecture DispatchRun pour métadonnées)
        """
        # Analyser les raisons détaillées de non-assignation
        unassigned_reasons = self._analyze_unassigned_reasons(
            problem, final_assignments, unassigned_ids
        )

        # Mesures de performance agrégées si disponibles
        h_calls = 0
        h_avg = 0
        h_time = 0
        if h_res is not None:
            try:
                h_calls = int(getattr(h_res, "osrm_calls", 0))
                h_avg = int(getattr(h_res, "osrm_avg_latency_ms", 0))
                h_time = int(getattr(h_res, "heuristic_time_ms", 0))
            except (ValueError, TypeError, AttributeError):
                # Erreurs attendues : valeurs invalides, attributs manquants
                pass
            except Exception:
                # Erreur inattendue : ignorer silencieusement (métriques non-critiques)
                pass
        s_time = 0
        if s_res is not None:
            try:
                s_time = int(getattr(s_res, "solver_time_ms", 0))
            except (ValueError, TypeError, AttributeError):
                # Erreurs attendues : valeurs invalides, attributs manquants
                pass
            except Exception:
                # Erreur inattendue : ignorer silencieusement (métriques non-critiques)
                pass

        metrics = {
            "assignments_count": len(final_assignments),
            "unassigned_count": len(unassigned_ids),
            "mode": mode,
            "regular_first": regular_first,
            "allow_emergency": allow_emg,
            "unassigned_reasons": unassigned_reasons,
            "osrm_calls": h_calls,
            "osrm_avg_latency_ms": h_avg,
            "heuristic_time_ms": h_time,
            "solver_time_ms": s_time,
        }

        debug_info: dict[str, Any] = {
            "heuristic": getattr(h_res, "debug", None) if h_res else None,
            "solver": getattr(s_res, "debug", None) if s_res else None,
            "settings": settings.to_dict() if hasattr(settings, "to_dict") else None,
            "for_date": for_date or day_str,
            "regular_first": regular_first,
            "allow_emergency": allow_emg,
            "unassigned_after": unassigned_ids,
            "phase": "regular_then_emergency"
            if used_emergency_pass and regular_first
            else ("regular_only" if regular_first else "direct"),
            "used_heuristic": used_heuristic,
            "used_solver": used_solver,
            "used_fallback": used_fallback,
        }

        try:
            if "matrix_provider" in problem:
                debug_info["matrix_provider"] = problem["matrix_provider"]
            if "matrix_units" in problem:
                debug_info["matrix_units"] = problem["matrix_units"]
        except (KeyError, AttributeError, TypeError):
            # Erreurs attendues : clés manquantes, attributs manquants
            pass
        except Exception:
            # Erreur inattendue : logger mais continuer (non-critique)
            logger.debug(
                "[MetricsFinalizer] Unexpected error accessing problem metadata"
            )
            pass

        drid = safe_int(getattr(dispatch_run, "id", None)) if dispatch_run else None
        debug_info["dispatch_run_id"] = drid

        # Finaliser le run - TX courte
        if dispatch_run:
            try:
                with _begin_tx():
                    dispatch_run.mark_completed(metrics)
            except (OperationalError, DBAPIError, IntegrityError) as e:
                # Erreurs DB attendues : connexion, contraintes
                logger.warning(
                    "[MetricsFinalizer] Failed to complete DispatchRun id=%s (DB error: %s)",
                    getattr(dispatch_run, "id", None),
                    type(e).__name__,
                )
            except Exception:
                # Erreur inattendue : logger avec trace complète
                logger.exception(
                    "[MetricsFinalizer] Failed to complete DispatchRun id=%s (unexpected error)",
                    getattr(dispatch_run, "id", None),
                )

        # Collecter les métriques analytics (asynchrone)
        try:
            from services.analytics.metrics_collector import collect_dispatch_metrics

            if drid is not None:
                collect_dispatch_metrics(
                    dispatch_run_id=drid,
                    company_id=company_id,
                    day=for_date
                    if isinstance(for_date, date)
                    else to_date_ymd(for_date or day_str),
                )
        except ImportError:
            # Erreur attendue : module optionnel non disponible
            pass
        except (ValueError, TypeError) as e:
            # Erreurs de validation : paramètres invalides
            logger.debug(
                "[MetricsFinalizer] Failed to collect analytics metrics (validation error: %s)",
                type(e).__name__,
            )
        except Exception:
            # Erreur inattendue : logger mais continuer (métriques non-critiques)
            logger.debug(
                "[MetricsFinalizer] Failed to collect analytics metrics (unexpected error)"
            )

        # Collecter les métriques de qualité du dispatch
        try:
            from services.unified_dispatch.metrics.dispatch import (
                collect_dispatch_metrics as collect_quality_metrics,
            )

            if drid is not None:
                quality_metrics = collect_quality_metrics(
                    dispatch_run_id=drid,
                    company_id=company_id,
                    day=for_date
                    if isinstance(for_date, date)
                    else to_date_ymd(for_date or day_str),
                )
                logger.info(
                    "[MetricsFinalizer] Dispatch quality score: %.1f/100 (assignment: %.1f%%, on-time: %.1f%%, pooling: %.1f%%)",
                    quality_metrics.quality_score,
                    quality_metrics.assignment_rate,
                    (
                        quality_metrics.on_time_bookings
                        / max(1, quality_metrics.total_bookings)
                    )
                    * 100,
                    quality_metrics.pooling_rate,
                )
                debug_info["quality_metrics"] = quality_metrics.to_summary()

                # KPI MONITOR : Vérifier les KPIs et déclencher backout si nécessaire
                if should_apply_rl and kpi_monitor:
                    try:
                        avg_delay_min = 0.0
                        if (
                            hasattr(quality_metrics, "total_late_minutes")
                            and hasattr(quality_metrics, "total_bookings")
                            and quality_metrics.total_bookings > 0
                        ):
                            avg_delay_min = (
                                quality_metrics.total_late_minutes
                                / quality_metrics.total_bookings
                            )

                        kpis = {
                            "quality_score": quality_metrics.quality_score,
                            "on_time_rate": quality_metrics.assignment_rate,
                            "avg_delay_min": avg_delay_min,
                        }

                        should_backout, reason = kpi_monitor.check_kpis(
                            company_id, kpis
                        )

                        if should_backout:
                            logger.error(
                                "[MetricsFinalizer] Company %d: BACKOUT TRIGGERED - %s",
                                company_id,
                                reason,
                            )
                            settings.features.enable_rl_apply = False
                            debug_info["rl_backout"] = {
                                "triggered": True,
                                "reason": reason,
                                "kpis": kpis,
                            }
                        else:
                            logger.info(
                                "[MetricsFinalizer] Company %d: KPIs OK (quality_score=%.1f, on_time_rate=%.1f%%, avg_delay=%.1f min)",
                                company_id,
                                kpis["quality_score"],
                                kpis["on_time_rate"],
                                kpis["avg_delay_min"],
                            )
                    except (ValueError, TypeError, AttributeError) as e:
                        # Erreurs de validation : KPIs invalides, attributs manquants
                        logger.warning(
                            "[MetricsFinalizer] Failed to check KPIs (validation error: %s): %s",
                            type(e).__name__,
                            e,
                        )
                    except Exception as e:
                        # Erreur inattendue : logger et continuer
                        logger.warning(
                            "[MetricsFinalizer] Failed to check KPIs (unexpected error): %s",
                            e,
                        )
        except (ValueError, TypeError, AttributeError, ImportError) as e:
            # Erreurs attendues : validation, imports manquants
            logger.warning(
                "[MetricsFinalizer] Failed to collect quality metrics (expected error: %s): %s",
                type(e).__name__,
                e,
            )
        except Exception as e:
            # Erreur inattendue : logger et continuer
            logger.warning(
                "[MetricsFinalizer] Failed to collect quality metrics (unexpected error): %s",
                e,
            )

        # Finaliser les métriques de performance
        feature_flags = {
            "enable_heuristics": settings.features.enable_heuristics,
            "enable_solver": settings.features.enable_solver,
            "enable_rl": settings.features.enable_rl,
            "enable_clustering": settings.features.enable_clustering,
            "enable_parallel_heuristics": settings.features.enable_parallel_heuristics,
        }

        algorithm_used = "unknown"
        if used_solver:
            algorithm_used = "solver"
        elif used_heuristic:
            algorithm_used = "heuristics"
        elif used_fallback:
            algorithm_used = "fallback"

        perf_metrics = None
        if perf_collector:
            perf_metrics = perf_collector.finalize(
                algorithm_used=algorithm_used, feature_flags=feature_flags
            )

            # Enregistrer métriques Prometheus
            self._record_prometheus_metrics(
                company_id=company_id,
                dispatch_run_id=drid,
                problem=problem,
                final_assignments=final_assignments,
                unassigned_ids=unassigned_ids,
                perf_metrics=perf_metrics,
                mode=mode,
            )

        # Construire le résultat final
        return self._result_builder.build(
            dispatch_run_id=drid,
            assignments=final_assignments,
            unassigned_ids=unassigned_ids,
            bookings=problem.get("bookings", []),
            drivers=problem.get("drivers", []),
            meta=metrics,
            debug=debug_info,
        )

    def _record_prometheus_metrics(
        self,
        company_id: int,
        dispatch_run_id: int | None,
        problem: dict[str, Any],
        final_assignments: list[Any],
        unassigned_ids: list[int],
        perf_metrics: Any | None,
        mode: str,
    ) -> None:
        """Enregistre les métriques Prometheus (optionnel).

        Enregistre les métriques Prometheus pour le monitoring si Prometheus
        est disponible. Gère gracieusement les erreurs (imports manquants,
        etc.) car Prometheus est optionnel.

        Args:
            company_id: ID de l'entreprise
            dispatch_run_id: ID du DispatchRun (peut être None)
            problem: Dict contenant les données du problème
            final_assignments: List des assignations finales
            unassigned_ids: List des IDs non assignés
            perf_metrics: Métriques de performance (peut être None)
            mode: Mode de dispatch

        Side-effects:
            - Métriques: Enregistrement Prometheus (si disponible)
        """
        try:
            from services.unified_dispatch.metrics.prometheus import (
                record_assignment_rate,
                record_assignments_created,
                record_bookings_processed,
                record_data_collection_time,
                record_db_conflicts,
                record_dispatch_duration,
                record_dispatch_quality,
                record_drivers_available,
                record_drivers_total,
                record_heuristics_time,
                record_persistence_time,
                record_solver_time,
                record_temporal_conflicts,
                record_unassigned_count,
            )

            if dispatch_run_id is not None and perf_metrics:
                # Qualité
                if perf_metrics.quality_score > 0:
                    record_dispatch_quality(
                        perf_metrics.quality_score, dispatch_run_id, company_id
                    )

                # Taux d'assignation
                if perf_metrics.assignment_rate > 0:
                    record_assignment_rate(
                        perf_metrics.assignment_rate, dispatch_run_id, company_id
                    )

                # Non assignés
                unassigned_count = len(unassigned_ids) if unassigned_ids else 0
                record_unassigned_count(unassigned_count, dispatch_run_id, company_id)

                # Conflits temporels
                if perf_metrics.temporal_conflicts_count > 0:
                    record_temporal_conflicts(
                        perf_metrics.temporal_conflicts_count,
                        dispatch_run_id,
                        company_id,
                    )

                # Conflits DB
                if perf_metrics.db_conflicts_count > 0:
                    record_db_conflicts(
                        perf_metrics.db_conflicts_count, dispatch_run_id, company_id
                    )

                # Durée totale
                if perf_metrics.total_time > 0:
                    record_dispatch_duration(perf_metrics.total_time, mode, company_id)

                # Métriques supplémentaires
                if perf_metrics.bookings_processed > 0:
                    record_bookings_processed(
                        perf_metrics.bookings_processed, dispatch_run_id, company_id
                    )

                if perf_metrics.drivers_available > 0:
                    record_drivers_available(
                        perf_metrics.drivers_available, dispatch_run_id, company_id
                    )

                drivers_list = problem.get("drivers", [])
                drivers_total = len(drivers_list)
                if drivers_total > 0:
                    record_drivers_total(drivers_total, dispatch_run_id, company_id)

                assignments_created = len(final_assignments) if final_assignments else 0
                if assignments_created > 0:
                    record_assignments_created(
                        assignments_created, dispatch_run_id, company_id
                    )

                if perf_metrics.data_collection_time > 0:
                    record_data_collection_time(
                        perf_metrics.data_collection_time, company_id
                    )

                if perf_metrics.heuristics_time > 0:
                    record_heuristics_time(perf_metrics.heuristics_time, company_id)

                if perf_metrics.solver_time > 0:
                    record_solver_time(perf_metrics.solver_time, company_id)

                if perf_metrics.persistence_time > 0:
                    record_persistence_time(perf_metrics.persistence_time, company_id)

        except (ValueError, TypeError, AttributeError, ImportError) as e:
            # Erreurs attendues : validation, imports manquants (Prometheus optionnel)
            logger.debug(
                "[MetricsFinalizer] Failed to record Prometheus metrics (expected error: %s): %s",
                type(e).__name__,
                e,
            )
        except Exception as e:
            # Erreur inattendue : logger et continuer (métriques non-critiques)
            logger.warning(
                "[MetricsFinalizer] Failed to record Prometheus metrics (unexpected error): %s",
                e,
            )

    def _analyze_unassigned_reasons(
        self,
        problem: dict[str, Any],
        assignments: list[Any],
        unassigned_ids: list[int],
    ) -> dict[str, Any]:
        """Analyse les raisons détaillées de non-assignation.

        Utilise UnassignedAnalyzer pour analyser pourquoi certains bookings
        n'ont pas été assignés. Retourne un dict avec les raisons par booking ID.

        Args:
            problem: Dict contenant les données du problème
            assignments: List des assignations créées
            unassigned_ids: List des IDs de bookings non assignés

        Returns:
            Dict avec clés = booking IDs (str) et valeurs = List[str] de raisons,
            ou dict vide si analyse échoue

        Side-effects:
            - Logging: Warnings si analyse échoue
        """
        try:
            from services.unified_dispatch.analysis import UnassignedAnalyzer

            analyzer = UnassignedAnalyzer()
            result = analyzer.analyze(problem, assignments, unassigned_ids)
            # Convertir Dict[int, List[str]] en Dict[str, Any] pour compatibilité
            return {str(k): v for k, v in result.items()} if result else {}
        except (ValueError, TypeError, AttributeError, ImportError) as e:
            # Erreurs attendues : validation, imports manquants
            logger.debug(
                "[MetricsFinalizer] UnassignedAnalyzer failed (expected error: %s)",
                type(e).__name__,
            )
            return {}
        except Exception:
            # Erreur inattendue : logger et retourner dict vide
            logger.warning(
                "[MetricsFinalizer] UnassignedAnalyzer failed (unexpected error), returning empty dict"
            )
            return {}
