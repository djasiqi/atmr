# backend/services/unified_dispatch/orchestration/problem_builder.py
"""Constructeur de problème VRPTW.

Ce module gère la construction du problème VRPTW (Vehicle Routing Problem
with Time Windows) pour le dispatch. Il est responsable de :
- La construction des données du problème via build_problem_data
- La validation des coordonnées géographiques des bookings
- La gestion des cas d'erreur (pas de données, erreurs de construction)
- L'intégration avec OpenTelemetry pour le tracing

Side-effects:
    - Accès DB (lecture bookings, drivers via build_problem_data)
    - Transactions DB (marquage DispatchRun FAILED en cas d'erreur)
    - Métriques: Performance data collection
    - Tracing: OpenTelemetry spans (optionnel)
"""

from __future__ import annotations  # noqa: I001

import logging
from contextlib import nullcontext
from typing import Any, Dict

from models import DispatchStatus
from services.unified_dispatch import data
from services.unified_dispatch.orchestration.dispatch_run_manager import (
    DispatchRunManager,
)
from services.unified_dispatch.orchestration.utils import safe_int
from services.unified_dispatch.transaction_helpers import _begin_tx
from services.unified_dispatch.core.types import DispatchResult
from shared.constants import GeoConstants
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError

try:
    from shared.otel_setup import get_tracer
except ImportError:
    get_tracer = None

logger = logging.getLogger(__name__)


class ProblemBuilder:
    """Constructeur de problème VRPTW pour le dispatch.

    Cette classe centralise la logique de construction du problème VRPTW :
    - Appel à build_problem_data pour récupérer bookings et drivers
    - Validation des coordonnées géographiques
    - Gestion des cas d'erreur (pas de données, erreurs DB, validation)

    Exemple:
        >>> builder = ProblemBuilder()
        >>> problem, error = builder.build(
        ...     company=company,
        ...     company_id=1,
        ...     dispatch_run=dispatch_run,
        ...     settings=settings,
        ...     for_date="2025-01-14",
        ...     day_str="2025-01-14",
        ...     regular_first=True,
        ...     allow_emg=True,
        ...     overrides=None,
        ...     perf_collector=perf_collector
        ... )
        >>> if problem:
        ...     validation = builder.validate_geographic_coordinates(problem)
        ...     print(f"Bookings sans coords: {validation['bookings_without_coords']}")
    """

    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        """Initialise le constructeur de problème."""
        self._dispatch_run_manager = DispatchRunManager()

    def build(
        self,
        _company: Any | None,  # Unused but kept for API consistency
        company_id: int,
        dispatch_run: Any | None,
        settings: Any,
        for_date: str | None,
        day_str: str,
        regular_first: bool,
        allow_emg: bool,
        overrides: dict[str, Any] | None,
        perf_collector: Any | None,
    ) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
        """Construit le problème VRPTW.

        Construit le problème VRPTW en appelant build_problem_data et en
        validant les données. Gère les erreurs et retourne un résultat
        structuré en cas d'échec.

        Args:
            company: Objet Company (non utilisé mais conservé pour API consistency)
            company_id: ID de l'entreprise
            dispatch_run: DispatchRun (peut être None, utilisé pour propagation ID)
            settings: Settings de dispatch
            for_date: Date du dispatch
            day_str: Date au format YYYY-MM-DD
            regular_first: Prioriser les courses régulières
            allow_emg: Autoriser les courses d'urgence
            overrides: Overrides de configuration
            perf_collector: Collecteur de métriques de performance

        Returns:
            Tuple (problem, error_result) où :
            - problem: Dict avec données du problème (bookings, drivers, etc.)
              si succès, None sinon
            - error_result: Dict avec résultat d'erreur structuré si échec,
              None si succès

        Side-effects:
            - Accès DB (lecture via build_problem_data)
            - Transactions DB (marquage DispatchRun FAILED si erreur)
            - Métriques: Performance data collection (via perf_collector)
            - Tracing: OpenTelemetry span "data_prep" (optionnel)
        """
        # Tracer OpenTelemetry (optionnel)
        if get_tracer:
            try:
                tracer = get_tracer("orchestrator")
            except Exception:
                tracer = None
        else:
            tracer = None

        # Construire les données "problème"
        problem: Dict[str, Any] = {}
        try:
            # ✅ D1: Span data_prep (si tracer disponible)
            if tracer:
                span_context = tracer.start_as_current_span("data_prep")
            else:
                span_context = nullcontext()
                data_span = None

            with span_context as data_span:
                problem = (
                    data.build_problem_data(
                        company_id=company_id,
                        settings=settings,
                        for_date=for_date or day_str,
                        regular_first=bool(regular_first),
                        allow_emergency=allow_emg,
                        overrides=overrides or {},
                    )
                    or {}
                )
                n_b = len(problem.get("bookings", []))
                n_d = len(problem.get("drivers", []))
                if data_span:
                    data_span.set_attribute("bookings_count", n_b)
                    data_span.set_attribute("drivers_count", n_d)
                logger.info(
                    "[ProblemBuilder] Problem built: bookings=%d drivers=%d for_date=%s",
                    n_b,
                    n_d,
                    for_date or day_str,
                )

            # ✅ 17. Valider coordonnées géographiques avant dispatch
            self.validate_geographic_coordinates(problem)

            # Propager le dispatch_run_id dans le problem pour qu'il arrive
            # jusqu'au solver
            if dispatch_run:
                drid = safe_int(getattr(dispatch_run, "id", None))
                if drid is not None:
                    problem["dispatch_run_id"] = drid
                    logger.info(
                        "[ProblemBuilder] Added dispatch_run_id=%s to problem", drid
                    )

            # Arrêter le timer de collecte de données
            if perf_collector:
                perf_collector.end_timer("data_collection")
        except (OperationalError, DBAPIError) as e:
            # Erreurs DB transitoires : connexion, timeout
            logger.error(
                "[ProblemBuilder] build_problem_data failed (DB error: %s) for company=%s",
                type(e).__name__,
                company_id,
            )
            logger.exception("[ProblemBuilder] build_problem_data DB error details")
            # Retourner un dict vide en cas d'erreur DB
            problem = {}
        except (ValueError, TypeError, AttributeError) as e:
            # Erreurs de validation : données invalides, attributs manquants
            logger.error(
                "[ProblemBuilder] build_problem_data failed (validation error: %s) for company=%s",
                type(e).__name__,
                company_id,
            )
            logger.exception(
                "[ProblemBuilder] build_problem_data validation error details"
            )
            # Retourner un dict vide en cas d'erreur de validation
            problem = {}
        except Exception:
            # Erreur inattendue : logger avec trace complète
            logger.exception(
                "[ProblemBuilder] build_problem_data failed (unexpected error) for company=%s",
                company_id,
            )
            if dispatch_run:
                # ✅ TX courte pour marquer le run en échec, même si la
                # session a été salie
                try:
                    self._dispatch_run_manager.update_status(
                        dispatch_run, DispatchStatus.FAILED
                    )
                except (OperationalError, DBAPIError, IntegrityError) as e:
                    # Erreurs DB attendues : connexion, contraintes
                    logger.warning(
                        "[ProblemBuilder] Failed to mark DispatchRun FAILED (DB error: %s) after build_problem_data error",
                        type(e).__name__,
                    )
                except Exception:
                    # Erreur inattendue : logger avec trace complète
                    logger.exception(
                        "[ProblemBuilder] Failed to mark DispatchRun FAILED (unexpected error) after build_problem_data error"
                    )
                # ✅ FIX: Calculer dispatch_run_id avant retour
                drid_on_error = (
                    safe_int(getattr(dispatch_run, "id", None))
                    if dispatch_run
                    else None
                )
                # ✅ Standardisation: Utiliser DispatchResult
                error_result = DispatchResult(
                    dispatch_run_id=drid_on_error,
                    assignments=[],
                    unassigned=[],
                    bookings=[],
                    drivers=[],
                    meta={
                        "reason": "problem_build_failed",
                        "for_date": for_date or day_str,
                        "dispatch_run_id": drid_on_error,
                    },
                    debug={
                        "reason": "problem_build_failed",
                        "for_date": for_date or day_str,
                        "dispatch_run_id": drid_on_error,
                    },
                ).to_dict()
                # ✅ Retourner immédiatement si build_problem_data a échoué
                if dispatch_run:
                    try:
                        with _begin_tx():
                            dispatch_run.mark_failed("problem_build_failed")
                    except (OperationalError, DBAPIError, IntegrityError) as e:
                        # Erreurs DB attendues : connexion, contraintes
                        logger.warning(
                            "[ProblemBuilder] Failed to mark DispatchRun FAILED (DB error: %s)",
                            type(e).__name__,
                        )
                    except Exception:
                        # Erreur inattendue : logger avec trace complète
                        logger.exception(
                            "[ProblemBuilder] Failed to mark DispatchRun FAILED (unexpected error)"
                        )
                return None, error_result

        # Continuer avec le traitement normal si problem est défini
        # Si problem n'a pas été défini (erreur dans except), utiliser {}
        if not problem:
            problem = {}
        if not problem or not problem.get("bookings") or not problem.get("drivers"):
            logger.info(
                "[ProblemBuilder] Pas de données à dispatcher (company=%s)", company_id
            )
            if dispatch_run:
                # ✅ TX courte pour compléter proprement le run "no_data"
                try:
                    self._dispatch_run_manager.finalize(
                        dispatch_run, assignments_count=0, unassigned_count=0
                    )
                except (OperationalError, DBAPIError, IntegrityError) as e:
                    # Erreurs DB attendues : connexion, contraintes
                    logger.warning(
                        "[ProblemBuilder] Failed to mark DispatchRun COMPLETED (DB error: %s) for no_data",
                        type(e).__name__,
                    )
                except Exception:
                    # Erreur inattendue : logger avec trace complète
                    logger.exception(
                        "[ProblemBuilder] Failed to mark DispatchRun COMPLETED (unexpected error) for no_data"
                    )
            # Si problem est vide (erreur DB ou validation), retourner problem, None
            # Si problem a des données mais pas de bookings/drivers, retourner None, error_result
            if not problem:
                # Erreur DB ou validation : retourner problem vide, pas d'error_result
                return problem, None
            # ✅ Standardisation: Utiliser DispatchResult
            drid_no_data = (
                safe_int(getattr(dispatch_run, "id", None)) if dispatch_run else None
            )
            error_result = DispatchResult(
                dispatch_run_id=drid_no_data,
                assignments=[],
                unassigned=[],
                bookings=problem.get("bookings", []),
                drivers=problem.get("drivers", []),
                meta={
                    "reason": "no_data",
                    "for_date": for_date or day_str,
                    "dispatch_run_id": drid_no_data,
                },
                debug={
                    "reason": "no_data",
                    "for_date": for_date or day_str,
                    "dispatch_run_id": drid_no_data,
                },
            ).to_dict()
            return None, error_result

        return problem, None

    def validate_geographic_coordinates(
        self, problem: dict[str, Any]
    ) -> dict[str, list[int]]:
        """Valide les coordonnées géographiques des bookings.

        Vérifie que tous les bookings ont des coordonnées valides (pickup et dropoff).
        Identifie les bookings sans coordonnées et ceux avec coordonnées invalides
        (hors plages valides).

        Args:
            problem: Dict contenant les données du problème avec clé "bookings"

        Returns:
            Dict avec clés :
            - bookings_without_coords: List[int] des IDs de bookings sans coordonnées
            - bookings_with_invalid_coords: List[int] des IDs avec coordonnées invalides

        Side-effects:
            - Logging: Warnings pour bookings sans/invalides coordonnées
        """
        # Constantes pour validation géographique
        LATITUDE_MIN = GeoConstants.LATITUDE_MIN
        LATITUDE_MAX = GeoConstants.LATITUDE_MAX
        LONGITUDE_MIN = GeoConstants.LONGITUDE_MIN
        LONGITUDE_MAX = GeoConstants.LONGITUDE_MAX
        MAX_BOOKING_IDS_TO_LOG = 20  # Limite le nombre de booking IDs dans les logs

        bookings_list = problem.get("bookings", [])
        if not bookings_list:
            return {"bookings_without_coords": [], "bookings_with_invalid_coords": []}

        bookings_without_coords = []
        bookings_with_invalid_coords = []
        for booking in bookings_list:
            pickup_lat = getattr(booking, "pickup_lat", None)
            pickup_lon = getattr(booking, "pickup_lon", None)
            dropoff_lat = getattr(booking, "dropoff_lat", None)
            dropoff_lon = getattr(booking, "dropoff_lon", None)

            # Vérifier pickup
            pickup_missing = pickup_lat is None or pickup_lon is None
            pickup_invalid = False
            if not pickup_missing and pickup_lat is not None and pickup_lon is not None:
                try:
                    lat_float = float(pickup_lat)
                    lon_float = float(pickup_lon)
                    # Valider les plages de coordonnées
                    if not (LATITUDE_MIN <= lat_float <= LATITUDE_MAX) or not (
                        LONGITUDE_MIN <= lon_float <= LONGITUDE_MAX
                    ):
                        pickup_invalid = True
                except (ValueError, TypeError):
                    pickup_invalid = True

            # Vérifier dropoff
            dropoff_missing = dropoff_lat is None or dropoff_lon is None
            dropoff_invalid = False
            if (
                not dropoff_missing
                and dropoff_lat is not None
                and dropoff_lon is not None
            ):
                try:
                    lat_float = float(dropoff_lat)
                    lon_float = float(dropoff_lon)
                    # Valider les plages de coordonnées
                    if not (LATITUDE_MIN <= lat_float <= LATITUDE_MAX) or not (
                        LONGITUDE_MIN <= lon_float <= LONGITUDE_MAX
                    ):
                        dropoff_invalid = True
                except (ValueError, TypeError):
                    dropoff_invalid = True

            booking_id = getattr(booking, "id", None)
            if booking_id:
                if pickup_missing or dropoff_missing:
                    bookings_without_coords.append(booking_id)
                elif pickup_invalid or dropoff_invalid:
                    bookings_with_invalid_coords.append(booking_id)

        if bookings_without_coords:
            logger.warning(
                "[ProblemBuilder] ⚠️ %d booking(s) sans coordonnées géographiques (pickup ou dropoff manquantes) : %s",
                len(bookings_without_coords),
                bookings_without_coords[:MAX_BOOKING_IDS_TO_LOG],
            )

        if bookings_with_invalid_coords:
            logger.warning(
                "[ProblemBuilder] ⚠️ %d booking(s) avec coordonnées géographiques invalides (hors plages valides) : %s",
                len(bookings_with_invalid_coords),
                bookings_with_invalid_coords[:MAX_BOOKING_IDS_TO_LOG],
            )

        return {
            "bookings_without_coords": bookings_without_coords,
            "bookings_with_invalid_coords": bookings_with_invalid_coords,
        }
