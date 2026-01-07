# backend/infrastructure/dispatch/unified_dispatch_engine.py
from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import asdict
from datetime import date, datetime
from typing import Any, Dict, List, cast

from ext import db
from models import Company
from repositories.company_repository import CompanyRepository
from services.unified_dispatch import data
from services.unified_dispatch.core import settings as ud_settings
from services.unified_dispatch.validation.analysis import UnassignedAnalyzer
from services.unified_dispatch.assignment.assignment_applier import AssignmentApplier
from services.unified_dispatch.locking import RedisLockManager
from shared.constants import GeoConstants, NumericConstants
from shared.otel_setup import get_tracer  # D1: OpenTelemetry

# pyright: reportUnusedImport=false
# pyright: reportUnusedVariable=false
# pyright: reportUnusedFunction=false
# Nombreuses variables conditionnelles utilisées dans des blocs try/except

# ✅ Refactoring: Instances des modules extraits
_lock_manager = RedisLockManager()
_unassigned_analyzer = UnassignedAnalyzer()
_assignment_applier = AssignmentApplier()

# Constantes pour éviter les valeurs magiques
# ✅ REFACTORING: Utilisation de constantes centralisées
DISTANCE_ZERO = NumericConstants.ZERO
DISTANCE_THRESHOLD_KM = 0.1  # ~1km en degrés
ECART_THRESHOLD = NumericConstants.TWO
DATE_FORMAT_LENGTH = 10  # Longueur du format YYYY-MM-DD
CLUSTERING_BOOKINGS_THRESHOLD = 100  # Seuil pour activer le clustering géographique
LATITUDE_MIN = GeoConstants.LATITUDE_MIN
LATITUDE_MAX = GeoConstants.LATITUDE_MAX
LONGITUDE_MIN = GeoConstants.LONGITUDE_MIN
LONGITUDE_MAX = GeoConstants.LONGITUDE_MAX
MAX_BOOKING_IDS_TO_LOG = 20  # Limite le nombre de booking IDs dans les logs

logger = logging.getLogger(__name__)

# ✅ D1: Tracer OpenTelemetry pour traces E2E
tracer = get_tracer("engine")


# ---------- Helpers typage/runtime ----------


def _to_date_ymd(s: str) -> date:
    # accepte 'YYYY-MM-DD' et ISO full (on ne garde que la date)
    try:
        if len(s) == DATE_FORMAT_LENGTH and s[4] == "-" and s[7] == "-":
            return date.fromisoformat(s)
        return datetime.fromisoformat(s).date()
    except (ValueError, TypeError) as err:
        # Erreurs de parsing de date attendues
        msg = f"for_date invalide: {s!r} (attendu 'YYYY-MM-DD')"
        raise ValueError(msg) from err
    except Exception as err:
        # Erreur inattendue : logger et re-lever avec contexte
        logger.exception("Erreur inattendue lors de la conversion de date: %s", s)
        msg = f"for_date invalide: {s!r} (attendu 'YYYY-MM-DD')"
        raise ValueError(msg) from err


def _safe_int(v: Any) -> int | None:
    """Convertit n'importe quelle valeur (y compris un InstrumentedAttribute/Column)
    en int Python ou retourne None. Typé pour apaiser Pylance.
    """
    try:
        return int(v)
    except (ValueError, TypeError, OverflowError):
        # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
        return None
    except Exception:
        # Erreur inattendue : logger et retourner None
        logger.debug("Unexpected error converting to int: %s, returning None", v)
        return None


# ------------------------------------------------------------------
# Verrous distribués Redis pour environnement multi-workers
# ✅ Refactoring: Utilisation de RedisLockManager


def _acquire_day_lock(company_id: int, day_str: str) -> bool:
    """Acquiert un verrou distribue Redis pour eviter les runs concurrents.

    Refactoring: Delegue a RedisLockManager.
    """
    return _lock_manager.acquire(company_id, day_str)


def _release_day_lock(company_id: int, day_str: str) -> None:
    """Libere le verrou distribue Redis.

    Refactoring: Delegue a RedisLockManager.
    """
    _lock_manager.release(company_id, day_str)


def _analyze_unassigned_reasons(
    problem: Dict[str, Any],
    assignments: List[Any],  # Argument conservé pour compatibilité API
    unassigned_ids: List[int],
) -> Dict[int, List[str]]:
    """Analyse les raisons detaillees pour lesquelles certaines courses n'ont
    pas pu etre assignees.

    Refactoring: Delegue a UnassignedAnalyzer.
    """
    return _unassigned_analyzer.analyze(problem, assignments, unassigned_ids)


def run(
    company_id: int,
    mode: str = "auto",
    custom_settings: ud_settings.Settings | None = None,
    *,
    for_date: str | None = None,
    regular_first: bool = True,
    allow_emergency: bool | None = None,
    overrides: dict[str, Any] | None = None,
    existing_dispatch_run_id: int | None = None,
    raise_on_company_not_found: bool = False,
) -> Dict[str, Any]:
    """Exécute un dispatch avec métriques Prometheus intégrées.

    Cette fonction orchestre l'optimisation de dispatch pour une entreprise sur une date donnée.
    Elle crée un enregistrement DispatchRun et lie les assignations à celui-ci.

    **COMPORTEMENT IMPORTANT - ROLLBACK DÉFENSIF** :
    - Cette fonction effectue un rollback défensif au début pour garantir
      un état de session propre, même si une transaction précédente a échoué.
    - Ce rollback peut expirer les objets SQLAlchemy non commités dans la session.
    - **IMPLICATION** : Les objets (Company, Booking, Driver) DOIVENT être commités
      avant d'appeler cette fonction, sinon ils seront expirés et invisibles.

    **📝 UTILISATION DANS LES TESTS** :
    - Utiliser les fixtures `company`, `drivers`, `bookings` qui garantissent le commit
    - Ou appeler `db.session.commit()` explicitement avant `engine.run()`
    - Après `engine.run()`, recharger les objets depuis la DB si nécessaire

    **🔄 GESTION DES TRANSACTIONS** :
    - La fonction crée sa propre transaction pour le DispatchRun et les assignments
    - Les objets commités avant l'appel restent visibles dans le savepoint du test
    - Le rollback défensif n'affecte que les objets non commités dans la session

    Args:
        company_id: ID de l'entreprise pour laquelle exécuter le dispatch
        mode: Mode de dispatch ("auto", "manual", etc.)
        custom_settings: Paramètres personnalisés (None = utiliser paramètres par défaut)
        for_date: Date au format YYYY-MM-DD (None = aujourd'hui)
        regular_first: Prioriser les courses régulières
        allow_emergency: Autoriser les courses d'urgence (None = selon paramètres entreprise)
        overrides: Dict d'overrides pour personnaliser le comportement
        existing_dispatch_run_id: ID d'un DispatchRun existant (pour reprise/continuation)
        raise_on_company_not_found: Si True, lève CompanyNotFoundError au lieu de retourner erreur

    Returns:
        Dict contenant :
        - "success": bool - Indique si le dispatch a réussi
        - "dispatch_run_id": int - ID du DispatchRun créé
        - "assignments": List[Dict] - Liste des assignations créées
        - "assignments_count": int - Nombre d'assignations
        - "unassigned_ids": List[int] - IDs des courses non assignées
        - "metrics": Dict - Métriques de performance
        - "error": str - Message d'erreur si échec

    Raises:
        ValueError: Si for_date invalide ou paramètres invalides
        CompanyNotFoundError: Si company introuvable et raise_on_company_not_found=True
        Exception: En cas d'erreur DB ou dispatch (rollback automatique)

    Side-effects:
        - DB: Crée DispatchRun, Assignment, met à jour Booking.driver_id
        - Redis: Verrou distribué pour éviter runs concurrents
        - Socket.IO: Émissions d'événements temps réel (assignations)
        - Métriques: Prometheus, logging, traces OpenTelemetry
        - DB: Commit transaction (ou rollback en cas d'erreur)

    Exemple:
        >>> from services.unified_dispatch.core.engine import run
        >>> # Dispatch automatique pour aujourd'hui
        >>> result = run(company_id=1, mode="auto")
        >>> if result["success"]:
        ...     print(f"Assignations: {result['assignments_count']}")
        >>> # Dispatch pour une date spécifique
        >>> result = run(
        ...     company_id=1,
        ...     for_date="2025-01-13",
        ...     mode="auto",
        ...     regular_first=True
        ... )
        >>> # Reprise d'un dispatch existant
        >>> result = run(
        ...     company_id=1,
        ...     existing_dispatch_run_id=123,
        ...     mode="manual"
        ... )
    """
    # REFACTORING: Delegation complete a DispatchOrchestrator
    from services.unified_dispatch.orchestration.dispatch_orchestrator import (
        DispatchOrchestrator,
    )

    # D1: Creer span racine pour le dispatch
    with tracer.start_as_current_span("dispatch.run") as root_span:
        root_span.set_attribute("company_id", company_id)
        root_span.set_attribute("mode", mode)
        root_span.set_attribute("for_date", str(for_date) if for_date else "today")

        # Rollback défensif : ignorer silencieusement toute erreur
        # (la session peut déjà être rollback, ou être dans un état invalide)
        # Acceptable: suppress(Exception) ici car rollback défensif non-critique
        with suppress(Exception):
            db.session.rollback()

    # Delegation a l'orchestrateur
    orchestrator = DispatchOrchestrator()
    return orchestrator.execute(
        company_id=company_id,
        mode=mode,
        custom_settings=custom_settings,
        for_date=for_date,
        regular_first=regular_first,
        allow_emergency=allow_emergency,
        overrides=overrides,
        existing_dispatch_run_id=existing_dispatch_run_id,
        raise_on_company_not_found=raise_on_company_not_found,
    )

    # SUPPRIME - Ancienne implementation deplacee vers DispatchOrchestrator
    # La logique complete est maintenant dans orchestration/dispatch_orchestrator.py
    # Cette fonction a ete supprimee pour reduire engine.py a < 1000 lignes
    #
    # Si vous avez besoin de référencer l'ancienne implémentation, consultez :
    # - backend/services/unified_dispatch/orchestration/dispatch_orchestrator.py
    # - Les méthodes extraites : _find_and_validate_company, _configure_settings,
    #   _create_or_reuse_dispatch_run, _build_problem, _execute_dispatch_pipeline,
    #   _apply_assignments, _finalize_metrics_and_result


# ------------------------------------------------------------
# Helpers internes
# ------------------------------------------------------------


def _filter_problem(
    problem: Dict[str, Any], booking_ids: List[int], s: ud_settings.Settings
) -> Dict[str, Any]:
    """Reconstruit un sous-problème avec les mêmes settings que le run principal."""
    bookings_map = {b.id: b for b in problem.get("bookings", [])}
    new_bookings = [bookings_map[bid] for bid in booking_ids if bid in bookings_map]
    drivers = problem.get("drivers", [])
    company_id = problem.get("company_id") or getattr(
        problem.get("company"), "id", None
    )
    company_id = _safe_int(company_id)
    if company_id is None:
        company_id = _safe_int(getattr(problem, "company_id", None))
        # repli : utiliser l'objet company reçu en param de run() si nécessaire
        # (on évite un N+1 en DB, mais on reste safe)
        problem_company_id = getattr(problem, "company_id", None)
        if problem_company_id:
            # ✅ FIX: S'assurer que la Company est visible dans la session
            db.session.flush()  # S'assurer que les objets flushés sont visibles
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            company_repo = CompanyRepository()
            company_dto = company_repo.find_by_id(problem_company_id)
            company_obj = Company.query.get(company_dto.id) if company_dto else None
            company_id = getattr(company_obj, "id", None) if company_obj else None
        else:
            company_id = None

    # Propager for_date et dispatch_run_id
    for_date = problem.get("for_date")
    dispatch_run_id = problem.get("dispatch_run_id")

    # ✅ FIX: S'assurer que la Company est visible dans la session
    if company_id:
        db.session.flush()  # S'assurer que les objets flushés sont visibles
    # ✅ Utilisation du repository pour découpler de SQLAlchemy
    company_repo = CompanyRepository()
    company = None
    if company_id:
        company_dto = company_repo.find_by_id(company_id)
        if company_dto:
            company = cast("Company", Company.query.get(company_dto.id))
    if not company:
        # Si company est None, on ne peut pas continuer
        raise ValueError(f"Company with id {company_id} not found")
    result = data.build_vrptw_problem(
        company,
        new_bookings,
        drivers,
        settings=s,
        base_time=problem.get("base_time"),
        for_date=problem.get("for_date"),
    )

    # Assurer que for_date et dispatch_run_id sont propagés
    if for_date:
        result["for_date"] = for_date
    if dispatch_run_id:
        result["dispatch_run_id"] = dispatch_run_id

    # 📌 CRUCIAL: Propager les états de disponibilité des chauffeurs
    if "busy_until" in problem:
        result["busy_until"] = problem["busy_until"]
    if "driver_scheduled_times" in problem:
        result["driver_scheduled_times"] = problem["driver_scheduled_times"]
    if "proposed_load" in problem:
        result["proposed_load"] = problem["proposed_load"]

    # ⚡ CRUCIAL: Propager preferred_driver_id, company_coords, driver_load_multipliers
    if "preferred_driver_id" in problem:
        result["preferred_driver_id"] = problem["preferred_driver_id"]
    if "company_coords" in problem:
        result["company_coords"] = problem["company_coords"]
    if "driver_load_multipliers" in problem:
        result["driver_load_multipliers"] = problem["driver_load_multipliers"]

    return result


def _apply_and_emit(
    company: Company, assignments: List[Any], dispatch_run_id: int | None
) -> None:
    """Applique les assignations en base et émet événements/notifications.

    Validation temporelle stricte: Si des conflits sont detectes et que
    strict_temporal_validation est active, rollback automatique.

    Refactoring: Delegue a AssignmentApplier.
    """
    _assignment_applier.apply_and_emit(company, assignments, dispatch_run_id)


def _serialize_assignment(a: Any) -> Dict[str, Any]:
    """Sérialise une assignation (SolverAssignment ou autre) en dict API.
    Assure que dispatch_run_id est inclus.
    """
    if hasattr(a, "to_dict"):
        return cast(Dict[str, Any], a.to_dict())

    # Fallback manuel si pas de to_dict()
    out = {}
    for field in [
        "booking_id",
        "driver_id",
        "status",
        "estimated_pickup_arrival",
        "estimated_dropoff_arrival",
        "reason",
        "route_index",
        "dispatch_run_id",
    ]:
        if hasattr(a, field):
            out[field] = getattr(a, field)
    return out


def _serialize_booking(b: Any) -> Dict[str, Any]:
    """Serialisation legere et stable cote API pour diagnostics/front.
    Adaptee si b est un SQLA model ou un objet dataclass.
    """
    try:
        if hasattr(b, "to_dict"):
            return cast(Dict[str, Any], b.to_dict())
    except (AttributeError, TypeError, ValueError) as e:
        # Erreurs attendues : attribut manquant, type incorrect, valeur invalide
        logger.debug(
            "Failed to serialize booking via to_dict (expected): %s", type(e).__name__
        )
    except Exception:
        # Erreur inattendue : logger mais continuer avec fallback
        logger.debug("Unexpected error serializing booking via to_dict")
    try:
        # dataclass support éventuel
        if hasattr(b, "__dataclass_fields__"):
            return asdict(b)
    except (TypeError, ValueError) as e:
        # Erreurs attendues : conversion dataclass échouée, valeurs invalides
        logger.debug(
            "Failed to serialize booking via asdict (expected): %s", type(e).__name__
        )
    except Exception:
        # Erreur inattendue : logger mais continuer avec fallback manuel
        logger.debug("Unexpected error serializing booking via asdict")

    fields = (
        "id",
        "customer_name",
        "pickup_location",
        "dropoff_location",
        "scheduled_time",
        "amount",
        "status",
        "pickup_lat",
        "pickup_lon",
        "dropoff_lat",
        "dropoff_lon",
        "is_return",
        "is_urgent",
        "medical_facility",
        "hospital_service",
        "parent_booking_id",
    )
    out: Dict[str, Any] = {}
    for f in fields:
        if hasattr(b, f):
            out[f] = getattr(b, f)
    return out


def _serialize_driver(d: Any) -> Dict[str, Any]:
    """Sérialisation légère driver pour diagnostics/front."""
    try:
        if hasattr(d, "to_dict"):
            return cast(Dict[str, Any], d.to_dict())
    except (AttributeError, TypeError, ValueError) as e:
        # Erreurs attendues : attribut manquant, type incorrect, valeur invalide
        logger.debug(
            "Failed to serialize driver via to_dict (expected): %s", type(e).__name__
        )
    except Exception:
        # Erreur inattendue : logger mais continuer avec fallback manuel
        logger.debug("Unexpected error serializing driver via to_dict")
    fields = (
        "id",
        "is_active",
        "is_available",
        "latitude",
        "longitude",
        "vehicle_assigned",
        "brand",
        "company_id",
    )
    out: Dict[str, Any] = {}
    for f in fields:
        if hasattr(d, f):
            out[f] = getattr(d, f)
    return out
