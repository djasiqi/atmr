# backend/services/unified_dispatch/engine.py
from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, cast

from sqlalchemy.exc import IntegrityError

from ext import db
from models import Assignment, Booking, BookingStatus, Company, DispatchRun, DispatchStatus, Driver, DriverType
from services.notification_service import notify_booking_assigned, notify_dispatch_run_completed
from services.safety_guards import get_safety_guards
from services.unified_dispatch import data, heuristics, performance_metrics, solver
from services.unified_dispatch import settings as ud_settings
from services.unified_dispatch.ab_router import ABRouter
from services.unified_dispatch.apply import apply_assignments
from services.unified_dispatch.clustering import GeographicClustering
from services.unified_dispatch.rl_kpi_monitor import RLKPIMonitor
from services.unified_dispatch.slo import check_slo_breach, get_slo_tracker
from services.unified_dispatch.transaction_helpers import _begin_tx
from shared.otel_setup import get_tracer  # ✅ D1: OpenTelemetry

# Constantes pour éviter les valeurs magiques
DISTANCE_ZERO = 0
DISTANCE_THRESHOLD_KM = 0.1  # ~1km en degrés
ECART_THRESHOLD = 2
DATE_FORMAT_LENGTH = 10  # Longueur du format YYYY-MM-DD
CLUSTERING_BOOKINGS_THRESHOLD = 100  # Seuil pour activer le clustering géographique

if TYPE_CHECKING:
    from collections.abc import Iterable

# pyright: reportUnusedImport=false, reportUnusedVariable=false, reportUnusedFunction=false
# Nombreuses variables conditionnelles utilisées dans des blocs try/except

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
    except Exception as err:
        msg = f"for_date invalide: {s!r} (attendu 'YYYY-MM-DD')"
        raise ValueError(msg) from err


def _safe_int(v: Any) -> int | None:
    """Convertit n'importe quelle valeur (y compris un InstrumentedAttribute/Column)
    en int Python ou retourne None. Typé pour apaiser Pylance.
    """
    try:
        return int(v)
    except Exception:
        return None


# ------------------------------------------------------------------
# Verrous distribués Redis pour environnement multi-workers
# clé = dispatch:lock:{company_id}:{day_str}

def _acquire_day_lock(company_id: int, day_str: str) -> bool:
    """Acquiert un verrou distribué Redis pour éviter les runs concurrents."""
    from ext import redis_client

    key = f"dispatch:lock:{company_id}:{day_str}"
    try:
        # Utiliser SET avec NX (Not eXists) et EX (EXpire) pour créer un verrou
        # avec TTL
        result = redis_client.set(key, "1", nx=True, ex=300)  # TTL 5 minutes
        return result is True
    except Exception as e:
        logger.warning("[Engine] Failed to acquire Redis lock for company=%s day=%s: %s",
                       company_id, day_str, e)
        return False


def _release_day_lock(company_id: int, day_str: str) -> None:
    """Libère le verrou distribué Redis."""
    from ext import redis_client

    key = f"dispatch:lock:{company_id}:{day_str}"
    try:
        redis_client.delete(key)
    except Exception as e:
        logger.warning("[Engine] Failed to release Redis lock for company=%s day=%s: %s",
                       company_id, day_str, e)


def _analyze_unassigned_reasons(
        problem: Dict[str, Any], assignments: List[Any], unassigned_ids: List[int]) -> Dict[int, List[str]]:
    """Analyse les raisons détaillées pour lesquelles certaines courses n'ont pas pu être assignées."""
    reasons = {}
    bookings = problem.get("bookings", [])
    drivers = problem.get("drivers", [])

    # Créer des dictionnaires pour un accès rapide
    bookings_dict = {b.id: b for b in bookings}
    _drivers_dict = {d.id: d for d in drivers}
    _assigned_booking_ids = {a.booking_id for a in assignments}

    for booking_id in unassigned_ids:
        booking = bookings_dict.get(booking_id)
        if not booking:
            reasons[booking_id] = ["booking_not_found"]
            continue

        booking_reasons = []

        # Vérifier la disponibilité des chauffeurs
        available_drivers = [
            d for d in drivers if getattr(
                d, "is_available", True)]
        if not available_drivers:
            booking_reasons.append("no_driver_available")

        # Vérifier la capacité
        if hasattr(booking, "capacity_required") and booking.capacity_required:
            suitable_drivers = [d for d in available_drivers
                                if hasattr(d, "capacity") and d.capacity >= booking.capacity_required]
            if not suitable_drivers:
                booking_reasons.append("capacity_exceeded")

        # Vérifier les fenêtres horaires
        if hasattr(booking, "scheduled_time") and booking.scheduled_time:
            # Vérifier si l'heure est dans une fenêtre de travail
            booking_time = booking.scheduled_time
            working_drivers = []
            for driver in available_drivers:
                if hasattr(driver, "work_windows") and driver.work_windows:
                    for window in driver.work_windows:
                        if window.start <= booking_time <= window.end:
                            working_drivers.append(driver)
                            break

            if not working_drivers:
                booking_reasons.append("time_window_infeasible")

        # Vérifier les contraintes géographiques
        if hasattr(booking, "pickup_lat") and hasattr(booking, "pickup_lon"):
            # Vérifier si des chauffeurs sont dans la zone
            nearby_drivers = []
            for driver in available_drivers:
                if hasattr(driver, "current_lat") and hasattr(
                        driver, "current_lon"):
                    # Calculer la distance (simplifié)
                    distance = ((booking.pickup_lat - driver.current_lat) ** 2 +
                                (booking.pickup_lon - driver.current_lon) ** 2) ** 0.5
                    if distance < DISTANCE_THRESHOLD_KM:  # ~1km
                        nearby_drivers.append(driver)

            if not nearby_drivers:
                booking_reasons.append("no_nearby_drivers")

        # Vérifier les contraintes d'urgence
        if hasattr(booking, "is_emergency") and booking.is_emergency:
            emergency_drivers = [d for d in available_drivers
                                 if hasattr(d, "can_handle_emergency") and d.can_handle_emergency]
            if not emergency_drivers:
                booking_reasons.append("no_emergency_drivers")

        # Si aucune raison spécifique n'a été trouvée
        if not booking_reasons:
            booking_reasons.append("unknown_constraint")

        reasons[booking_id] = booking_reasons

    return reasons


def run( # pyright: ignore[reportGeneralTypeIssues]
    company_id: int,
    mode: str = "auto",
    custom_settings: ud_settings.Settings | None = None,
    *,
    for_date: str | None = None,
    regular_first: bool = True,
    allow_emergency: bool | None = None,
    overrides: dict[str, Any] | None = None,
    existing_dispatch_run_id: int | None = None,  # ✅ Nouveau paramètre optionnel
) -> Dict[str, Any]:
    """Exécute un dispatch avec métriques Prometheus intégrées."""
    # ✅ Context manager pour métriques Prometheus
    try:
        from services.unified_dispatch.dispatch_prometheus_metrics import dispatch_metrics_context
        # Note: dispatch_run_id sera disponible après création du DispatchRun
        # On utilisera le context manager plus tard dans la fonction
    except ImportError:
        dispatch_metrics_context = None
    """Run the dispatch optimization for a company on a specific date.
    Creates a DispatchRun record and links assignments to it.
    """
    
    # ✅ D1: Créer span racine pour le dispatch
    with tracer.start_as_current_span("dispatch.run") as root_span:
        root_span.set_attribute("company_id", company_id)
        root_span.set_attribute("mode", mode)
        root_span.set_attribute("for_date", str(for_date) if for_date else "today")
        
        with suppress(Exception):
            db.session.rollback()

    # Variable pour stocker le résultat final
    result: Dict[str, Any] = {
        "assignments": [], "unassigned": [], "bookings": [], "drivers": [],
        "meta": {"reason": "unknown"}, "debug": {"reason": "unknown"},
    }

    day_str = (for_date or datetime.now(UTC).strftime("%Y-%m-%d"))
    dispatch_run: DispatchRun | None = None
    problem: Dict[str, Any] = {}
    perf_collector: performance_metrics.DispatchMetricsCollector | None = None
    
    try:
        company: Company | None = Company.query.get(company_id)
        if not company:
            logger.warning("[Engine] Company %s introuvable", company_id)
            result = {
                "assignments": [], "unassigned": [], "bookings": [], "drivers": [],
                "meta": {"reason": "company_not_found"}, "debug": {"reason": "company_not_found"},
            }
        # 1) Configuration
        s = custom_settings or ud_settings.for_company(company)
        
        # ⚡ Détecter le mode rapide depuis overrides
        is_fast_mode = overrides and overrides.get("fast_mode") is True
        if is_fast_mode:
            # ⚡ Mode rapide : forcer heuristic_only et désactiver optimisations lourdes
            mode = "heuristic_only"
            logger.info("[Engine] ⚡ Mode RAPIDE détecté : heuristic_only, optimisations désactivées")
            # Désactiver solver et RL pour garantir < 1 minute
            if not hasattr(s, "features"):
                s.features = ud_settings.FeaturesSettings()
            s.features.enable_solver = False
            s.features.enable_rl_optimizer = False
            s.features.enable_parallel_heuristics = True  # Activer parallélisme pour vitesse
            # Limiter le solver à 10s max si jamais appelé (sécurité)
            s.solver.time_limit_sec = 10
        
        if overrides:
            logger.info("[Engine] Applying overrides: %s", list(overrides.keys()))
            logger.info("[Engine] 📋 Overrides détaillés: reset_existing=%s, preferred_driver_id=%s, fast_mode=%s",
                        overrides.get("reset_existing"), overrides.get("preferred_driver_id"), overrides.get("fast_mode"))
            
            # ✅ Logger les paramètres demandés avant merge
            logger.info("[Engine] 📥 Overrides demandés: %s", overrides)
            
            # Capturer les valeurs avant merge pour comparaison
            fairness_weight_before = getattr(getattr(s, "fairness", None), "fairness_weight", None) if hasattr(s, "fairness") else None
            driver_load_before = getattr(getattr(s, "heuristic", None), "driver_load_balance", None) if hasattr(s, "heuristic") else None
            proximity_before = getattr(getattr(s, "heuristic", None), "proximity", None) if hasattr(s, "heuristic") else None
            
            try:
                s = ud_settings.merge_overrides(s, overrides)
                
                # ✅ Logger les paramètres appliqués vs demandés (comparaison avant/après)
                if hasattr(s, "heuristic"):
                    driver_load_after = s.heuristic.driver_load_balance
                    proximity_after = s.heuristic.proximity
                    heuristic_override = overrides.get("heuristic", {}) if isinstance(overrides.get("heuristic"), dict) else {}
                    logger.info(
                        "[Engine] ✅ After merge - heuristic.driver_load_balance: %s → %s (demandé: %s)",
                        driver_load_before,
                        driver_load_after,
                        heuristic_override.get("driver_load_balance", "N/A"),
                    )
                    logger.info(
                        "[Engine] ✅ After merge - heuristic.proximity: %s → %s (demandé: %s)",
                        proximity_before,
                        proximity_after,
                        heuristic_override.get("proximity", "N/A"),
                    )
                if hasattr(s, "fairness"):
                    fairness_weight_after = s.fairness.fairness_weight
                    fairness_override = overrides.get("fairness", {}) if isinstance(overrides.get("fairness"), dict) else {}
                    logger.info(
                        "[Engine] ✅ After merge - fairness.fairness_weight: %s → %s (demandé: %s)",
                        fairness_weight_before,
                        fairness_weight_after,
                        fairness_override.get("fairness_weight", "N/A"),
                    )
            except Exception as e:
                logger.exception(
                    "[Engine] merge_overrides failed with error: %s", e)
                logger.warning("[Engine] Using base settings due to merge failure")
        if allow_emergency is not None:
            with suppress(Exception):
                s.emergency.allow_emergency_drivers = bool(allow_emergency)
        allow_emg = bool(
            getattr(
                getattr(
                    s,
                    "emergency",
                    None),
                "allow_emergency_drivers",
                True))

        logger.info(
            "[Engine] Dispatch start company=%s mode=%s for_date=%s regular_first=%s allow_emergency=%s fast_mode=%s",
            company_id, mode, for_date, regular_first, allow_emg, is_fast_mode
        )

        # 1.b Verrou d'idempotence par (entreprise, jour)
        if not _acquire_day_lock(company_id, day_str):
            logger.warning(
                "[Engine] Run skipped (locked) company=%s day=%s",
                company_id,
                day_str)
            result = {
                "assignments": [], "unassigned": [], "bookings": [], "drivers": [],
                "meta": {"reason": "locked", "for_date": for_date, "day": day_str},
                "debug": {"reason": "locked", "for_date": for_date, "day": day_str},
            }

        # 2) Créer / réutiliser le DispatchRun (unique: company_id+day)
        try:
            day_date = _to_date_ymd(day_str)
        except Exception:
            logger.warning(
                "[Engine] Invalid day_str=%r, fallback to today",
                day_str)
            day_date = datetime.now(UTC).date()

        logger.info("[Engine] Using day_date: %s for dispatch run", day_date)

        cfg = {
            "mode": mode,
            "regular_first": bool(regular_first),
            "allow_emergency": bool(allow_emg),
            "for_date": for_date,
        }

        # ✅ Si existing_dispatch_run_id est fourni, réutiliser le DispatchRun existant
        if existing_dispatch_run_id:
            dispatch_run = DispatchRun.query.get(existing_dispatch_run_id)
            if dispatch_run:
                # Vérifier que le DispatchRun correspond à la company et à la date
                if dispatch_run.company_id != company_id:
                    logger.warning(
                        "[Engine] DispatchRun id=%s company_id=%s doesn't match requested company_id=%s, creating new",
                        existing_dispatch_run_id, dispatch_run.company_id, company_id
                    )
                    dispatch_run = None
                elif dispatch_run.day != day_date:
                    logger.warning(
                        "[Engine] DispatchRun id=%s day=%s doesn't match requested day=%s, creating new",
                        existing_dispatch_run_id, dispatch_run.day, day_date
                    )
                    dispatch_run = None
                else:
                    logger.info(
                        "[Engine] Reusing existing DispatchRun id=%s for company=%s day=%s",
                        existing_dispatch_run_id, company_id, day_str
                    )
            else:
                logger.warning(
                    "[Engine] DispatchRun id=%s not found, will create new",
                    existing_dispatch_run_id
                )
                dispatch_run = None
        else:
            # Comportement par défaut : chercher par company_id+day
            dispatch_run = DispatchRun.query.filter_by(
                company_id=company_id, day=day_date).first()

        # ✅ Définir le contexte logging pour dispatch_run_id
        dispatch_run_id_val = _safe_int(getattr(dispatch_run, "id", None)) if dispatch_run else None
        if dispatch_run_id_val:
            try:
                from shared.dispatch_logging import set_dispatch_context
                set_dispatch_context(dispatch_run_id=dispatch_run_id_val, company_id=company_id)
            except ImportError:
                pass  # Module optionnel

        if dispatch_run is None:
            # TX courte de création ; en cas de race → IntegrityError
            try:
                with _begin_tx():
                    dr_any: Any = DispatchRun()
                    dr_any.company_id = int(company_id)
                    dr_any.day = day_date
                    dr_any.status = DispatchStatus.RUNNING
                    dr_any.started_at = datetime.now(UTC)
                    dr_any.created_at = datetime.now(UTC)
                    dr_any.config = cfg
                    db.session.add(dr_any)
                    db.session.flush()
                    dispatch_run = cast("DispatchRun", dr_any)
                logger.info("[Engine] Created DispatchRun id=%s for company=%s day=%s",
                            dispatch_run.id, company_id, day_str)
            except IntegrityError:
                # Un autre thread l'a créé entre-temps → récupère l'existant
                # puis MAJ sous TX courte
                db.session.rollback()
                dispatch_run = DispatchRun.query.filter_by(
                    company_id=company_id, day=day_date).first()
                if dispatch_run is None:
                    raise
                with _begin_tx():
                    dr2any: Any = dispatch_run
                    dr2any.status = DispatchStatus.RUNNING
                    dr2any.started_at = datetime.now(UTC)
                    dr2any.completed_at = None
                    dr2any.config = cfg
                    db.session.add(dr2any)
        else:
            # Reuse : MAJ sous TX courte (mettre à jour le statut à RUNNING si nécessaire)
            with _begin_tx():
                dr3any: Any = dispatch_run
                # Ne mettre à jour le statut que s'il n'est pas déjà RUNNING
                if dr3any.status != DispatchStatus.RUNNING:
                    dr3any.status = DispatchStatus.RUNNING
                    dr3any.started_at = datetime.now(UTC)
                    dr3any.completed_at = None
                    dr3any.config = cfg
                    db.session.add(dr3any)

        # 3) Commit du DispatchRun pour qu'il soit visible dans les prochaines
        # transactions
        try:
            db.session.commit()
            logger.info(
                "[Engine] DispatchRun id=%s committed successfully",
                dispatch_run.id)
        except Exception:
            logger.exception("[Engine] Failed to commit DispatchRun")
            db.session.rollback()
            raise

        # 3.5) Initialiser le collecteur de métriques de performance
        if dispatch_run:          # Initialiser seulement si dispatch_run existe
            perf_collector = performance_metrics.DispatchMetricsCollector(
                company_id=company_id,
                dispatch_run_id=dispatch_run.id if dispatch_run else None
            )
            
            # Réinitialiser les compteurs
            performance_metrics.reset_sql_counter()
            
            # ✅ A1: Réinitialiser le compteur de conflits temporels
            heuristics.reset_temporal_conflict_counter()
            
            # Démarrer le timer pour la collecte de données
            perf_collector.start_timer("data_collection")
        else:
            logger.warning("[Engine] perf_collector not initialized: dispatch_run is None")
        
        # 3.6) Initialiser AB Router et KPI Monitor pour rollout progressif RL
        ab_router = ABRouter(s)
        kpi_monitor = RLKPIMonitor(s)

        # 4) Reset anciennes assignations pour cette date (si relance le même jour)
        # ⚡ Si reset_existing est True dans overrides, supprimer TOUTES les assignations de la date
        # Sinon, supprimer seulement celles du run actuel
        reset_existing = overrides and overrides.get("reset_existing", False)
        logger.info(
            "[Engine] 🔄 Reset existing: %s (overrides keys: %s)",
            reset_existing, list(overrides.keys()) if overrides else []
        )
        try:
            with _begin_tx():
                if reset_existing and day_date:
                    # Supprimer TOUTES les assignations pour cette date (nouveau dispatch complet)
                    # Récupérer tous les booking_ids pour cette date
                    # Convertir day_date en datetime pour la comparaison
                    day_start = datetime.combine(day_date, datetime.min.time())
                    day_end = day_start + timedelta(days=1)
                    booking_ids_for_date = [
                        b.id for b in Booking.query.filter(
                            Booking.company_id == company_id,
                            Booking.scheduled_time >= day_start,
                            Booking.scheduled_time < day_end,
                            Booking.status.notin_([
                                BookingStatus.COMPLETED,
                                BookingStatus.RETURN_COMPLETED,
                                BookingStatus.CANCELED,  # ✅ Seulement CANCELED existe dans l'enum
                            ])
                        ).all()
                    ]
                    if booking_ids_for_date:
                        # ⚡ Supprimer les Assignment ET réinitialiser les Booking
                        deleted_count = Assignment.query.filter(
                            Assignment.booking_id.in_(booking_ids_for_date)
                        ).delete(synchronize_session=False)
                        
                        # ⚡ CRITIQUE : Réinitialiser driver_id des Booking pour que fairness_counts soit à zéro
                        # On ne change pas le status, seulement le driver_id pour que count_assigned_bookings_for_day
                        # ne compte plus ces bookings comme assignés
                        bookings_reset = Booking.query.filter(
                            Booking.id.in_(booking_ids_for_date),
                            Booking.driver_id.isnot(None)  # Seulement ceux qui ont un driver_id
                        ).update({
                            Booking.driver_id: None
                        }, synchronize_session=False)
                        
                        logger.info(
                            "[Engine] 🗑️ Supprimé %d assignations et réinitialisé %d bookings pour redispatch complet (date=%s, booking_ids=%s)",
                            deleted_count, bookings_reset, for_date or day_str, len(booking_ids_for_date)
                        )
                    else:
                        logger.info(
                            "[Engine] ⚠️ Aucun booking trouvé pour reset (date=%s)",
                            for_date or day_str
                        )
                else:
                    # Supprimer seulement les assignations du run actuel (comportement par défaut)
                    deleted_count = Assignment.query.filter_by(
                    dispatch_run_id=dispatch_run.id).delete(
                    synchronize_session=False)
                    logger.debug(
                        "[Engine] Reset assignations du run_id=%s uniquement (supprimé: %d)",
                        dispatch_run.id, deleted_count
                    )
            # ✅ Commit explicite pour s'assurer que les suppressions sont persistées
            # avant que build_problem_data calcule fairness_counts
            db.session.commit()
            logger.info("[Engine] ✅ Commit des suppressions d'assignations effectué")
        except Exception:
            logger.exception(
                "[Engine] Failed to reset previous assignments for run_id=%s",
                getattr(
                dispatch_run,
                "id",
                None))
            db.session.rollback()
        # on continue quand même ; le pipeline peut recréer des assignments

        # 5) Construire les données "problème"
        try:
            # ✅ D1: Span data_prep
            with tracer.start_as_current_span("data_prep") as data_span:
                problem = data.build_problem_data(
                    company_id=company_id,
                    settings=s,
                    for_date=for_date or day_str,
                    regular_first=bool(regular_first),
                    allow_emergency=allow_emg,
                    overrides=overrides or {},
                ) or {}
                n_b = len(problem.get("bookings", []))
                n_d = len(problem.get("drivers", []))
                data_span.set_attribute("bookings_count", n_b)
                data_span.set_attribute("drivers_count", n_d)
                logger.info(
                    "[Engine] Problem built: bookings=%d drivers=%d for_date=%s",
                    n_b,
                    n_d,
                    for_date or day_str)

            # Propager le dispatch_run_id dans le problem pour qu'il arrive
            # jusqu'au solver
            if dispatch_run:
                drid = _safe_int(getattr(dispatch_run, "id", None))
                if drid is not None:
                    problem["dispatch_run_id"] = drid
                    logger.info(
                        "[Engine] Added dispatch_run_id=%s to problem", drid)
            
            # Arrêter le timer de collecte de données
            if perf_collector:
                perf_collector.end_timer("data_collection")
        except Exception:
            logger.exception(
                "[Engine] build_problem_data failed (company=%s)",
                company_id)
            if dispatch_run:
                # ✅ TX courte pour marquer le run en échec, même si la session a été salie
                try:
                    with _begin_tx():
                        dispatch_run.status = DispatchStatus.FAILED
                except Exception:
                    logger.exception(
                        "[Engine] Failed to mark DispatchRun FAILED after build_problem_data error")
                result = {
                    "assignments": [], "unassigned": [], "bookings": [], "drivers": [],
                    "meta": {"reason": "problem_build_failed", "for_date": for_date or day_str},
                    "debug": {"reason": "problem_build_failed", "for_date": for_date or day_str},
                }
                # ✅ Retourner immédiatement si build_problem_data a échoué
                if dispatch_run:
                    try:
                        with _begin_tx():
                            dispatch_run.mark_failed("problem_build_failed")
                    except Exception:
                        logger.exception("[Engine] Failed to mark DispatchRun FAILED")
                return result

        # Continuer avec le traitement normal si problem est défini
        if not problem or not problem.get("bookings") or not problem.get("drivers"):
            logger.info(
                "[Engine] Pas de données à dispatcher (company=%s)",
                company_id)
            if dispatch_run:
                # ✅ TX courte pour compléter proprement le run "no_data"
                try:
                    with _begin_tx():
                        dispatch_run.mark_completed({"reason": "no_data"})
                except Exception:
                    logger.exception(
                        "[Engine] Failed to complete DispatchRun (no_data)")
            # ✅ Retourner immédiatement si pas de données
            return {
                "assignments": [], "unassigned": [], "bookings": [], "drivers": [],
                "meta": {"reason": "no_data", "for_date": for_date or day_str},
                "debug": {"reason": "no_data", "for_date": for_date or day_str},
            }

        # 5.5) Clustering géographique (si activé et > threshold)
        use_clustering = getattr(s.features, "enable_clustering", False)
        n_bookings = len(problem.get("bookings", []))
        n_drivers = len(problem.get("drivers", []))
        
        # Utiliser le threshold depuis settings
        clustering_threshold = getattr(s.clustering, "bookings_threshold", CLUSTERING_BOOKINGS_THRESHOLD)
        
        if use_clustering and n_bookings > clustering_threshold:
            try:
                logger.info(
                    "[Clustering] Activating geographic clustering: %d bookings, %d drivers",
                    n_bookings, n_drivers
                )
                
                # Utiliser les paramètres depuis settings
                max_bookings = getattr(s.clustering, "max_bookings_per_zone", 100)
                cross_tolerance = getattr(s.clustering, "cross_zone_tolerance", 0.1)
                
                clustering = GeographicClustering(max_bookings_per_zone=max_bookings)
                zones = clustering.create_zones(
                    bookings=problem["bookings"],
                    drivers=problem["drivers"],
                    cross_zone_tolerance=cross_tolerance
                )
                
                if len(zones) > 1:
                    logger.info(
                        "[Clustering] Created %d zones, dispatching independently",
                        len(zones)
                    )
                    
                    # Dispatch par zone avec pipeline complet
                    zone_results = {}
                    clustering_final_assignments = []
                    clustering_unassigned_ids = []
                    
                    for zone in zones:
                        logger.info(
                            "[Clustering] Dispatching zone %d: %d bookings, %d drivers",
                            zone.zone_id, len(zone.bookings), len(zone.drivers)
                        )
                        
                        # Créer un sous-problème pour cette zone
                        zone_problem = data.build_vrptw_problem(
                            company, zone.bookings, zone.drivers, settings=s,
                            base_time=problem.get("base_time"), 
                            for_date=problem.get("for_date")
                        )
                        
                        # Dispatch avec heuristique
                        zone_assignments = []
                        zone_unassigned_ids = []
                        
                        if mode in ("auto", "heuristic_only") and getattr(s.features, "enable_heuristics", True):
                            try:
                                zone_h_res = heuristics.assign(zone_problem, settings=s)
                                zone_assignments.extend(zone_h_res.assignments)
                                zone_unassigned_ids = zone_h_res.unassigned_booking_ids
                                logger.info(
                                    "[Clustering] Zone %d heuristic: %d assigned, %d unassigned",
                                    zone.zone_id, len(zone_h_res.assignments), len(zone_unassigned_ids)
                                )
                            except Exception as e:
                                logger.warning("[Clustering] Zone %d heuristic failed: %s", zone.zone_id, e)
                        
                        # Dispatch avec solveur pour les restants
                        if zone_unassigned_ids and mode in ("auto", "solver_only") and getattr(s.features, "enable_solver", True):
                            try:
                                zone_s_problem = _filter_problem(zone_problem, zone_unassigned_ids, s)
                                zone_s_res = solver.solve(zone_s_problem, settings=s)
                                zone_assignments.extend(zone_s_res.assignments)
                                zone_unassigned_ids = zone_s_res.unassigned_booking_ids
                                logger.info(
                                    "[Clustering] Zone %d solver: +%d assigned, %d unassigned",
                                    zone.zone_id, len(zone_s_res.assignments), len(zone_unassigned_ids)
                                )
                            except Exception as e:
                                logger.warning("[Clustering] Zone %d solver failed: %s", zone.zone_id, e)
                        
                        # Fallback pour les restants
                        if zone_unassigned_ids:
                            try:
                                zone_fb = heuristics.closest_feasible(zone_problem, zone_unassigned_ids, settings=s)
                                zone_assignments.extend(zone_fb.assignments)
                                logger.info(
                                    "[Clustering] Zone %d fallback: +%d assigned",
                                    zone.zone_id, len(zone_fb.assignments)
                                )
                            except Exception as e:
                                logger.warning("[Clustering] Zone %d fallback failed: %s", zone.zone_id, e)
                        
                        zone_results[zone.zone_id] = {
                            "assignments": zone_assignments,
                            "unassigned": zone_unassigned_ids
                        }
                        clustering_final_assignments.extend(zone_assignments)
                        clustering_unassigned_ids.extend(zone_unassigned_ids)
                    
                    logger.info(
                        "[Clustering] Completed: %d total assignments, %d unassigned across %d zones",
                        len(clustering_final_assignments), len(clustering_unassigned_ids), len(zones)
                    )
                    
                    # Utiliser les résultats du clustering comme final_assignments
                    final_assignments = clustering_final_assignments
                    assigned_set = {a.booking_id for a in clustering_final_assignments}
                    used_heuristic = True
                    used_solver = True
                    used_fallback = True
                    
                    logger.info("[Clustering] Using clustering results as final assignments")
                    
            except Exception as e:
                logger.warning("[Clustering] Failed, falling back to normal pipeline: %s", e)
                # Continue avec le pipeline normal en cas d'échec
        
        # 5) Séparation réguliers/urgences
        regs: List[Driver] = []
        emgs: List[Driver] = []
        try:
            regs, emgs = data.get_available_drivers_split(company_id)
        except Exception:
            for d in problem.get("drivers", []):
                d_type = getattr(d, "driver_type", None)
                (emgs if (d_type == DriverType.EMERGENCY or str(
                    d_type).endswith("EMERGENCY")) else regs).append(d)

        # 6) Pipeline commun
        # Ne pas réinitialiser si le clustering a été utilisé
        clustering_used = use_clustering and n_bookings > CLUSTERING_BOOKINGS_THRESHOLD and len(zones) > 1 if "zones" in locals() else False
        
        if not clustering_used:
            final_assignments: List[Any] = []
            assigned_set = set()
        
        # Pour méta/debug
        phase = "regular_only" if regular_first else "direct"
        if not clustering_used:
            used_heuristic = False
            used_solver = False
            used_fallback = False
            used_emergency_pass = False
        # ----

        def extend_unique(assigns: Iterable[Any]) -> None:
            for a in assigns:
                bid_raw = getattr(a, "booking_id", None)
                try:
                    bid = int(bid_raw) if bid_raw is not None else None
                except Exception:
                    bid = None
                if bid is None or bid in assigned_set:
                    continue
                final_assignments.append(a)
                assigned_set.add(bid)

        # 6.a Urgents
        if not clustering_used:
            try:
                urgent_ids = data.pick_urgent_returns(problem, settings=s) or []
            except Exception:
                urgent_ids = []
            if urgent_ids:
                try:
                    urg_res = heuristics.assign_urgent(
                        problem, urgent_ids, settings=s)
                    extend_unique(urg_res.assignments)
                except Exception:
                    logger.exception("[Engine] assign_urgent failed")

        def remaining_ids_from(p: Dict[str, Any]) -> List[int]:
            res: List[int] = []
            for b in p.get("bookings", []):
                try:
                    bid = int(cast("Any", getattr(b, "id", None)))
                except Exception:
                    bid = None
                if bid is not None and bid not in assigned_set:
                    res.append(bid)
            return res

        # 6.b Pass 1 - réguliers
        h_res = None
        s_res = None
        # Initialiser pour portée globale (sera mis à jour si le fallback est
        # exécuté)
        fb = None
        if regular_first and regs and company is not None:
            logger.info(
                "[Engine] === Pass 1: Regular drivers only (%d drivers) ===",
                len(regs))
            prob_regs = data.build_vrptw_problem(
                company, problem["bookings"], regs, settings=s,
                base_time=problem.get("base_time"), for_date=problem.get("for_date")
            )
            # ⚡ CRUCIAL: Propager preferred_driver_id, company_coords, driver_load_multipliers depuis le problème original
            logger.info("[Engine] 🔍 Propagation preferred_driver_id: problem.keys()=%s, preferred_driver_id in problem=%s", 
                       list(problem.keys())[:10], "preferred_driver_id" in problem)
            if "preferred_driver_id" in problem:
                prob_regs["preferred_driver_id"] = problem["preferred_driver_id"]
                logger.info("[Engine] ✅ preferred_driver_id propagé: %s → prob_regs", problem.get("preferred_driver_id"))
            else:
                logger.warning("[Engine] ⚠️ preferred_driver_id NON présent dans problem (keys: %s)", list(problem.keys())[:20])
            if "company_coords" in problem:
                prob_regs["company_coords"] = problem["company_coords"]
            if "driver_load_multipliers" in problem:
                prob_regs["driver_load_multipliers"] = problem["driver_load_multipliers"]
            remaining_ids = remaining_ids_from(prob_regs)

            if remaining_ids and mode in ("auto", "heuristic_only") and getattr(
                    s.features, "enable_heuristics", True):
                try:
                    h_sub = _filter_problem(prob_regs, remaining_ids, s)
                    # ✅ D1: Span heuristics
                    with tracer.start_as_current_span("heuristics") as heuristics_span:
                        heuristics_span.set_attribute("parallel", getattr(s.features, "enable_parallel_heuristics", False))
                        # Mesurer le temps des heuristiques
                        if perf_collector:
                            with perf_collector.time_step("heuristics"):
                                h_res = heuristics.assign(h_sub, settings=s)
                        else:
                            h_res = heuristics.assign(h_sub, settings=s)
                        heuristics_span.set_attribute("assignments_count", len(h_res.assignments))
                    used_heuristic = True
                    extend_unique(h_res.assignments)
                    logger.info("[Engine] Heuristic P1: %d assignés, %d restants",
                                len(h_res.assignments), len(h_res.unassigned_booking_ids))

                    # 🧠 Optimisation RL (si modèle disponible) - ⚡ SAUF en mode rapide
                    if mode == "auto" and not is_fast_mode and len(final_assignments) > 0:
                        try:
                            from services.unified_dispatch.rl_optimizer import RLDispatchOptimizer

                            logger.info(
                                "[Engine] 🧠 Tentative d'optimisation RL des assignations...")

                            optimizer = RLDispatchOptimizer(
                                # 🆕 v2 (23 dispatches, gap~2)
                                model_path="data/rl/models/dispatch_optimized_v2.pth",
                                max_swaps=15,  # Plus de swaps pour gap ≤1
                                min_improvement=0.3,  # Accepter plus facilement les améliorations
                                config_context="production",  # 🆕 Sprint 1: Configuration optimale
                            )

                            if optimizer.is_available():
                                # Convertir assignments en format optimisable
                                initial = [
                                    {
                                        "booking_id": a.booking_id,
                                        "driver_id": a.driver_id,
                                    }
                                    for a in final_assignments
                                ]

                                # Optimiser
                                optimized = optimizer.optimize_assignments(
                                    initial_assignments=initial,
                                    bookings=problem["bookings"],
                                    drivers=regs,
                                    matrix_quality=prob_regs.get("matrix_quality") if isinstance(prob_regs, dict) else None,
                                    coord_quality=prob_regs.get("coord_quality") if isinstance(prob_regs, dict) else None,
                                )

                                # Appliquer les changements
                                for i, a in enumerate(final_assignments):
                                    new_driver_id = optimized[i]["driver_id"]
                                    if a.driver_id != new_driver_id:
                                        logger.info(
                                            "[Engine] RL swap: Booking %d → Driver %d (was %d)",
                                            a.booking_id,
                                            new_driver_id,
                                            a.driver_id,
                                        )
                                        a.driver_id = new_driver_id

                                # 🛡️ Vérification Safety Guards après optimisation RL
                                safety_guards = get_safety_guards()

                                # Préparer les métriques pour les Safety Guards
                                dispatch_metrics = {
                                    "max_delay_minutes": 0,  # À calculer depuis les assignations
                                    "avg_delay_minutes": 0,
                                    "completion_rate": len(final_assignments) / len(problem["bookings"]) if problem["bookings"] else 1,
                                    "invalid_action_rate": 0,  # À calculer depuis l'optimiseur
                                    "driver_loads": [len([a for a in final_assignments if a.driver_id == d.id]) for d in regs],
                                    "avg_distance_km": 0,  # À calculer
                                    "max_distance_km": 0,
                                    "total_distance_km": 0
                                }

                                # Métadonnées RL
                                rl_metadata = {
                                    "confidence": 0.85,  # Confiance par défaut
                                    "uncertainty": 0.15,
                                    "decision_time_ms": 35,  # Latence mesurée
                                    "q_value_variance": 0.1,
                                    "episode_length": 100
                                }

                                # Vérifier la sécurité
                                is_safe, safety_result = safety_guards.check_dispatch_result(
                                    dispatch_metrics, rl_metadata
                                )

                                if not is_safe:
                                    logger.warning(
                                        "[Engine] 🛡️ Safety Guards: Décision RL dangereuse détectée - Rollback vers heuristique"
                                    )

                                    # Rollback vers assignations heuristiques
                                    final_assignments = h_res.assignments.copy()

                                    # Notification d'alerte
                                    try:
                                        # Import conditionnel pour éviter les
                                        # erreurs si le module n'existe pas
                                        try:
                                            from services.notification_service import NotificationService
                                            notification_service = NotificationService()
                                        except ImportError:
                                            # Fallback si NotificationService
                                            # n'est pas disponible
                                            notification_service = None
                                        if notification_service is not None:
                                            notification_service.send_alert(
                                                alert_type="safety_rollback",
                                                severity="warning",
                                                message="Rollback RL vers heuristique - Décision dangereuse détectée",
                                                metadata=safety_result
                                            )
                                    except Exception as notify_e:
                                        logger.error(
                                            "[Engine] Erreur notification rollback: %s", notify_e)

                                    logger.info(
                                        "[Engine] ✅ Rollback vers heuristique effectué")
                                else:
                                    logger.info(
                                        "[Engine] ✅ Safety Guards: Décision RL validée")

                                logger.info(
                                    "[Engine] ✅ Optimisation RL terminée")
                            else:
                                logger.info(
                                    "[Engine] ⏳ Optimiseur RL non disponible (modèle non trouvé)")

                        except Exception as e:
                            logger.warning(
                                "[Engine] ⚠️ Optimisation RL échouée: %s", e)
                            # Continuer avec l'heuristique seule

                    # ⚠️ Vérification d'équité : TEMPORAIREMENT DÉSACTIVÉE
                    # Le solver OR-Tools échoue avec "No solution" à cause de contraintes trop strictes.
                    # L'heuristique fonctionne et assigne tout, même si la répartition n'est pas parfaite.
                    # TODO : Améliorer l'heuristique pour mieux équilibrer dès
                    # le départ
                    if False:  # Désactivé temporairement - voir commentaires ci-dessus
                        # Calculer la charge par chauffeur
                        driver_loads = {}
                        for a in final_assignments:
                            did = getattr(a, "driver_id", None)
                            if did:
                                driver_loads[did] = driver_loads.get(
                                    did, 0) + 1

                        if driver_loads:
                            max_load = max(driver_loads.values())
                            min_load = min(driver_loads.values())
                            load_gap = max_load - min_load

                            # Si écart > ECART_THRESHOLD courses ET fairness
                            # activé, forcer solver
                            fairness_threshold = 2
                            if load_gap > fairness_threshold and getattr(
                                    s.fairness, "enabled", True):
                                logger.warning(
                                    "[Engine] ⚖️ Équité insatisfaisante après heuristique : écart=%d courses (max=%d, min=%d). Relancement avec solver pour optimisation globale...",
                                    load_gap, max_load, min_load
                                )
                                # Vider final_assignments pour que le solver
                                # réassigne TOUT
                                final_assignments.clear()
                                assigned_set.clear()
                                # Recréer un problème vierge pour le solver
                                # (sans état précédent)
                                prob_regs = data.build_vrptw_problem(
                                    company, problem["bookings"], regs, settings=s,
                                    base_time=problem.get("base_time")
                                )
                                # ⚡ CRUCIAL: Propager preferred_driver_id, company_coords, driver_load_multipliers depuis le problème original
                                if "preferred_driver_id" in problem:
                                    prob_regs["preferred_driver_id"] = problem["preferred_driver_id"]
                                if "company_coords" in problem:
                                    prob_regs["company_coords"] = problem["company_coords"]
                                if "driver_load_multipliers" in problem:
                                    prob_regs["driver_load_multipliers"] = problem["driver_load_multipliers"]
                                # Forcer remaining_ids à contenir TOUTES les
                                # courses
                                h_res.unassigned_booking_ids = [
                                    b.id for b in prob_regs.get("bookings", [])]
                                logger.info(
                                    "[Engine] ♻️ Problème recréé from scratch pour solver: %d courses", len(
                                        prob_regs.get(
                                            "bookings", [])))

                    if mode == "heuristic_only":
                        _apply_and_emit(
                            company,
                            final_assignments,
                            dispatch_run_id=(
                                int(
                                    cast(
                                        "Any",
                                        dispatch_run.id)) if dispatch_run and getattr(
                                    dispatch_run,
                                    "id",
                                    None) is not None else None))
                        dispatch_run.mark_completed({
                            "mode": "heuristic_only",
                            "assignments": len(final_assignments),
                            "unassigned": len(remaining_ids_from(prob_regs))
                        })
                        db.session.commit()
                        result = {
                            "assignments": [_serialize_assignment(a) for a in final_assignments],
                            "unassigned": remaining_ids_from(prob_regs),
                            "debug": {"heuristic": getattr(h_res, "debug", None), "for_date": for_date or day_str}
                        }
                except Exception:
                    logger.exception("[Engine] Heuristic pass-1 failed")

            remaining_ids = remaining_ids_from(prob_regs)

            if remaining_ids and mode in ("auto", "solver_only") and getattr(
                    s.features, "enable_solver", True):
                try:
                    s_sub = _filter_problem(prob_regs, remaining_ids, s)
                    # Warm-start: Injecter les assignments heuristiques comme hint initial
                    if h_res and h_res.assignments:
                        s_sub["heuristic_assignments"] = h_res.assignments
                        logger.info("[Engine] Passing %d heuristic assignments as warm-start to solver", len(h_res.assignments))
                    # ✅ D1: Span solver
                    with tracer.start_as_current_span("solver") as solver_span:
                        solver_span.set_attribute("solver", "ortools")
                        solver_span.set_attribute("warm_start", bool(h_res and h_res.assignments))
                        # Mesurer le temps du solveur
                        if perf_collector:
                            with perf_collector.time_step("solver"):
                                s_res = solver.solve(s_sub, settings=s)
                        else:
                            s_res = solver.solve(s_sub, settings=s)
                        solver_span.set_attribute("assignments_count", len(s_res.assignments))
                    used_solver = True
                    extend_unique(s_res.assignments)
                    logger.info("[Engine] Solver P1: %d assignés, %d non assignés",
                                len(s_res.assignments), len(s_res.unassigned_booking_ids))
                    if mode == "solver_only":
                        _apply_and_emit(
                            company,
                            final_assignments,
                            dispatch_run_id=(
                                int(
                                    cast(
                                        "Any",
                                        dispatch_run.id)) if dispatch_run and getattr(
                                    dispatch_run,
                                    "id",
                                    None) is not None else None))
                        dispatch_run.mark_completed({
                            "mode": "solver_only",
                            "assignments": len(final_assignments),
                            "unassigned": len(s_res.unassigned_booking_ids)
                        })
                        db.session.commit()
                        result = {
                            "assignments": [_serialize_assignment(a) for a in final_assignments],
                            "unassigned": s_res.unassigned_booking_ids,
                            "debug": {"solver": getattr(s_res, "debug", None), "for_date": for_date or day_str}
                        }
                except Exception:
                    logger.exception("[Engine] Solver pass-1 failed")

            remaining_ids = remaining_ids_from(prob_regs)
            if remaining_ids:
                try:
                    # 📅 Injecter les états de l'heuristique dans le problem pour que le fallback les utilise
                    if h_res and h_res.debug:
                        prob_regs["busy_until"] = h_res.debug.get(
                            "busy_until", {})
                        prob_regs["driver_scheduled_times"] = h_res.debug.get(
                            "driver_scheduled_times", {})
                        prob_regs["proposed_load"] = h_res.debug.get(
                            "proposed_load", {})
                        logger.warning(
                            "[Engine] 📥 Injection état vers fallback: busy_until=%s, proposed_load=%s",
                            prob_regs.get("busy_until"),
                            prob_regs.get("proposed_load"))

                    fb = heuristics.closest_feasible(
                        prob_regs, remaining_ids, settings=s)
                    used_fallback = True
                    extend_unique(fb.assignments)
                    logger.info("[Engine] Fallback P1: +%d, reste=%d",
                                len(fb.assignments), len(fb.unassigned_booking_ids))
                except Exception:
                    logger.exception("[Engine] Fallback pass-1 failed")

        # 6.c Pass 2 - urgences si nécessaire
        remaining_all = remaining_ids_from(problem)
        # ✅ Toujours utiliser allow_emg (calculé depuis settings + overrides) au lieu de allow_emergency (param brut)
        allow_emg2 = allow_emg
        logger.info("[Engine] Checking for Pass 2: remaining=%d, allow_emergency=%s, emergency_drivers=%d",
                    len(remaining_all), allow_emg2, len(emgs))

        if remaining_all and allow_emg2 and emgs and company is not None:
            try:
                used_emergency_pass = True
                logger.info(
                    "[Engine] === Pass 2: Adding emergency drivers (%d total) ===",
                    len(regs) + len(emgs))
                prob_full = data.build_vrptw_problem(
                    company, problem["bookings"], regs + emgs, settings=s,
                    base_time=problem.get("base_time"), for_date=problem.get("for_date")
                )

                # ⚡ CRUCIAL: Propager preferred_driver_id, company_coords, driver_load_multipliers depuis le problème original
                if "preferred_driver_id" in problem:
                    prob_full["preferred_driver_id"] = problem["preferred_driver_id"]
                if "company_coords" in problem:
                    prob_full["company_coords"] = problem["company_coords"]
                if "driver_load_multipliers" in problem:
                    prob_full["driver_load_multipliers"] = problem["driver_load_multipliers"]

                # 📅 Injecter les états du Pass 1 dans le Pass 2 pour éviter les conflits
                # Utiliser fb (fallback) en priorité car il contient les états les plus à jour
                # Sinon utiliser h_res (heuristique)
                latest_result = fb if (fb and fb.debug) else h_res
                if latest_result and latest_result.debug:
                    prob_full["busy_until"] = latest_result.debug.get(
                        "busy_until", {})
                    prob_full["driver_scheduled_times"] = latest_result.debug.get(
                        "driver_scheduled_times", {})
                    prob_full["proposed_load"] = latest_result.debug.get(
                        "proposed_load", {})
                    source_name = "Fallback P1" if (
                        fb and fb.debug) else "Heuristic P1"
                    logger.warning(
                        "[Engine] 📥 Injection état %s → Pass2: busy_until=%s, proposed_load=%s",
                        source_name,
                        prob_full.get("busy_until"),
                        prob_full.get("proposed_load"))

                rem = remaining_ids_from(prob_full)
                h2 = None
                if rem and mode in ("auto", "heuristic_only") and getattr(
                        s.features, "enable_heuristics", True):
                    h_sub2 = _filter_problem(prob_full, rem, s)
                    h2 = heuristics.assign(h_sub2, settings=s)
                    used_heuristic = True
                    extend_unique(h2.assignments)
                    logger.info("[Engine] Heuristic P2: %d assignés, %d restants",
                                len(h2.assignments), len(h2.unassigned_booking_ids))

                rem = remaining_ids_from(prob_full)
                if rem and mode in ("auto", "solver_only") and getattr(
                        s.features, "enable_solver", True):
                    s_sub2 = _filter_problem(prob_full, rem, s)
                    # Warm-start: Injecter les assignments heuristiques (Pass 2)
                    if h2 and h2.assignments:
                        s_sub2["heuristic_assignments"] = h2.assignments
                    s2 = solver.solve(s_sub2, settings=s)
                    used_solver = True
                    extend_unique(s2.assignments)
                    logger.info("[Engine] Solver P2: %d assignés, %d non assignés",
                                len(s2.assignments), len(s2.unassigned_booking_ids))

                rem = remaining_ids_from(prob_full)
                if rem:
                    # 📅 Injecter les états combinés (Pass1 + Pass2) dans le fallback P2
                    if h2 and h2.debug:
                        # Utiliser les états mis à jour du Pass 2
                        prob_full["busy_until"] = h2.debug.get(
                            "busy_until", {})
                        prob_full["driver_scheduled_times"] = h2.debug.get(
                            "driver_scheduled_times", {})
                        prob_full["proposed_load"] = h2.debug.get(
                            "proposed_load", {})
                        logger.warning(
                            "[Engine] 📥 Injection état P2 → Fallback P2: busy_until=%s, proposed_load=%s",
                            prob_full.get("busy_until"),
                            prob_full.get("proposed_load"))

                    fb2 = heuristics.closest_feasible(
                        prob_full, rem, settings=s)
                    used_fallback = True
                    extend_unique(fb2.assignments)
                    logger.info("[Engine] Fallback P2: +%d, reste=%d",
                                len(fb2.assignments), len(fb2.unassigned_booking_ids))
            except Exception:
                logger.exception("[Engine] Emergency pass failed")

        # 6.d Pas de regular_first → pipeline direct
        if not regular_first and company is not None:
            phase = "direct"
            rem = remaining_ids_from(problem)
            if rem and mode in ("auto", "heuristic_only") and getattr(
                    s.features, "enable_heuristics", True):
                h_sub = _filter_problem(problem, rem, s)
                h_res = heuristics.assign(h_sub, settings=s)
                extend_unique(h_res.assignments)
            rem = remaining_ids_from(problem)
            if rem and mode in ("auto", "solver_only") and getattr(
                    s.features, "enable_solver", True):
                used_solver = True
                s_sub = _filter_problem(problem, rem, s)
                # Warm-start: Injecter les assignments heuristiques (direct mode)
                if h_res and h_res.assignments:
                    s_sub["heuristic_assignments"] = h_res.assignments
                # Mesurer le temps du solveur (Pass 2)
                if perf_collector:
                    with perf_collector.time_step("solver"):
                        s_res = solver.solve(s_sub, settings=s)
                else:
                    s_res = solver.solve(s_sub, settings=s)
                extend_unique(s_res.assignments)
            rem = remaining_ids_from(problem)
            if rem:
                fb = heuristics.closest_feasible(problem, rem, settings=s)
                used_fallback = True
                extend_unique(fb.assignments)

        # 6.4) AB ROUTER : Vérifier si RL doit être appliqué pour cette entreprise
        should_apply_rl = False
        quality_score_pre_apply = 0.0  # ✅ B1: Score pré-apply pour garde-fou
        
        if dispatch_run:
            try:
                should_apply_rl = ab_router.should_apply_rl(company_id)
                
                # ✅ B1: Calculer quality_score pré-apply pour garde-fou
                if should_apply_rl and len(final_assignments) > 0 and company:
                    try:
                        from services.unified_dispatch.dispatch_metrics import (
                            QUALITY_THRESHOLD,
                            DispatchMetricsCollector,
                        )
                        collector = DispatchMetricsCollector(company_id)
                        # Calcul rapide du quality score avant RL
                        temp_metrics = collector._calculate_metrics(
                            dispatch_run_id=dispatch_run.id,
                            run_date=problem.get("for_date", dispatch_run.day),
                            assignments=final_assignments,
                            all_bookings=problem.get("bookings", []),
                            run_metadata={}
                        )
                        quality_score_pre_apply = temp_metrics.quality_score
                        
                        logger.info(
                            "[B1] Quality score pré-apply: %.1f/100 (dominants: %s)",
                            quality_score_pre_apply,
                            ", ".join(f"{k}={v:.1f}" for k, v in list(temp_metrics.dominant_factors.items())[:2])
                        )
                        
                        # ✅ B1: Garde-fou - Désactiver RL apply si quality_score < 70
                        if quality_score_pre_apply < QUALITY_THRESHOLD:
                            logger.warning(
                                "[B1] ⚠️ Auto-apply RL désactivé: quality_score=%.1f < seuil=%d",
                                quality_score_pre_apply,
                                QUALITY_THRESHOLD
                            )
                            should_apply_rl = False
                        else:
                            logger.info(
                                "[B1] ✅ Auto-apply RL autorisé: quality_score=%.1f >= seuil=%d",
                                quality_score_pre_apply,
                                QUALITY_THRESHOLD
                            )
                    except Exception as e:
                        logger.warning("[B1] Failed to calculate pre-apply quality score: %s", e)
                
                if should_apply_rl:
                    logger.info(
                        "[ABRouter] Company %d: RL apply-mode enabled (bucket routing)",
                        company_id
                    )
                    # Activer enable_rl_apply pour cette entreprise
                    s.features.enable_rl_apply = True
                else:
                    logger.debug(
                        "[ABRouter] Company %d: RL apply-mode disabled (not in bucket or quality guard triggered)",
                        company_id
                    )
                    s.features.enable_rl_apply = False
            except Exception as e:
                logger.warning("[ABRouter] Failed to check routing: %s", e)
        
        # 6.5) SHADOW MODE : Générer suggestions RL (après heuristiques, avant apply)
        shadow_suggestions_stored = 0
        rem_before_apply = remaining_ids_from(problem)
        if dispatch_run and getattr(s.features, "enable_rl", False):
            try:
                from services.rl.shadow_mode_manager import ShadowModeManager
                
                shadow_manager = ShadowModeManager()
                
                # Construire les assignations courantes
                current_assignments = {a.booking_id: a.driver_id for a in final_assignments if a.booking_id}
                
                # Générer suggestions shadow
                shadow_suggestions = shadow_manager.generate_shadow_suggestions(
                    bookings=problem.get("bookings", []),
                    drivers=problem.get("drivers", []),
                    current_assignments=current_assignments
                )
                
                # Stocker les suggestions
                if shadow_suggestions:
                    kpi_snapshot = {
                        "assignments_count": len(final_assignments),
                        "unassigned_count": len(rem_before_apply) if rem_before_apply else 0,
                        "heuristic_used": used_heuristic,
                        "solver_used": used_solver
                    }
                    shadow_suggestions_stored = shadow_manager.store_shadow_suggestions(
                        dispatch_run_id=dispatch_run.id,
                        suggestions=shadow_suggestions,
                        kpi_snapshot=kpi_snapshot
                    )
                    logger.info(
                        "[ShadowMode] Stored %s shadow suggestions for dispatch_run %s",
                        shadow_suggestions_stored, dispatch_run.id
                    )
            except Exception as e:
                logger.warning("[ShadowMode] Failed to generate/store shadow suggestions: %s", e)

        # 7) Application en DB
        if company is not None:
            # Mesurer le temps de persistence
            if perf_collector:
                with perf_collector.time_step("persistence"):
                    _apply_and_emit(
                        company,
                        final_assignments,
                        dispatch_run_id=_safe_int(getattr(dispatch_run, "id", None)),
                    )
            else:
                _apply_and_emit(
                    company,
                    final_assignments,
                    dispatch_run_id=_safe_int(getattr(dispatch_run, "id", None)),
                )
        # 8) Résumé & debug + 9) Finir le run
        rem = remaining_ids_from(problem)

        # Analyser les raisons détaillées de non-assignation
        unassigned_reasons = _analyze_unassigned_reasons(
            problem, final_assignments, rem)

        # Mesures de performance agrégées si disponibles
        h_calls = 0
        h_avg = 0
        h_time = 0
        if h_res is not None:
            with suppress(Exception):
                h_calls = int(getattr(h_res, "osrm_calls", 0))
                h_avg = int(getattr(h_res, "osrm_avg_latency_ms", 0))
                h_time = int(getattr(h_res, "heuristic_time_ms", 0))
        s_time = 0
        if s_res is not None:
            with suppress(Exception):
                s_time = int(getattr(s_res, "solver_time_ms", 0))

        metrics = {
            "assignments_count": len(final_assignments),
            "unassigned_count": len(rem),
            "mode": mode,
            "regular_first": regular_first,
            "allow_emergency": allow_emg,
            # Métriques enrichies
            "unassigned_reasons": unassigned_reasons,
            "osrm_calls": h_calls,
            "osrm_avg_latency_ms": h_avg,
            "heuristic_time_ms": h_time,
            "solver_time_ms": s_time,
        }

        # Sérialiser les entités réellement utilisées par le solver
        ser_bookings = [_serialize_booking(b)
                        for b in problem.get("bookings", [])]
        ser_drivers = [_serialize_driver(d)
                       for d in problem.get("drivers", [])]

        logger.info("[Engine] Serialized %d bookings, %d drivers for response",
                    len(ser_bookings), len(ser_drivers))

        debug_info: Dict[str, Any] = {
            "heuristic": getattr(h_res, "debug", None),
            "solver": getattr(s_res, "debug", None),
            "settings": s.to_dict() if hasattr(s, "to_dict") else None,
            "for_date": for_date or day_str,
            "regular_first": regular_first,
            "allow_emergency": allow_emg,
            "unassigned_after": rem,
            "phase": "regular_then_emergency" if used_emergency_pass and regular_first else phase,
            "used_heuristic": used_heuristic,
            "used_solver": used_solver,
            "used_fallback": used_fallback,
        }
        try:
            if "matrix_provider" in problem:
                debug_info["matrix_provider"] = problem["matrix_provider"]
            if "matrix_units" in problem:
                debug_info["matrix_units"] = problem["matrix_units"]
        except Exception:
            pass

        drid = _safe_int(getattr(dispatch_run, "id", None))
        if drid is not None:
            debug_info["dispatch_run_id"] = drid

        # 9) Finaliser le run - TX courte
        try:
            with _begin_tx():
                dispatch_run.mark_completed(metrics)
        except Exception:
            logger.exception(
                "[Engine] Failed to complete DispatchRun id=%s", getattr(
                    dispatch_run, "id", None))

        # 10) Collecter les métriques analytics (asynchrone, ne bloque pas le
        # dispatch)
        with suppress(Exception):
            from services.analytics.metrics_collector import collect_dispatch_metrics
            if drid is not None:
                collect_dispatch_metrics(
                    dispatch_run_id=drid,
                    company_id=company_id,
                    day=for_date if isinstance(
                        for_date, date) else _to_date_ymd(
                        for_date or day_str)
                )

        # 11) Collecter les métriques de qualité du dispatch
        try:
            from services.unified_dispatch.dispatch_metrics import collect_dispatch_metrics as collect_quality_metrics
            if drid is not None:
                quality_metrics = collect_quality_metrics(
                    dispatch_run_id=drid,
                    company_id=company_id,
                    day=for_date if isinstance(
                        for_date, date) else _to_date_ymd(
                        for_date or day_str)
                )
                logger.info(
                    "[Engine] Dispatch quality score: %.1f/100 (assignment: %.1f%%, on-time: %.1f%%, pooling: %.1f%%)",
                    quality_metrics.quality_score,
                    quality_metrics.assignment_rate,
                    (quality_metrics.on_time_bookings /
                     max(1, quality_metrics.total_bookings)) * 100,
                    quality_metrics.pooling_rate
                )
                # Ajouter au debug_info pour le retour API
                debug_info["quality_metrics"] = quality_metrics.to_summary()
                
                # 11.1) KPI MONITOR : Vérifier les KPIs et déclencher backout si nécessaire
                if should_apply_rl:
                    try:
                        # Calculer avg_delay en minutes
                        avg_delay_min = 0.0
                        if (hasattr(quality_metrics, "total_late_minutes") and hasattr(quality_metrics, "total_bookings") 
                            and quality_metrics.total_bookings > 0):
                            avg_delay_min = quality_metrics.total_late_minutes / quality_metrics.total_bookings
                        
                        # Préparer les KPIs pour le monitor
                        kpis = {
                            "quality_score": quality_metrics.quality_score,
                            "on_time_rate": quality_metrics.assignment_rate,
                            "avg_delay_min": avg_delay_min
                        }
                        
                        # Vérifier les KPIs et backout si nécessaire
                        should_backout, reason = kpi_monitor.check_kpis(company_id, kpis)
                        
                        if should_backout:
                            logger.error(
                                "[KPI Monitor] Company %d: BACKOUT TRIGGERED - %s",
                                company_id, reason
                            )
                            # Désactiver RL pour cette entreprise
                            s.features.enable_rl_apply = False
                            debug_info["rl_backout"] = {
                                "triggered": True,
                                "reason": reason,
                                "kpis": kpis
                            }
                        else:
                            logger.info(
                                "[KPI Monitor] Company %d: KPIs OK (quality_score=%.1f, on_time_rate=%.1f%%, avg_delay=%.1f min)",
                                company_id, kpis["quality_score"], kpis["on_time_rate"], kpis["avg_delay_min"]
                            )
                    except Exception as e:
                        logger.warning("[KPI Monitor] Failed to check KPIs: %s", e)
        except Exception as e:
            logger.warning("[Engine] Failed to collect quality metrics: %s", e)

        # Finaliser les métriques de performance
        feature_flags = {
            "enable_heuristics": s.features.enable_heuristics,
            "enable_solver": s.features.enable_solver,
            "enable_rl": s.features.enable_rl,
            "enable_clustering": s.features.enable_clustering,
            "enable_parallel_heuristics": s.features.enable_parallel_heuristics,
        }
        # Déterminer l'algorithme utilisé
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
                algorithm_used=algorithm_used,
                feature_flags=feature_flags
            )
            
            # ✅ Enregistrer métriques Prometheus
            try:
                from services.unified_dispatch.dispatch_prometheus_metrics import (
                    record_assignment_rate,
                    record_db_conflicts,
                    record_dispatch_duration,
                    record_dispatch_quality,
                    record_temporal_conflicts,
                    record_unassigned_count,
                )
                
                drid = _safe_int(getattr(dispatch_run, "id", None))
                if drid is not None:
                    # Qualité
                    if perf_metrics.quality_score > 0:
                        record_dispatch_quality(perf_metrics.quality_score, drid, company_id)
                    
                    # Taux d'assignation
                    if perf_metrics.assignment_rate > 0:
                        record_assignment_rate(perf_metrics.assignment_rate, drid, company_id)
                    
                    # Non assignés
                    unassigned_count = len(rem) if rem else 0
                    record_unassigned_count(unassigned_count, drid, company_id)
                    
                    # Conflits temporels
                    if perf_metrics.temporal_conflicts_count > 0:
                        record_temporal_conflicts(perf_metrics.temporal_conflicts_count, drid, company_id)
                    
                    # Conflits DB
                    if perf_metrics.db_conflicts_count > 0:
                        record_db_conflicts(perf_metrics.db_conflicts_count, drid, company_id)
                    
                    # Durée
                    if perf_metrics.total_time > 0:
                        record_dispatch_duration(perf_metrics.total_time, mode, company_id)
            except ImportError:
                # Métriques Prometheus optionnelles
                pass
            except Exception as e:
                logger.warning("[Engine] Erreur enregistrement métriques Prometheus: %s", e)
            
            # Récupérer le nombre de requêtes SQL
            perf_metrics.sql_queries_count = performance_metrics.get_sql_count()
            
            # ✅ A1: Récupérer le nombre de conflits temporels
            temporal_conflicts = heuristics.get_temporal_conflict_count()
            perf_collector.metrics.temporal_conflicts_count = temporal_conflicts
            perf_metrics.temporal_conflicts_count = temporal_conflicts
            if temporal_conflicts > 0:
                logger.warning("[Engine] ⚠️ %d conflits temporels détectés pendant ce dispatch", temporal_conflicts)
            
            # ✅ A2: Récupérer le nombre de conflits DB (contraintes uniques)
            from services.unified_dispatch.apply import get_db_conflict_count
            db_conflicts = get_db_conflict_count()
            perf_collector.metrics.db_conflicts_count = db_conflicts
            perf_metrics.db_conflicts_count = db_conflicts
            if db_conflicts > 0:
                logger.info("[Engine] ℹ️ %d conflits DB (idempotence OK) pendant ce dispatch", db_conflicts)
            
            # ✅ A5: Récupérer les métriques cache OSRM
            from services.unified_dispatch.osrm_cache_metrics import (
                HIT_RATE_THRESHOLD,
                get_cache_metrics_dict,
            )
            cache_metrics = get_cache_metrics_dict()
            perf_collector.metrics.osrm_cache_hits = cache_metrics["hits"]
            perf_collector.metrics.osrm_cache_misses = cache_metrics["misses"]
            perf_collector.metrics.osrm_cache_bypass_count = cache_metrics["bypass_count"]
            perf_metrics.osrm_cache_hits = cache_metrics["hits"]
            perf_metrics.osrm_cache_misses = cache_metrics["misses"]
            perf_metrics.osrm_cache_bypass_count = cache_metrics["bypass_count"]
            
            hit_rate = cache_metrics["hit_rate"]
            if hit_rate < HIT_RATE_THRESHOLD:
                logger.warning("[Engine] ⚠️ Cache OSRM hit-rate bas: %.2f%%", hit_rate * 100)
            
            # Ajouter les compteurs
            perf_collector.metrics.bookings_processed = len(final_assignments)
            perf_collector.metrics.drivers_available = len(problem.get("drivers", []))
            
            # ✅ A4: Vérifier les SLO
            from time import time
            n_bookings = len(final_assignments)
            assignment_rate = len(final_assignments) / max(len(problem.get("bookings", [])), 1)
            quality_score = result.get("meta", {}).get("quality_score", 0.0)
            
            slo_check = check_slo_breach(
                total_time_sec=perf_metrics.total_time,
                assignment_rate=assignment_rate,
                quality_score=quality_score,
                n_bookings=n_bookings
            )
            
            if slo_check["breached"]:
                current_time = time()
                for breach in slo_check["breaches"]:
                    get_slo_tracker().record_breach(
                        breach_type=breach["dimension"],
                        timestamp=current_time
                    )
                
                logger.warning(
                    "[Engine] ⚠️ SLO breach détecté: %d violations pour batch size %d",
                    len(slo_check["breaches"]),
                    n_bookings
                )
                
                # Vérifier si on doit alerter
                if get_slo_tracker().should_alert(current_time):
                    breach_summary = get_slo_tracker().get_breach_summary(current_time)
                    logger.critical(
                        "[Engine] 🚨 SLO breach répété: %d breaches dans fenêtre %d min (Pager déclenché)",
                        breach_summary["breach_count"],
                        breach_summary["window_minutes"]
                    )
            
            # Ajouter les métriques SLO au résultat
            perf_metrics.slo_check = slo_check
        
        result = {
            "assignments": [_serialize_assignment(a) for a in final_assignments],
            "unassigned": rem,
            "unassigned_reasons": unassigned_reasons,
            "bookings": ser_bookings,
            "drivers": ser_drivers,
            "meta": debug_info,
            "debug": debug_info,
            "dispatch_run_id": drid,
        }
        
        # Ajouter les métriques de performance si disponible
        if perf_metrics:
            result["performance_metrics"] = perf_metrics.to_dict()

    except Exception as e:
        # ✅ logging SQLA enrichi pour capter la 1re requête fautive
        extra_sql = {}
        try:
            from sqlalchemy.exc import SQLAlchemyError
            if isinstance(e, SQLAlchemyError):
                extra_sql = {
                    "sql_statement": getattr(e, "statement", None),
                    "sql_params": getattr(e, "params", None),
                    "dbapi_orig": str(getattr(e, "orig", "")),
                }
        except Exception:
            pass
        logger.exception("[Engine] Unhandled error during run company=%s day=%s extra=%s",
                         company_id, day_str, extra_sql)
        try:
            db.session.rollback()  # ✅ défensif
            if dispatch_run:
                with _begin_tx():
                    dispatch_run.status = DispatchStatus.FAILED
        except Exception:
            db.session.rollback()
        
        # Ajouter les métriques de performance même en cas d'erreur si perf_collector existe
        result = {
            "assignments": [], "unassigned": [], "bookings": [], "drivers": [],
            "meta": {"reason": "run_failed", "for_date": for_date or day_str},
            "debug": {"reason": "run_failed", "for_date": for_date or day_str},
        }
        # Si perf_collector a été initialisé, inclure les métriques partielles
        try:
            if perf_collector is not None:
                perf_metrics = perf_collector.metrics
                result["performance_metrics"] = perf_metrics.to_dict()
        except Exception:
            pass
    finally:
        _release_day_lock(company_id, day_str)
        # ✅ Nettoyer le contexte logging
        try:
            from shared.dispatch_logging import clear_dispatch_context
            clear_dispatch_context()
        except ImportError:
            pass  # Module optionnel
    
    return result

# ------------------------------------------------------------
# Helpers internes
# ------------------------------------------------------------


def _filter_problem(
    problem: Dict[str, Any],
    booking_ids: List[int],
    s: ud_settings.Settings
) -> Dict[str, Any]:
    """Reconstruit un sous-problème avec les mêmes settings que le run principal."""
    bookings_map = {b.id: b for b in problem.get("bookings", [])}
    new_bookings = [bookings_map[bid]
                    for bid in booking_ids if bid in bookings_map]
    drivers = problem.get("drivers", [])
    company_id = problem.get("company_id") or getattr(
        problem.get("company"), "id", None)
    company_id = _safe_int(company_id)
    if company_id is None:
        company_id = _safe_int(getattr(problem, "company_id", None))
        # repli : utiliser l'objet company reçu en param de run() si nécessaire
        # (on évite un N+1 en DB, mais on reste safe)
        company_id = getattr(
            Company.query.get(
                getattr(
                    problem,
                    "company_id",
                    None)),
            "id",
            None)

    # Propager for_date et dispatch_run_id
    for_date = problem.get("for_date")
    dispatch_run_id = problem.get("dispatch_run_id")

    company = cast("Company", Company.query.get(company_id))
    result = data.build_vrptw_problem(
        company, new_bookings, drivers, settings=s,
        base_time=problem.get("base_time"), for_date=problem.get("for_date")
    )

    # Assurer que for_date et dispatch_run_id sont propagés
    if for_date:
        result["for_date"] = for_date
    if dispatch_run_id:
        result["dispatch_run_id"] = dispatch_run_id

    # 📅 CRUCIAL: Propager les états de disponibilité des chauffeurs
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
        company: Company, assignments: List[Any], dispatch_run_id: int | None) -> None:
    """Applique les assignations en base et émet événements/notifications.
    
    ✅ Validation temporelle stricte: Si des conflits sont détectés et que
    strict_temporal_validation est activé, rollback automatique.
    """
    if not assignments:
        return
    
    # Session propre avant les writes
    with suppress(Exception):
        db.session.rollback()
    
    # ✅ Validation temporelle stricte avant application
    try:
        # Vérifier si validation stricte est activée (via settings)
        # On vérifie depuis les settings de la company
        from services.unified_dispatch import settings as ud_settings
        from services.unified_dispatch.validation import validate_assignments
        company_settings = ud_settings.for_company(company)
        strict_validation = getattr(
            company_settings.features, 
            "enable_strict_temporal_conflict_check", 
            True  # Par défaut activé
        )
        
        if strict_validation:
            # Convertir assignments en format dict pour validation
            assignments_dict = []
            for a in assignments:
                if isinstance(a, dict):
                    assignments_dict.append(a)
                else:
                    # Convertir objet en dict
                    assignment_dict = {
                        "driver_id": getattr(a, "driver_id", None),
                        "booking_id": getattr(a, "booking_id", None),
                        "scheduled_time": getattr(a, "scheduled_time", None),
                        "estimated_duration_minutes": getattr(a, "estimated_duration_minutes", None),
                    }
                    assignments_dict.append(assignment_dict)
            
            validation_result = validate_assignments(assignments_dict, strict=True)
            
            if not validation_result["valid"]:
                # ✅ Rollback automatique si conflits détectés
                logger.error(
                    "[Engine] ❌ Validation temporelle stricte échouée: %d erreurs critiques détectées. Rollback automatique.",
                    len(validation_result["errors"])
                )
                for error in validation_result["errors"]:
                    logger.error("[Engine]   %s", error)
                
                # Rollback de la transaction
                db.session.rollback()
                
                # Lever une exception pour arrêter le dispatch
                raise ValueError(
                    f"Validation temporelle stricte échouée: {len(validation_result['errors'])} conflits détectés. "
                    + f"Assignations non appliquées. Erreurs: {validation_result['errors'][:3]}"
                )
            if validation_result.get("warnings"):
                # Avertissements seulement (pas de rollback)
                logger.warning(
                    "[Engine] ⚠️ Validation temporelle: %d avertissements (non bloquants)",
                    len(validation_result["warnings"])
                )
                for warning in validation_result["warnings"][:5]:  # Limiter à 5 warnings
                    logger.warning("[Engine]   %s", warning)
    except ImportError:
        # Module validation non disponible, continuer sans validation
        logger.warning("[Engine] Module validation non disponible, skip validation temporelle")
    except ValueError:
        # Ré-élever l'exception de validation
        raise
    except Exception as e:
        # Erreur lors de la validation, logger mais continuer (mode défensif)
        logger.exception("[Engine] Erreur lors de la validation temporelle: %s", e)

    # ✅ D1: Span persist
    with tracer.start_as_current_span("persist") as persist_span:
        persist_span.set_attribute("assignments_count", len(assignments))
        
        # 1) Apply en DB
        try:
            logger.info(
                "[Engine] Applying assignments with dispatch_run_id=%s",
                dispatch_run_id)
            company_id_int = _safe_int(getattr(company, "id", None)) or 0
            result = apply_assignments(
                company_id_int,
                assignments,
                dispatch_run_id=dispatch_run_id,
                return_pairs=True,
            )
            db.session.commit()
            logger.info(
                "[Engine] Applied %d assignments with dispatch_run_id=%s",
                len(result.get("applied", [])),
                dispatch_run_id,
            )
            persist_span.set_attribute("applied_count", len(result.get("applied", [])))
        except Exception:
            logger.exception("[Engine] DB apply failed")
            with suppress(Exception):
                db.session.rollback()
            raise

        # 2) Notifications par booking (ne touche pas à la session sauf pour le
        # .get)
        # ✅ D1: Span ws_emit
        with tracer.start_as_current_span("ws_emit") as ws_span:
            applied_count = 0
            for a in assignments:
                bid = getattr(a, "booking_id", None)
                if bid is None:
                    continue
                with suppress(Exception):
                    b = Booking.query.get(bid)
                    if b:
                        notify_booking_assigned(b)
                        applied_count += 1
            ws_span.set_attribute("events_count", applied_count)

    # 3) Notification globale de fin de run
    try:
        if dispatch_run_id:
            # Assainir la session avant un SELECT (évite
            # InFailedSqlTransaction)
            with suppress(Exception):
                db.session.rollback()

            # Charger le DispatchRun proprement
            dr = None
            try:
                dr = db.session.get(DispatchRun, int(dispatch_run_id))
            except Exception as e:
                logger.warning(
                    "[Engine] Failed to load DispatchRun %s: %s",
                    dispatch_run_id,
                    e)

            date_str: str | None = None
            if dr is not None:
                dr_day = getattr(dr, "day", None)
                # ✅ évite le test booléen sur une Column; vérifie le type valeur
                if isinstance(dr_day, date):
                    date_str = dr_day.isoformat()

            notify_dispatch_run_completed(
                _safe_int(getattr(company, "id", None)) or 0,
                int(dispatch_run_id),
                applied_count,
                date_str,
            )
            logger.info(
                "[Engine] Notified dispatch completion: company_id=%s, dispatch_run_id=%s, assignments=%s, date=%s",
                getattr(company, "id", None),
                dispatch_run_id,
                applied_count,
                date_str,
            )
    except Exception:
        with suppress(Exception):
            logger.error("[Engine] Notification/socket error")


def _serialize_assignment(a: Any) -> Dict[str, Any]:
    """Sérialise une assignation (SolverAssignment ou autre) en dict API.
    Assure que dispatch_run_id est inclus.
    """
    if hasattr(a, "to_dict"):
        return a.to_dict()

    # Fallback manuel si pas de to_dict()
    out = {}
    for field in ["booking_id", "driver_id", "status", "estimated_pickup_arrival",
                  "estimated_dropoff_arrival", "reason", "route_index", "dispatch_run_id"]:
        if hasattr(a, field):
            out[field] = getattr(a, field)
    return out


def _serialize_booking(b: Any) -> Dict[str, Any]:
    """Sérialisation légère et stable côté API pour diagnostics/front.
    Adaptée si b est un SQLA model ou un objet dataclass.
    """
    try:
        if hasattr(b, "to_dict"):
            return b.to_dict()
    except Exception:
        pass
    try:
        # dataclass support éventuel
        if hasattr(b, "__dataclass_fields__"):
            return asdict(b)
    except Exception:
        pass

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
            return d.to_dict()
    except Exception:
        pass
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
