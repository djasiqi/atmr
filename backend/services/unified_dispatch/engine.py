# backend/services/unified_dispatch/engine.py
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Dict, Any, List, Optional, Iterable, Callable, cast
import threading
from datetime import datetime, date, timezone 
from sqlalchemy.exc import IntegrityError
from ext import db
from models import Company, Booking, Driver, DriverType, DispatchRun, Assignment, DispatchStatus
from services.unified_dispatch import data, heuristics, solver, settings
from services.unified_dispatch import settings as ud_settings
from services.unified_dispatch.apply import apply_assignments
from services.notification_service import notify_booking_assigned, notify_dispatch_run_completed
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def utcnow():
    return datetime.now(timezone.utc)

# ---------- Helpers typage/runtime ----------
def _to_date_ymd(s: str) -> date:
    # accepte 'YYYY-MM-DD' et ISO full (on ne garde que la date)
    try:
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return date.fromisoformat(s)
        return datetime.fromisoformat(s).date()
    except Exception:
        raise ValueError(f"for_date invalide: {s!r} (attendu 'YYYY-MM-DD')")
    
def _safe_int(v: Any) -> Optional[int]:
    """
    Convertit n'importe quelle valeur (y compris un InstrumentedAttribute/Column)
    en int Python ou retourne None. Typé pour apaiser Pylance.
    """
    try:
        # ignore[arg-type] pour éviter l'avertissement sur Column[int]
        return int(v)  # type: ignore[arg-type]
    except Exception:
        return None

def _in_tx() -> bool:
    """
    Détecte proprement une transaction active sur la session SQLAlchemy,
    sans dépendre d'un stub précis (Pylance-friendly).
    """
    try:
        meth: Optional[Callable[[], Any]] = getattr(db.session, "in_transaction", None)
        if callable(meth):
            return bool(meth())
        get_tx: Optional[Callable[[], Any]] = getattr(db.session, "get_transaction", None)
        if callable(get_tx):
            return get_tx() is not None
    except Exception:
        pass
    # Fallback raisonnable
    return bool(getattr(db.session, "is_active", False))

@contextmanager
def _begin_tx():
    """
    Ouvre une transaction en s'adaptant à l'état courant de la Session.
    - Si une transaction est déjà ouverte (implicitement ou non), on utilise un savepoint (begin_nested).
    - Sinon, on ouvre une transaction normale (begin).
    """
    if _in_tx():
        cm = db.session.begin_nested()
    else:
        cm = db.session.begin()
    with cm:
        yield


# ------------------------------------------------------------------
# Simple verrou "un run par jour et par entreprise"
# clé = (company_id, day_str "YYYY-MM-DD")
_DAY_LOCKS: dict[tuple[int, str], threading.Lock] = {}

def _acquire_day_lock(company_id: int, day_str: str) -> bool:
    key = (int(company_id), str(day_str))
    lock = _DAY_LOCKS.get(key)
    if lock is None:
        lock = threading.Lock()
        _DAY_LOCKS[key] = lock
    return lock.acquire(blocking=False)

def _release_day_lock(company_id: int, day_str: str) -> None:
    key = (int(company_id), str(day_str))
    lock = _DAY_LOCKS.get(key)
    if lock and lock.locked():
        lock.release()



def run(
    company_id: int,
    mode: str = "auto",
    custom_settings: Optional[settings.Settings] = None,
    *,
    for_date: Optional[str] = None,
    date_range: Optional[dict] = None,
    regular_first: bool = True,
    allow_emergency: Optional[bool] = None,
    overrides: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Run the dispatch optimization for a company on a specific date.
    Creates a DispatchRun record and links assignments to it.
    """
    try:
        db.session.rollback()
    except Exception:
        pass

    company: Optional[Company] = Company.query.get(company_id)
    if not company:
        logger.warning("[Engine] Company %s introuvable", company_id)
        return {
            "assignments": [], "unassigned": [], "bookings": [], "drivers": [],
            "meta": {"reason": "company_not_found"}, "debug": {"reason": "company_not_found"},
        }

    # 1) Configuration
    s = custom_settings or settings.for_company(company)
    if overrides:
        try:
            s = ud_settings.merge_overrides(s, overrides)
        except Exception:
            logger.warning("[Engine] merge_overrides failed; using base settings", exc_info=True)
    if allow_emergency is not None:
        try:
            s.emergency.allow_emergency_drivers = bool(allow_emergency)
        except Exception:
            pass
    allow_emg = bool(getattr(getattr(s, "emergency", None), "allow_emergency_drivers", True))

    logger.info(
        "[Engine] Dispatch start company=%s mode=%s for_date=%s regular_first=%s allow_emergency=%s",
        company_id, mode, for_date, regular_first, allow_emg
    )

    # 1.b Verrou d'idempotence par (entreprise, jour)
    day_str = (for_date or datetime.now().strftime("%Y-%m-%d"))
    if not _acquire_day_lock(company_id, day_str):
        logger.warning("[Engine] Run skipped (locked) company=%s day=%s", company_id, day_str)
        return {
            "assignments": [], "unassigned": [], "bookings": [], "drivers": [],
            "meta": {"reason": "locked", "for_date": for_date, "day": day_str},
            "debug": {"reason": "locked", "for_date": for_date, "day": day_str},
        }

    dispatch_run: Optional[DispatchRun] = None
    try:
        # 2) Créer / réutiliser le DispatchRun (unique: company_id+day)

        try:
            day_date = _to_date_ymd(day_str)
        except Exception:
            logger.warning("[Engine] Invalid day_str=%r, fallback to today", day_str)
            day_date = date.today()

        logger.info(f"[Engine] Using day_date: {day_date} for dispatch run")

        dispatch_run = DispatchRun.query.filter_by(company_id=company_id, day=day_date).first()

        cfg = {
            "mode": mode,
            "regular_first": bool(regular_first),
            "allow_emergency": bool(allow_emg),
            "for_date": for_date,
        }

        if dispatch_run is None:
            # TX courte de création ; en cas de race → IntegrityError
            try:
                with _begin_tx():
                    dr_any: Any = DispatchRun()  # type: ignore[call-arg]
                    dr_any.company_id = int(company_id)
                    dr_any.day = day_date
                    dr_any.status = DispatchStatus.RUNNING
                    dr_any.started_at = utcnow()
                    dr_any.created_at = utcnow()
                    dr_any.config = cfg
                    db.session.add(dr_any)
                    db.session.flush()
                    dispatch_run = cast(DispatchRun, dr_any)
                logger.info("[Engine] Created DispatchRun id=%s for company=%s day=%s",
                            dispatch_run.id, company_id, day_str)
            except IntegrityError:
                # Un autre thread l'a créé entre-temps → récupère l'existant puis MAJ sous TX courte
                db.session.rollback()
                dispatch_run = DispatchRun.query.filter_by(company_id=company_id, day=day_date).first()
                if dispatch_run is None:
                    raise
                with _begin_tx():
                    dr2_any: Any = dispatch_run
                    dr2_any.status = DispatchStatus.RUNNING
                    dr2_any.started_at = utcnow()
                    dr2_any.completed_at = None
                    dr2_any.config = cfg
                    db.session.add(dr2_any)
        else:
            # Reuse : MAJ sous TX courte
            with _begin_tx():
                dr3_any: Any = dispatch_run
                dr3_any.status = DispatchStatus.RUNNING
                dr3_any.started_at = utcnow()
                dr3_any.completed_at = None
                dr3_any.config = cfg
                db.session.add(dr3_any)

        # 3) Reset anciennes assignations de CE run (si relance le même jour) — TX courte
        try:
            with _begin_tx():
                Assignment.query.filter_by(dispatch_run_id=dispatch_run.id).delete(synchronize_session=False)
        except Exception:
            logger.exception("[Engine] Failed to reset previous assignments for run_id=%s", getattr(dispatch_run, "id", None))
            # on continue quand même ; le pipeline peut recréer des assignments


        # 4) Construire les données "problème"
        try:
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
            logger.info("[Engine] Problem built: bookings=%d drivers=%d for_date=%s", n_b, n_d, for_date or day_str)
            
            # Propager le dispatch_run_id dans le problem pour qu'il arrive jusqu'au solver
            if dispatch_run:
                drid = _safe_int(getattr(dispatch_run, "id", None))
                if drid is not None:
                    problem["dispatch_run_id"] = drid
                    logger.info("[Engine] Added dispatch_run_id=%s to problem", drid)
        except Exception:
            logger.exception("[Engine] build_problem_data failed (company=%s)", company_id)
            if dispatch_run:
                # ✅ TX courte pour marquer le run en échec, même si la session a été salie
                try:
                    with _begin_tx():
                        dispatch_run.status = DispatchStatus.FAILED
                except Exception:
                    logger.exception("[Engine] Failed to mark DispatchRun FAILED after build_problem_data error")
            return {
                "assignments": [], "unassigned": [], "bookings": [], "drivers": [],
                "meta": {"reason": "problem_build_failed", "for_date": for_date or day_str},
                "debug": {"reason": "problem_build_failed", "for_date": for_date or day_str},
            }

        if not problem or not problem.get("bookings") or not problem.get("drivers"):
            logger.info("[Engine] Pas de données à dispatcher (company=%s)", company_id)
            if dispatch_run:
                # ✅ TX courte pour compléter proprement le run "no_data"
                try:
                    with _begin_tx():
                        dispatch_run.mark_completed({"reason": "no_data"})
                except Exception:
                    logger.exception("[Engine] Failed to complete DispatchRun (no_data)")    
            return {
                "assignments": [], "unassigned": [], "bookings": [], "drivers": [],
                "meta": {"reason": "no_data", "for_date": for_date or day_str},
                "debug": {"reason": "no_data", "for_date": for_date or day_str},
            }

        # 5) Séparation réguliers/urgences
        regs: List[Driver] = []
        emgs: List[Driver] = []
        try:
            regs, emgs = data.get_available_drivers_split(company_id)
        except Exception:
            for d in problem.get("drivers", []):
                d_type = getattr(d, "driver_type", None)
                (emgs if (d_type == DriverType.EMERGENCY or str(d_type).endswith("EMERGENCY")) else regs).append(d)

        # 6) Pipeline commun
        final_assignments: List[Any] = []
        assigned_set = set()
        # Pour méta/debug
        phase = "regular_only" if regular_first else "direct"
        used_heuristic = False
        used_solver = False
        used_fallback = False
        used_emergency_pass = False
        # ----

        def _extend_unique(assigns: Iterable[Any]) -> None:
            for a in assigns:
                bid_raw = getattr(a, "booking_id", None)
                try:
                    bid = int(cast(Any, bid_raw)) if bid_raw is not None else None
                except Exception:
                    bid = None
                if bid is None or bid in assigned_set:
                    continue
                final_assignments.append(a)
                assigned_set.add(bid)

        # 6.a Urgents
        try:
            urgent_ids = data.pick_urgent_returns(problem, settings=s) or []
        except Exception:
            urgent_ids = []
        if urgent_ids:
            try:
                urg_res = heuristics.assign_urgent(problem, urgent_ids, settings=s)
                _extend_unique(urg_res.assignments)
            except Exception:
                logger.exception("[Engine] assign_urgent failed")

        def remaining_ids_from(p: Dict[str, Any]) -> List[int]:
            res: List[int] = []
            for b in p.get("bookings", []):
                try:
                    bid = int(cast(Any, getattr(b, "id", None)))
                except Exception:
                    bid = None
                if bid is not None and bid not in assigned_set:
                    res.append(bid)
            return res

        # 6.b Pass 1 — réguliers
        h_res = None
        s_res = None
        if regular_first and regs:
            logger.info("[Engine] === Pass 1: Regular drivers only (%d drivers) ===", len(regs))
            prob_regs = data.build_vrptw_problem(
                company, problem["bookings"], regs, settings=s,
                base_time=problem.get("base_time"), for_date=problem.get("for_date")
            )
            remaining_ids = remaining_ids_from(prob_regs)

            if remaining_ids and mode in ("auto", "heuristic_only") and getattr(s.features, "enable_heuristics", True):
                try:
                    h_sub = _filter_problem(prob_regs, remaining_ids, s)
                    h_res = heuristics.assign(h_sub, settings=s)
                    used_heuristic = True
                    _extend_unique(h_res.assignments)
                    logger.info("[Engine] Heuristic P1: %d assignés, %d restants",
                                len(h_res.assignments), len(h_res.unassigned_booking_ids))
                    if mode == "heuristic_only":
                        _apply_and_emit(company, final_assignments, dispatch_run_id=(int(cast(Any, dispatch_run.id)) if dispatch_run and getattr(dispatch_run, "id", None) is not None else None))
                        dispatch_run.mark_completed({
                            "mode": "heuristic_only",
                            "assignments": len(final_assignments),
                            "unassigned": len(remaining_ids_from(prob_regs))
                        })
                        db.session.commit()
                        return {
                            "assignments": [_serialize_assignment(a) for a in final_assignments],
                            "unassigned": remaining_ids_from(prob_regs),
                            "debug": {"heuristic": getattr(h_res, "debug", None), "for_date": for_date or day_str}
                        }
                except Exception:
                    logger.exception("[Engine] Heuristic pass-1 failed")

            remaining_ids = remaining_ids_from(prob_regs)
            if remaining_ids and mode in ("auto", "solver_only") and getattr(s.features, "enable_solver", True):
                try:
                    s_sub = _filter_problem(prob_regs, remaining_ids, s)
                    s_res = solver.solve(s_sub, settings=s)
                    used_solver = True
                    _extend_unique(s_res.assignments)
                    logger.info("[Engine] Solver P1: %d assignés, %d non assignés",
                                len(s_res.assignments), len(s_res.unassigned_booking_ids))
                    if mode == "solver_only":
                        _apply_and_emit(company, final_assignments, dispatch_run_id=(int(cast(Any, dispatch_run.id)) if dispatch_run and getattr(dispatch_run, "id", None) is not None else None))
                        dispatch_run.mark_completed({
                            "mode": "solver_only",
                            "assignments": len(final_assignments),
                            "unassigned": len(s_res.unassigned_booking_ids)
                        })
                        db.session.commit()
                        return {
                            "assignments": [_serialize_assignment(a) for a in final_assignments],
                            "unassigned": s_res.unassigned_booking_ids,
                            "debug": {"solver": getattr(s_res, "debug", None), "for_date": for_date or day_str}
                        }
                except Exception:
                    logger.exception("[Engine] Solver pass-1 failed")

            remaining_ids = remaining_ids_from(prob_regs)
            if remaining_ids:
                try:
                    fb = heuristics.closest_feasible(prob_regs, remaining_ids, settings=s)
                    used_fallback = True
                    _extend_unique(fb.assignments)
                    logger.info("[Engine] Fallback P1: +%d, reste=%d",
                                len(fb.assignments), len(fb.unassigned_booking_ids))
                except Exception:
                    logger.exception("[Engine] Fallback pass-1 failed")

        # 6.c Pass 2 — urgences si nécessaire
        remaining_all = remaining_ids_from(problem)
        allow_emg2 = allow_emg if allow_emergency is None else bool(allow_emergency)
        logger.info("[Engine] Checking for Pass 2: remaining=%d, allow_emergency=%s, emergency_drivers=%d",
                    len(remaining_all), allow_emg2, len(emgs))

        if remaining_all and allow_emg2 and emgs:
            try:
                used_emergency_pass = True
                logger.info("[Engine] === Pass 2: Adding emergency drivers (%d total) ===", len(regs) + len(emgs))
                prob_full = data.build_vrptw_problem(
                    company, problem["bookings"], regs + emgs, settings=s,
                    base_time=problem.get("base_time"), for_date=problem.get("for_date")
                )

                rem = remaining_ids_from(prob_full)
                if rem and mode in ("auto", "heuristic_only") and getattr(s.features, "enable_heuristics", True):
                    h_sub2 = _filter_problem(prob_full, rem, s)
                    h2 = heuristics.assign(h_sub2, settings=s)
                    used_heuristic = True
                    _extend_unique(h2.assignments)
                    logger.info("[Engine] Heuristic P2: %d assignés, %d restants",
                                len(h2.assignments), len(h2.unassigned_booking_ids))

                rem = remaining_ids_from(prob_full)
                if rem and mode in ("auto", "solver_only") and getattr(s.features, "enable_solver", True):
                    s_sub2 = _filter_problem(prob_full, rem, s)
                    s2 = solver.solve(s_sub2, settings=s)
                    used_solver = True
                    _extend_unique(s2.assignments)
                    logger.info("[Engine] Solver P2: %d assignés, %d non assignés",
                                len(s2.assignments), len(s2.unassigned_booking_ids))

                rem = remaining_ids_from(prob_full)
                if rem:
                    fb2 = heuristics.closest_feasible(prob_full, rem, settings=s)
                    used_fallback = True
                    _extend_unique(fb2.assignments)
                    logger.info("[Engine] Fallback P2: +%d, reste=%d",
                                len(fb2.assignments), len(fb2.unassigned_booking_ids))
            except Exception:
                logger.exception("[Engine] Emergency pass failed")

        # 6.d Pas de regular_first → pipeline direct
        if not regular_first:
            phase = "direct"
            rem = remaining_ids_from(problem)
            if rem and mode in ("auto", "heuristic_only") and getattr(s.features, "enable_heuristics", True):
                h_sub = _filter_problem(problem, rem, s)
                h_res = heuristics.assign(h_sub, settings=s)
                _extend_unique(h_res.assignments)
            rem = remaining_ids_from(problem)
            if rem and mode in ("auto", "solver_only") and getattr(s.features, "enable_solver", True):
                used_solver = True
                s_sub = _filter_problem(problem, rem, s)
                s_res = solver.solve(s_sub, settings=s)
                _extend_unique(s_res.assignments)
            rem = remaining_ids_from(problem)
            if rem:
                fb = heuristics.closest_feasible(problem, rem, settings=s)
                used_fallback = True
                _extend_unique(fb.assignments)

        # 7) Application en DB
        _apply_and_emit(
            company,
            final_assignments,
            dispatch_run_id=_safe_int(getattr(dispatch_run, "id", None)),
        )
        # 8) Résumé & debug + 9) Finir le run
        rem = remaining_ids_from(problem)
        metrics = {
            "assignments_count": len(final_assignments),
            "unassigned_count": len(rem),
            "mode": mode,
            "regular_first": regular_first,
            "allow_emergency": allow_emg,
        }
        # reasons placeholder (enrichissable plus tard)
        unassigned_reasons = {bid: ["unknown"] for bid in rem}

        # Sérialiser les entités réellement utilisées par le solver
        ser_bookings = [_serialize_booking(b) for b in problem.get("bookings", [])]
        ser_drivers  = [_serialize_driver(d)  for d in problem.get("drivers", [])]
        
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
            if isinstance(problem, dict):
                if "matrix_provider" in problem:
                    debug_info["matrix_provider"] = problem["matrix_provider"]
                if "matrix_units" in problem:
                    debug_info["matrix_units"] = problem["matrix_units"]
        except Exception:
            pass

        drid = _safe_int(getattr(dispatch_run, "id", None))
        if drid is not None:
            debug_info["dispatch_run_id"] = drid

        # 9) Finaliser le run — TX courte
        try:
            with _begin_tx():
                dispatch_run.mark_completed(metrics)
        except Exception:
            logger.exception("[Engine] Failed to complete DispatchRun id=%s", getattr(dispatch_run, "id", None))


        return {
            "assignments": [_serialize_assignment(a) for a in final_assignments],
            
            "unassigned": rem,
            "unassigned_reasons": unassigned_reasons,
            "bookings": ser_bookings,
            "drivers": ser_drivers,
            "meta": debug_info,
            "debug": debug_info,
            "dispatch_run_id": drid,
        }

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
        logger.error("[Engine] Unhandled error during run company=%s day=%s extra=%s",
                     company_id, day_str, extra_sql, exc_info=True)
        try:
            db.session.rollback()  # ✅ défensif
            if dispatch_run:
                with _begin_tx():
                    dispatch_run.status = DispatchStatus.FAILED
        except Exception:
            db.session.rollback()
        return {
            "assignments": [], "unassigned": [], "bookings": [], "drivers": [],
            "meta": {"reason": "run_failed", "for_date": for_date or day_str},
            "debug": {"reason": "run_failed", "for_date": for_date or day_str},
        }
    finally:
        _release_day_lock(company_id, day_str)

# ------------------------------------------------------------
# Helpers internes
# ------------------------------------------------------------

def _filter_problem(
    problem: Dict[str, Any],
    booking_ids: List[int],
    s: settings.Settings
) -> Dict[str, Any]:
    """
    Reconstruit un sous-problème avec les mêmes settings que le run principal.
    """
    bookings_map = {b.id: b for b in problem.get("bookings", [])}
    new_bookings = [bookings_map[bid] for bid in booking_ids if bid in bookings_map]
    drivers = problem.get("drivers", [])
    company_id = problem.get("company_id") or getattr(problem.get("company"), "id", None)
    company_id = _safe_int(company_id)
    if company_id is None:
        company_id = _safe_int(getattr(problem, "company_id", None))
        # repli : utiliser l'objet company reçu en param de run() si nécessaire
        # (on évite un N+1 en DB, mais on reste safe)
        company_id = getattr(Company.query.get(getattr(problem, "company_id", None)), "id", None)
    
    # Propager for_date et dispatch_run_id
    for_date = problem.get("for_date")
    dispatch_run_id = problem.get("dispatch_run_id")
  

    company = cast(Company, Company.query.get(company_id))
    result = data.build_vrptw_problem(
        company, new_bookings, drivers, settings=s,
        base_time=problem.get("base_time"), for_date=problem.get("for_date")
    )
    
    # Assurer que for_date et dispatch_run_id sont propagés
    if for_date:
        result["for_date"] = for_date
    if dispatch_run_id:
        result["dispatch_run_id"] = dispatch_run_id
    
    return result

def _apply_and_emit(company: Company, assignments: List[Any], dispatch_run_id: Optional[int]) -> None:
    """
    Applique les assignations en base et émet événements/notifications.
    """
    if not assignments:
        return

    # Session propre avant les writes
    try:
        db.session.rollback()
    except Exception:
        pass

    # 1) Apply en DB
    try:
        logger.info(f"[Engine] Applying assignments with dispatch_run_id={dispatch_run_id}")
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
    except Exception:
        logger.exception("[Engine] DB apply failed")
        try:
            db.session.rollback()
        except Exception:
            pass
        raise

    # 2) Notifications par booking (ne touche pas à la session sauf pour le .get)
    applied_count = 0
    for a in assignments:
        bid = getattr(a, "booking_id", None)
        if bid is None:
            continue
        try:
            b = Booking.query.get(bid)
        except Exception:
            b = None
        if not b:
            continue
        try:
            notify_booking_assigned(b)
            applied_count += 1
        except Exception as e:
            logger.error("[Engine] Notification/socket error: %s", e)

    # 3) Notification globale de fin de run
    try:
        if dispatch_run_id:
            # Assainir la session avant un SELECT (évite InFailedSqlTransaction)
            try:
                db.session.rollback()
            except Exception:
                pass

            # Charger le DispatchRun proprement
            dr = None
            try:
                dr = db.session.get(DispatchRun, int(dispatch_run_id))
            except Exception as e:
                logger.warning("[Engine] Failed to load DispatchRun %s: %s", dispatch_run_id, e)

            date_str: Optional[str] = None
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
    except Exception as e:
        logger.error("[Engine] Notification/socket error: %s", e)


def _serialize_assignment(a: Any) -> Dict[str, Any]:
    """
    Sérialise une assignation (SolverAssignment ou autre) en dict API.
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
    """
    Sérialisation légère et stable côté API pour diagnostics/front.
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
    """
    Sérialisation légère driver pour diagnostics/front.
    """
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