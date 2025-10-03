# backend/services/unified_dispatch/queue.py
from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Any, cast

from celery.result import AsyncResult
from flask import current_app

logger = logging.getLogger(__name__)

# ============================================================
# Valeurs par défaut raisonnables, surchargées via ENV.
# ============================================================

DEBOUNCE_MS = int(os.getenv("UD_RTC_DEBOUNCE_MS", "800"))
COALESCE_MS = int(os.getenv("UD_RTC_COALESCE_MS", "800"))
LOCK_TTL_SEC = int(os.getenv("UD_RTC_LOCK_TTL_SEC", "30"))
MAX_BACKLOG = int(os.getenv("UD_RTC_MAX_QUEUE_BACKLOG", "100"))

# ============================================================
# app Flask global
# ============================================================

_APP: Optional[Any] = None

def init_app(app):
    """À appeler depuis create_app(app)."""
    global _APP
    _APP = app

# ============================================================
# State par entreprise
# ============================================================

@dataclass
class CompanyDispatchState:
    company_id: int
    # Sémaphore/lock pour empêcher deux runs concurrents sur la même entreprise
    lock: Any = field(default_factory=lambda: __import__('threading').Lock())
    # Timer de déclenchement différé (coalescing)
    timer: Optional[Any] = None
    # Indique si un run est en cours
    running: bool = False
    # Pour éviter un run bloqué : timestamp du dernier start
    last_start: Optional[datetime] = None
    # Backlog de raisons (debug)
    backlog: List[str] = field(default_factory=list)
    # Nombre d'échecs récents (pour backoff)
    recent_failures: int = 0
    # 🔴 NEW: paramètres cumulés pour le prochain run (for_date, overrides, ...)
    params: Dict[str, Any] = field(default_factory=dict)
    # Référence à l'app Flask (capturée sur trigger() si contexte dispo)
    app_ref: Optional[Any] = None  # type: ignore
    # 🔴 NEW: ID de la dernière tâche Celery
    last_task_id: Optional[str] = None


# Mémoire globale in‑process (une entrée par company_id)
_STATE: Dict[int, CompanyDispatchState] = {}
# Statut observable par l'API /status
_LAST_RESULT: Dict[int, Dict[str, Any]] = {}
_LAST_ERROR: Dict[int, Optional[str]] = {}
_RUNNING: Dict[int, bool] = {}
_PROGRESS: Dict[int, int] = {}  # 0..100 approximation de progression
_CELERY_STATE: Dict[int, str] = {}  # État Celery (PENDING, STARTED, SUCCESS, FAILURE, etc.)

# Lock global pour l'accès au dict
_STATE_LOCK = __import__('threading').Lock()
# Interrupteur global (stop propre)
_STOP_EVENT = __import__('threading').Event()


def _get_state(company_id: int) -> CompanyDispatchState:
    with _STATE_LOCK:
        st = _STATE.get(company_id)
        if st is None:
            st = CompanyDispatchState(company_id=company_id)
            _STATE[company_id] = st
        return st


def get_status(company_id: int) -> Dict[str, Any]:
    """
    Utilisé par GET /company_dispatch/status
    Enrichi avec des informations de diagnostic plus détaillées
    """
    last = _LAST_RESULT.get(company_id) or {}
    last_error = _LAST_ERROR.get(company_id)
    
    # Get counts from the last result
    bookings_count = len(last.get("bookings", []))
    drivers_count = len(last.get("drivers", []))
    assignments_count = len(last.get("assignments", []))
    
    # Check Celery task status if we have a task_id
    celery_state = "UNKNOWN"
    st = _get_state(company_id)
    task_id = st.last_task_id
    
    if task_id:
        try:
            # Import here to avoid circular imports
            from celery_app import celery
            task_result = AsyncResult(task_id, app=celery)
            celery_state = task_result.state
            
            # Update running state based on Celery task state
            is_running = celery_state in ('PENDING', 'RECEIVED', 'STARTED')
            _RUNNING[company_id] = is_running
            _CELERY_STATE[company_id] = celery_state
            
            # If task has failed, get the error
            if celery_state == 'FAILURE' and task_result.failed():
                _LAST_ERROR[company_id] = str(task_result.result)
                last_error = _LAST_ERROR[company_id]
            
            # If task has succeeded, update the last result
            if celery_state == 'SUCCESS' and task_result.ready():
                try:
                    result = task_result.get()
                    if isinstance(result, dict):
                        _LAST_RESULT[company_id] = result
                        last = result
                        bookings_count = len(last.get("bookings", []))
                        drivers_count = len(last.get("drivers", []))
                        assignments_count = len(last.get("assignments", []))
                except Exception as e:
                    logger.exception(f"[Queue] Error getting task result: {e}")
        
        except Exception as e:
            logger.exception(f"[Queue] Error checking task status: {e}")
    
    # Determine reason if there are no assignments
    reason = None
    if assignments_count == 0:
        if bookings_count == 0:
            reason = "no_bookings_for_day"
        elif drivers_count == 0:
            reason = "no_drivers"
        elif last_error:
            reason = "apply_failed"
        else:
            reason = "unknown"
            
    return {
        "is_running": bool(_RUNNING.get(company_id, False)),
        "progress": int(_PROGRESS.get(company_id, 0)),
        "last_result": last,
        "last_result_meta": last.get("meta"),
        "last_error": last_error,
        "reason": reason,
        "counters": {
            "bookings": bookings_count,
            "drivers": drivers_count,
            "assignments": assignments_count,
        },
        "dispatch_run_id": last.get("dispatch_run_id") or (last.get("meta", {}) or {}).get("dispatch_run_id"),
        "celery_state": _CELERY_STATE.get(company_id, celery_state),
        "last_task_id": st.last_task_id,
    }


def trigger_job(company_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Utilisé par POST /company_dispatch/run (async). 
    Enfile un job (coalescé) et renvoie un job_id.
    """
    job_id = str(uuid.uuid4())
    # On passe par trigger() pour bénéficier du coalescing/debounce
    trigger(company_id, reason="manual_trigger", mode="auto", params=params)
    return {"id": job_id, "company_id": company_id}


def trigger(company_id: int, reason: str = "generic", mode: str = "auto", params: Optional[Dict[str, Any]] = None) -> None:
    """
    Appel léger depuis les routes ou services :
      - Empile la demande (pour debug),
      - Programme/relance un timer de coalescence,
      - Garantit qu'un seul run partira après DEBOUNCE+COALESCE.
    """
    st = _get_state(company_id)

    # Anti-tempête : limiter la taille du backlog
    if len(st.backlog) >= MAX_BACKLOG:
        # Remplacer la dernière raison par une agrégation
        st.backlog[-1] = f"{st.backlog[-1]} | (saturated)"
    else:
        st.backlog.append(f"{datetime.now(timezone.utc).isoformat()} {reason}")
    # Coalesce: on mémorise/merge les derniers params (on garde la dernière valeur pour chaque clé)
    if params:
        # merge "dernier gagnant" (les clés fournies écrasent les anciennes)
        st.params = dict(params)

    try:
        # si on est dans un contexte requête, mémorise l'app pour le worker
        # (LocalProxy → objet réel)
        get_obj = getattr(current_app, "_get_current_object", None)
        if callable(get_obj):
            st.app_ref = get_obj()  # type: ignore[misc]
        else:
            st.app_ref = current_app  # fallback typé Any
    except Exception:
        # pas de contexte ; le worker utilisera _APP injectée par init_app
        st.app_ref = st.app_ref  # no-op, garde l'existante si présente
    # (Re)programmer le timer
    _schedule_run(st, mode=mode)


def stop_all() -> None:
    """
    Arrête proprement tous les timers (à appeler lors du shutdown).
    """
    _STOP_EVENT.set()
    with _STATE_LOCK:
        for st in _STATE.values():
            if st.timer is not None:
                try:
                    st.timer.cancel()
                except Exception:
                    pass
            st.timer = None


# ============================================================
# Internals
# ============================================================

def _schedule_run(st: CompanyDispatchState, mode: str) -> None:
    """
    Programme (ou reprogramme) un timer pour exécuter _try_run après DEBOUNCE+COALESCE.
    """
    delay_sec = (DEBOUNCE_MS + COALESCE_MS) / 1000.0

    # Si un timer existe déjà, on le remplace pour prolonger la fenêtre de coalescence.
    if st.timer is not None:
        try:
            st.timer.cancel()
        except Exception:
            pass

    # Utiliser threading.Timer pour le debounce/coalesce
    timer_cls = __import__('threading').Timer
    t = timer_cls(delay_sec, _try_run, kwargs={"st": st, "mode": mode})
    t.daemon = True
    t.start()
    st.timer = t


def _try_run(st: CompanyDispatchState, mode: str) -> None:
    """
    Tente de lancer un run pour l'entreprise si aucune exécution concurrente.
    Gère un TTL basique au cas où un run serait resté bloqué.
    """
    if _STOP_EVENT.is_set():
        return

    # Vérifier/renouveler le TTL si running
    now = datetime.now(timezone.utc)
    if st.running and st.last_start:
        if now - st.last_start > timedelta(seconds=LOCK_TTL_SEC):
            # On considère le run précédent comme bloqué (TTL expiré)
            logger.warning("[Queue] TTL expired for company=%s, forcing unlock", st.company_id)
            st.running = False

    # Essayer de prendre le lock
    acquired = st.lock.acquire(blocking=False)
    if not acquired:
        # Un autre thread tente de lancer (rare grâce au timer unique)
        _schedule_run(st, mode)  # replanifie un essai
        return

    try:
        if st.running:
            # Déjà en cours (double sécurité). On replanifie.
            _schedule_run(st, mode)
            return

        st.running = True
        st.last_start = now
        _RUNNING[st.company_id] = True
        _PROGRESS[st.company_id] = 5
    finally:
        st.lock.release()

    # Lancer la tâche Celery au lieu d'un thread
    _enqueue_celery_task(st, mode)


def _enqueue_celery_task(st: CompanyDispatchState, mode: str) -> None:
    """
    Enqueue a Celery task instead of running in a thread.
    """
    company_id = st.company_id
    reasons = list(st.backlog)
    st.backlog.clear()
    
    # Choisit l'app : celle capturée sur trigger() ou celle injectée globalement
    app = getattr(st, "app_ref", None) or _APP
    if app is None:
        logger.error("[Queue] No Flask app available for company=%s; aborting run", company_id)
        st.running = False
        st.last_start = None
        _RUNNING[company_id] = False
        _PROGRESS[company_id] = 0
        return
    
    try:
        with app.app_context():
            logger.info(
                "[Queue] Dispatch start company=%s mode=%s reasons=%s params_keys=%s",
                company_id, mode, reasons[-3:], list(getattr(st, "params", {}).keys()),
            )
            
            # Déballer proprement les params coalescés
            run_kwargs = dict(getattr(st, "params", {}))
            # Garantir company_id (sécurité)
            run_kwargs["company_id"] = company_id
            # Ajouter mode si absent
            run_kwargs.setdefault("mode", mode)
            
            # Log the parameters being used for the run
            logger.info(
                "[Queue] Running dispatch with params: company_id=%s, for_date=%s, regular_first=%s, allow_emergency=%s, mode=%s",
                company_id, 
                run_kwargs.get("for_date", "None"),
                run_kwargs.get("regular_first", True),
                run_kwargs.get("allow_emergency", None),
                run_kwargs.get("mode", "auto")
            )
            
            # Import here to avoid circular imports
            from tasks.dispatch_tasks import run_dispatch_task
            
            # Enqueue Celery task
            TaskCallable = cast(Any, run_dispatch_task)  # .delay non typé dans stubs
            task = TaskCallable.delay(**run_kwargs)
            st.last_task_id = task.id
            _CELERY_STATE[company_id] = task.state
            
            logger.info(
                "[Queue] Enqueued Celery task company=%s task_id=%s",
                company_id, task.id
            )
            
            # Update state
            _PROGRESS[company_id] = 20
            
    except Exception as e:
        logger.exception("[Queue] Failed to enqueue Celery task company=%s: %s", company_id, e)
        st.running = False
        st.last_start = None
        _RUNNING[company_id] = False
        _PROGRESS[company_id] = 0
        _LAST_ERROR[company_id] = str(e)