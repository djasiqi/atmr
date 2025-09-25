# backend/services/unified_dispatch/queue.py
from __future__ import annotations

import logging
import faulthandler, signal, sys
import threading
import time
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Any  # <-- Any ajouté

from flask import current_app

from services.unified_dispatch import engine

logger = logging.getLogger(__name__)

# ============================================================
# Native threading (bypass Eventlet monkey patch)
# ============================================================
try:
    # Si Eventlet a monkey-patché threading, on récupère les originaux
    from eventlet import patcher as _el_patcher  # type: ignore
    _orig_threading = _el_patcher.original('threading')
    NativeThread = _orig_threading.Thread
    NativeTimer = _orig_threading.Timer
except Exception:
    # Fallback: pas d'Eventlet, on garde threading standard
    NativeThread = threading.Thread
    NativeTimer = threading.Timer

# coalescing + lock par company_id (anti-collisions)

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

_APP = None

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
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Timer de déclenchement différé (coalescing)
    timer: Optional[threading.Timer] = None
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


# Mémoire globale in‑process (une entrée par company_id)
_STATE: Dict[int, CompanyDispatchState] = {}
# Statut observable par l'API /status
_LAST_RESULT: Dict[int, Dict[str, Any]] = {}
_LAST_ERROR: Dict[int, Optional[str]] = {}
_RUNNING: Dict[int, bool] = {}
_PROGRESS: Dict[int, int] = {}  # 0..100 approximation de progression

# Lock global pour l'accès au dict
_STATE_LOCK = threading.Lock()
# Interrupteur global (stop propre)
_STOP_EVENT = threading.Event()


def _get_state(company_id: int) -> CompanyDispatchState:
    with _STATE_LOCK:
        st = _STATE.get(company_id)
        if st is None:
            st = CompanyDispatchState(company_id=company_id)
            _STATE[company_id] = st
        return st


# ============================================================
# API publique
# ============================================================

def get_status(company_id: int) -> Dict[str, Any]:
    """
    Utilisé par GET /company_dispatch/status
    """
    last = _LAST_RESULT.get(company_id) or {}
    return {
        "is_running": bool(_RUNNING.get(company_id, False)),
        "progress": int(_PROGRESS.get(company_id, 0)),
        "last_result": last,
        "last_result_meta": last.get("meta"),
        "last_error": _LAST_ERROR.get(company_id),
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
        st.app_ref = current_app._get_current_object()
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

    st.timer = NativeTimer(delay_sec, _try_run, kwargs={"st": st, "mode": mode})
    st.timer.daemon = True
    st.timer.start()


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
    finally:
        st.lock.release()

    # Lancer le worker dans un thread dédié (non bloquant pour l'appelant)
    t = NativeThread(target=_run_worker, kwargs={"st": st, "mode": mode}, daemon=True)
    t.start()


def _run_worker(st: CompanyDispatchState, mode: str) -> None:
    start_ts = time.time()
    company_id = st.company_id
    reasons = list(st.backlog)
    st.backlog.clear()
    _PROGRESS[company_id] = 5

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
            # Dump auto si crash natif (SIGABRT/SIGSEGV) pour laisser une trace dans les logs
            try:
                faulthandler.enable(file=sys.stderr, all_threads=True)
                for _sig in (signal.SIGABRT, signal.SIGSEGV):
                    try:
                        faulthandler.register(_sig, file=sys.stderr, all_threads=True)
                    except Exception:
                        pass
            except Exception:
                pass

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
            # Attend: company_id, for_date, regular_first, allow_emergency, overrides
            _PROGRESS[company_id] = max(_PROGRESS.get(company_id, 5), 20)
            
            # Log the parameters being used for the run
            logger.info(
                "[Queue] Running dispatch with params: company_id=%s, for_date=%s, regular_first=%s, allow_emergency=%s, mode=%s",
                company_id, 
                run_kwargs.get("for_date", "None"),
                run_kwargs.get("regular_first", True),
                run_kwargs.get("allow_emergency", None),
                run_kwargs.get("mode", "auto")
            )
            
            result = engine.run(**run_kwargs)
            if not result:
                result = {}
            if not isinstance(result, dict):
                result = {"meta": {"raw": result}}
            # Fallbacks pour une structure toujours définie
            result.setdefault("assignments", [])
            result.setdefault("bookings", [])
            result.setdefault("drivers", [])
            result.setdefault("meta", {})
            result.setdefault("dispatch_run_id", None)
            _PROGRESS[company_id] = max(_PROGRESS.get(company_id, 20), 90)
            assigned = len(result.get("assignments", []))
            unassigned = len(result.get("unassigned", []))
            logger.info("[Queue] Dispatch done company=%s assigned=%d unassigned=%d duration=%.3fs",
                        company_id, assigned, unassigned, time.time() - start_ts)
            st.recent_failures = 0
            _LAST_RESULT[company_id] = result
            
            # Émettre un événement Socket.IO pour signaler la fin du run
            try:
                from services.notification_service import notify_dispatch_run_completed
                dispatch_run_id = result.get("dispatch_run_id") or result.get("meta", {}).get("dispatch_run_id")
                if dispatch_run_id:
                    logger.info("[Queue] Emitting dispatch_run_completed event for company=%s run_id=%s", 
                                company_id, dispatch_run_id)
                    
                    # Récupérer la date du run pour l'inclure dans la notification
                    day_str = None
                    try:
                        from models import DispatchRun
                        dispatch_run = DispatchRun.query.get(dispatch_run_id)
                        if dispatch_run and dispatch_run.day:
                            day_str = dispatch_run.day.isoformat()
                    except Exception as e:
                        logger.warning("[Queue] Failed to get day_str from DispatchRun: %s", e)
                    
                    # Envoyer la notification avec la date si disponible
                    notify_dispatch_run_completed(company_id, dispatch_run_id, assigned, day_str)
            except Exception as e:
                logger.warning("[Queue] Failed to emit dispatch_run_completed: %s", e)
  
            _LAST_ERROR[company_id] = None
            _PROGRESS[company_id] = 100
    except Exception as e:
        st.recent_failures += 1
        delay = min(30, 2 ** min(st.recent_failures, 4))
        logger.exception("[Queue] Dispatch error company=%s failures=%d -> retry in %ss",
                         company_id, st.recent_failures, delay)
        _LAST_ERROR[company_id] = str(e)
        _PROGRESS[company_id] = 0
        if not _STOP_EVENT.is_set():
            try:
                if st.timer is not None:
                    st.timer.cancel()
            except Exception:
                pass
            st.timer = NativeTimer(delay, _try_run, kwargs={"st": st, "mode": mode})
            st.timer.daemon = True
            st.timer.start()
    finally:
        # Important: nettoyer la session SQLAlchemy quand on sort du thread
        try:
            from ext import db
            db.session.remove()
        except Exception:
            pass
        try:
            st.params.clear()
        except Exception:
            st.params = {}
        st.running = False
        st.last_start = None
        _RUNNING[company_id] = False
        # Ne pas supprimer _PROGRESS ici pour garder une valeur finale 0/100 lisible



# ============================================================
# Hooks utilitaires (optionnels)
# ============================================================

def trigger_on_booking_change(company_id: int, action: str) -> None:
    """
    À appeler depuis vos routes/modèles :
      - action ∈ {"create","update","cancel","return_request"}
    Regroupe les changements et déclenche un run "auto".
    """
    reason = f"booking_{action}"
    trigger(company_id, reason=reason, mode="auto")


def trigger_on_driver_status(company_id: int, action: str) -> None:
    """
    À appeler lorsqu'un chauffeur change de statut (dispo/actif/localisation).
      - action ∈ {"availability","activation","location"}
    """
    reason = f"driver_{action}"
    trigger(company_id, reason=reason, mode="auto")