# backend/services/unified_dispatch/queue.py
from __future__ import annotations

import logging
import os
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Dict, List, cast

from cachetools import (  # pyright: ignore[reportMissingModuleSource]  # ✅ P1: Limiter caches in-memory
    LRUCache,
    TTLCache,
)
from celery.result import AsyncResult  # pyright: ignore[reportMissingImports]
from flask import current_app  # pyright: ignore[reportMissingImports]
from sqlalchemy.exc import IntegrityError

from ext import db
from models import Company, DispatchRun, DispatchStatus
from models.base import _as_dt, _iso
from repositories.company_repository import CompanyRepository
from repositories.dispatch_run_repository import DispatchRunRepository

logger = logging.getLogger(__name__)


def _serialize_datetimes(obj: Any) -> Any:
    """Sérialise récursivement tous les objets datetime/date en chaînes ISO.

    Args:
        obj: Objet à sérialiser (dict, list, datetime, date, ou autre)

    Returns:
        Objet avec tous les datetime/date sérialisés en chaînes ISO
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize_datetimes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_datetimes(item) for item in obj]
    return obj


# ============================================================
# Valeurs par défaut raisonnables, surchargées via ENV.
# ============================================================

DEBOUNCE_MS = int(os.getenv("UD_RTC_DEBOUNCE_MS", "800"))
COALESCE_MS = int(os.getenv("UD_RTC_COALESCE_MS", "800"))
LOCK_TTL_SEC = int(os.getenv("UD_RTC_LOCK_TTL_SEC", "30"))
MAX_BACKLOG = int(os.getenv("UD_RTC_MAX_QUEUE_BACKLOG", "100"))

# ✅ P1: Limites pour les caches in-memory (évite explosion mémoire)
# Limite le nombre d'entreprises en cache (max 1000 entreprises actives)
CACHE_MAX_COMPANIES = int(os.getenv("UD_CACHE_MAX_COMPANIES", "1000"))
# TTL pour les données de statut (1 heure)
CACHE_STATUS_TTL_SEC = int(os.getenv("UD_CACHE_STATUS_TTL_SEC", "3600"))

# ============================================================
# app Flask global
# ============================================================

_APP: Any | None = None


def init_app(app):
    """À appeler depuis create_app(app)."""
    global _APP  # noqa: PLW0603
    _APP = app


# ============================================================
# State par entreprise
# ============================================================


@dataclass
class CompanyDispatchState:
    company_id: int
    # Sémaphore/lock pour empêcher deux runs concurrents sur la même entreprise
    lock: Any = field(default_factory=lambda: __import__("threading").Lock())
    # Timer de déclenchement différé (coalescing)
    timer: Any | None = None
    # Indique si un run est en cours
    running: bool = False
    # Pour éviter un run bloqué : timestamp du dernier start
    last_start: datetime | None = None
    # Backlog de raisons (debug)
    backlog: List[str] = field(default_factory=list)
    # Nombre d'échecs récents (pour backoff)
    recent_failures: int = 0
    # 🔴 NEW: paramètres cumulés pour le prochain run (for_date, overrides, ...)
    params: Dict[str, Any] = field(default_factory=dict)
    # Référence à l'app Flask (capturée sur trigger() si contexte dispo)
    app_ref: Any | None = None
    # 🔴 NEW: ID de la dernière tâche Celery
    last_task_id: str | None = None


# ✅ P1: Mémoire globale in-process avec caches limités (évite explosion mémoire)
# LRUCache pour l'état actif des entreprises (max 1000 entreprises)
_STATE: LRUCache[int, CompanyDispatchState] = LRUCache(maxsize=CACHE_MAX_COMPANIES)
# TTLCache pour les données de statut (expire après 1h)
_LAST_RESULT: TTLCache[int, Dict[str, Any]] = TTLCache(
    maxsize=CACHE_MAX_COMPANIES, ttl=CACHE_STATUS_TTL_SEC
)
_LAST_ERROR: TTLCache[int, str | None] = TTLCache(
    maxsize=CACHE_MAX_COMPANIES, ttl=CACHE_STATUS_TTL_SEC
)
_RUNNING: TTLCache[int, bool] = TTLCache(
    maxsize=CACHE_MAX_COMPANIES, ttl=CACHE_STATUS_TTL_SEC
)
_PROGRESS: TTLCache[int, int] = TTLCache(
    maxsize=CACHE_MAX_COMPANIES, ttl=CACHE_STATUS_TTL_SEC
)  # 0..100 approximation de progression
# État Celery (PENDING, STARTED, SUCCESS, FAILURE, etc.)
_CELERY_STATE: TTLCache[int, str] = TTLCache(
    maxsize=CACHE_MAX_COMPANIES, ttl=CACHE_STATUS_TTL_SEC
)

# Lock global pour l'accès au dict
_STATE_LOCK = __import__("threading").Lock()
# Interrupteur global (stop propre)
_STOP_EVENT = __import__("threading").Event()


def _get_state(company_id: int) -> CompanyDispatchState:
    with _STATE_LOCK:
        st = _STATE.get(company_id)
        if st is None:
            st = CompanyDispatchState(company_id=company_id)
            _STATE[company_id] = st
        return st


def _get_redis_for_status() -> Any | None:
    """Récupère un client Redis pour le cache statut dispatch.

    ✅ P1: Support Redis pour partage entre instances.

    Returns:
        Client Redis ou None si indisponible
    """
    try:
        from ext import redis_client as ext_redis_client

        if ext_redis_client is not None:
            ext_redis_client.ping()
            return ext_redis_client
    except Exception:
        pass

    # Fallback : essayer de créer depuis REDIS_URL
    try:
        redis_url = os.getenv("REDIS_URL", None)
        if redis_url:
            import redis  # pyright: ignore[reportMissingImports]

            socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
            socket_connect_timeout = int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
            client = redis.from_url(
                redis_url,
                decode_responses=False,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
            )
            client.ping()
            return client
    except Exception:
        pass

    return None


def get_status(company_id: int, for_date: str | None = None) -> Dict[str, Any]:
    """Utilisé par GET /company_dispatch/status
    Enrichi avec des informations de diagnostic plus détaillées.

    ✅ P1: Support Redis (L2) + TTLCache in-memory (L1) pour partage entre instances.

    Args:
        company_id: ID de l'entreprise
        for_date: Date optionnelle (YYYY-MM-DD) pour obtenir le statut
            d'un dispatch spécifique
    """
    # ✅ P1: Vérifier cache Redis (L2 cache) pour partage entre instances
    redis_client = _get_redis_for_status()
    cache_key = f"dispatch:status:{company_id}:{for_date or 'default'}"
    last: Dict[str, Any] = {}
    last_error: str | None = None

    if redis_client:
        try:
            import json

            cached_data = redis_client.get(cache_key)
            if cached_data:
                cached_str: str | None = None
                if isinstance(cached_data, (bytes, bytearray)):
                    cached_str = cached_data.decode("utf-8", errors="ignore")
                elif isinstance(cached_data, str):
                    cached_str = cached_data

                if cached_str:
                    try:
                        cached_result = json.loads(cached_str)
                        if isinstance(cached_result, dict):
                            last = cached_result.get("last_result", {})
                            last_error = cached_result.get("last_error")
                            logger.debug(
                                "[Queue] ✅ Redis cache hit for dispatch status (company=%s)",
                                company_id,
                            )
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.debug("[Queue] Failed to decode cached status: %s", e)
        except Exception as e:
            logger.debug("[Queue] Redis cache check failed: %s", e)

    # ✅ P1: Fallback vers cache local (L1 cache) si Redis miss
    if not last:
        last = _LAST_RESULT.get(company_id) or {}
        last_error = _LAST_ERROR.get(company_id)

    # Get counts from the last result
    bookings_count = len(last.get("bookings", []))
    drivers_count = len(last.get("drivers", []))
    assignments_count = len(last.get("assignments", []))

    # ✅ Récupérer le DispatchRun actif pour cette date si fournie
    active_dispatch_run_id = None
    active_dispatch_status = None
    active_assignments_count = 0
    dispatch_run = None  # ✅ Déclarer dispatch_run pour utilisation ultérieure

    if for_date:
        try:
            from datetime import date as date_type

            day_date = date_type.fromisoformat(for_date)
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            dispatch_run_repo = DispatchRunRepository()
            dispatch_run_dto = dispatch_run_repo.find_by_company_and_day(
                company_id, day_date
            )
            # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
            dispatch_run = (
                DispatchRun.query.get(dispatch_run_dto.id) if dispatch_run_dto else None
            )

            if dispatch_run:
                active_dispatch_run_id = dispatch_run.id
                active_dispatch_status = (
                    dispatch_run.status.value
                    if hasattr(dispatch_run.status, "value")
                    else str(dispatch_run.status)
                )

                # ✅ Compter les assignments pour ce DispatchRun
                active_assignments_count = (
                    len(dispatch_run.assignments)
                    if hasattr(dispatch_run, "assignments")
                    else 0
                )

                logger.debug(
                    (
                        "[Queue] Found active DispatchRun id=%s status=%s "
                        "assignments=%s for company=%s date=%s"
                    ),
                    active_dispatch_run_id,
                    active_dispatch_status,
                    active_assignments_count,
                    company_id,
                    for_date,
                )
        except Exception as e:
            logger.exception(
                "[Queue] Error fetching DispatchRun for date=%s: %s", for_date, e
            )

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
            is_running = celery_state in ("PENDING", "RECEIVED", "STARTED")
            _RUNNING[company_id] = is_running
            _CELERY_STATE[company_id] = celery_state

            # If task has failed, get the error
            if celery_state == "FAILURE" and task_result.failed():
                _LAST_ERROR[company_id] = str(task_result.result)
                last_error = _LAST_ERROR[company_id]

            # If task has succeeded, update the last result
            if celery_state == "SUCCESS" and task_result.ready():
                try:
                    result = task_result.get()
                    if isinstance(result, dict):
                        _LAST_RESULT[company_id] = result
                        last = result
                        bookings_count = len(last.get("bookings", []))
                        drivers_count = len(last.get("drivers", []))
                        assignments_count = len(last.get("assignments", []))

                        # ✅ P1: Mettre à jour cache Redis (L2 cache) pour partage entre instances
                        redis_client = _get_redis_for_status()
                        if redis_client:
                            try:
                                import json

                                cache_key = f"dispatch:status:{company_id}:{for_date or 'default'}"
                                cache_value = json.dumps(
                                    {
                                        "last_result": result,
                                        "last_error": last_error,
                                        "is_running": False,
                                        "progress": 100,
                                    }
                                )
                                redis_client.setex(
                                    cache_key, CACHE_STATUS_TTL_SEC, cache_value
                                )
                                logger.debug(
                                    "[Queue] ✅ Redis cache updated for dispatch status (company=%s)",
                                    company_id,
                                )
                            except Exception as e:
                                logger.debug(
                                    "[Queue] Failed to update Redis cache: %s", e
                                )
                except Exception as e:
                    logger.exception("[Queue] Error getting task result: %s", e)

        except Exception as e:
            logger.exception("[Queue] Error checking task status: %s", e)

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

    # ✅ Utiliser le dispatch_run_id actif si disponible,
    # sinon celui du dernier résultat
    dispatch_run_id = (
        active_dispatch_run_id
        or last.get("dispatch_run_id")
        or (last.get("meta", {}) or {}).get("dispatch_run_id")
    )

    # ✅ Construire active_dispatch_run avec sérialisation des dates
    # si dispatch_run existe
    active_dispatch_run_dict = None
    if dispatch_run and active_dispatch_run_id:
        active_dispatch_run_dict = {
            "id": active_dispatch_run_id,
            "status": active_dispatch_status,
            "assignments_count": active_assignments_count,
            "day": dispatch_run.day.isoformat() if dispatch_run.day else None,
            "created_at": _iso(_as_dt(dispatch_run.created_at))
            if dispatch_run.created_at
            else None,
            "started_at": _iso(_as_dt(dispatch_run.started_at))
            if dispatch_run.started_at
            else None,
            "completed_at": _iso(_as_dt(dispatch_run.completed_at))
            if dispatch_run.completed_at
            else None,
        }

    # ✅ Sérialiser récursivement tous les objets datetime/date
    # pour éviter les erreurs JSON
    serialized_last = _serialize_datetimes(last) if last else {}
    serialized_meta = (
        _serialize_datetimes(last.get("meta")) if last and last.get("meta") else None
    )

    return {
        "is_running": bool(_RUNNING.get(company_id, False)),
        "progress": int(_PROGRESS.get(company_id, 0)),
        "last_result": serialized_last,
        "last_result_meta": serialized_meta,
        "last_error": last_error,
        "reason": reason,
        "counters": {
            "bookings": bookings_count,
            "drivers": drivers_count,
            "assignments": assignments_count,
        },
        "dispatch_run_id": dispatch_run_id,
        "active_dispatch_run": active_dispatch_run_dict,
        "celery_state": _CELERY_STATE.get(company_id, celery_state),
        "last_task_id": st.last_task_id,
    }


def trigger_job(company_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
    """Utilisé par POST /company_dispatch/run (async).
    Enfile un job (coalescé) et renvoie un job_id.
    Crée le DispatchRun avec statut PENDING avant l'enfilage
    pour avoir un dispatch_run_id immédiatement.
    """
    job_id = str(uuid.uuid4())
    mode = str((params or {}).get("mode", "auto")).strip().lower()

    logger.info(
        "[Queue] trigger_job called for company_id=%s params_keys=%s",
        company_id,
        list(params.keys()) if params else [],
    )

    snapshot: Dict[str, Any] = {
        "for_date": params.get("for_date"),
        "mode": params.get("mode"),
        "regular_first": params.get("regular_first"),
        "allow_emergency": params.get("allow_emergency"),
    }
    # ✅ P1: Protéger accès dictionnaires pour éviter KeyError
    overrides = params.get("overrides")
    if isinstance(overrides, dict):
        snapshot["overrides_keys"] = sorted(overrides.keys())
    dispatch_overrides = params.get("dispatch_overrides")
    if isinstance(dispatch_overrides, dict):
        snapshot["dispatch_overrides_keys"] = sorted(dispatch_overrides.keys())
    logger.info("[Queue] trigger_job params snapshot=%s", snapshot)

    # Créer le DispatchRun avec statut PENDING avant l'enfilage
    dispatch_run_id = None
    try:
        for_date_str = params.get("for_date")
        logger.debug("[Queue] trigger_job: for_date_str=%s", for_date_str)
        if for_date_str:
            # Parser la date
            try:
                day_date = date.fromisoformat(for_date_str)
            except (ValueError, TypeError):
                logger.warning(
                    "[Queue] Invalid for_date=%s, cannot create DispatchRun early",
                    for_date_str,
                )
                day_date = None
        else:
            # Utiliser aujourd'hui par défaut
            day_date = datetime.now(UTC).date()
            logger.warning(
                "[Queue] No for_date in params, using today=%s for DispatchRun",
                day_date,
            )

        if day_date:
            logger.info(
                (
                    "[Queue] trigger_job: day_date=%s, "
                    "attempting to create/reuse DispatchRun"
                ),
                day_date,
            )
            # Créer ou réutiliser le DispatchRun avec statut PENDING
            # Utiliser une transaction courte pour éviter les race conditions
            try:
                # ✅ Flask/SQLAlchemy gère automatiquement les transactions
                # - pas besoin de begin()
                # ✅ Utilisation du repository pour découpler de SQLAlchemy
                dispatch_run_repo = DispatchRunRepository()
                existing_run_dto = dispatch_run_repo.find_by_company_and_day(
                    company_id, day_date
                )
                # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
                existing_run = (
                    DispatchRun.query.get(existing_run_dto.id)
                    if existing_run_dto
                    else None
                )

                if existing_run and existing_run.day != day_date:
                    existing_run = None

                if existing_run:
                    # Réutiliser le DispatchRun existant
                    existing_run.status = DispatchStatus.PENDING
                    existing_run.started_at = None
                    existing_run.completed_at = None
                    existing_run.config = {
                        "mode": mode,
                        "regular_first": params.get("regular_first", True),
                        "allow_emergency": params.get("allow_emergency"),
                        "for_date": for_date_str,
                    }
                    dispatch_run_id = existing_run.id
                    logger.info(
                        (
                            "[Queue] Reusing existing DispatchRun id=%s "
                            "for company=%s day=%s"
                        ),
                        dispatch_run_id,
                        company_id,
                        day_date,
                    )
                else:
                    # Créer un nouveau DispatchRun
                    new_run = DispatchRun()
                    new_run.company_id = company_id
                    new_run.day = day_date
                    new_run.status = DispatchStatus.PENDING
                    new_run.created_at = datetime.now(UTC)
                    new_run.config = {
                        "mode": mode,
                        "regular_first": params.get("regular_first", True),
                        "allow_emergency": params.get("allow_emergency"),
                        "for_date": for_date_str,
                    }
                    db.session.add(new_run)
                    db.session.flush()  # Pour obtenir l'ID
                    dispatch_run_id = new_run.id
                    logger.info(
                        (
                            "[Queue] Created DispatchRun id=%s with status PENDING "
                            "for company=%s day=%s"
                        ),
                        dispatch_run_id,
                        company_id,
                        day_date,
                    )

                # ✅ Commit explicite pour persister la transaction
                db.session.commit()
                logger.debug(
                    "[Queue] DispatchRun id=%s committed successfully", dispatch_run_id
                )
            except IntegrityError as e:
                # ✅ P2.2: Track métrique IntegrityError (race condition)
                from services.unified_dispatch.metrics.errors import (
                    track_integrity_error,
                )

                error_code = (
                    getattr(e.orig, "pgcode", None) if hasattr(e, "orig") else None
                )
                track_integrity_error(
                    error_code=str(error_code) if error_code else "unknown",
                    company_id=company_id,
                    dispatch_run_id=None,
                )

                # Race condition : un autre thread a créé le DispatchRun entre temps
                db.session.rollback()
                # ✅ Utilisation du repository pour découpler de SQLAlchemy
                dispatch_run_repo = DispatchRunRepository()
                existing_run_dto = dispatch_run_repo.find_by_company_and_day(
                    company_id, day_date
                )
                # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
                existing_run = (
                    DispatchRun.query.get(existing_run_dto.id)
                    if existing_run_dto
                    else None
                )
                if existing_run and existing_run.day != day_date:
                    existing_run = None
                if existing_run:
                    dispatch_run_id = existing_run.id
                    logger.info(
                        "[Queue] Race condition: using existing DispatchRun id=%s",
                        dispatch_run_id,
                    )
                else:
                    logger.error(
                        "[Queue] Failed to create/reuse DispatchRun after IntegrityError"
                    )
            except Exception as e:
                db.session.rollback()
                logger.exception("[Queue] Failed to create DispatchRun early: %s", e)
                # Continuer sans dispatch_run_id (fallback vers comportement actuel)
    except Exception as e:
        logger.exception("[Queue] Error creating DispatchRun early: %s", e)
        # Continuer sans dispatch_run_id (fallback vers comportement actuel)

    # Harmoniser les overrides (legacy dispatch_overrides → overrides)
    if params.get("dispatch_overrides") and "overrides" not in params:
        params = dict(params)
        params["overrides"] = params.pop("dispatch_overrides")

    # Appliquer les overrides de company.autonomous_config si absents
    if "overrides" not in params:
        try:
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            company_repo = CompanyRepository()
            company_dto = company_repo.find_by_id(company_id)
            company_obj = Company.query.get(company_dto.id) if company_dto else None
            if company_obj:
                auto_cfg = company_obj.get_autonomous_config()
                dispatch_defaults = auto_cfg.get("dispatch_overrides")
                if isinstance(dispatch_defaults, dict) and dispatch_defaults:
                    params = dict(params)
                    params["overrides"] = dispatch_defaults
                    logger.info(
                        (
                            "[Queue] trigger_job applied company dispatch_overrides "
                            "(keys=%s)"
                        ),
                        sorted(dispatch_defaults.keys()),
                    )
        except Exception as exc:
            logger.warning(
                (
                    "[Queue] Failed to load company dispatch_overrides "
                    "for company_id=%s: %s"
                ),
                company_id,
                exc,
            )

    # Passer le dispatch_run_id dans les params pour la tâche Celery
    if dispatch_run_id:
        params = dict(params)  # Copie pour éviter mutation
        params["dispatch_run_id"] = dispatch_run_id

    trigger(company_id, reason="manual_trigger", mode=mode, params=params)
    return {
        "id": job_id,
        "company_id": company_id,
        "status": "queued",
        "dispatch_run_id": dispatch_run_id,  # Retourner le dispatch_run_id si créé
    }


def trigger(
    company_id: int,
    reason: str = "generic",
    mode: str = "auto",
    params: Dict[str, Any] | None = None,
) -> None:
    """Appel léger depuis les routes ou services :
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
        st.backlog.append(f"{datetime.now(UTC).isoformat()} {reason}")
    # Coalesce: on mémorise/merge les derniers params (on garde la dernière
    # valeur pour chaque clé)
    if params:
        st.params = dict(params)

    try:
        # si on est dans un contexte requête, mémorise l'app pour le worker
        # (LocalProxy → objet réel)
        get_obj = getattr(current_app, "_get_current_object", None)
        if callable(get_obj):
            st.app_ref = get_obj()
        else:
            st.app_ref = current_app  # fallback typé Any
    except Exception:
        # pas de contexte ; le worker utilisera _APP injectée par init_app
        st.app_ref = st.app_ref  # no-op, garde l'existante si présente
    # (Re)programmer le timer
    _schedule_run(st, mode=mode)


def trigger_on_booking_change(
    company_id: int,
    mode: str = "auto",
    params: Dict[str, Any] | None = None,
) -> None:
    """Déclenche un dispatch suite à un changement de booking.

    Args:
        company_id: ID de l'entreprise
        mode: Mode de dispatch (défaut: "auto")
        params: Paramètres additionnels pour le dispatch
    """
    trigger(company_id=company_id, reason="booking_change", mode=mode, params=params)


def stop_all() -> None:
    """Arrête proprement tous les timers (à appeler lors du shutdown)."""
    _STOP_EVENT.set()
    with _STATE_LOCK:
        for st in _STATE.values():
            if st.timer is not None:
                with suppress(Exception):
                    st.timer.cancel()
            st.timer = None


# ============================================================
# Internals
# ============================================================


def _schedule_run(st: CompanyDispatchState, mode: str) -> None:
    """Programme (ou reprogramme) un timer pour exécuter _try_run
    après DEBOUNCE+COALESCE.

    ✅ P1: Tous les accès à st.timer sont protégés par st.lock pour éviter
    les race conditions.
    """
    delay_sec = (DEBOUNCE_MS + COALESCE_MS) / 1000.0

    # ✅ P1: Protéger tous les accès à st.timer avec le lock de l'état
    with st.lock:
        # Si un timer existe déjà, on le remplace pour prolonger la fenêtre de
        # coalescence.
        if st.timer is not None:
            with suppress(Exception):
                st.timer.cancel()

        # Utiliser threading.Timer pour le debounce/coalesce
        timer_cls = __import__("threading").Timer
        t = timer_cls(delay_sec, _try_run, kwargs={"st": st, "mode": mode})
        t.daemon = True
        t.start()
        st.timer = t


def _try_run(st: CompanyDispatchState, mode: str) -> None:
    """Tente de lancer un run pour l'entreprise si aucune exécution concurrente.
    Gère un TTL basique au cas où un run serait resté bloqué.
    """
    if _STOP_EVENT.is_set():
        return

    # Vérifier/renouveler le TTL si running
    now = datetime.now(UTC)
    if (
        st.running
        and st.last_start
        and now - st.last_start > timedelta(seconds=LOCK_TTL_SEC)
    ):
        # On considère le run précédent comme bloqué (TTL expiré)
        logger.warning(
            "[Queue] TTL expired for company=%s, forcing unlock", st.company_id
        )
        st.running = False

    # Essayer de prendre le lock
    acquired = st.lock.acquire(blocking=False)
    if not acquired:
        # Un autre thread tente de lancer (rare grâce au timer unique)
        # ✅ P1: Pas besoin de lock ici car _schedule_run() l'acquiert lui-même
        _schedule_run(st, mode)  # replanifie un essai
        return

    lock_released = False
    try:
        if st.running:
            # Déjà en cours (double sécurité). On replanifie.
            # ✅ P1: Libérer le lock avant d'appeler _schedule_run() pour éviter deadlock
            st.lock.release()
            lock_released = True
            _schedule_run(st, mode)
            return

        st.running = True
        st.last_start = now
        _RUNNING[st.company_id] = True
        _PROGRESS[st.company_id] = 5

        # ✅ P1: Invalider cache Redis statut dispatch lors du démarrage
        redis_client = _get_redis_for_status()
        if redis_client:
            try:
                from services.infrastructure.cache import (
                    invalidate_dispatch_status_cache,
                )

                invalidate_dispatch_status_cache(
                    st.company_id, st.params.get("for_date")
                )
            except Exception as e:
                logger.debug(
                    "[Queue] Failed to invalidate dispatch status cache: %s", e
                )
    finally:
        # ✅ P1: Ne libérer que si le lock n'a pas été libéré dans le if st.running
        if not lock_released:
            st.lock.release()

    # Lancer la tâche Celery au lieu d'un thread
    _enqueue_celery_task(st, mode)


def _enqueue_celery_task(st: CompanyDispatchState, mode: str) -> None:
    """Enqueue a Celery task instead of running in a thread."""
    company_id = st.company_id
    reasons = list(st.backlog)
    st.backlog.clear()

    # Choisit l'app : celle capturée sur trigger() ou celle injectée
    # globalement
    app = getattr(st, "app_ref", None) or _APP
    if app is None:
        logger.error(
            "[Queue] No Flask app available for company=%s; aborting run", company_id
        )
        st.running = False
        st.last_start = None
        _RUNNING[company_id] = False
        _PROGRESS[company_id] = 0
        return

    try:
        with app.app_context():
            logger.info(
                "[Queue] Dispatch start company=%s mode=%s reasons=%s params_keys=%s",
                company_id,
                mode,
                reasons[-3:],
                list(getattr(st, "params", {}).keys()),
            )

            # Déballer proprement les params coalescés
            run_kwargs = dict(getattr(st, "params", {}))
            # Garantir company_id (sécurité)
            run_kwargs["company_id"] = company_id
            # Ajouter mode si absent
            run_kwargs.setdefault("mode", mode)

            # Anti-duplication: vérifier si un run identique est déjà en cours
            import hashlib
            import json

            from ext import redis_client

            params_str = json.dumps(run_kwargs, sort_keys=True)
            # SHA-256 au lieu de MD5 pour meilleures pratiques de sécurité
            params_hash = hashlib.sha256(params_str.encode()).hexdigest()
            dedup_key = f"dispatch:enqueued:{company_id}:{params_hash}"

            if redis_client:
                if not redis_client.setnx(dedup_key, 1):
                    logger.info(
                        "[Queue] Duplicate run ignored for company=%s (same params)",
                        company_id,
                    )
                    st.running = False
                    st.last_start = None
                    _RUNNING[company_id] = False
                    _PROGRESS[company_id] = 0
                    return

                # TTL 5 minutes pour éviter les blocages
                redis_client.expire(dedup_key, 300)

            # Log the parameters being used for the run
            logger.info(
                (
                    "[Queue] Running dispatch with params: company_id=%s, "
                    "for_date=%s, regular_first=%s, allow_emergency=%s, mode=%s"
                ),
                company_id,
                run_kwargs.get("for_date", "None"),
                run_kwargs.get("regular_first", True),
                run_kwargs.get("allow_emergency"),
                run_kwargs.get("mode", "auto"),
            )

            # Import here to avoid circular imports
            from celery_app import celery as celery_app
            from tasks.dispatch_tasks import run_dispatch_task

            # ✅ Vérifier et forcer Redis comme transport
            broker_url = celery_app.conf.broker_url

            # Constante pour la longueur max de l'URL du broker à afficher
            MAX_BROKER_URL_DISPLAY_LENGTH = 50

            # ✅ Forcer explicitement le transport Redis (toujours, pour être sûr)
            celery_app.conf.broker_transport = "redis"

            # ✅ Forcer aussi broker_write_url pour s'assurer que Redis est utilisé
            celery_app.conf.broker_write_url = broker_url
            celery_app.conf.broker_read_url = broker_url

            # ✅ Vérifier que le broker_url commence bien par "redis://"
            if broker_url and not broker_url.startswith("redis://"):
                logger.error(
                    "[Queue] ⚠️ broker_url ne commence pas par 'redis://': %s",
                    broker_url[:MAX_BROKER_URL_DISPLAY_LENGTH] + "***"
                    if len(broker_url) > MAX_BROKER_URL_DISPLAY_LENGTH
                    else broker_url,
                )
                # Reconstruire l'URL Redis depuis les variables d'environnement
                import os
                from urllib.parse import quote_plus

                redis_host = os.getenv("REDIS_HOST", "redis")
                redis_port = os.getenv("REDIS_PORT", "6379")
                redis_db = os.getenv("REDIS_DB", "0")
                redis_password = os.getenv("REDIS_PASSWORD", "")
                if redis_password:
                    redis_password_escaped = quote_plus(redis_password)
                    broker_url = f"redis://:{redis_password_escaped}@{redis_host}:{redis_port}/{redis_db}"
                else:
                    broker_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
                celery_app.conf.broker_url = broker_url
                logger.info(
                    "[Queue] ✅ broker_url reconstruit pour Redis: %s",
                    broker_url[:MAX_BROKER_URL_DISPLAY_LENGTH] + "***"
                    if len(broker_url) > MAX_BROKER_URL_DISPLAY_LENGTH
                    else broker_url,
                )

            # ✅ FORCER la réinitialisation COMPLÈTE de la connexion Celery
            # Le problème : Celery/Kombu utilise AMQP en cache même si on configure Redis
            # Solution : Fermer TOUTES les connexions et forcer une nouvelle connexion Redis
            with suppress(Exception):
                # 1. Fermer la connexion principale si elle existe
                connection = getattr(celery_app, "_connection", None)
                if connection:
                    logger.info(
                        "[Queue] 🔄 Fermeture de la connexion Celery principale"
                    )
                    with suppress(Exception):
                        connection.close()
                    celery_app._connection = None

                # 2. Forcer la suppression du cache de connexion broker
                broker_connection = getattr(celery_app, "broker_connection", None)
                if broker_connection is not None:
                    logger.info("[Queue] 🔄 Suppression du cache broker_connection")
                    with suppress(Exception):
                        broker_connection.close()
                    celery_app.broker_connection = None

                # 3. Forcer la suppression de tous les caches de connexion Kombu
                # Kombu peut mettre en cache la connexion dans plusieurs endroits
                if hasattr(celery_app, "pool"):
                    pool = getattr(celery_app, "pool", None)
                    if pool and hasattr(pool, "_connection"):
                        logger.info("[Queue] 🔄 Suppression du cache pool._connection")
                        with suppress(Exception):
                            if pool._connection:
                                pool._connection.close()
                        pool._connection = None

            # 4. FORCER la création d'une nouvelle connexion avec Redis explicitement
            # en passant broker_url et transport directement
            logger.info("[Queue] 🔄 Création d'une nouvelle connexion Redis explicite")

            # Initialiser redis_connection à None
            redis_connection = None

            try:
                # Forcer la création d'une nouvelle connexion avec Redis
                # en passant broker_url et transport explicitement
                from kombu import Connection  # pyright: ignore[reportMissingImports]

                # Créer une nouvelle connexion Kombu avec Redis explicitement
                redis_connection = Connection(
                    broker_url,
                    transport="redis",
                    transport_options={},
                )

                logger.info("[Queue] ✅ Nouvelle connexion Redis créée explicitement")
            except Exception as conn_err:
                logger.warning(
                    "[Queue] ⚠️ Erreur lors de la création explicite de la connexion Redis: %s",
                    conn_err,
                )
                # Continuer quand même, peut-être que la connexion par défaut fonctionnera

            logger.info(
                "[Queue] Enqueuing task with broker_url=%s, transport=%s",
                broker_url[:MAX_BROKER_URL_DISPLAY_LENGTH] + "***"
                if broker_url and len(broker_url) > MAX_BROKER_URL_DISPLAY_LENGTH
                else broker_url,
                celery_app.conf.broker_transport,
            )

            # Enqueue Celery task avec Celery normalement
            # ✅ Forcer Celery à utiliser Redis en remplaçant broker_connection
            logger.info(
                "[Queue] 📤 Enfilage de la tâche via Celery avec connexion Redis forcée"
            )

            # Initialiser old_connection pour pouvoir la restaurer
            old_connection = None
            old_connection_for_write = None

            try:
                # Si on a créé une connexion Redis, forcer Celery à l'utiliser
                if redis_connection is not None:
                    logger.info(
                        "[Queue] 🔄 Remplacement de broker_connection pour forcer Redis"
                    )
                    # Sauvegarder l'ancienne connexion et les méthodes internes
                    old_connection = getattr(celery_app, "broker_connection", None)
                    old_connection_for_write = getattr(
                        celery_app, "_connection_for_write", None
                    )

                    # Forcer la connexion Redis
                    celery_app.broker_connection = redis_connection

                    # Forcer aussi _connection_for_write pour éviter qu'il crée une nouvelle connexion AMQP
                    def force_redis_connection(*_args: Any, **_kwargs: Any) -> Any:
                        logger.info(
                            "[Queue] 🔄 Utilisation de la connexion Redis forcée"
                        )
                        return redis_connection

                    celery_app._connection_for_write = force_redis_connection

                    logger.info(
                        "[Queue] ✅ broker_connection et _connection_for_write remplacés"
                    )

                # Utiliser send_task() qui utilisera notre connexion Redis
                task_name = "tasks.dispatch_tasks.run_dispatch_task"

                # Vérifier la connexion avant d'envoyer
                logger.info(
                    "[Queue] 🔍 Vérification connexion avant send_task: broker_transport=%s",
                    celery_app.conf.broker_transport,
                )

                result = celery_app.send_task(
                    task_name,
                    kwargs=run_kwargs,
                    queue="default",
                )

                task = result
                logger.info(
                    "[Queue] ✅ Tâche envoyée via Celery.send_task() (task_id=%s)",
                    result.id,
                )
            except Exception as send_err:
                logger.error(
                    "[Queue] ❌ Erreur avec send_task(), fallback apply_async(): %s",
                    send_err,
                )
                logger.exception("[Queue] Stack trace complète:")
                # Fallback : utiliser apply_async() normalement
                TaskCallable = cast("Any", run_dispatch_task)
                task = TaskCallable.apply_async(kwargs=run_kwargs, queue="default")
            finally:
                # Restaurer l'ancienne connexion si on l'a remplacée
                if old_connection is not None:
                    celery_app.broker_connection = old_connection
                    logger.info("[Queue] 🔄 Ancienne connexion restaurée")
                # Restaurer _connection_for_write si on l'a remplacé
                if (
                    redis_connection is not None
                    and old_connection_for_write is not None
                ):
                    celery_app._connection_for_write = old_connection_for_write
                    logger.info("[Queue] 🔄 _connection_for_write restauré")
                elif redis_connection is not None and hasattr(
                    celery_app, "_connection_for_write"
                ):
                    # Supprimer l'attribut si on l'a ajouté
                    with suppress(Exception):
                        delattr(celery_app, "_connection_for_write")
            st.last_task_id = task.id
            _CELERY_STATE[company_id] = task.state

            logger.info(
                "[Queue] Enqueued Celery task company=%s task_id=%s",
                company_id,
                task.id,
            )

            # Update state
            _PROGRESS[company_id] = 20

    except Exception as e:
        logger.exception(
            "[Queue] Failed to enqueue Celery task company=%s: %s", company_id, e
        )
        st.running = False
        st.last_start = None
        _RUNNING[company_id] = False
        _PROGRESS[company_id] = 0
        _LAST_ERROR[company_id] = str(e)
