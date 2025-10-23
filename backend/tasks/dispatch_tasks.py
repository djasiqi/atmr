# backend/tasks/dispatch_tasks.py
from __future__ import annotations

import logging
import time
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, Dict, cast

from celery import shared_task
from celery.exceptions import MaxRetriesExceededError
from flask import current_app as app
from sqlalchemy import exc as sa_exc

from ext import db
from models import Company
from services.unified_dispatch import engine

logger = logging.getLogger(__name__)

# ---- Helpers typage ----
def _safe_int(v: Any) -> int | None:
    """Convertit en int Python si possible, sinon None (compatible Column/InstrumentedAttribute)."""
    try:
        return int(v)  # type: ignore[arg-type]
    except Exception:
        return None



@shared_task(
    bind=True,
    acks_late=True,  # ✅ Ne pas ack avant traitement complet
    task_time_limit=300,  # 5 minutes max
    task_soft_time_limit=270,
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    name="tasks.dispatch_tasks.run_dispatch_task"
)
def run_dispatch_task(
    self,
    company_id: int,
    for_date: str,
    mode: str = "auto",
    regular_first: bool = True,
    allow_emergency: bool | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Exécuté par un worker Celery.
    - Nettoie/normalise la session DB avant/après.
    - Normalise le payload (mode/overrides).
    - Ne laisse jamais la session en état 'aborted'.
    - Retourne toujours un dict cohérent.
    """
    start_time = time.time()
    task_id = getattr(self.request, "id", None)

    # --- Normalisation d'entrée ---
    mode = (mode or "auto").strip().lower()
    if mode not in {"auto", "solver_only", "heuristic_only"}:
        mode = "auto"

    ov = dict(overrides or {})
    if "mode" not in ov:
        ov["mode"] = mode  # garde une source unique pour le moteur

    # On fait tout sous app_context pour une session/teardown propre
    with app.app_context():
        # Assainit d'abord la session si elle a été "polluée" par un appel précédent
        with suppress(Exception):
            db.session.rollback()

        logger.info(
            "[Celery] Starting dispatch task company_id=%s for_date=%s mode=%s task_id=%s",
            company_id, for_date, mode, task_id,
            extra={
                "task_id": task_id,
                "company_id": company_id,
                "for_date": for_date,
                "mode": mode,
                "regular_first": regular_first,
                "allow_emergency": allow_emergency,
            },
        )

        try:
            # -------- Appel moteur --------
            run_kwargs: Dict[str, Any] = {
                "company_id": company_id,
                "for_date": for_date,
                "mode": mode,
                "regular_first": regular_first,
                "allow_emergency": allow_emergency,
                "overrides": ov,
            }
            raw_result: Any = engine.run(**run_kwargs)

            # -------- Normalisation résultat --------
            result: Dict[str, Any]
            if isinstance(raw_result, dict):
                result = cast(Dict[str, Any], raw_result)
            elif raw_result is None:
                result = {}
            else:
                result = {"meta": {"raw": raw_result}}
            # assignments
            _assignments = result.get("assignments")
            if not isinstance(_assignments, list):
                result["assignments"] = []
            # unassigned
            _unassigned = result.get("unassigned")
            if not isinstance(_unassigned, list):
                result["unassigned"] = []
            # bookings
            _bookings = result.get("bookings")
            if not isinstance(_bookings, list):
                result["bookings"] = []
            # drivers
            _drivers = result.get("drivers")
            if not isinstance(_drivers, list):
                result["drivers"] = []
            # meta
            _meta = result.get("meta")
            _meta = {} if not isinstance(_meta, dict) else dict(_meta)
            result["meta"] = _meta
            # dispatch_run_id (laisser None si absent)
            if "dispatch_run_id" not in result:
                result["dispatch_run_id"] = None

            # remonte dispatch_run_id depuis meta si présent
            if not result.get("dispatch_run_id") and "dispatch_run_id" in _meta:
                result["dispatch_run_id"] = _meta["dispatch_run_id"]

            meta = cast(Dict[str, Any], result["meta"])
            meta["task_id"] = task_id
            meta["execution_time"] = float(time.time() - start_time)
            if result.get("dispatch_run_id"):
                meta["dispatch_run_id"] = result["dispatch_run_id"]

            assigned = len(cast(list, result.get("assignments") or []))
            unassigned = len(cast(list, result.get("unassigned") or []))
            dispatch_run_id = result.get("dispatch_run_id")

            logger.info(
                "[Celery] Dispatch completed successfully company_id=%s for_date=%s "
                "assigned=%s unassigned=%s dispatch_run_id=%s duration=%.3fs",
                company_id, for_date, assigned, unassigned, dispatch_run_id, time.time() - start_time,
                extra={
                    "task_id": task_id,
                    "company_id": company_id,
                    "for_date": for_date,
                    "assigned": assigned,
                    "unassigned": unassigned,
                    "dispatch_run_id": dispatch_run_id,
                    "duration": time.time() - start_time,
                },
            )

            # Pas de commit ici : on suppose que les écritures DB (si besoin)
            # sont gérées/commit par les couches appelées. On garde la session propre.
            with suppress(Exception):
                db.session.expunge_all()

            return result

        except Exception as e:
            # Nettoie la session pour éviter l'état 'InFailedSqlTransaction'
            with suppress(Exception):
                db.session.rollback()

            import traceback
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            # Si c’est une erreur SQLA, expose aussi .orig / .statement
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
            logger.error(
                "[Celery] Dispatch FAILED company_id=%s for_date=%s type=%s msg=%s extra=%s\n%s",
                company_id, for_date, type(e).__name__, str(e), extra_sql, tb,
                extra={
                    "task_id": task_id,
                    "company_id": company_id,
                    "for_date": for_date,
                    "error": str(e),
                    "retry_count": getattr(self.request, "retries", 0),
                    "max_retries": getattr(self, "max_retries", 0),
                },
            )

            # Décide si on retente (réseau/transient) ou si on renvoie un résultat 'run_failed'
            transient = isinstance(e, (sa_exc.OperationalError, sa_exc.DBAPIError)) and getattr(e, "connection_invalidated", False)

            if transient:
                with suppress(MaxRetriesExceededError):
                    raise self.retry(exc=e) from e

            # Retour déterministe (UI friendly)
            return {
                "assignments": [],
                "unassigned": [],
                "bookings": [],
                "drivers": [],
                "meta": {
                    "reason": "run_failed",
                    "error": str(e),
                    "task_id": task_id,
                    "company_id": company_id,
                    "for_date": for_date,
                    "retries_exhausted": bool(transient),
                },
                "dispatch_run_id": None,
            }

        finally:
            # Toujours remettre la session à zéro côté worker
            with suppress(Exception):
                db.session.remove()

@shared_task(
    name="tasks.dispatch_tasks.autorun_tick",
    acks_late=True,
    task_time_limit=600  # 10 minutes (peut lancer plusieurs dispatch)
)
def autorun_tick() -> Dict[str, Any]:
    start_time = time.time()
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    results: Dict[str, Any] = {"triggered": 0, "skipped": 0, "errors": 0, "companies": []}

    # Défensif
    with suppress(Exception):
        db.session.rollback()

    try:
        companies = Company.query.filter_by(dispatch_enabled=True).all()

        for company in companies:
            # Pylance-safe + robustesse runtime
            company_id = _safe_int(getattr(company, "id", None))
            if company_id is None:
                logger.warning("[Celery] Autorun: company sans id utilisable, skip.")
                results["skipped"] += 1
                continue

            try:
                # 🆕 Utiliser le gestionnaire autonome pour décider si le dispatch doit tourner
                from services.unified_dispatch.autonomous_manager import get_manager_for_company
                manager = get_manager_for_company(company_id)

                # Vérifier si l'autorun doit s'exécuter selon le mode
                if not manager.should_run_autorun():
                    logger.debug(
                        "[Celery] Autorun skipped for company_id=%s (mode: %s, autorun disabled)",
                        company_id, manager.mode.value
                    )
                    results["skipped"] += 1
                    continue

                logger.info(
                    "[Celery] Autorun triggering dispatch for company_id=%s mode=%s date=%s",
                    company_id, manager.mode.value, today
                )
                task = cast(Any, run_dispatch_task).delay(
                    company_id=company_id,
                    for_date=str(today),
                    mode="auto",
                    regular_first=True,
                    allow_emergency=None,
                )

                results["triggered"] += 1
                cast(list, results["companies"]).append({
                    "company_id": company_id,
                    "dispatch_mode": manager.mode.value,
                    "task_id": task.id,
                    "for_date": today
                })

            except Exception as e:
                # 🔁 Sur erreur pendant la boucle → rollback pour la suite
                with suppress(Exception):
                    db.session.rollback()

                logger.exception(f"[Celery] Autorun error for company_id={company_id}: {e}")
                results["errors"] += 1
                cast(list, results["companies"]).append({
                    "company_id": company_id,
                    "error": str(e)
                })

    except Exception as e:
        with suppress(Exception):
            db.session.rollback()
        logger.exception(f"[Celery] Autorun tick failed: {e}")
        results["error"] = str(e)
    finally:
        db.session.remove()

    results["duration"] = float(time.time() - start_time)
    logger.info(
        f"[Celery] Autorun tick completed: triggered={results['triggered']} "
        f"skipped={results['skipped']} errors={results['errors']} "
        f"duration={results['duration']:.3f}s"
    )
    return results


@shared_task(
    name="tasks.dispatch_tasks.realtime_monitoring_tick",
    acks_late=True,
    task_time_limit=300  # 5 minutes max
)
def realtime_monitoring_tick() -> Dict[str, Any]:
    """
    Tâche Celery périodique pour le monitoring temps réel du dispatch.
    Remplace le thread RealtimeOptimizer pour survivre aux redémarrages serveur.

    Cette tâche :
    - Vérifie les opportunités d'optimisation pour chaque entreprise
    - Applique automatiquement les suggestions selon le mode (fully_auto)
    - S'exécute toutes les 2 minutes via Celery Beat

    Returns:
        Statistiques du monitoring (entreprises vérifiées, opportunités, actions)
    """
    start_time = time.time()
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    results: Dict[str, Any] = {
        "companies_checked": 0,
        "total_opportunities": 0,
        "auto_applied": 0,
        "manual_required": 0,
        "errors": 0,
        "companies": []
    }

    # Défensif : nettoyer la session
    with suppress(Exception):
        db.session.rollback()

    try:
        # Récupérer toutes les entreprises avec dispatch activé
        companies = Company.query.filter(
            Company.dispatch_enabled == True  # noqa: E712
        ).all()

        for company in companies:
            company_id = _safe_int(getattr(company, "id", None))
            if company_id is None:
                continue

            try:
                # Créer le gestionnaire autonome
                from services.unified_dispatch.autonomous_manager import get_manager_for_company
                manager = get_manager_for_company(company_id)

                # Vérifier si le monitoring doit tourner pour cette entreprise
                if not manager.should_run_realtime_optimizer():
                    logger.debug(
                        "[RealtimeMonitoring] Skipped for company %s (mode: %s, optimizer disabled)",
                        company_id, manager.mode.value
                    )
                    continue

                results["companies_checked"] += 1

                # Vérifier les opportunités d'optimisation
                from services.unified_dispatch.realtime_optimizer import check_opportunities_manual
                opportunities = check_opportunities_manual(
                    company_id=company_id,
                    for_date=today,
                    app=None  # Le contexte Flask est fourni par ContextTask
                )

                results["total_opportunities"] += len(opportunities)

                # Traiter les opportunités (appliquer si mode fully_auto)
                if opportunities:
                    stats = manager.process_opportunities(opportunities, dry_run=False)
                    results["auto_applied"] += stats["auto_applied"]
                    results["manual_required"] += stats["manual_required"]

                    logger.info(
                        "[RealtimeMonitoring] Company %s: %d opportunities, %d auto-applied, %d manual",
                        company_id, len(opportunities), stats["auto_applied"], stats["manual_required"]
                    )

                # Ajouter aux résultats
                results["companies"].append({
                    "company_id": company_id,
                    "mode": manager.mode.value,
                    "opportunities": len(opportunities),
                    "auto_applied": stats.get("auto_applied", 0) if opportunities else 0,
                    "manual_required": stats.get("manual_required", 0) if opportunities else 0
                })

            except Exception as e:
                results["errors"] += 1
                logger.exception(
                    "[RealtimeMonitoring] Error for company %s: %s",
                    company_id, e
                )

                results["companies"].append({
                    "company_id": company_id,
                    "error": str(e)
                })

                # Rollback pour ne pas polluer la suite
                with suppress(Exception):
                    db.session.rollback()

    except Exception as e:
        logger.exception("[RealtimeMonitoring] Tick failed: %s", e)
        results["error"] = str(e)
        with suppress(Exception):
            db.session.rollback()

    finally:
        db.session.remove()

    results["duration"] = float(time.time() - start_time)

    logger.info(
        "[RealtimeMonitoring] Tick completed: companies=%d opportunities=%d auto_applied=%d "
        "manual=%d errors=%d duration=%.3fs",
        results["companies_checked"],
        results["total_opportunities"],
        results["auto_applied"],
        results["manual_required"],
        results["errors"],
        results["duration"]
    )

    return results
