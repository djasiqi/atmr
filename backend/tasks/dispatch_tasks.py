# backend/tasks/dispatch_tasks.py
from __future__ import annotations

import logging
import json
from typing import Dict, Any, Optional, cast
from datetime import datetime, timezone
import time

from celery import shared_task
from celery.exceptions import MaxRetriesExceededError
from sqlalchemy import exc as sa_exc
from flask import current_app as app
from models import Company
from services.unified_dispatch import engine


from ext import db


logger = logging.getLogger(__name__)

# ---- Helpers typage ----
def _safe_int(v: Any) -> Optional[int]:
    """Convertit en int Python si possible, sinon None (compatible Column/InstrumentedAttribute)."""
    try:
        return int(v)  # type: ignore[arg-type]
    except Exception:
        return None



@shared_task(
    bind=True,
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
    allow_emergency: Optional[bool] = None,
    overrides: Optional[Dict[str, Any]] = None,
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
        try:
            db.session.rollback()
        except Exception:
            # au pire on ignore — remove() en finally nettoiera
            pass

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
            if isinstance(raw_result, dict):
                result: Dict[str, Any] = cast(Dict[str, Any], raw_result)
            elif raw_result is None:
                result: Dict[str, Any] = {}
            else:
                result: Dict[str, Any] = {"meta": {"raw": raw_result}}
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
            if not isinstance(_meta, dict):
                _meta = {}
            else:
                # on copie pour éviter d'éditer une éventuelle mapping-proxy/etc.
                _meta = dict(_meta)
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
            try:
                db.session.expunge_all()
            except Exception:
                pass

            return result

        except Exception as e:
            # Nettoie la session pour éviter l'état 'InFailedSqlTransaction'
            try:
                db.session.rollback()
            except Exception:
                pass

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
                try:
                    raise self.retry(exc=e)
                except MaxRetriesExceededError:
                    pass  # tombera dans le retour 'run_failed' ci-dessous

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
                    "retries_exhausted": True if transient else False,
                },
                "dispatch_run_id": None,
            }

        finally:
            # Toujours remettre la session à zéro côté worker
            try:
                db.session.remove()
            except Exception:
                pass

@shared_task(name="tasks.dispatch_tasks.autorun_tick")
def autorun_tick() -> Dict[str, Any]:
    start_time = time.time()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results: Dict[str, Any] = {"triggered": 0, "skipped": 0, "errors": 0, "companies": []}

    # Défensif
    try:
        db.session.rollback()
    except Exception:
        pass

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
                autorun_enabled = True
                if hasattr(company, 'dispatch_settings') and company.dispatch_settings:
                    try:
                        settings_data = json.loads(company.dispatch_settings)
                        autorun_enabled = settings_data.get('autorun_enabled', True)
                    except (json.JSONDecodeError, AttributeError):
                        pass

                if not autorun_enabled:
                    logger.debug(f"[Celery] Autorun skipped for company_id={company_id} (disabled)")
                    results["skipped"] += 1
                    continue

                logger.info(f"[Celery] Autorun triggering dispatch for company_id={company_id} date={today}")
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
                    "task_id": task.id,
                    "for_date": today
                })

            except Exception as e:
                # 🔁 Sur erreur pendant la boucle → rollback pour la suite
                try:
                    db.session.rollback()
                except Exception:
                    pass

                logger.exception(f"[Celery] Autorun error for company_id={company_id}: {e}")
                results["errors"] += 1
                cast(list, results["companies"]).append({
                    "company_id": company_id,
                    "error": str(e)
                })

    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
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
