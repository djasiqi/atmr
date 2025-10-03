# backend/tasks/dispatch_tasks.py
from __future__ import annotations

import logging
import json
from typing import Dict, Any, Optional, cast
from datetime import datetime, timezone
import time

from celery import shared_task
from celery.exceptions import MaxRetriesExceededError

from models import Company, DispatchRun
from services.unified_dispatch import engine

logger = logging.getLogger(__name__)


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
    Execute the dispatch optimization for a company on a specific date.
    
    Args:
        company_id: ID of the company
        for_date: Date string in YYYY-MM-DD format
        mode: Mode of operation (auto, manual, etc.)
        regular_first: Whether to prioritize regular drivers
        allow_emergency: Whether to allow emergency drivers
        overrides: Settings overrides
        
    Returns:
        Dictionary with dispatch results
    """
    start_time = time.time()
    task_id = self.request.id
    
    # Structured logging for task start
    logger.info(
        f"[Celery] Starting dispatch task company_id={company_id} for_date={for_date} "
        f"mode={mode} task_id={task_id}",
        extra={
            "task_id": task_id,
            "company_id": company_id,
            "for_date": for_date,
            "mode": mode,
            "regular_first": regular_first,
            "allow_emergency": allow_emergency,
        }
    )
    
    try:
        # Prepare parameters for engine.run
        run_kwargs: Dict[str, Any] = {
            "company_id": company_id,
            "for_date": for_date,
            "mode": mode,
            "regular_first": regular_first,
            "allow_emergency": allow_emergency,
        }
        
        if overrides:
            run_kwargs["overrides"] = overrides
            
        # Execute the dispatch engine
        raw_result: Any = engine.run(**run_kwargs)
        result: Dict[str, Any] = {}
        
        # Normalize to a dict
        if isinstance(raw_result, dict):
            result = cast(Dict[str, Any], raw_result)
        elif raw_result is None:
            result = {}
        else:
            result = {"meta": {"raw": raw_result}}
            
        # Ensure consistent structure (avoid setdefault typing issues)
        assignments = result.get("assignments")
        if not isinstance(assignments, list):
            result["assignments"] = []
        bookings = result.get("bookings")
        if not isinstance(bookings, list):
            result["bookings"] = []
        drivers = result.get("drivers")
        if not isinstance(drivers, list):
            result["drivers"] = []
        if not isinstance(result.get("meta"), dict):
            result["meta"] = {}
        if "dispatch_run_id" not in result:
            result["dispatch_run_id"] = None
        
        # Add task metadata
        meta = cast(Dict[str, Any], result["meta"])
        meta["task_id"] = task_id
        meta["execution_time"] = float(time.time() - start_time)
        
        # Log success
        assigned = len(cast(list, result.get("assignments") or []))
        unassigned = len(cast(list, result.get("unassigned") or []))
        dispatch_run_id = result.get("dispatch_run_id") or result.get("meta", {}).get("dispatch_run_id")
        
        logger.info(
            f"[Celery] Dispatch completed successfully company_id={company_id} "
            f"for_date={for_date} assigned={assigned} unassigned={unassigned} "
            f"dispatch_run_id={dispatch_run_id} duration={time.time() - start_time:.3f}s",
            extra={
                "task_id": task_id,
                "company_id": company_id,
                "for_date": for_date,
                "assigned": assigned,
                "unassigned": unassigned,
                "dispatch_run_id": dispatch_run_id,
                "duration": time.time() - start_time,
            }
        )
        
        # Emit notification (moved from queue.py)
        try:
            from services.notification_service import notify_dispatch_run_completed
            if dispatch_run_id:
                # Get the day string from the dispatch run
                day_str = None
                try:
                    dispatch_run = DispatchRun.query.get(dispatch_run_id)
                    if dispatch_run and dispatch_run.day:
                        day_str = dispatch_run.day.isoformat()
                except Exception as e:
                    logger.warning(f"[Celery] Failed to get day_str from DispatchRun: {e}")
                
                # Send notification
                notify_dispatch_run_completed(company_id, dispatch_run_id, assigned, day_str)
        except Exception as e:
            logger.warning(f"[Celery] Failed to emit dispatch_run_completed: {e}")
        
        # Return serializable result
        return result
        
    except Exception as e:
        # Log error with structured data
        logger.exception(
            f"[Celery] Dispatch task failed company_id={company_id} for_date={for_date} "
            f"error={str(e)} retry={self.request.retries}/{self.max_retries}",
            extra={
                "task_id": task_id,
                "company_id": company_id,
                "for_date": for_date,
                "error": str(e),
                "retry_count": self.request.retries,
                "max_retries": self.max_retries,
            }
        )
        
        # Retry with exponential backoff
        try:
            raise self.retry(exc=e)
        except MaxRetriesExceededError:
            # Return error result after all retries are exhausted
            return {
                "assignments": [],
                "bookings": [],
                "drivers": [],
                "meta": {
                    "error": str(e),
                    "task_id": task_id,
                    "company_id": company_id,
                    "for_date": for_date,
                    "retries_exhausted": True,
                },
            }


@shared_task(name="tasks.dispatch_tasks.autorun_tick")
def autorun_tick() -> Dict[str, Any]:
    """
    Periodic task that triggers dispatch for companies with autorun enabled.
    
    Returns:
        Dictionary with results summary
    """
    start_time = time.time()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results: Dict[str, Any] = {"triggered": 0, "skipped": 0, "errors": 0, "companies": []}
    
    try:
        # Get companies with dispatch enabled (Company.active n'existe pas)
        companies = Company.query.filter_by(dispatch_enabled=True).all()
        
        for company in companies:
            company_id = company.id
            try:
                # Check if autorun is enabled for this company
                # First check company.dispatch_settings if available
                autorun_enabled = True  # Default to True
                
                # Try to get from dispatch_settings if available
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
                
                # Trigger dispatch task for today
                logger.info(f"[Celery] Autorun triggering dispatch for company_id={company_id} date={today}")
                task = cast(Any, run_dispatch_task).delay(
                    company_id=int(company_id),
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
                logger.exception(f"[Celery] Autorun error for company_id={company_id}: {e}")
                results["errors"] += 1
                cast(list, results["companies"]).append({
                    "company_id": company_id,
                    "error": str(e)
                })
    
    except Exception as e:
        logger.exception(f"[Celery] Autorun tick failed: {e}")
        results["error"] = str(e)
    
    results["duration"] = float(time.time() - start_time)
    logger.info(
        f"[Celery] Autorun tick completed: triggered={results['triggered']} "
        f"skipped={results['skipped']} errors={results['errors']} "
        f"duration={results['duration']:.3f}s"
    )
    
    return results