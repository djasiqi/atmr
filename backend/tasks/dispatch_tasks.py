# backend/tasks/dispatch_tasks.py
from __future__ import annotations

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import time

from celery import shared_task
from celery.exceptions import MaxRetriesExceededError
from sqlalchemy import func

from models import Company, DispatchRun
from services.unified_dispatch import engine
from services.unified_dispatch import settings as ud_settings

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
        run_kwargs = {
            "company_id": company_id,
            "for_date": for_date,
            "mode": mode,
            "regular_first": regular_first,
            "allow_emergency": allow_emergency,
        }
        
        if overrides:
            run_kwargs["overrides"] = overrides
            
        # Execute the dispatch engine
        result = engine.run(**run_kwargs)
        
        # Ensure result is a dictionary with expected structure
        if not result:
            result = {}
        if not isinstance(result, dict):
            result = {"meta": {"raw": result}}
            
        # Add fallbacks for a consistent structure
        result.setdefault("assignments", [])
        result.setdefault("bookings", [])
        result.setdefault("drivers", [])
        result.setdefault("meta", {})
        result.setdefault("dispatch_run_id", None)
        
        # Add task metadata
        result["meta"]["task_id"] = task_id
        result["meta"]["execution_time"] = time.time() - start_time
        
        # Log success
        assigned = len(result.get("assignments", []))
        unassigned = len(result.get("unassigned", []))
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
    results = {"triggered": 0, "skipped": 0, "errors": 0, "companies": []}
    
    try:
        # Get companies with dispatch enabled
        companies = Company.query.filter_by(active=True).all()
        
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
                task = run_dispatch_task.delay(
                    company_id=company_id,
                    for_date=today,
                    mode="auto"
                )
                
                results["triggered"] += 1
                results["companies"].append({
                    "company_id": company_id,
                    "task_id": task.id,
                    "for_date": today
                })
                
            except Exception as e:
                logger.exception(f"[Celery] Autorun error for company_id={company_id}: {e}")
                results["errors"] += 1
                results["companies"].append({
                    "company_id": company_id,
                    "error": str(e)
                })
    
    except Exception as e:
        logger.exception(f"[Celery] Autorun tick failed: {e}")
        results["error"] = str(e)
    
    results["duration"] = time.time() - start_time
    logger.info(
        f"[Celery] Autorun tick completed: triggered={results['triggered']} "
        f"skipped={results['skipped']} errors={results['errors']} "
        f"duration={results['duration']:.3f}s"
    )
    
    return results