from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def handle_dispatch_run_completed(event: dict[str, Any]) -> None:
    from services.notification_service import notify_dispatch_run_completed

    company_id = int(event.get("company_id") or 0)
    dispatch_run_id = int(event.get("dispatch_run_id") or 0)
    assignments_count = int(event.get("assignments_count") or 0)
    date_str = event.get("date_str")
    notify_dispatch_run_completed(
        company_id,
        dispatch_run_id,
        assignments_count,
        str(date_str) if date_str else None,
    )


def handle_dispatch_requested(event: dict[str, Any]) -> None:
    company_id = int(event.get("company_id") or 0)
    action = str(event.get("action") or "update")
    reason = event.get("reason")

    try:
        from services.unified_dispatch import queue as _queue
    except ImportError as e:
        # Erreur d'import attendue : module non disponible
        logger.warning("[EventBus] queue import failed (ImportError): %s", e)
        return
    except Exception:
        # Erreur inattendue lors de l'import
        logger.exception("[EventBus] queue import failed")
        return

    trigger1: Any = getattr(_queue, "trigger_on_booking_change", None)
    if callable(trigger1):
        trigger1(company_id, action=action)
        return

    trigger2: Any = getattr(_queue, "trigger", None)
    if callable(trigger2):
        trigger2(company_id, reason=str(reason or f"event_{action}"), mode="auto")
        return

    logger.warning(
        "[EventBus] Aucun trigger compatible trouvé dans services.unified_dispatch.queue"
    )
