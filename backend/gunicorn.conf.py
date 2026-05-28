"""Configuration Gunicorn (hooks worker timeout / abort)."""

from __future__ import annotations

import faulthandler
import logging
import sys

logger = logging.getLogger("gunicorn.error")


def worker_int(worker):  # noqa: ARG001
    """Signal WORKER TIMEOUT — log + dump stacks pour post-mortem."""
    logger.warning("WORKER TIMEOUT pid=%s", worker.pid)
    try:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except Exception:
        logger.exception("faulthandler.dump_traceback failed")


def worker_abort(worker):
    """Worker tué après timeout — dernier log avant recycle."""
    logger.warning("WORKER ABORT pid=%s", worker.pid)
    try:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except Exception:
        logger.exception("faulthandler.dump_traceback failed on worker_abort")
