"""Configuration Gunicorn (hooks worker timeout / abort / prometheus multiproc)."""

from __future__ import annotations

import faulthandler
import logging
import sys

logger = logging.getLogger("gunicorn.error")


def worker_int(worker):
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


def child_exit(server, worker):  # noqa: ARG001
    """Nettoie les fichiers multiproc Prometheus du worker mort."""
    try:
        from prometheus_client import multiprocess

        multiprocess.mark_process_dead(worker.pid)
    except Exception:
        logger.exception("prometheus mark_process_dead failed pid=%s", worker.pid)
