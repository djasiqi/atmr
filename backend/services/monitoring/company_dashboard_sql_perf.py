"""Tracing SQL par requête HTTP pour l'audit dashboard entreprise (headers de réponse)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from flask import Flask, g, request
from sqlalchemy import event
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

DASHBOARD_PATH_PREFIXES = (
    "/api/v1/companies/me",
    "/api/v1/company_dispatch/",
    "/api/v1/company/request-offers",
    "/api/v1/companies/notifications",
    "/api/v1/messages/",
)


def _sql_perf_enabled() -> bool:
    return os.getenv("COMPANY_DASH_PERF_SQL", "").strip() in ("1", "true", "yes")


def _is_dashboard_route(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in DASHBOARD_PATH_PREFIXES)


def _reset_sql_counters() -> None:
    g._dash_sql_count = 0
    g._dash_sql_duration_ms = 0.0
    g._dash_sql_slow: list[dict[str, Any]] = []


@event.listens_for(Engine, "before_cursor_execute")
def _before_cursor_execute(
    conn, cursor, statement, parameters, context, executemany
):  # noqa: ARG001
    if not getattr(g, "_dash_sql_trace", False):
        return
    g._dash_sql_pending_start = time.perf_counter()


@event.listens_for(Engine, "after_cursor_execute")
def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany):  # noqa: ARG001
    if not getattr(g, "_dash_sql_trace", False):
        return
    started = getattr(g, "_dash_sql_pending_start", None)
    duration_ms = 0.0
    if started is not None:
        duration_ms = (time.perf_counter() - started) * 1000.0
    g._dash_sql_count = getattr(g, "_dash_sql_count", 0) + 1
    g._dash_sql_duration_ms = getattr(g, "_dash_sql_duration_ms", 0.0) + duration_ms
    slow = getattr(g, "_dash_sql_slow", None)
    if slow is not None and duration_ms >= 5.0:
        stmt = " ".join(str(statement).split())
        slow.append(
            {
                "duration_ms": round(duration_ms, 2),
                "statement": stmt[:500],
            }
        )
        slow.sort(key=lambda x: x["duration_ms"], reverse=True)
        del slow[10:]


def init_company_dashboard_sql_perf(app: Flask) -> None:
    """Active le tracing SQL + headers X-SQL-* sur routes dashboard critiques."""

    @app.before_request
    def _start_dashboard_sql_trace():  # pyright: ignore
        if not _sql_perf_enabled():
            return None
        path = request.path or ""
        if not _is_dashboard_route(path):
            return None
        g._dash_sql_trace = True
        _reset_sql_counters()
        return None

    @app.after_request
    def _attach_dashboard_sql_headers(response):  # pyright: ignore
        if not getattr(g, "_dash_sql_trace", False):
            return response
        count = int(getattr(g, "_dash_sql_count", 0))
        duration_ms = round(float(getattr(g, "_dash_sql_duration_ms", 0.0)), 2)
        response.headers["X-SQL-Query-Count"] = str(count)
        response.headers["X-SQL-Duration-Ms"] = str(duration_ms)
        slow = getattr(g, "_dash_sql_slow", []) or []
        if slow:
            top = slow[0]
            response.headers["X-SQL-Slowest-Ms"] = str(top.get("duration_ms", 0))
        g._dash_sql_trace = False
        return response

    app.logger.info(
        "[CompanyDashboardSQLPerf] Middleware actif (COMPANY_DASH_PERF_SQL=1)"
    )
