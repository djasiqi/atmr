"""Agrégation lecture seule pour GET /api/v1/platform/status (Admin Ops / Platform)."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

HTTP_OK = 200
READY_PATH = "/api/v1/ready"
WS_PATH = "/api/v1/health/websocket"

# Contrat public : criticité par identifiant de check (Phase 1A)
CHECK_CRITICALITY: dict[str, str] = {
    "ready": "critical",
    "database": "critical",
    "redis": "critical",
    "websocket": "high",
}

CRITICAL_CHECK_KEYS = frozenset({"ready", "database", "redis"})
CRITICAL_UNKNOWN_ROLLUP_THRESHOLD = 2

# Ordre d’affichage UI (criticité puis métier) — aligné plan §4.4
CHECK_DISPLAY_ORDER: tuple[str, ...] = ("ready", "database", "redis", "websocket")


def _iso_z(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def _norm_check_value(raw: Any) -> str:
    """Mappe une valeur de check (ex. checks.database) vers ok|degraded|down|unknown."""
    if raw is None:
        return "unknown"
    s = str(raw).strip().lower()
    if s == "ok":
        return "ok"
    if "not_configured" in s:
        return "degraded"
    if "warning" in s:
        return "degraded"
    if s.startswith("error") or "error" in s:
        return "down"
    return "unknown"


def _norm_ws_status(raw: Any) -> str:
    if raw is None:
        return "unknown"
    s = str(raw).strip().lower()
    if s == "ok":
        return "ok"
    if s == "degraded":
        return "degraded"
    if s == "error":
        return "down"
    return "unknown"


def _rollup_env_status(checks: dict[str, str], *, is_prod: bool) -> str:
    """Agrège les statuts textuels des checks en statut d’environnement (contrat public)."""
    vals = list(checks.values())
    if "down" in vals:
        return "down"

    if is_prod:
        critical_unknowns = [
            k for k in CRITICAL_CHECK_KEYS if checks.get(k) == "unknown"
        ]
        if len(critical_unknowns) >= CRITICAL_UNKNOWN_ROLLUP_THRESHOLD:
            return "unknown"
        if len(critical_unknowns) == 1:
            return "degraded"
        if "degraded" in vals:
            return "degraded"
        if checks.get("websocket") == "unknown":
            return "degraded"
        if "unknown" in vals:
            return "unknown"
        return "ok"

    if "degraded" in vals or "unknown" in vals:
        return "degraded"
    return "ok"


def _fetch_json(
    base_url: str, path: str, timeout: float
) -> tuple[dict[str, Any] | None, int | None, float, str | None]:
    """Retourne (json_body, http_status, latency_ms, erreur)."""
    url = base_url.rstrip("/") + path
    t0 = datetime.now(UTC).timestamp()
    try:
        resp = requests.get(url, timeout=timeout)
        latency_ms = (datetime.now(UTC).timestamp() - t0) * 1000.0
        try:
            body = resp.json() if resp.content else {}
        except ValueError:
            body = None
        return body, resp.status_code, latency_ms, None
    except requests.RequestException as e:
        latency_ms = (datetime.now(UTC).timestamp() - t0) * 1000.0
        return None, None, latency_ms, str(e)


def _check_obj(
    name: str,
    status: str,
    *,
    criticality: str,
    latency_ms: float | None,
    checked_at: str | None,
    reason: str | None = None,
    detail: str | None = None,
    operator_hint: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "criticality": criticality,
        "latency_ms": round(latency_ms) if latency_ms is not None else None,
        "checked_at": checked_at,
        "reason": reason,
        "detail": detail,
        "operator_hint": operator_hint,
        "last_success_at": None,
        "last_failure_at": None,
        "state_changed_at": None,
    }


def _empty_checks_unknown() -> dict[str, dict[str, Any]]:
    return {
        "ready": _check_obj(
            "ready",
            "unknown",
            criticality=CHECK_CRITICALITY["ready"],
            latency_ms=None,
            checked_at=None,
            reason="not_monitored",
        ),
        "database": _check_obj(
            "database",
            "unknown",
            criticality=CHECK_CRITICALITY["database"],
            latency_ms=None,
            checked_at=None,
            reason="not_monitored",
        ),
        "redis": _check_obj(
            "redis",
            "unknown",
            criticality=CHECK_CRITICALITY["redis"],
            latency_ms=None,
            checked_at=None,
            reason="not_monitored",
        ),
        "websocket": _check_obj(
            "websocket",
            "unknown",
            criticality=CHECK_CRITICALITY["websocket"],
            latency_ms=None,
            checked_at=None,
            reason="not_monitored",
        ),
    }


def _build_env_block(
    label: str,
    display_name: str,
    base_url: str | None,
    monitored: bool,
    timeout: float,
    *,
    is_prod: bool,
) -> dict[str, Any]:
    """Construit environments.{prod|demo}."""
    if not monitored or not base_url:
        return {
            "name": display_name,
            "monitored": False,
            "status": "unknown",
            "latency_ms": None,
            "checks": _empty_checks_unknown(),
            "errors": [
                {
                    "type": "not_monitored",
                    "message": (
                        f"Collecte {label} désactivée ou URL non configurée "
                        f"(PLATFORM_API_URL_{label.upper()})."
                    ),
                }
            ],
        }

    errors: list[dict[str, str]] = []
    flat: dict[str, str] = {
        "ready": "unknown",
        "database": "unknown",
        "redis": "unknown",
        "websocket": "unknown",
    }
    latencies: list[float] = []
    ready_checked_at: str | None = None
    ws_checked_at: str | None = None
    ready_lat: float | None = None
    ws_lat: float | None = None

    def job_ready() -> tuple[str, Any]:
        body, code, lat, err = _fetch_json(base_url, READY_PATH, timeout)
        return "ready", (body, code, lat, err)

    def job_ws() -> tuple[str, Any]:
        body, code, lat, err = _fetch_json(base_url, WS_PATH, timeout)
        return "ws", (body, code, lat, err)

    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(job_ready), ex.submit(job_ws)]
        for fut in as_completed(futures):
            key, payload = fut.result()
            results[key] = payload

    ready_pack = results.get("ready")
    ws_pack = results.get("ws")

    derived_detail = "Derived from /api/v1/ready response"

    # Ready
    if ready_pack:
        body, code, lat, err = ready_pack
        t_done = datetime.now(UTC).timestamp()
        ready_checked_at = _iso_z(t_done)
        if lat is not None:
            latencies.append(lat)
            ready_lat = lat
        if err:
            errors.append({"type": "ready_fetch_error", "message": err})
            flat["ready"] = "unknown"
            flat["database"] = "unknown"
            flat["redis"] = "unknown"
        elif code is None:
            flat["ready"] = "unknown"
            flat["database"] = "unknown"
            flat["redis"] = "unknown"
        elif code != HTTP_OK:
            errors.append(
                {
                    "type": "ready_http_error",
                    "message": f"HTTP {code}",
                }
            )
            flat["ready"] = "down"
            if isinstance(body, dict):
                chk = body.get("checks") or {}
                flat["database"] = _norm_check_value(chk.get("database"))
                flat["redis"] = _norm_check_value(chk.get("redis"))
            else:
                flat["database"] = "down"
                flat["redis"] = "down"
        else:
            flat["ready"] = "ok"
            if isinstance(body, dict):
                chk = body.get("checks") or {}
                st = str(body.get("status", "")).lower()
                if st == "not_ready":
                    flat["ready"] = "down"
                flat["database"] = _norm_check_value(chk.get("database"))
                flat["redis"] = _norm_check_value(chk.get("redis"))
            else:
                flat["ready"] = "unknown"
                flat["database"] = "unknown"
                flat["redis"] = "unknown"

    # WebSocket
    if ws_pack:
        body, code, lat, err = ws_pack
        t_done = datetime.now(UTC).timestamp()
        ws_checked_at = _iso_z(t_done)
        if lat is not None:
            latencies.append(lat)
            ws_lat = lat
        if err:
            errors.append({"type": "websocket_fetch_error", "message": err})
            flat["websocket"] = "unknown"
        elif code is None:
            flat["websocket"] = "unknown"
        elif code != HTTP_OK:
            errors.append({"type": "websocket_http_error", "message": f"HTTP {code}"})
            flat["websocket"] = _norm_ws_status(
                isinstance(body, dict) and body.get("status")
            )
        elif isinstance(body, dict):
            flat["websocket"] = _norm_ws_status(body.get("status"))
        else:
            flat["websocket"] = "unknown"

    env_status = _rollup_env_status(flat, is_prod=is_prod)

    max_lat = max(latencies) if latencies else None

    checks_out: dict[str, dict[str, Any]] = {}
    for key in CHECK_DISPLAY_ORDER:
        st = flat[key]
        crit = CHECK_CRITICALITY[key]
        if key == "ready":
            lat, cat, detail = ready_lat, ready_checked_at, None
        elif key in ("database", "redis"):
            lat, cat = ready_lat, ready_checked_at
            detail = (
                derived_detail if ready_lat is not None and st != "unknown" else None
            )
        else:
            lat, cat, detail = ws_lat, ws_checked_at, None

        checks_out[key] = _check_obj(
            key,
            st,
            criticality=crit,
            latency_ms=lat,
            checked_at=cat,
            reason=None,
            detail=detail,
            operator_hint=None,
        )

    return {
        "name": display_name,
        "monitored": True,
        "status": env_status,
        "latency_ms": round(max_lat) if max_lat is not None else None,
        "checks": checks_out,
        "errors": errors,
    }


def _platform_setting(config: Any, key: str) -> str | None:
    """Lit une variable Admin Ops / Platform.

    Priorité à ``os.environ`` puis ``app.config`` : les attributs de classe
    ``Config`` sont figés à l'import ; ``os.getenv`` au moment de la requête
    reflète toujours l'environnement du conteneur (robuste avec Gunicorn ``--preload``).
    """
    raw = (os.getenv(key) or "").strip()
    if raw:
        return raw
    cfg_val = getattr(config, key, None)
    if isinstance(cfg_val, str) and cfg_val.strip():
        return cfg_val.strip()
    return None


def _platform_timeout_seconds(config: Any) -> float:
    raw = (os.getenv("PLATFORM_STATUS_TIMEOUT_SECONDS") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return float(getattr(config, "PLATFORM_STATUS_TIMEOUT_SECONDS", 2.5) or 2.5)


def compute_overall_status(prod: dict[str, Any], demo: dict[str, Any]) -> str:
    """Règle documentée : prod prioritaire ; demo non suivie n'empêche pas ok si prod OK.

    Si prod est non monitorée ou son statut est `unknown` (indécision sur la prod),
    `overall_status` / `global_status` est `unknown`.
    """
    if not prod.get("monitored"):
        return "unknown"
    ps = prod.get("status")
    if ps == "unknown":
        return "unknown"
    if ps == "down":
        return "down"
    if not demo.get("monitored"):
        return "ok" if ps == "ok" else "degraded"
    ds = demo.get("status")
    if ps == "ok":
        return "ok" if ds == "ok" else "degraded"
    return "degraded"


def _build_metadata_block(config: Any) -> dict[str, Any]:
    """Métadonnées légères (Phase 1B) — pas de Redis/Celery ; env optionnelle."""
    checked_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    commit = (_platform_setting(config, "PLATFORM_METADATA_GIT_COMMIT") or "").strip()
    if not commit:
        commit = (os.getenv("GIT_COMMIT_SHA") or os.getenv("GIT_COMMIT") or "").strip()
    version = (_platform_setting(config, "PLATFORM_METADATA_APP_VERSION") or "").strip()
    if not version:
        version = (os.getenv("APP_VERSION") or "").strip()
    if not commit and not version:
        return {
            "status": "not_configured",
            "reason": "missing_platform_metadata",
            "checked_at": checked_at,
            "data": None,
        }
    data: dict[str, Any] = {}
    if commit:
        data["git_commit"] = commit[:64]
    if version:
        data["app_version"] = version[:128]
    return {
        "status": "ok",
        "reason": None,
        "checked_at": checked_at,
        "data": data,
    }


def _build_deep_links(links: dict[str, str | None]) -> dict[str, Any]:
    """Liens structurés (Phase 1B) — mêmes URLs que ``links``, regroupées par domaine."""
    return {
        "observability": {
            "grafana": links.get("grafana"),
            "prometheus": links.get("prometheus"),
            "alertmanager": links.get("alertmanager"),
        }
    }


def _count_summary(environments: dict[str, Any]) -> dict[str, int]:
    """Compte les checks par statut sur les environnements monitorés."""
    totals = {
        "total_checks": 0,
        "ok_checks": 0,
        "degraded_checks": 0,
        "down_checks": 0,
        "unknown_checks": 0,
    }
    for env in environments.values():
        if not isinstance(env, dict) or not env.get("monitored"):
            continue
        checks = env.get("checks") or {}
        for c in checks.values():
            if not isinstance(c, dict):
                continue
            st = str(c.get("status") or "").lower()
            totals["total_checks"] += 1
            if st == "ok":
                totals["ok_checks"] += 1
            elif st == "degraded":
                totals["degraded_checks"] += 1
            elif st == "down":
                totals["down_checks"] += 1
            elif st == "unknown":
                totals["unknown_checks"] += 1
    return totals


def build_platform_status_payload(config: Any) -> dict[str, Any]:
    """Construit le corps JSON pour GET /api/v1/platform/status."""
    timeout = _platform_timeout_seconds(config)
    prod_url = _platform_setting(config, "PLATFORM_API_URL_PROD")
    demo_url = _platform_setting(config, "PLATFORM_API_URL_DEMO")
    prod_mon = bool(prod_url)
    demo_mon = bool(demo_url)

    prod_block = _build_env_block(
        "prod", "ATMR Production", prod_url, prod_mon, timeout, is_prod=True
    )
    demo_block = _build_env_block(
        "demo", "ATMR Demo", demo_url, demo_mon, timeout, is_prod=False
    )

    overall = compute_overall_status(prod_block, demo_block)

    links = {
        "grafana": _platform_setting(config, "PLATFORM_LINK_GRAFANA"),
        "prometheus": _platform_setting(config, "PLATFORM_LINK_PROMETHEUS"),
        "alertmanager": _platform_setting(config, "PLATFORM_LINK_ALERTMANAGER"),
    }

    environments = {"prod": prod_block, "demo": demo_block}
    summary = _count_summary(environments)

    metadata_block = _build_metadata_block(config)

    return {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "global_status": overall,
        "overall_status": overall,
        "summary": summary,
        "environments": environments,
        "links": links,
        "deep_links": _build_deep_links(links),
        "metadata": metadata_block,
    }
