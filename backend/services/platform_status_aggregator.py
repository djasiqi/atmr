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


def _norm_check_value(raw: Any) -> str:
    """Mappe une valeur de check (ex. checks.database) vers ok|degraded|unavailable|unknown."""
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
        return "unavailable"
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
        return "unavailable"
    return "unknown"


def _rollup_env_status(checks: dict[str, str], *, is_prod: bool) -> str:
    """Agrège les checks en statut d'environnement.

    Pour **prod** : toute part `unknown` (timeout, réponse invalide, etc.) rend
    l'environnement `unknown` — on ne conclut pas « dégradé » sans visibilité.

    Pour **demo** : `unknown` sur un check se comporte comme avant (avec
    `degraded`) pour ne pas sur-interpréter la démo.
    """
    vals = list(checks.values())
    if "unavailable" in vals:
        return "unavailable"
    if is_prod:
        if "unknown" in vals:
            return "unknown"
        if "degraded" in vals:
            return "degraded"
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


def _build_env_block(
    label: str,
    base_url: str | None,
    monitored: bool,
    timeout: float,
    *,
    is_prod: bool,
) -> dict[str, Any]:
    """Construit environments.{prod|demo}."""
    if not monitored or not base_url:
        return {
            "monitored": False,
            "status": "unknown",
            "latency_ms": None,
            "checks": {
                "ready": {"status": "unknown"},
                "database": {"status": "unknown"},
                "redis": {"status": "unknown"},
                "websocket": {"status": "unknown"},
            },
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
    checks: dict[str, dict[str, str]] = {
        "ready": {"status": "unknown"},
        "database": {"status": "unknown"},
        "redis": {"status": "unknown"},
        "websocket": {"status": "unknown"},
    }
    latencies: list[float] = []

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

    # Ready
    if ready_pack:
        body, code, lat, err = ready_pack
        if lat is not None:
            latencies.append(lat)
        if err:
            errors.append({"type": "ready_fetch_error", "message": err})
            checks["ready"]["status"] = "unknown"
        elif code is None:
            checks["ready"]["status"] = "unknown"
        elif code != HTTP_OK:
            errors.append(
                {
                    "type": "ready_http_error",
                    "message": f"HTTP {code}",
                }
            )
            checks["ready"]["status"] = "unavailable"
            if isinstance(body, dict):
                chk = body.get("checks") or {}
                checks["database"]["status"] = _norm_check_value(chk.get("database"))
                checks["redis"]["status"] = _norm_check_value(chk.get("redis"))
            else:
                checks["database"]["status"] = "unavailable"
                checks["redis"]["status"] = "unavailable"
        else:
            checks["ready"]["status"] = "ok"
            if isinstance(body, dict):
                chk = body.get("checks") or {}
                st = str(body.get("status", "")).lower()
                if st == "not_ready":
                    checks["ready"]["status"] = "unavailable"
                checks["database"]["status"] = _norm_check_value(chk.get("database"))
                checks["redis"]["status"] = _norm_check_value(chk.get("redis"))
            else:
                checks["database"]["status"] = "unknown"
                checks["redis"]["status"] = "unknown"

    # WebSocket
    if ws_pack:
        body, code, lat, err = ws_pack
        if lat is not None:
            latencies.append(lat)
        if err:
            errors.append({"type": "websocket_fetch_error", "message": err})
            checks["websocket"]["status"] = "unknown"
        elif code is None:
            checks["websocket"]["status"] = "unknown"
        elif code != HTTP_OK:
            errors.append(
                {"type": "websocket_http_error", "message": f"HTTP {code}"}
            )
            checks["websocket"]["status"] = _norm_ws_status(
                isinstance(body, dict) and body.get("status")
            )
        elif isinstance(body, dict):
            checks["websocket"]["status"] = _norm_ws_status(body.get("status"))
        else:
            checks["websocket"]["status"] = "unknown"

    flat = {k: v["status"] for k, v in checks.items()}
    env_status = _rollup_env_status(flat, is_prod=is_prod)

    max_lat = max(latencies) if latencies else None
    return {
        "monitored": True,
        "status": env_status,
        "latency_ms": round(max_lat) if max_lat is not None else None,
        "checks": checks,
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
    if cfg_val is not None and str(cfg_val).strip():
        return str(cfg_val).strip()
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
    `overall_status` est `unknown`.
    """
    if not prod.get("monitored"):
        return "unknown"
    ps = prod.get("status")
    if ps == "unknown":
        return "unknown"
    if ps == "unavailable":
        return "unavailable"
    if not demo.get("monitored"):
        return "ok" if ps == "ok" else "degraded"
    ds = demo.get("status")
    if ps == "ok":
        return "ok" if ds == "ok" else "degraded"
    return "degraded"


def build_platform_status_payload(config: Any) -> dict[str, Any]:
    """Construit le corps JSON pour GET /api/v1/platform/status."""
    timeout = _platform_timeout_seconds(config)
    prod_url = _platform_setting(config, "PLATFORM_API_URL_PROD")
    demo_url = _platform_setting(config, "PLATFORM_API_URL_DEMO")
    prod_mon = bool(prod_url)
    demo_mon = bool(demo_url)

    # Séquentiel prod puis demo pour éviter imbrication de ThreadPoolExecutor.
    prod_block = _build_env_block("prod", prod_url, prod_mon, timeout, is_prod=True)
    demo_block = _build_env_block("demo", demo_url, demo_mon, timeout, is_prod=False)

    overall = compute_overall_status(prod_block, demo_block)

    links = {
        "grafana": _platform_setting(config, "PLATFORM_LINK_GRAFANA"),
        "prometheus": _platform_setting(config, "PLATFORM_LINK_PROMETHEUS"),
        "alertmanager": _platform_setting(config, "PLATFORM_LINK_ALERTMANAGER"),
    }

    return {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "overall_status": overall,
        "environments": {"prod": prod_block, "demo": demo_block},
        "links": links,
    }
