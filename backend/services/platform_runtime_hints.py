"""Hints runtime pour enrichir observed_state (GPS / services)."""

from __future__ import annotations

from typing import Any

from services.platform_runtime import build_platform_runtime_payload


def gps_pipeline_hint() -> dict[str, Any]:
    """Extrait un résumé du pipeline GPS depuis le payload runtime (best-effort)."""
    try:
        payload = build_platform_runtime_payload()
    except Exception:
        return {"status": "unknown", "reason": "runtime_unavailable"}
    sec = (payload.get("sections") or {}).get("gps_pipeline") or {}
    return {
        "status": sec.get("status"),
        "checked_at": sec.get("checked_at"),
        "reason": sec.get("reason"),
    }
