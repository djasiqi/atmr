"""Mock backend pour validation Phase 2.

Simule UNIQUEMENT les endpoints nécessaires :
- POST /api/internal/tracking/ingest (cible GPS forward ws-service)
- GET  /health
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request

app = FastAPI(title="mock-backend (Phase 2 validation)")

_INTERNAL_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "validation-token")

# Compteurs in-memory pour assertions
state: dict[str, Any] = {
    "ingest_calls": 0,
    "ingest_points": 0,
    "ingest_last_driver": None,
    "ingest_unauthorized": 0,
    "ingest_invalid": 0,
    "last_batch_size": 0,
}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "mock-backend", "state": state}


@app.post("/api/internal/tracking/ingest")
async def ingest(
    request: Request,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> dict[str, Any]:
    if _INTERNAL_TOKEN and x_internal_token != _INTERNAL_TOKEN:
        state["ingest_unauthorized"] += 1
        raise HTTPException(status_code=401, detail="unauthorized")

    payload = await request.json()
    driver_id = payload.get("driver_id")
    points = payload.get("points")
    if not isinstance(driver_id, int) or not isinstance(points, list):
        state["ingest_invalid"] += 1
        raise HTTPException(status_code=400, detail="invalid_payload")

    state["ingest_calls"] += 1
    state["ingest_points"] += len(points)
    state["ingest_last_driver"] = driver_id
    state["last_batch_size"] = len(points)
    return {"ok": True, "accepted": len(points), "ts": int(time.time() * 1000)}


@app.post("/reset")
def reset() -> dict[str, Any]:
    for k in list(state.keys()):
        state[k] = 0 if isinstance(state[k], int) else None
    return {"ok": True}
