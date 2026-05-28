"""Harness Phase 2 — utilitaires partagés (JWT, clients Socket.IO, health)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any

import jwt  # type: ignore[import-untyped]
import socketio  # type: ignore[import-untyped]

WS_URL = "http://127.0.0.1:8001"
MOCK_BACKEND_URL = "http://127.0.0.1:8080"
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6380
JWT_SECRET = "validation-jwt-secret-only-for-local"
JWT_ALG = "HS256"
RELAY_CHANNEL = "ws:relay:events"


def make_token(
    *,
    role: str = "driver",
    user_id: int | None = None,
    sub: str | None = None,
    company_id: int | None = None,
    driver_id: int | None = None,
    ttl_sec: int = 300,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "iat": now,
        "exp": now + ttl_sec,
        "role": role,
    }
    if user_id is not None:
        payload["user_id"] = user_id
    if sub is not None:
        payload["sub"] = sub
    elif user_id is not None:
        payload["sub"] = str(user_id)
    else:
        payload["sub"] = str(uuid.uuid4())
    if company_id is not None:
        payload["company_id"] = company_id
    if driver_id is not None:
        payload["driver_id"] = driver_id
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def http_get_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(url: str, body: dict[str, Any] | None = None, timeout: float = 3.0) -> dict[str, Any]:
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def wait_for(predicate, *, timeout: float = 10.0, interval: float = 0.2, message: str = "") -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(interval)
    if message:
        print(f"[wait_for] timeout: {message}")
    return False


@dataclass
class CapturedClient:
    """Wrapper Socket.IO client + capture événements."""

    sio: socketio.Client
    received: list[tuple[str, Any]] = field(default_factory=list)
    connected_payload: dict[str, Any] | None = None
    authority_payload: dict[str, Any] | None = None
    connect_ok: bool = False
    connect_error: str | None = None

    def event_count(self, event_type: str) -> int:
        return sum(1 for e, _ in self.received if e == event_type)

    def payloads(self, event_type: str) -> list[Any]:
        return [p for e, p in self.received if e == event_type]


def new_client(
    *,
    token: str,
    transports: list[str] | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float = 8.0,
) -> CapturedClient:
    """Crée un client Socket.IO connecté à ws-service avec capture événements."""
    sio = socketio.Client(reconnection=False, logger=False, engineio_logger=False)
    cap = CapturedClient(sio=sio)

    @sio.event
    def connect() -> None:  # noqa: ARG001
        cap.connect_ok = True

    @sio.event
    def connect_error(data: Any) -> None:
        cap.connect_error = str(data)

    @sio.on("connected")
    def on_connected(data: Any) -> None:
        cap.connected_payload = data if isinstance(data, dict) else {"_raw": data}

    @sio.on("connection.authority")
    def on_authority(data: Any) -> None:
        cap.authority_payload = data if isinstance(data, dict) else {"_raw": data}

    @sio.on("*")
    def on_any(event: str, data: Any) -> None:  # noqa: ARG001
        cap.received.append((event, data))

    sio.connect(
        WS_URL,
        auth={"token": token},
        transports=transports or ["websocket"],
        headers=extra_headers or {},
        wait=True,
        wait_timeout=timeout,
    )
    return cap


def close_client(cap: CapturedClient) -> None:
    try:
        cap.sio.disconnect()
    except Exception:
        pass


def reset_mock_backend() -> None:
    try:
        http_post_json(f"{MOCK_BACKEND_URL}/reset")
    except Exception:
        pass


def get_mock_state() -> dict[str, Any]:
    return http_get_json(f"{MOCK_BACKEND_URL}/health")["state"]


def get_ws_health() -> dict[str, Any]:
    return http_get_json(f"{WS_URL}/health")
