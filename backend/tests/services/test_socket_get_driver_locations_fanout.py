from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from sockets import chat


class _DummySocketIO:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}

    def on(self, event: str, namespace: str = "/"):  # type: ignore[no-untyped-def]
        def _decorator(fn):  # type: ignore[no-untyped-def]
            self.handlers[event] = fn
            return fn

        return _decorator


class _UserQuery:
    def __init__(self, user: object) -> None:
        self._user = user

    def filter_by(self, **_kwargs):  # type: ignore[no-untyped-def]
        return self

    def first(self):  # type: ignore[no-untyped-def]
        return self._user


class _DriverQuery:
    def __init__(self, drivers: list[object]) -> None:
        self._drivers = drivers

    def filter_by(self, **_kwargs):  # type: ignore[no-untyped-def]
        return self

    def all(self):  # type: ignore[no-untyped-def]
        return self._drivers


@dataclass
class _DriverObj:
    id: int
    company_id: int
    latitude: float | None
    longitude: float | None
    is_active: bool = True
    user: object | None = None


class _FakeRedis:
    def __init__(self, payloads: dict[str, dict[bytes, bytes]]) -> None:
        self._payloads = payloads

    def hgetall(self, key: str):  # type: ignore[no-untyped-def]
        return self._payloads.get(key, {})

    def get(self, _key: str):  # type: ignore[no-untyped-def]
        return None


def test_snapshot_reconnect_never_forces_canonical_without_canonical_source(
    monkeypatch,
) -> None:
    dummy_socket = _DummySocketIO()
    chat.init_chat_socket(dummy_socket)
    handler = dummy_socket.handlers["get_driver_locations"]

    # driver 1: only legacy redis key available
    # driver 2: DB fallback (no redis hash)
    driver1 = _DriverObj(
        id=101,
        company_id=77,
        latitude=None,
        longitude=None,
        user=SimpleNamespace(first_name="D1"),
    )
    driver2 = _DriverObj(
        id=202,
        company_id=77,
        latitude=46.21,
        longitude=6.12,
        user=SimpleNamespace(first_name="D2"),
    )
    now_iso = datetime.now(UTC).isoformat()
    fake_redis = _FakeRedis(
        {
            "driver:101:loc": {
                b"lat": b"46.2",
                b"lon": b"6.1",
                b"ts": now_iso.encode(),
                b"location_mode": b"mission_live",
            }
        }
    )

    chat._SID_INDEX["sid-company"] = {
        "user_public_id": "company-public-id",
        "role": "company",
        "company_id": 77,
    }

    monkeypatch.setattr(chat, "_get_sid", lambda fallback_request=None: "sid-company")
    monkeypatch.setattr(chat, "redis_client", fake_redis)
    monkeypatch.setattr(
        chat, "User", SimpleNamespace(query=_UserQuery(SimpleNamespace(id=999)))
    )
    monkeypatch.setattr(
        chat, "Driver", SimpleNamespace(query=_DriverQuery([driver1, driver2]))
    )
    monkeypatch.setattr(
        chat, "_resolve_mission_status_for_driver", lambda _driver_id: None
    )
    monkeypatch.setattr(
        chat,
        "_resolve_driver_status",
        lambda **_kwargs: "available",
    )

    with patch(
        "services.realtime.socketio.fanout_driver_location_update"
    ) as fanout_mock:
        handler()

    assert fanout_mock.call_count == 2
    statuses = [call.kwargs.get("accept_status") for call in fanout_mock.call_args_list]
    assert statuses == [
        "accepted_observability_only",
        "accepted_observability_only",
    ]
