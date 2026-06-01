"""Tests pour les helpers introduits sur send_silent_data_update.

Couvre :
- la dédup des DeviceToken par valeur `token` (garde la ligne la plus récente),
- le throttle Redis SET NX EX par (driver_id, sync_type),
- les modes "fail-open" du throttle (Redis indisponible).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from services.events.fanout import (
    _dedup_device_tokens_by_token,
    _should_throttle_silent_update,
)


def _mk_token(token: str, updated_at: datetime | None = None) -> MagicMock:
    dt = MagicMock()
    dt.token = token
    dt.updated_at = updated_at
    return dt


def test_dedup_keeps_most_recent_per_token():
    older = _mk_token("tok-a", datetime(2026, 1, 1, tzinfo=timezone.utc))
    newer = _mk_token("tok-a", datetime(2026, 6, 1, tzinfo=timezone.utc))
    other = _mk_token("tok-b", datetime(2026, 5, 1, tzinfo=timezone.utc))

    deduped = _dedup_device_tokens_by_token([older, newer, other])
    tokens = sorted(dt.token for dt in deduped)
    assert tokens == ["tok-a", "tok-b"]
    # Le DeviceToken conservé pour tok-a est `newer`.
    by_token = {dt.token: dt for dt in deduped}
    assert by_token["tok-a"] is newer


def test_dedup_drops_empty_tokens():
    empty = _mk_token("", datetime(2026, 1, 1, tzinfo=timezone.utc))
    keep = _mk_token("tok-a", datetime(2026, 1, 1, tzinfo=timezone.utc))
    deduped = _dedup_device_tokens_by_token([empty, keep])
    assert [dt.token for dt in deduped] == ["tok-a"]


def test_dedup_keeps_first_when_updated_at_missing():
    first = _mk_token("tok-a", None)
    second = _mk_token("tok-a", None)
    deduped = _dedup_device_tokens_by_token([first, second])
    assert len(deduped) == 1
    assert deduped[0] is first


def test_throttle_returns_true_when_redis_already_set():
    fake_redis = MagicMock()
    fake_redis.set.return_value = None  # SET NX a échoué : clé déjà présente
    with patch("ext.redis_client", fake_redis):
        throttled = _should_throttle_silent_update(7135, "profile")
    assert throttled is True
    fake_redis.set.assert_called_once()
    args, kwargs = fake_redis.set.call_args
    assert args[0] == "silent_update:7135:profile"
    assert kwargs.get("nx") is True
    assert kwargs.get("ex", 0) > 0


def test_throttle_returns_false_when_redis_sets_key():
    fake_redis = MagicMock()
    fake_redis.set.return_value = True  # SET NX a réussi : on autorise l'envoi
    with patch("ext.redis_client", fake_redis):
        throttled = _should_throttle_silent_update(7135, "profile")
    assert throttled is False


def test_throttle_fail_open_when_redis_unavailable():
    """Si redis_client vaut None, on doit laisser passer le silent_update."""
    with patch("ext.redis_client", None):
        assert _should_throttle_silent_update(7135, "profile") is False


def test_throttle_fail_open_when_redis_raises():
    fake_redis = MagicMock()
    fake_redis.set.side_effect = RuntimeError("redis down")
    with patch("ext.redis_client", fake_redis):
        assert _should_throttle_silent_update(7135, "profile") is False
