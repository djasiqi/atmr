"""Tests dedup/throttle atomiques Redis."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.notifications.dedup_throttle import should_skip_dedup, should_skip_throttle


def test_should_skip_dedup_first_call_not_duplicate():
    mock_redis = MagicMock()
    mock_redis.set.return_value = True

    with patch("services.notifications.dedup_throttle._get_redis", return_value=mock_redis):
        assert should_skip_dedup("driver", 1, "key-a") is False

    mock_redis.set.assert_called_once_with(
        "push:dedup:driver:1:key-a", "1", nx=True, ex=300
    )


def test_should_skip_dedup_second_call_is_duplicate():
    mock_redis = MagicMock()
    mock_redis.set.return_value = None

    with patch("services.notifications.dedup_throttle._get_redis", return_value=mock_redis):
        assert should_skip_dedup("driver", 1, "key-a") is True


def test_should_skip_throttle_uses_lua_script():
    mock_redis = MagicMock()
    mock_redis.eval.return_value = 1

    with patch("services.notifications.dedup_throttle._get_redis", return_value=mock_redis):
        assert should_skip_throttle("driver", 1, "scope", 60, 1) is False

    mock_redis.eval.assert_called_once()
    assert mock_redis.incr.call_count == 0


def test_should_skip_throttle_blocks_when_over_limit():
    mock_redis = MagicMock()
    mock_redis.eval.return_value = 2

    with patch("services.notifications.dedup_throttle._get_redis", return_value=mock_redis):
        assert should_skip_throttle("driver", 1, "scope", 60, 1) is True
