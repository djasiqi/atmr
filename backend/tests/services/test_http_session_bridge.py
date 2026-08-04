"""Tests ensure_http_tracking_session_fields (HTTP → Kafka outbox)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.tracking.http_session_bridge import ensure_http_tracking_session_fields


def _flask_app():
    app = MagicMock()
    ctx = MagicMock()
    app.app_context.return_value = ctx
    ctx.__enter__ = MagicMock(return_value=None)
    ctx.__exit__ = MagicMock(return_value=False)
    return app


def test_ensure_uses_payload_session_and_sequence():
    redis = MagicMock()
    db = MagicMock()
    app = _flask_app()
    with (
        patch("celery_app.get_flask_app", return_value=app),
        patch("ext.db", db),
        patch("ext.redis_client", redis),
        patch(
            "services.tracking.session_registry.register_tracking_session",
            return_value={
                "tracking_session_id": "sess-a",
                "session_generation": 7,
                "first_sequence_id": 1,
                "status": "active",
            },
        ) as reg,
    ):
        out = ensure_http_tracking_session_fields(
            driver_id=7514,
            company_id=1,
            payload={
                "latitude": 46.2,
                "longitude": 6.1,
                "tracking_session_id": "sess-a",
                "sequence_id": 42,
            },
        )
    assert out["tracking_session_id"] == "sess-a"
    assert out["sequence_id"] == 42
    assert out["session_generation"] == 7
    reg.assert_called_once()
    redis.incr.assert_not_called()


def test_ensure_allocates_sequence_via_redis():
    redis = MagicMock()
    # Clé Redis déjà présente → pas de seed PG, juste INCR.
    redis.get.side_effect = lambda key: (
        b"2" if key.startswith("tracking:http_seq:") else None
    )
    redis.incr.return_value = 3
    db = MagicMock()
    app = _flask_app()
    with (
        patch("celery_app.get_flask_app", return_value=app),
        patch("ext.db", db),
        patch("ext.redis_client", redis),
        patch(
            "services.tracking.session_registry.register_tracking_session",
            return_value={
                "tracking_session_id": "http-legacy-9",
                "session_generation": 1,
                "first_sequence_id": 1,
                "status": "active",
            },
        ),
    ):
        out = ensure_http_tracking_session_fields(
            driver_id=9,
            company_id=1,
            payload={"latitude": 46.2, "longitude": 6.1},
        )
    assert out["tracking_session_id"] == "http-legacy-9"
    assert out["sequence_id"] == 3
    redis.incr.assert_called_once()
    redis.set.assert_not_called()


def test_ensure_seeds_redis_sequence_from_pg_max_seen():
    """Après expire Redis, seed depuis max_seen PG puis INCR (évite recyclage)."""
    redis = MagicMock()
    redis.get.return_value = None
    redis.incr.return_value = 11
    db = MagicMock()
    db.session.execute.return_value.mappings.return_value.first.return_value = {
        "max_seen": 10
    }
    app = _flask_app()
    with (
        patch("celery_app.get_flask_app", return_value=app),
        patch("ext.db", db),
        patch("ext.redis_client", redis),
        patch(
            "services.tracking.session_registry.register_tracking_session",
            return_value={
                "tracking_session_id": "http-legacy-3",
                "session_generation": 118,
                "first_sequence_id": 1,
                "status": "active",
            },
        ),
    ):
        out = ensure_http_tracking_session_fields(
            driver_id=3,
            company_id=1,
            payload={"latitude": 46.2, "longitude": 6.1},
        )
    assert out["sequence_id"] == 11
    redis.set.assert_called_once_with(
        "tracking:http_seq:3:http-legacy-3", 10, nx=True, ex=86400
    )
    redis.incr.assert_called_once_with("tracking:http_seq:3:http-legacy-3")
