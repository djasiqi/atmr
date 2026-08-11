"""Régression : POST registre et http_session_bridge convergent sur le même SID."""

from __future__ import annotations

import uuid
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

from services.tracking.http_session_bridge import ensure_http_tracking_session_fields
from services.tracking.session_registry import register_tracking_session
from tests.factories import create_driver_with_position


@pytest.mark.integration
def test_register_and_http_bridge_converge_same_sid(app, db, sample_company):
    """Route registre + bridge PUT/location → même génération, 1 session, 1 state."""
    driver = create_driver_with_position(company=sample_company)
    db.session.flush()
    driver_id = int(driver.id)
    company_id = int(sample_company.id)
    sid = f"trk_bridge_{uuid.uuid4().hex[:12]}"

    direct = register_tracking_session(
        db.session,
        driver_id=driver_id,
        company_id=company_id,
        tracking_session_id=sid,
        tracking_session_started_at="2026-08-11T20:09:12.558Z",
    )
    db.session.flush()

    redis = MagicMock()
    redis.get.return_value = None
    redis.incr.return_value = 1

    # Évite commit/remove du bridge (savepoint) et un 2e app_context
    # qui ouvrirait une autre connexion → deadlock advisory xact.
    db.session.commit = MagicMock()  # type: ignore[method-assign]
    db.session.remove = MagicMock()  # type: ignore[method-assign]
    db.session.rollback = MagicMock()  # type: ignore[method-assign]

    with (
        patch("celery_app.get_flask_app", return_value=app),
        patch.object(app, "app_context", return_value=nullcontext()),
        patch("ext.db", db),
        patch("ext.redis_client", redis),
    ):
        bridged = ensure_http_tracking_session_fields(
            driver_id=driver_id,
            company_id=company_id,
            payload={
                "latitude": 46.2,
                "longitude": 6.1,
                "tracking_session_id": sid,
                "sequence_id": 1,
            },
        )

    assert bridged["tracking_session_id"] == sid
    assert bridged["session_generation"] == direct["session_generation"]

    n_sess = db.session.execute(
        text(
            """
            SELECT COUNT(*) FROM tracking_sessions
            WHERE driver_id = :d AND tracking_session_id = :sid
            """
        ),
        {"d": driver_id, "sid": sid},
    ).scalar_one()
    n_state = db.session.execute(
        text(
            """
            SELECT COUNT(*) FROM tracking_session_state
            WHERE driver_id = :d AND tracking_session_id = :sid
            """
        ),
        {"d": driver_id, "sid": sid},
    ).scalar_one()
    assert int(n_sess) == 1
    assert int(n_state) == 1
