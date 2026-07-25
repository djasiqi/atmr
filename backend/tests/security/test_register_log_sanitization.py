"""Tests Lot 0 P0 — sanitisation logs inscription (SEC-01)."""

from __future__ import annotations

import logging


def test_register_does_not_log_password(client, caplog, db):
    password = "SuperSecretP0Pass1!"
    with caplog.at_level(logging.INFO):
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "sec01_user",
                "email": "sec01_user@example.com",
                "password": password,
                "phone": "+41791234567",
                "first_name": "Sec",
                "last_name": "One",
            },
            headers={"Content-Type": "application/json"},
        )

    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert password not in joined
    assert "Données reçues dans /auth/register" not in joined
    assert "Données validées :" not in joined
