"""Intégration API indicatif portail (estimate + admin), hors preview réservation."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from flask_jwt_extended import create_access_token
from sqlalchemy import inspect

from ext import db as ext_db
from models import PlatformClientIndicativeFareConfig
from models.enums import UserRole


def _table_exists(app) -> bool:
    with app.app_context():
        insp = inspect(ext_db.engine)
        return "platform_client_indicative_fare_config" in insp.get_table_names()


def _client_bearer_headers(app, sample_client) -> dict[str, str]:
    claims = {
        "role": UserRole.client.value,
        "company_id": None,
        "driver_id": None,
        "aud": "atmr-api",
    }
    with app.app_context():
        token = create_access_token(
            identity=str(sample_client.user.public_id),
            additional_claims=claims,
        )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _ensure_singleton_row(db, **overrides) -> PlatformClientIndicativeFareConfig:
    row = db.session.get(PlatformClientIndicativeFareConfig, 1)
    if row is None:
        row = PlatformClientIndicativeFareConfig(
            id=1,
            is_enabled=True,
            min_fare_chf=Decimal("45"),
            base_chf=Decimal("18"),
            per_minute_chf=Decimal("0.35"),
            ref_km=Decimal("13.5"),
            ref_min=Decimal("20"),
            config_version=1,
            calibration_note=None,
        )
        db.session.add(row)
    for k, v in overrides.items():
        setattr(row, k, v)
    db.session.flush()
    return row


@pytest.mark.integration
def test_indicative_fare_estimate_disabled_412(
    client, app, db, sample_client, monkeypatch
):
    if not _table_exists(app):
        pytest.skip("Table platform_client_indicative_fare_config absente (flask db upgrade).")

    row = _ensure_singleton_row(db, is_enabled=False, config_version=3)
    assert row.is_enabled is False

    def _boom(_p, _d):
        raise AssertionError("get_optimized_route ne doit pas être appelé si désactivé")

    monkeypatch.setattr("routes.clients.get_optimized_route", _boom)

    headers = _client_bearer_headers(app, sample_client)
    r = client.post(
        "/api/v1/clients/me/indicative-fare/estimate",
        data=json.dumps(
            {
                "pickup_location": "Rue de la Gare 1, 1200 Genève",
                "dropoff_location": "HUG, Genève",
            }
        ),
        headers=headers,
    )
    assert r.status_code == 412
    body = r.get_json()
    assert body.get("error") == "indicative_fare_disabled"


@pytest.mark.integration
def test_admin_put_bumps_config_version_then_estimate_matches(
    client, app, db, sample_client, admin_headers, monkeypatch
):
    if not _table_exists(app):
        pytest.skip("Table platform_client_indicative_fare_config absente (flask db upgrade).")

    row = _ensure_singleton_row(
        db,
        is_enabled=True,
        min_fare_chf=Decimal("45"),
        config_version=10,
    )
    v_before = int(row.config_version or 0)

    r_put = client.put(
        "/api/v1/admin/client-indicative-fare",
        data=json.dumps({"min_fare_chf": 50}),
        headers={**admin_headers, "Content-Type": "application/json"},
    )
    assert r_put.status_code == 200, r_put.get_data(as_text=True)
    put_body = r_put.get_json()
    assert int(put_body["config_version"]) == v_before + 1
    assert float(put_body["min_fare_chf"]) == 50.0

    def _fake_route(_pickup, _dropoff):
        return {
            "distance_m": 13_500,
            "duration_s": 20 * 60,
            "polyline": "mock",
        }

    monkeypatch.setattr("routes.clients.get_optimized_route", _fake_route)

    headers = _client_bearer_headers(app, sample_client)
    r_est = client.post(
        "/api/v1/clients/me/indicative-fare/estimate",
        data=json.dumps(
            {
                "pickup_location": "A",
                "dropoff_location": "B",
            }
        ),
        headers=headers,
    )
    assert r_est.status_code == 200, r_est.get_data(as_text=True)
    est = r_est.get_json()
    assert est.get("is_contractual") is False
    assert int(est["config_version"]) == v_before + 1
    assert float(est["indicative_amount_chf"]) == 50.0
    assert est["distance_m"] == 13_500
    assert est["duration_s"] == 20 * 60
