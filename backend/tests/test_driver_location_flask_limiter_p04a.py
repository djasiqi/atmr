"""P0.4-A : exemption Flask-Limiter sur la vraie view RESTX PUT GPS.

Preuve que :
- A : PUT /driver/me/location n'est jamais 429 global Flask-Limiter
- B : le limiteur métier Lua/mémoire reste actif
- C : un autre endpoint reste soumis au limiteur global
- D : l'endpoint enregistré (pas Resource.put) est bien exempté
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from flask_jwt_extended import create_access_token
from flask_limiter import Limiter
from flask_limiter._limits import Limit
from flask_limiter.util import get_qualified_name, get_remote_address
from flask_restx import Api, Namespace, Resource

from routes_api import exempt_driver_location_registered_views

LOCATION_PATH = "/api/v1/driver/me/location"
_GPS_ENV = {"REMOTE_ADDR": "10.66.4.101"}


@pytest.fixture
def app() -> Iterator[Flask]:
    """Crée une application neuve avant l'enregistrement des hooks Limiter."""
    from app import create_app

    return create_app(config_name="testing")


class _BusinessLimitStub:
    """Compteur métier local (limit appels autorisés)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count = 0

    def __call__(self, _driver_id: int) -> tuple[bool, int | None, str | None]:
        self.count += 1
        if self.count > self.limit:
            return False, 7, "short_window"
        return True, None, None


@pytest.fixture
def strict_global_limiter(app: Flask) -> Iterator[Limiter]:
    """Active Flask-Limiter global à 2/min pour la durée du test."""
    from ext import limiter

    previous_enabled = bool(app.config.get("RATELIMIT_ENABLED", False))
    previous_limiter_enabled = bool(getattr(limiter, "enabled", False))
    previous_defaults = list(limiter.limit_manager._default_limits)

    # TestingConfig pose RATELIMIT_ENABLED=False : forcer True avant init_app.
    app.config["RATELIMIT_ENABLED"] = True
    limiter.enabled = True
    limiter.init_app(app)
    assert limiter.enabled is True
    assert app.extensions.get("limiter") is not None

    limiter.limit_manager.set_default_limits(
        [Limit("2 per minute", key_function=get_remote_address)]
    )
    # Ré-appliquer l'exemption GPS (idempotent) après reconfig
    exempt_driver_location_registered_views(app, limiter_instance=limiter)

    yield limiter

    limiter.limit_manager.set_default_limits(previous_defaults)
    app.config["RATELIMIT_ENABLED"] = previous_enabled
    limiter.enabled = previous_limiter_enabled
    if not previous_enabled:
        limiter.enabled = False
        app.config["RATELIMIT_ENABLED"] = False


def _is_flask_global_429(response: Any) -> bool:
    """True si 429 Flask-Limiter global (pas le 429 métier GPS)."""
    if response.status_code != 429:
        return False
    body = response.get_json(silent=True) or {}
    return not (isinstance(body, dict) and body.get("error") == "rate_limit_exceeded")


def _has_location_route(app: Flask) -> bool:
    return any(
        str(r.rule).endswith("/driver/me/location") and "PUT" in (r.methods or set())
        for r in app.url_map.iter_rules()
    )


@pytest.mark.integration
def test_a_put_location_never_flask_global_429(
    app: Flask, client, strict_global_limiter: Limiter
) -> None:
    """5 PUT sans JWT : 401 OK, jamais 429 Flask-Limiter global."""
    _ = strict_global_limiter
    if not _has_location_route(app):
        pytest.skip("Route driver location non enregistrée")

    statuses: list[int] = []
    for _ in range(5):
        resp = client.put(
            LOCATION_PATH,
            json={"latitude": 46.2, "longitude": 6.1},
            environ_overrides=_GPS_ENV,
        )
        statuses.append(resp.status_code)
        assert not _is_flask_global_429(resp), (
            f"429 Flask-Limiter global inattendu sur GPS: statuses={statuses} "
            f"body={resp.get_json(silent=True)}"
        )

    assert all(s == 401 for s in statuses), f"Attendu 401×5, got={statuses}"


@pytest.mark.integration
def test_b_business_limiter_still_active(
    app: Flask, client, sample_driver, strict_global_limiter: Limiter, monkeypatch
) -> None:
    """Flask global exempt ; limiteur métier actif → 429 métier au 3e appel."""
    _ = strict_global_limiter
    if not _has_location_route(app):
        pytest.skip("Route driver location non enregistrée")

    monkeypatch.setattr(
        "routes.driver.check_http_driver_location_rate_limit",
        _BusinessLimitStub(limit=2),
    )

    claims = {
        "role": sample_driver.user.role.value,
        "company_id": sample_driver.company_id,
        "driver_id": sample_driver.id,
        "aud": "atmr-api",
    }
    with app.app_context():
        token = create_access_token(
            identity=str(sample_driver.user.public_id), additional_claims=claims
        )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "latitude": 46.2044,
        "longitude": 6.1432,
        "location_mode": "availability_presence",
        "recorded_at": "2026-08-07T10:00:00Z",
    }

    uc_result = MagicMock(
        dedup_skipped=False,
        snapped_lat=46.2044,
        snapped_lon=6.1432,
        source="raw",
        geofence_events=[],
        accept_status="accepted_canonical",
        accept_reason="",
        received_at="2026-08-07T10:00:01Z",
        canonical_updated=True,
        db_persisted=True,
        location_event_id="evt_test_p04a",
    )
    mock_uc = MagicMock()
    mock_uc.execute.return_value = uc_result

    statuses: list[int] = []
    bodies: list[Any] = []
    with (
        patch(
            "application.drivers.update_driver_location.UpdateDriverLocationUseCase",
            return_value=mock_uc,
        ),
        patch("services.realtime.socketio.fanout_driver_location_update"),
    ):
        for _ in range(3):
            resp = client.put(
                LOCATION_PATH,
                json=body,
                headers=headers,
                environ_overrides=_GPS_ENV,
            )
            statuses.append(resp.status_code)
            bodies.append(resp.get_json(silent=True))

    assert statuses[0] != 429, f"1er appel ne doit pas être 429: {statuses} {bodies}"
    assert statuses[1] != 429, f"2e appel ne doit pas être 429: {statuses} {bodies}"
    assert statuses[2] == 429, f"3e appel = 429 métier, got={statuses} {bodies}"
    body_429 = bodies[2] or {}
    assert body_429.get("error") == "rate_limit_exceeded"
    assert "retry_after_seconds" in body_429
    assert body_429.get("rate_limit_reason") is not None


def test_c_other_endpoint_still_globally_limited() -> None:
    """P0.4-A n'a pas désactivé Flask-Limiter : une autre route reste limitée."""
    flask_app = Flask("p04a_other")
    flask_app.config["RATELIMIT_ENABLED"] = True
    lim = Limiter(
        key_func=get_remote_address,
        default_limits=["2 per minute"],
        storage_uri="memory://",
    )
    lim.init_app(flask_app)

    ns = Namespace("driver")

    class DriverLocation(Resource):
        def put(self):
            return {"error": "unauthorized"}, 401

    ns.add_resource(DriverLocation, "/me/location")
    api = Api(flask_app, prefix="/api/v1", doc=False)
    api.add_namespace(ns, path="/driver")
    exempt_driver_location_registered_views(flask_app, limiter_instance=lim)

    @flask_app.route("/__p04a_other")
    def other_probe():
        return {"ok": True}, 200

    client = flask_app.test_client()
    gps_statuses = [
        client.put(
            LOCATION_PATH, environ_overrides={"REMOTE_ADDR": "10.66.4.201"}
        ).status_code
        for _ in range(5)
    ]
    assert gps_statuses == [401, 401, 401, 401, 401], gps_statuses

    other_statuses = [
        client.get(
            "/__p04a_other", environ_overrides={"REMOTE_ADDR": "10.66.4.202"}
        ).status_code
        for _ in range(4)
    ]
    assert other_statuses[:2] == [200, 200], other_statuses
    assert any(s == 429 for s in other_statuses[2:]), other_statuses


@pytest.mark.integration
def test_d_registered_view_is_exempt_not_only_resource_put(
    app: Flask, strict_global_limiter: Limiter
) -> None:
    """Introspection : endpoint driver_driver_location exempt, pas seulement put()."""
    limiter = strict_global_limiter
    if not _has_location_route(app):
        pytest.skip("Route driver location non enregistrée")

    matched = [
        (str(rule.rule), str(rule.endpoint))
        for rule in app.url_map.iter_rules()
        if "PUT" in (rule.methods or set())
        and str(rule.rule).endswith("/driver/me/location")
    ]
    assert matched, "Aucune règle PUT /driver/me/location"
    assert any(
        ep == "driver_driver_location" or ep.startswith("driver_driver_location_")
        for _, ep in matched
    )

    for rule_str, endpoint in matched:
        view = app.view_functions[endpoint]
        qual = get_qualified_name(view)
        assert qual in limiter._route_exemptions, (
            f"View enregistrée non exemptée: rule={rule_str} endpoint={endpoint} "
            f"qual={qual} exemptions={list(limiter._route_exemptions)}"
        )


def test_helper_exempts_restx_as_view_not_resource_method() -> None:
    """Preuve unitaire isolée du bug RESTX + correctif post-enregistrement."""
    flask_app = Flask("p04a_restx")
    flask_app.config["RATELIMIT_ENABLED"] = True
    lim = Limiter(
        key_func=get_remote_address,
        default_limits=["2 per minute"],
        storage_uri="memory://",
    )
    lim.init_app(flask_app)

    ns = Namespace("driver")

    class DriverLocation(Resource):
        @lim.exempt
        def put(self):
            return {"error": "unauthorized"}, 401

    ns.add_resource(DriverLocation, "/me/location")
    api = Api(flask_app, prefix="/api/v1", doc=False)
    api.add_namespace(ns, path="/driver")

    client = flask_app.test_client()
    before = [
        client.put(
            LOCATION_PATH, environ_overrides={"REMOTE_ADDR": "10.66.4.9"}
        ).status_code
        for _ in range(4)
    ]
    assert 429 in before, f"Sans helper, 429 global attendu: {before}"

    flask_app2 = Flask("p04a_restx_fixed")
    flask_app2.config["RATELIMIT_ENABLED"] = True
    lim2 = Limiter(
        key_func=get_remote_address,
        default_limits=["2 per minute"],
        storage_uri="memory://",
    )
    lim2.init_app(flask_app2)
    ns2 = Namespace("driver")

    class DriverLocation2(Resource):
        @lim2.exempt
        def put(self):
            return {"error": "unauthorized"}, 401

    ns2.add_resource(DriverLocation2, "/me/location")
    api2 = Api(flask_app2, prefix="/api/v1", doc=False)
    api2.add_namespace(ns2, path="/driver")

    matched = exempt_driver_location_registered_views(flask_app2, limiter_instance=lim2)
    assert matched
    assert all(rule.endswith("/driver/me/location") for rule, _ in matched)
    assert any("driver_location" in ep for _, ep in matched), matched

    client2 = flask_app2.test_client()
    after = [
        client2.put(
            LOCATION_PATH, environ_overrides={"REMOTE_ADDR": "10.66.4.10"}
        ).status_code
        for _ in range(5)
    ]
    assert after == [401, 401, 401, 401, 401], f"Avec helper: jamais 429, got={after}"

    endpoint = matched[0][1]
    view = flask_app2.view_functions[endpoint]
    assert get_qualified_name(view) in lim2._route_exemptions
    # Le décorateur sur Resource.put seul ne suffit pas
    put_qual = get_qualified_name(DriverLocation2.put)
    assert put_qual != get_qualified_name(view)
