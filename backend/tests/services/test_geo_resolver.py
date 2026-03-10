from __future__ import annotations

from services.geo.geo_resolver import PickupAdminResolution, resolve_pickup_admin


def test_resolve_pickup_admin_uses_db_fallback(monkeypatch) -> None:
    import services.geo.geo_resolver as mod

    monkeypatch.setattr(mod, "_cache_get_json", lambda _k: None)
    monkeypatch.setattr(mod, "_cache_set_json", lambda _k, _p, _t: None)
    monkeypatch.setattr(mod, "_geoadmin_reverse", lambda _lat, _lng, _lang: None)
    monkeypatch.setattr(
        mod,
        "_try_db_resolution",
        lambda **_kwargs: PickupAdminResolution(
            token="commune:6630",
            canton_code="GE",
            source="db",
            confidence="authoritative",
            label="Anieres (GE)",
        ),
    )
    monkeypatch.setattr(mod, "_photon_reverse", lambda _lat, _lng: None)

    payload = resolve_pickup_admin(
        lat=46.27,
        lng=6.24,
        pickup_zip="1247",
        pickup_text="Anieres",
    )

    assert payload["token"] == "commune:6630"
    assert payload["canton_code"] == "GE"
    assert payload["source"] == "db"


def test_resolve_pickup_admin_returns_unknown_without_signals(monkeypatch) -> None:
    import services.geo.geo_resolver as mod

    monkeypatch.setattr(mod, "_try_db_resolution", lambda **_kwargs: None)

    payload = resolve_pickup_admin(
        lat=None,
        lng=None,
        pickup_zip=None,
        pickup_text=None,
    )

    assert payload["token"] is None
    assert payload["source"] == "unknown"
