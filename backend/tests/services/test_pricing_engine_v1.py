from decimal import Decimal

import pytest

from services.pricing import pricing_engine as pricing_engine_module
from services.pricing.pricing_engine import (
    compute_price,
    validate_company_pricing_rules,
)


class _DummyModel:
    def __init__(self, model_type: str):
        self.model_type = type("V", (), {"value": model_type})()


class _DummyProfile:
    def __init__(self, model_type: str):
        self.model_type = type("V", (), {"value": model_type})()
        self.currency = "CHF"


class _DummyVersion:
    def __init__(self, rules_json: dict, model_type: str):
        self.rules_json = rules_json
        self.pricing_profile = _DummyProfile(model_type)


def test_compute_price_flat_is_deterministic():
    version = _DummyVersion(
        {
            "model": "flat",
            "base_fee": 45.0,
            "surcharges": [{"type": "after_time", "after": "20:00", "amount": 8.0}],
            "minimum": 0,
        },
        "flat",
    )
    amount, breakdown = compute_price(
        booking={},
        pricing_profile_version=version,
        context={"pickup_local_time": "20:30"},
    )
    assert amount == Decimal("53.00")
    assert breakdown["model"] == "flat"
    assert breakdown["total"] == "53.00"
    assert list(breakdown.keys()) == ["base", "extras", "minimum", "model", "total"]


def test_compute_price_zone_weekend_roundtrip():
    version = _DummyVersion(
        {
            "model": "zone",
            "pricing": {
                "weekday": {"one_way": 45.0, "round_trip": 85.0},
                "weekend": {"one_way": 60.0, "round_trip": 120.0},
            },
            "extras": [{"type": "after_time_per_zone", "after": "20:00", "amount": 8.0}],
        },
        "zone",
    )
    amount, breakdown = compute_price(
        booking={},
        pricing_profile_version=version,
        context={
            "is_weekend": True,
            "is_round_trip": True,
            "pickup_local_time": "21:00",
            "zones_count": 2,
        },
    )
    assert amount == Decimal("136.00")
    assert breakdown["base"]["rule"] == "weekend_round_trip"


def test_compute_price_distance_with_minimum():
    version = _DummyVersion(
        {
            "model": "distance",
            "base_fee": 20.0,
            "per_km": 3.5,
            "minimum": 40.0,
        },
        "distance",
    )
    amount, breakdown = compute_price(
        booking={},
        pricing_profile_version=version,
        context={"distance_km": 2},
    )
    assert amount == Decimal("40.00")
    assert breakdown["minimum"]["applied"] is True


def test_compute_price_zone_matrix_uses_pair_transition():
    version = _DummyVersion(
        {
            "model": "zone_matrix",
            "zones": [
                {"id": "z1", "code": "A", "label": "Centre", "tokens": ["commune:100"]},
                {"id": "z2", "code": "B", "label": "Rive", "tokens": ["commune:200"]},
            ],
            "matrix": {"z1": {"z1": 45.0, "z2": 60.0}, "z2": {"z1": 60.0, "z2": 50.0}},
            "extras": [{"type": "after_time_per_zone", "after": "20:00", "amount": 4.0}],
        },
        "zone",
    )
    amount, breakdown = compute_price(
        booking={},
        pricing_profile_version=version,
        context={
            "pickup_admin_token": "commune:100",
            "dropoff_admin_token": "commune:200",
            "pickup_local_time": "21:05",
        },
    )
    assert amount == Decimal("68.00")
    assert breakdown["model"] == "zone_matrix"
    assert breakdown["from_zone"]["id"] == "z1"
    assert breakdown["to_zone"]["id"] == "z2"
    assert breakdown["warnings"] == []


def test_compute_price_zone_matrix_fallback_when_unassigned():
    version = _DummyVersion(
        {
            "model": "zone_matrix",
            "zones": [{"id": "z1", "code": "A", "label": "Centre", "tokens": ["commune:100"]}],
            "matrix": {"z1": {"z1": 45.0}},
            "pricing": {"weekday": {"one_way": 30.0}},
        },
        "zone",
    )
    amount, breakdown = compute_price(
        booking={},
        pricing_profile_version=version,
        context={
            "pickup_admin_token": "commune:999",
            "dropoff_admin_token": "commune:100",
        },
    )
    assert amount == Decimal("30.00")
    assert breakdown["model"] == "zone_matrix"
    assert "unassigned_commune_fallback" in breakdown["warnings"]


def test_validate_company_pricing_rules_requires_zone_set_for_zone_model():
    rules = {
        "v": 1,
        "model": "zone_count",
        "currency": "CHF",
        "components": {
            "base": {"enabled": True, "amount": 30},
            "zone_count": {"enabled": True, "unit_price": 8, "max_units": 4},
            "distance": {"enabled": False, "per_km": 0, "included_km": 0},
        },
        "extras": {},
        "caps": {"minimum": 0, "maximum": None},
    }
    with pytest.raises(ValueError, match="zone_set_id"):
        validate_company_pricing_rules(rules)


def test_compute_price_zone_count_uses_zone_set_resolver(monkeypatch):
    monkeypatch.setattr(
        pricing_engine_module,
        "resolve_zone_id",
        lambda token, zone_set_id: "A" if token == "commune:100" else "B",
    )
    version = _DummyVersion(
        {
            "v": 1,
            "model": "zone_count",
            "currency": "CHF",
            "zone_set_id": "zoneset_ge_v1",
            "components": {
                "base": {"enabled": True, "amount": 30},
                "zone_count": {"enabled": True, "unit_price": 10, "strategy": "pickup_dropoff_diff_or_same", "max_units": 10},
                "distance": {"enabled": False, "per_km": 0, "included_km": 0, "rounding": "ceil_0_1"},
            },
            "extras": {},
            "caps": {"minimum": 0, "maximum": None},
        },
        "zone",
    )
    amount, breakdown = compute_price(
        booking={},
        pricing_profile_version=version,
        context={"pickup_admin_token": "commune:100", "dropoff_admin_token": "commune:200"},
    )
    assert amount == Decimal("40.00")
    assert breakdown["model"] == "zone_count"


@pytest.mark.parametrize(
    ("zones_traversed", "expected_amount"),
    [
        (1, Decimal("45.00")),
        (2, Decimal("45.00")),
        (3, Decimal("50.00")),
        (4, Decimal("55.00")),
        (5, Decimal("60.00")),
    ],
)
def test_compute_price_zone_count_formula_by_traversed_zones(monkeypatch, zones_traversed, expected_amount):
    monkeypatch.setattr(
        pricing_engine_module,
        "estimate_zones_traversed",
        lambda **_kwargs: zones_traversed,
    )
    version = _DummyVersion(
        {
            "v": 1,
            "model": "zone_count",
            "currency": "CHF",
            "zone_set_id": "zoneset_ge_v1",
            "components": {
                "base": {"enabled": True, "amount": 45},
                "zone_count": {
                    "enabled": True,
                    "unit_price": 5,
                    "strategy": "pickup_dropoff_diff_or_same",
                    "included_zones": 2,
                    "max_units": 10,
                },
                "distance": {"enabled": False, "per_km": 0, "included_km": 0, "rounding": "ceil_0_1"},
            },
            "extras": {},
            "caps": {"minimum": 0, "maximum": None},
        },
        "zone",
    )
    amount, breakdown = compute_price(
        booking={},
        pricing_profile_version=version,
        context={"pickup_admin_token": "commune:100", "dropoff_admin_token": "commune:200"},
    )
    assert amount == expected_amount
    assert breakdown["zones_traversees"] == zones_traversed
    assert breakdown["zones_incluses"] == 2
    assert breakdown["zones_facturables"] == max(zones_traversed - 2, 0)
