"""7B.2 — PRICING ENGINE PARITY GATE.

Invariant : le candidat plafond d'une entreprise = EXACTEMENT
``compute_price(...)`` (même Decimal quantize 0.01), sans formule parallèle.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from services.pricing.portal_carrier_ceiling import (
    ERROR_PORTAL_PRICING_CEILING_UNAVAILABLE,
    compute_portal_carrier_ceiling,
)
from services.pricing.pricing_engine import compute_price

CEILING_SRC = (
    Path(__file__).resolve().parents[2]
    / "services"
    / "pricing"
    / "portal_carrier_ceiling.py"
)


class _DummyProfile:
    def __init__(self, model_type: str):
        self.model_type = type("V", (), {"value": model_type})()
        self.currency = "CHF"
        self.id = 10


class _DummyVersion:
    def __init__(self, rules_json: dict, model_type: str, vid: int = 20):
        self.id = vid
        self.rules_json = rules_json
        self.pricing_profile = _DummyProfile(model_type)


def _company(cid: int, name: str):
    return SimpleNamespace(id=cid, name=name, is_approved=True, dispatch_enabled=True)


def _run_ceiling_with_real_engine(
    *,
    company,
    profile,
    version,
    context: dict,
):
    """Plafond avec compute_price RÉEL — seuls l'éligibilité / profil sont mockés."""
    with (
        patch(
            "services.pricing.portal_carrier_ceiling._eligible_companies",
            return_value=([company], []),
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._geo_unit_from_mission",
            return_value=None,
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._distance_meters",
            return_value=int(float(context.get("distance_km", 0) or 0) * 1000),
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._active_profile_version",
            return_value=(profile, version),
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._build_pricing_context",
            return_value=context,
        ),
    ):
        return compute_portal_carrier_ceiling(
            pickup_location="Genève",
            dropoff_location="HUG",
            is_round_trip=bool(context.get("is_round_trip")),
        )


def test_no_duplicated_pricing_formulas_in_ceiling_module():
    """Le module plafond ne doit pas réimplémenter flat/km/zones."""
    src = CEILING_SRC.read_text(encoding="utf-8")
    assert "compute_price" in src
    # Pas de formule parallèle évidente
    assert "base_fee +" not in src
    assert "per_km *" not in src
    assert "unit_price *" not in src
    assert "_compute_flat" not in src
    assert "_compute_distance" not in src
    assert "_compute_zone_count" not in src
    # Exclusions financières hors moteur trajet
    for forbidden in (
        "overdue_fee",
        "material_delivery_price_fixed",
        "payment_terms",
        "cancellation_fee",
        "vat_rate",
    ):
        assert forbidden not in src


def test_parity_flat_exact_decimal():
    rules = {"model": "flat", "base_fee": 45.0, "minimum": 0}
    version = _DummyVersion(rules, "flat")
    profile = SimpleNamespace(id=10, currency="CHF")
    context = {"pickup_local_time": "14:00", "is_weekend": False, "distance_km": 12.0}

    engine_amount, engine_bd = compute_price({}, version, context)
    assert engine_amount == Decimal("45.00")
    assert engine_bd["model"] == "flat"

    ceiling = _run_ceiling_with_real_engine(
        company=_company(1, "A-flat"),
        profile=profile,
        version=version,
        context=context,
    )
    assert Decimal(str(ceiling.maximum_accepted_amount)) == engine_amount
    assert ceiling.quotes[0].amount == float(engine_amount)
    assert ceiling.quotes[0].model == "flat"


def test_parity_distance_v1_exact_decimal():
    """Modèle UI actuel : components.distance (sans base)."""
    # 12 km × 3.20 = 38.40
    rules = {
        "model": "distance",
        "components": {
            "base": {"enabled": False, "amount": 0},
            "zone_count": {"enabled": False, "unit_price": 0, "included_zones": 1},
            "distance": {
                "enabled": True,
                "per_km": 3.20,
                "included_km": 0,
                "rounding": "ceil_0_1",
            },
        },
        "caps": {"minimum": 0},
    }
    version = _DummyVersion(rules, "distance")
    profile = SimpleNamespace(id=11, currency="CHF")
    context = {"distance_km": 12.0, "pickup_local_time": "14:00", "is_weekend": False}

    engine_amount, engine_bd = compute_price({}, version, context)
    assert engine_amount == Decimal("38.40")
    assert engine_bd["model"] == "distance"

    ceiling = _run_ceiling_with_real_engine(
        company=_company(2, "B-distance"),
        profile=profile,
        version=version,
        context=context,
    )
    assert Decimal(str(ceiling.maximum_accepted_amount)) == engine_amount
    assert ceiling.quotes[0].amount == float(engine_amount)


def test_parity_zone_count_exact_decimal(monkeypatch):
    """base 40 + 1 zone facturable × 12 = 52 (2 zones, 1 incluse)."""
    rules = {
        "model": "zone_count",
        "zone_set_id": "test_set",
        "components": {
            "base": {"enabled": True, "amount": 40},
            "zone_count": {
                "enabled": True,
                "unit_price": 12,
                "included_zones": 1,
                "max_units": 10,
                "strategy": "pickup_dropoff_diff_or_same",
            },
            "distance": {"enabled": False, "per_km": 0, "included_km": 0},
        },
        "caps": {"minimum": 0},
    }
    version = _DummyVersion(rules, "zone_count")
    profile = SimpleNamespace(id=12, currency="CHF")
    context = {
        "distance_km": 8.0,
        "zones_count": 2,
        "pickup_local_time": "14:00",
        "is_weekend": False,
        "pickup_admin_token": "commune:100",
        "dropoff_admin_token": "commune:200",
    }

    # Forcer la résolution zones à 2 (évite dépendance PostGIS)
    monkeypatch.setattr(
        "services.pricing.pricing_engine._compute_zones_count_from_rules",
        lambda _rules, _ctx: 2,
    )

    engine_amount, engine_bd = compute_price({}, version, context)
    assert engine_amount == Decimal("52.00")
    assert engine_bd["model"] == "zone_count"

    ceiling = _run_ceiling_with_real_engine(
        company=_company(3, "C-zone"),
        profile=profile,
        version=version,
        context=context,
    )
    assert Decimal(str(ceiling.maximum_accepted_amount)) == engine_amount
    assert ceiling.quotes[0].amount == float(engine_amount)


def test_parity_multi_model_max_45_38_40_52():
    """A flat 45, B distance 38.40, C zone_count 52 → plafond 52."""
    flat_v = _DummyVersion({"model": "flat", "base_fee": 45.0}, "flat", vid=21)
    dist_v = _DummyVersion(
        {
            "model": "distance",
            "components": {
                "base": {"enabled": False, "amount": 0},
                "zone_count": {"enabled": False, "unit_price": 0, "included_zones": 1},
                "distance": {
                    "enabled": True,
                    "per_km": 3.20,
                    "included_km": 0,
                },
            },
            "caps": {"minimum": 0},
        },
        "distance",
        vid=22,
    )
    zone_v = _DummyVersion(
        {
            "model": "zone_count",
            "zone_set_id": "test_set",
            "components": {
                "base": {"enabled": True, "amount": 40},
                "zone_count": {
                    "enabled": True,
                    "unit_price": 12,
                    "included_zones": 1,
                    "max_units": 10,
                },
                "distance": {"enabled": False, "per_km": 0, "included_km": 0},
            },
            "caps": {"minimum": 0},
        },
        "zone_count",
        vid=23,
    )

    companies = [
        _company(1, "A"),
        _company(2, "B"),
        _company(3, "C"),
    ]
    profiles = {
        1: (SimpleNamespace(id=1, currency="CHF"), flat_v),
        2: (SimpleNamespace(id=2, currency="CHF"), dist_v),
        3: (SimpleNamespace(id=3, currency="CHF"), zone_v),
    }
    context = {
        "distance_km": 12.0,
        "zones_count": 2,
        "pickup_local_time": "14:00",
        "is_weekend": False,
    }

    # Quotes moteur de référence (même context)
    with patch(
        "services.pricing.pricing_engine._compute_zones_count_from_rules",
        lambda _r, _c: 2,
    ):
        a, _ = compute_price({}, flat_v, context)
        b, _ = compute_price({}, dist_v, context)
        c, _ = compute_price({}, zone_v, context)
    assert (a, b, c) == (Decimal("45.00"), Decimal("38.40"), Decimal("52.00"))

    def fake_active(cid: int):
        return profiles[cid]

    with (
        patch(
            "services.pricing.portal_carrier_ceiling._eligible_companies",
            return_value=(companies, []),
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._geo_unit_from_mission",
            return_value=None,
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._distance_meters",
            return_value=12000,
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._active_profile_version",
            side_effect=fake_active,
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._build_pricing_context",
            return_value=context,
        ),
        patch(
            "services.pricing.pricing_engine._compute_zones_count_from_rules",
            lambda _r, _c: 2,
        ),
    ):
        ceiling = compute_portal_carrier_ceiling(
            pickup_location="Genève",
            dropoff_location="HUG",
        )

    assert ceiling.maximum_accepted_amount == 52.0
    by_id = {q.company_id: q.amount for q in ceiling.quotes}
    assert by_id == {1: 45.0, 2: 38.4, 3: 52.0}


def test_roundtrip_doubles_flat_distance_zone_count_same_as_engine():
    """AR : flat/distance/zone_count appliquent ×2 — plafond = moteur."""
    version = _DummyVersion({"model": "flat", "base_fee": 45.0}, "flat")
    profile = SimpleNamespace(id=10, currency="CHF")
    ctx_one = {"pickup_local_time": "14:00", "is_round_trip": False, "distance_km": 10}
    ctx_rt = {**ctx_one, "is_round_trip": True}

    one, _ = compute_price({}, version, ctx_one)
    rt, bd = compute_price({}, version, ctx_rt)
    assert one == Decimal("45.00")
    assert rt == Decimal("90.00")
    assert bd.get("round_trip", {}).get("applied") is True

    c1 = _run_ceiling_with_real_engine(
        company=_company(1, "A"), profile=profile, version=version, context=ctx_one
    )
    c2 = _run_ceiling_with_real_engine(
        company=_company(1, "A"), profile=profile, version=version, context=ctx_rt
    )
    assert c1.maximum_accepted_amount == 45.0
    assert c2.maximum_accepted_amount == 90.0


def test_financial_exclusions_not_in_engine_breakdown():
    """overdue / livraison / annulation / TVA hors compute_price trajet."""
    version = _DummyVersion({"model": "flat", "base_fee": 45.0}, "flat")
    amount, breakdown = compute_price(
        {},
        version,
        {
            "pickup_local_time": "14:00",
            # Champs « bruit » qui ne doivent pas influencer
            "overdue_fee": 15,
            "material_delivery_price_fixed": 25,
            "cancellation_fee": 30,
            "vat_rate": 8.1,
            "payment_terms_days": 30,
        },
    )
    assert amount == Decimal("45.00")
    blob = str(breakdown).lower()
    assert "overdue" not in blob
    assert "material" not in blob
    assert "cancel" not in blob
    assert "vat" not in blob
    assert "payment_terms" not in blob


def test_series_occurrences_multiplies_ceiling():
    """Récurrence : une demande × N passages → plafond = MAX(quotes) × N."""
    version = _DummyVersion({"model": "flat", "base_fee": 45.0}, "flat")
    profile = SimpleNamespace(id=10, currency="CHF")
    context = {"pickup_local_time": "14:00", "is_round_trip": False, "distance_km": 10}
    company = _company(1, "A")

    with (
        patch(
            "services.pricing.portal_carrier_ceiling._eligible_companies",
            return_value=([company], []),
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._geo_unit_from_mission",
            return_value=None,
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._distance_meters",
            return_value=10000,
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._active_profile_version",
            return_value=(profile, version),
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._build_pricing_context",
            return_value=context,
        ),
    ):
        ceiling = compute_portal_carrier_ceiling(
            pickup_location="Genève",
            dropoff_location="HUG",
            series_occurrences=11,
        )
    assert ceiling.per_trip_maximum_accepted_amount == 45.0
    assert ceiling.series_occurrences == 11
    assert ceiling.maximum_accepted_amount == 495.0
    with (
        patch(
            "services.pricing.portal_carrier_ceiling._eligible_companies",
            return_value=([], []),
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._geo_unit_from_mission",
            return_value=None,
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._distance_meters",
            return_value=None,
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._build_pricing_context",
            return_value={},
        ),
        pytest.raises(
            ValueError, match=ERROR_PORTAL_PRICING_CEILING_UNAVAILABLE
        ) as exc,
    ):
        compute_portal_carrier_ceiling(
            pickup_location="Genève",
            dropoff_location="HUG",
        )
    assert str(exc.value) == ERROR_PORTAL_PRICING_CEILING_UNAVAILABLE
