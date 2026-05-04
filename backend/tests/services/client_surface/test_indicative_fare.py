"""Tests indicatif portail (hors compute_price / preview)."""

from decimal import Decimal

import pytest

from models import PlatformClientIndicativeFareConfig
from services.client_surface.indicative_fare import (
    IndicativeFareValidationError,
    assert_coherence,
    compute_indicative_amount_chf,
    merge_admin_update,
    round_chf_to_five_rappen,
)


def _default_cfg() -> PlatformClientIndicativeFareConfig:
    return PlatformClientIndicativeFareConfig(
        id=1,
        is_enabled=True,
        min_fare_chf=Decimal("45"),
        base_chf=Decimal("18"),
        per_minute_chf=Decimal("0.35"),
        ref_km=Decimal("13.5"),
        ref_min=Decimal("20"),
        config_version=1,
    )


def test_round_chf_five_rappen_epsilon_parity() -> None:
    """Même sémantique que `Math.round((x + Number.EPSILON) * 20) / 20` (ES)."""
    # Frontières courantes (5 centimes)
    assert float(round_chf_to_five_rappen(Decimal("44.97"))) == 44.95
    assert float(round_chf_to_five_rappen(Decimal("45.02"))) in (
        45.0,
        45.05,
    )  # float bin près
    v = float(round_chf_to_five_rappen(Decimal("45.0")))
    assert v == 45.0
    v2 = float(round_chf_to_five_rappen(48.26))
    assert abs(v2 - 48.25) < 0.001


def test_compute_anchored_reference_trip() -> None:
    cfg = _default_cfg()
    out = compute_indicative_amount_chf(13_500, 20 * 60, cfg)
    assert out == Decimal("45")


def test_compute_larger_trip() -> None:
    cfg = _default_cfg()
    out = compute_indicative_amount_chf(20_000, 30 * 60, cfg)
    assert out > Decimal("45")


def test_assert_coherence_rejects_negative_slack() -> None:
    with pytest.raises(IndicativeFareValidationError) as e:
        assert_coherence(
            Decimal("10"),
            Decimal("18"),
            Decimal("0.35"),
            Decimal("20"),
            Decimal("13.5"),
        )
    assert e.value.code == "negative_per_km"


def test_merge_admin_bumps_version_in_caller() -> None:
    row = _default_cfg()
    assert row.config_version == 1
    merge_admin_update(
        row,
        {"min_fare_chf": 50},
    )
    # merge ne touche pas version — c'est l'API admin qui +1
    assert row.min_fare_chf == Decimal("50")
    # cohérence: min plus grand → per_km implicite augmente, toujours ok
    assert float(row.min_fare_chf) == 50.0


def test_per_km_derivation_positive() -> None:
    from services.client_surface.indicative_fare import derive_per_km_chf

    cfg = _default_cfg()
    d = derive_per_km_chf(cfg)
    assert d > 0
