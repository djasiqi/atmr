"""7B.2 — plafond PORTAL = MAX(quotes transporteurs).

Cas couverts :
1. Plusieurs modèles → MAX réel
2. Non éligible exclu du MAX
3. Sans grille / quote ≤ 0 → exclu (jamais 0 silencieux)
4. Aucune quote → erreur explicite
5. (create_booking) payload client maximum_accepted_amount ignoré
6. Snapshot figé (preuve evidence ; non réécrit par recalcul)
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from services.pricing.portal_carrier_ceiling import (
    ERROR_PORTAL_PRICING_CEILING_UNAVAILABLE,
    CarrierQuote,
    ExcludedCarrier,
    PortalCarrierCeiling,
    compute_portal_carrier_ceiling,
)


def _company(cid: int, name: str, *, approved: bool = True, dispatch: bool = True):
    return SimpleNamespace(
        id=cid, name=name, is_approved=approved, dispatch_enabled=dispatch
    )


def _profile(pid: int = 10, currency: str = "CHF"):
    return SimpleNamespace(id=pid, currency=currency)


def _version(vid: int = 20):
    return SimpleNamespace(id=vid)


def _patch_base(*, companies, amounts_or_side_effect, profile=None, version=None):
    """Context manager patches communs pour compute_portal_carrier_ceiling."""
    profile = profile or _profile()
    version = version or _version()
    return (
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
            return_value=(profile, version),
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._build_pricing_context",
            return_value={"distance_km": 12.0},
        ),
        patch(
            "services.pricing.portal_carrier_ceiling.compute_price",
            side_effect=amounts_or_side_effect,
        ),
    )


def test_case1_max_across_flat_km_zone_models():
    """A forfait 45, B km 38.40, C zone 52 → plafond 52."""
    companies = [
        _company(1, "A"),
        _company(2, "B"),
        _company(3, "C"),
    ]
    amounts = [
        (Decimal("45.00"), {"model": "flat"}),
        (Decimal("38.40"), {"model": "distance"}),
        (Decimal("52.00"), {"model": "zone"}),
    ]
    patches = _patch_base(companies=companies, amounts_or_side_effect=amounts)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        ceiling = compute_portal_carrier_ceiling(
            pickup_location="Genève",
            dropoff_location="HUG",
        )

    assert ceiling.maximum_accepted_amount == 52.0
    assert ceiling.quoted_carrier_count == 3
    assert [q.amount for q in ceiling.quotes] == [52.0, 45.0, 38.4]
    assert ceiling.currency == "CHF"


def test_case2_ineligible_carrier_not_in_max():
    """Transporteur hors zone / non éligible : pas dans le MAX."""
    eligible = [_company(1, "A"), _company(2, "B")]
    excluded_zone = [
        ExcludedCarrier(company_id=99, company_name="Hors zone", reason="not_in_zone")
    ]
    amounts = [
        (Decimal("45.00"), {"model": "flat"}),
        (Decimal("38.40"), {"model": "distance"}),
    ]
    with (
        patch(
            "services.pricing.portal_carrier_ceiling._eligible_companies",
            return_value=(eligible, excluded_zone),
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
            return_value=(_profile(), _version()),
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._build_pricing_context",
            return_value={"distance_km": 12.0},
        ),
        patch(
            "services.pricing.portal_carrier_ceiling.compute_price",
            side_effect=amounts,
        ),
    ):
        ceiling = compute_portal_carrier_ceiling(
            pickup_location="Genève",
            dropoff_location="HUG",
        )

    assert ceiling.maximum_accepted_amount == 45.0
    assert all(q.company_id != 99 for q in ceiling.quotes)
    assert any(e.company_id == 99 for e in ceiling.excluded)


def test_case3_no_grid_or_non_positive_excluded_never_silent_zero():
    """Sans grille / quote ≤ 0 → exclu ; pas de plafond 0 inventé."""
    companies = [
        _company(1, "Sans grille"),
        _company(2, "Quote zero"),
        _company(3, "OK"),
    ]

    def fake_active(company_id: int):
        if company_id == 1:
            return None, None
        return _profile(), _version()

    # Ordre d'appel après skip company 1 : company 2 puis 3
    price_results = [
        (Decimal("0"), {"model": "flat"}),  # company 2 → exclu non_positive
        (Decimal("52.00"), {"model": "zone"}),  # company 3
    ]

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
            return_value={"distance_km": 12.0},
        ),
        patch(
            "services.pricing.portal_carrier_ceiling.compute_price",
            side_effect=price_results,
        ),
    ):
        ceiling = compute_portal_carrier_ceiling(
            pickup_location="Genève",
            dropoff_location="HUG",
        )

    assert ceiling.maximum_accepted_amount == 52.0
    assert ceiling.quoted_carrier_count == 1
    reasons = {e.reason for e in ceiling.excluded}
    assert "no_active_pricing_profile" in reasons
    assert "non_positive_quote" in reasons
    assert ceiling.maximum_accepted_amount != 0.0


def test_case4_no_quotes_raises_explicit_error():
    """Aucune quote → portal_pricing_ceiling_unavailable (pas de plafond inventé)."""
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


def test_case3b_all_non_positive_also_raises():
    companies = [_company(1, "Zero")]
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
            return_value=1000,
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._active_profile_version",
            return_value=(_profile(), _version()),
        ),
        patch(
            "services.pricing.portal_carrier_ceiling._build_pricing_context",
            return_value={},
        ),
        patch(
            "services.pricing.portal_carrier_ceiling.compute_price",
            return_value=(Decimal("0"), {"model": "flat"}),
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


def test_case6_evidence_snapshot_is_self_contained():
    """La preuve figée contient le plafond ; un recalcul ultérieur ne la mute pas."""
    ceiling = PortalCarrierCeiling(
        maximum_accepted_amount=52.0,
        currency="CHF",
        pricing_calculated_at="2026-09-24T15:00:00+00:00",
        eligible_carrier_count=3,
        quoted_carrier_count=3,
        quotes=[
            CarrierQuote(3, "C", 52.0, "CHF", 10, 20, "zone"),
            CarrierQuote(1, "A", 45.0, "CHF", 11, 21, "flat"),
            CarrierQuote(2, "B", 38.4, "CHF", 12, 22, "distance"),
        ],
        excluded=[],
        distance_meters=12000,
    )
    frozen = ceiling.to_evidence_dict()
    # Simulation : une grille change après coup — l'objet evidence reste inchangé
    frozen_copy = {
        "pricing_ceiling": dict(frozen["pricing_ceiling"]),
    }
    frozen_copy["pricing_ceiling"]["quotes"] = list(frozen["pricing_ceiling"]["quotes"])
    # Muter un nouveau calcul n'altère pas la copie figée
    new_ceiling = PortalCarrierCeiling(
        maximum_accepted_amount=60.0,
        currency="CHF",
        pricing_calculated_at="2026-09-25T10:00:00+00:00",
        eligible_carrier_count=3,
        quoted_carrier_count=3,
        quotes=[CarrierQuote(3, "C", 60.0, "CHF", 10, 20, "zone")],
        excluded=[],
        distance_meters=12000,
    )
    assert frozen["pricing_ceiling"]["maximum_accepted_amount"] == 52.0
    assert new_ceiling.maximum_accepted_amount == 60.0
    assert frozen_copy["pricing_ceiling"]["maximum_accepted_amount"] == 52.0


def test_eligible_companies_excludes_unapproved_keeps_manual():
    """is_approved + profil actif ; dispatch_enabled/MANUAL n'exclut pas."""
    from services.pricing.portal_carrier_ceiling import _eligible_companies

    c_ok = _company(1, "OK")
    c_manual = _company(3, "Manuel", dispatch=False)
    c_no = _company(2, "Non approuvé", approved=False)

    priced = [c_ok, c_manual, c_no]
    with (
        patch("services.pricing.portal_carrier_ceiling.Company") as CompanyMock,
        patch(
            "services.pricing.portal_carrier_ceiling.PricingProfile",
            create=True,
        ),
    ):
        # Chaîne .join().filter().distinct().all()
        CompanyMock.query.join.return_value.filter.return_value.distinct.return_value.all.return_value = priced
        eligible, excluded = _eligible_companies(pickup_geo=None, dropoff_geo=None)

    assert {c.id for c in eligible} == {1, 3}
    reasons = {e.company_id: e.reason for e in excluded}
    assert reasons[2] == "not_approved"
    assert 3 not in reasons
