"""Montants A/R : alignement apairage / aperçu période (amount_ht_fn)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

from application.invoices.round_trip_booking_pairs import (
    find_round_trip_merge_booking_pairs,
)


def _booking(
    bid: int,
    cid: int | None,
    t1: datetime,
    pu: str,
    do: str,
    amount: Decimal,
    *,
    user_id: int = 7001,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=bid,
        client_id=cid,
        user_id=user_id,
        scheduled_time=t1,
        pickup_location=pu,
        dropoff_location=do,
        amount=amount,
        estimated_amount=None,
        status="COMPLETED",
        parent_booking_id=None,
        is_return=False,
    )


def test_hub_pair_skipped_when_raw_amounts_differ_but_preview_amounts_match():
    """Sans amount_ht_fn, amount brut incompatible → pas de paire ; avec HT aperçu → fusion."""
    day_am = datetime(2026, 3, 15, 10, 0, 0)
    day_pm = datetime(2026, 3, 15, 14, 0, 0)
    foyer = "Foyer test, Route 1, Anières"
    coll = "Chemin des Ramiers 9, Collonge-Bellerive"
    clin = "Clinique test, Chemin 9, Anières"
    b1 = _booking(1, 42, day_am, foyer, coll, Decimal("12.00"))
    b2 = _booking(2, 42, day_pm, clin, foyer, Decimal("78.00"))

    assert find_round_trip_merge_booking_pairs([b1, b2]) == []

    pairs = find_round_trip_merge_booking_pairs(
        [b1, b2],
        amount_ht_fn=lambda _b: Decimal("45.00"),
    )
    assert pairs == [(1, 2)]


def test_three_leg_route_group_is_not_merged_as_round_trip():
    """A→B, B→C, C→A : ni le parent du retour, ni la chaîne du même jour."""
    group = "18b0975a-221b-455c-a526-f5574b4122fc"
    day = datetime(2026, 9, 29, 22, 50, 0)
    later = datetime(2026, 9, 29, 23, 45, 0)
    pictet = "Avenue Ernest-Pictet 9, 1203, Genève"
    hug = "Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4, 1205 Genève"
    joli = "Clinique de Joli-Mont, Avenue Trembley 45, 1209, Genève"
    ab = _booking(46797, 42, day, pictet, hug, Decimal("40.00"))
    bc = _booking(46798, 42, later, hug, joli, Decimal("40.00"))
    ca = _booking(46799, 42, later, joli, pictet, Decimal("40.00"))
    ca.scheduled_time = None
    ca.parent_booking_id = 46798
    ca.is_return = True
    for booking in (ab, bc, ca):
        booking.route_group_id = group
    ab.is_round_trip = True
    assert find_round_trip_merge_booking_pairs([ab, bc, ca]) == []


def test_two_leg_parent_round_trip_still_merges():
    day = datetime(2026, 9, 29, 10, 0, 0)
    back = datetime(2026, 9, 29, 16, 0, 0)
    aller = _booking(10, 42, day, "Domicile", "Hôpital", Decimal("40.00"))
    retour = _booking(11, 42, back, "Hôpital", "Domicile", Decimal("40.00"))
    retour.parent_booking_id = 10
    retour.is_return = True
    aller.route_group_id = "pair-only"
    retour.route_group_id = "pair-only"
    assert find_round_trip_merge_booking_pairs([aller, retour]) == [(10, 11)]


def test_chain_segment_pair_dropoff_first_equals_pickup_second():
    """Clinique→foyer puis foyer→domicile : chaîne (pas inverse ni hub classique)."""
    day1 = datetime(2026, 3, 15, 11, 0, 0)
    day2 = datetime(2026, 3, 15, 14, 30, 0)
    clin = "Clinique les Hauts, Chemin des Courbes, Anières"
    foyer = "Foyer de jour Aux Cinq Colosses, Route d'Hermance, Anières"
    ramiers = "Chemin des Ramiers 9, 1245 Collonge-Bellerive"
    b1 = _booking(10, 42, day1, clin, foyer, Decimal("45.00"))
    b2 = _booking(11, 42, day2, foyer, ramiers, Decimal("45.00"))
    pairs = find_round_trip_merge_booking_pairs([b1, b2])
    assert pairs == [(10, 11)]


def test_hub_pair_when_client_id_null_groups_by_user_id():
    """Sans client_id, le groupement utilise user_id — indispensable pour l’aperçu période / clinique."""
    day_am = datetime(2026, 3, 15, 10, 0, 0)
    day_pm = datetime(2026, 3, 15, 14, 0, 0)
    foyer = "Foyer test, Route 1, Anières"
    coll = "Chemin des Ramiers 9, Collonge-Bellerive"
    clin = "Clinique test, Chemin 9, Anières"
    b1 = _booking(1, None, day_am, foyer, coll, Decimal("45.00"), user_id=42)
    b2 = _booking(2, None, day_pm, clin, foyer, Decimal("45.00"), user_id=42)
    pairs = find_round_trip_merge_booking_pairs([b1, b2])
    assert pairs == [(1, 2)]
