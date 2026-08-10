"""Résolveur d'unités de facturation (simple / A/R strict).

Politique facturation (pas l'heuristique hub/chaîne d'affichage) :
1. parent_booking_id explicite, même subject_key ;
2. route_group_id avec exactement 2 segments aller+retour, même sujet ;
3. fallback A→B / B→A strict, même sujet, même jour de service ;
4. jamais A→B→C, hub, composante > 2, cross-subject.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from application.invoices.round_trip_booking_pairs import (
    normalize_address_for_round_trip_comparison,
)
from application.invoices.subject_identity import resolve_subject_identity

UnitKind = Literal["single", "round_trip"]


@dataclass(frozen=True)
class BookingUnit:
    unit_key: str
    kind: UnitKind
    primary_booking_id: int
    booking_ids: tuple[int, ...]
    subject_key: str
    billing_party_id: int | None
    amount_ht: Decimal
    period_anchor_booking_id: int


def _bid(b: Any) -> int:
    return int(b.id)


def _subject_key(b: Any, subject_key_fn: Callable[[Any], str] | None) -> str:
    if subject_key_fn is not None:
        return subject_key_fn(b)
    return resolve_subject_identity(b).key


def _amount(b: Any, amount_ht_fn: Callable[[Any], Decimal] | None) -> Decimal:
    if amount_ht_fn is not None:
        return Decimal(str(amount_ht_fn(b)))
    return Decimal(str(getattr(b, "amount", None) or 0))


def _sched(b: Any) -> datetime | None:
    st = getattr(b, "scheduled_time", None)
    return st if isinstance(st, datetime) else None


def _service_day(b: Any) -> str | None:
    st = _sched(b)
    if st is None:
        return None
    return st.strftime("%Y-%m-%d")


def _is_strict_reverse(a: Any, b: Any) -> bool:
    a_pick = normalize_address_for_round_trip_comparison(
        getattr(a, "pickup_location", "") or ""
    )
    a_drop = normalize_address_for_round_trip_comparison(
        getattr(a, "dropoff_location", "") or ""
    )
    b_pick = normalize_address_for_round_trip_comparison(
        getattr(b, "pickup_location", "") or ""
    )
    b_drop = normalize_address_for_round_trip_comparison(
        getattr(b, "dropoff_location", "") or ""
    )
    if not (a_pick and a_drop and b_pick and b_drop):
        return False
    return a_pick == b_drop and a_drop == b_pick


def _order_pair(a: Any, b: Any) -> tuple[Any, Any]:
    """Retourne (primary, secondary) — primaire = non-retour / plus tôt."""
    a_ret = bool(getattr(a, "is_return", False))
    b_ret = bool(getattr(b, "is_return", False))
    if a_ret and not b_ret:
        return b, a
    if b_ret and not a_ret:
        return a, b
    sa, sb = _sched(a), _sched(b)
    if sa is not None and sb is not None and sa != sb:
        return (a, b) if sa <= sb else (b, a)
    return (a, b) if _bid(a) <= _bid(b) else (b, a)


def _same_billing_destination(a: Any, b: Any) -> bool:
    if (getattr(a, "billed_to_type", None) or "") != (
        getattr(b, "billed_to_type", None) or ""
    ):
        return False
    bpa = getattr(a, "billing_party_id", None)
    bpb = getattr(b, "billing_party_id", None)
    if bpa is not None or bpb is not None:
        return bpa == bpb
    return True


def expand_explicit_peer_ids(
    selected_ids: set[int],
    scope_bookings: list[Any],
    *,
    extra_bookings: list[Any] | None = None,
) -> set[int]:
    """Étend la sélection aux parents/enfants explicites présents dans le scope élargi."""
    by_id: dict[int, Any] = {}
    for b in list(scope_bookings) + list(extra_bookings or []):
        try:
            by_id[_bid(b)] = b
        except Exception:
            continue
    expanded = set(selected_ids)
    # parent -> child
    for bid in list(expanded):
        b = by_id.get(bid)
        if b is None:
            continue
        pid = getattr(b, "parent_booking_id", None)
        if pid is not None and int(pid) in by_id:
            expanded.add(int(pid))
    # child of selected parents
    for bid, b in by_id.items():
        pid = getattr(b, "parent_booking_id", None)
        if pid is not None and int(pid) in expanded:
            expanded.add(bid)
    return expanded


def collect_explicit_peer_ids_to_load(bookings: list[Any]) -> set[int]:
    """IDs de pairs explicites à charger hors filtre période."""
    needed: set[int] = set()
    present = {_bid(b) for b in bookings if getattr(b, "id", None) is not None}
    for b in bookings:
        try:
            _bid(b)
        except Exception:
            continue
        pid = getattr(b, "parent_booking_id", None)
        if pid is not None and int(pid) not in present:
            needed.add(int(pid))
        # enfants : chargés séparément via query parent_booking_id IN present
    return needed


def resolve_invoice_booking_units(
    *,
    selected_ids: set[int] | None,
    scope_bookings: list[Any],
    subject_key_fn: Callable[[Any], str] | None = None,
    amount_ht_fn: Callable[[Any], Decimal] | None = None,
    expand_explicit_peers: bool = True,
) -> list[BookingUnit]:
    """Construit les unités facturables à partir du scope (déjà élargi hors période si besoin)."""
    by_id: dict[int, Any] = {}
    for b in scope_bookings:
        try:
            by_id[_bid(b)] = b
        except Exception:
            continue
    if not by_id:
        return []

    work_ids = set(by_id.keys())
    if selected_ids is not None:
        work_ids = set(selected_ids) & set(by_id.keys())
        if expand_explicit_peers:
            work_ids = expand_explicit_peer_ids(work_ids, list(by_id.values()))
            work_ids &= set(by_id.keys())

    used: set[int] = set()
    units: list[BookingUnit] = []

    # 1) Liens parent explicites
    for bid in sorted(work_ids):
        if bid in used:
            continue
        b = by_id[bid]
        pid = getattr(b, "parent_booking_id", None)
        if pid is None:
            continue
        pid_i = int(pid)
        if pid_i not in work_ids or pid_i not in by_id:
            continue
        parent = by_id[pid_i]
        sk_a = _subject_key(b, subject_key_fn)
        sk_p = _subject_key(parent, subject_key_fn)
        if sk_a != sk_p or not _same_billing_destination(b, parent):
            continue
        primary, secondary = _order_pair(parent, b)
        ids = (_bid(primary), _bid(secondary))
        if ids[0] in used or ids[1] in used:
            continue
        amt = _amount(primary, amount_ht_fn) + _amount(secondary, amount_ht_fn)
        bp = getattr(primary, "billing_party_id", None)
        units.append(
            BookingUnit(
                unit_key=f"unit:round_trip:{ids[0]}:{ids[1]}",
                kind="round_trip",
                primary_booking_id=ids[0],
                booking_ids=ids,
                subject_key=sk_p,
                billing_party_id=int(bp) if bp is not None else None,
                amount_ht=amt,
                period_anchor_booking_id=ids[0],
            )
        )
        used.add(ids[0])
        used.add(ids[1])

    # 2) route_group_id exactement 2 segments
    by_rg: dict[str, list[Any]] = defaultdict(list)
    for bid in work_ids:
        if bid in used:
            continue
        b = by_id[bid]
        rg = getattr(b, "route_group_id", None)
        if rg is None:
            continue
        by_rg[str(rg)].append(b)
    for _rg, segs in by_rg.items():
        if len(segs) != 2:
            continue
        a, b = segs[0], segs[1]
        if _bid(a) in used or _bid(b) in used:
            continue
        sk_a = _subject_key(a, subject_key_fn)
        sk_b = _subject_key(b, subject_key_fn)
        if sk_a != sk_b or not _same_billing_destination(a, b):
            continue
        # Un aller + un retour (is_return) ou inversion stricte
        returns = sum(1 for s in segs if bool(getattr(s, "is_return", False)))
        if returns != 1 and not _is_strict_reverse(a, b):
            continue
        primary, secondary = _order_pair(a, b)
        ids = (_bid(primary), _bid(secondary))
        amt = _amount(primary, amount_ht_fn) + _amount(secondary, amount_ht_fn)
        bp = getattr(primary, "billing_party_id", None)
        units.append(
            BookingUnit(
                unit_key=f"unit:round_trip:{ids[0]}:{ids[1]}",
                kind="round_trip",
                primary_booking_id=ids[0],
                booking_ids=ids,
                subject_key=sk_a,
                billing_party_id=int(bp) if bp is not None else None,
                amount_ht=amt,
                period_anchor_booking_id=ids[0],
            )
        )
        used.add(ids[0])
        used.add(ids[1])

    # 3) Fallback legacy reverse strict même jour + même sujet
    remaining = [by_id[i] for i in sorted(work_ids) if i not in used]
    paired: set[int] = set()
    for i, a in enumerate(remaining):
        if _bid(a) in paired:
            continue
        for b in remaining[i + 1 :]:
            if _bid(b) in paired:
                continue
            if _subject_key(a, subject_key_fn) != _subject_key(b, subject_key_fn):
                continue
            if not _same_billing_destination(a, b):
                continue
            if _service_day(a) != _service_day(b) or _service_day(a) is None:
                continue
            if not _is_strict_reverse(a, b):
                continue
            primary, secondary = _order_pair(a, b)
            ids = (_bid(primary), _bid(secondary))
            amt = _amount(primary, amount_ht_fn) + _amount(secondary, amount_ht_fn)
            bp = getattr(primary, "billing_party_id", None)
            sk = _subject_key(primary, subject_key_fn)
            units.append(
                BookingUnit(
                    unit_key=f"unit:round_trip:{ids[0]}:{ids[1]}",
                    kind="round_trip",
                    primary_booking_id=ids[0],
                    booking_ids=ids,
                    subject_key=sk,
                    billing_party_id=int(bp) if bp is not None else None,
                    amount_ht=amt,
                    period_anchor_booking_id=ids[0],
                )
            )
            paired.add(ids[0])
            paired.add(ids[1])
            used.add(ids[0])
            used.add(ids[1])
            break

    # 4) Singles
    for bid in sorted(work_ids):
        if bid in used:
            continue
        b = by_id[bid]
        sk = _subject_key(b, subject_key_fn)
        bp = getattr(b, "billing_party_id", None)
        units.append(
            BookingUnit(
                unit_key=f"unit:single:{bid}",
                kind="single",
                primary_booking_id=bid,
                booking_ids=(bid,),
                subject_key=sk,
                billing_party_id=int(bp) if bp is not None else None,
                amount_ht=_amount(b, amount_ht_fn),
                period_anchor_booking_id=bid,
            )
        )
        used.add(bid)

    units.sort(key=lambda u: u.primary_booking_id)
    return units
