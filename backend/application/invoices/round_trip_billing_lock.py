"""Verrouillage éligibilité facturation (FK propre + utilitaires A/R).

L'éligibilité finale « non déjà facturé » repose sur la claim active
(``active_invoice_claim`` / ``covered_booking_ids``), pas sur un blocage
heuristique de pairs DSU.

``filter_bookings_open_for_new_invoice_line`` délègue à ce filet claim.
Les helpers DSU / ``peer_blocked_booking_ids`` restent disponibles pour
d'autres usages, mais ne définissent plus l'ouverture preview/génération.
"""

from __future__ import annotations

# pyright: reportImportCycles=false
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

from application.invoices.active_invoice_claim import (
    BLOCKING_INVOICE_STATUSES_FOR_CLAIM,
)
from application.invoices.round_trip_booking_pairs import (
    find_round_trip_merge_booking_pairs,
    normalize_address_for_round_trip_comparison,
)
from models import Invoice, InvoiceLine

# Aligné sur active_invoice_claim (CANCELLED non bloquant).
_BLOCKING_INVOICE_STATUSES = BLOCKING_INVOICE_STATUSES_FOR_CLAIM

_MAX_HOURS_SAME_DAY_CLUSTER = 12
_MIN_ROUND_TRIP_GROUP_SIZE = 2


class _DSU:
    """Union-find pour composantes connexes."""

    def __init__(self, ids: list[int]) -> None:
        super().__init__()
        self._parent = {i: i for i in ids}

    def find(self, x: int) -> int:
        p = self._parent.get(x, x)
        if p != x:
            self._parent[x] = self.find(p)
        return self._parent.get(x, x)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _client_day_key(booking: Any) -> tuple[str, int | None] | None:
    cid = getattr(booking, "client_id", None)
    if cid is not None:
        return ("c", int(cid))
    uid = getattr(booking, "user_id", None)
    if uid is not None:
        return ("u", int(uid))
    return None


def _scheduled_dt(booking: Any) -> datetime | None:
    st = getattr(booking, "scheduled_time", None)
    return st if isinstance(st, datetime) else None


def _within_hours(a: Any, b: Any, max_hours: float) -> bool:
    da = _scheduled_dt(a)
    db = _scheduled_dt(b)
    if not da or not db:
        return True
    return abs((db - da).total_seconds() / 3600.0) <= max_hours


def _share_normalized_endpoint(a: Any, b: Any) -> bool:
    """Au moins une extrémité (prise ou dépose) normalisée en commun."""
    ends_a = {
        normalize_address_for_round_trip_comparison(
            getattr(a, "pickup_location", "") or ""
        ),
        normalize_address_for_round_trip_comparison(
            getattr(a, "dropoff_location", "") or ""
        ),
    } - {""}
    ends_b = {
        normalize_address_for_round_trip_comparison(
            getattr(b, "pickup_location", "") or ""
        ),
        normalize_address_for_round_trip_comparison(
            getattr(b, "dropoff_location", "") or ""
        ),
    } - {""}
    return bool(ends_a & ends_b)


def build_round_trip_billing_dsu(
    bookings: list[Any],
    *,
    amount_ht_fn: Callable[[Any], Decimal] | None = None,
) -> tuple[_DSU, dict[int, int]]:
    """Construit le DSU des segments appartenant à la même unité A/R logique.

    Retourne ``(dsu, id -> racine)`` après union de tous les membres.
    """
    ids: list[int] = []
    by_id: dict[int, Any] = {}
    for b in bookings:
        try:
            bid = int(b.id)
        except Exception:
            continue
        ids.append(bid)
        by_id[bid] = b
    if not ids:
        return _DSU([]), {}

    dsu = _DSU(ids)

    # 1) Parent / retour explicite
    for b in bookings:
        try:
            bid = int(b.id)
        except Exception:
            continue
        pid = getattr(b, "parent_booking_id", None)
        if pid is not None and int(pid) in by_id:
            dsu.union(bid, int(pid))

    # 2) Paires fusion affichage / facture (inverse, chaîne, hub…)
    for pri, sec in find_round_trip_merge_booking_pairs(
        bookings,
        amount_ht_fn=amount_ht_fn,
    ):
        if pri in by_id and sec in by_id:
            dsu.union(int(pri), int(sec))

    # 3) Même jour + même patient / utilisateur : segments qui partagent une extrémité (réseau clinique / foyer)
    by_bucket: dict[tuple[Any, str], list[Any]] = defaultdict(list)
    for b in bookings:
        ck = _client_day_key(b)
        st = _scheduled_dt(b)
        if ck is None or st is None:
            continue
        date_key = st.strftime("%Y-%m-%d")
        by_bucket[(ck, date_key)].append(b)

    for _key, lst in by_bucket.items():
        if len(lst) < _MIN_ROUND_TRIP_GROUP_SIZE:
            continue
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                bi, bj = lst[i], lst[j]
                if not _within_hours(bi, bj, _MAX_HOURS_SAME_DAY_CLUSTER):
                    continue
                if _share_normalized_endpoint(bi, bj):
                    try:
                        dsu.union(int(bi.id), int(bj.id))
                    except Exception:
                        continue

    roots = {i: dsu.find(i) for i in ids}
    return dsu, roots


def booking_has_blocking_invoice_line(booking: Any) -> bool:
    """True si la réservation est liée à une ligne sur une facture non annulée."""
    from ext import db

    lid = getattr(booking, "invoice_line_id", None)
    if lid is None:
        return False
    try:
        lid_int = int(lid)
    except (TypeError, ValueError):
        return False
    line = db.session.get(InvoiceLine, lid_int)
    if line is None:
        return False
    inv = db.session.get(Invoice, line.invoice_id)
    if inv is None:
        return False
    return inv.status in _BLOCKING_INVOICE_STATUSES


def booking_open_for_new_invoice_line(booking: Any) -> bool:
    """Peut recevoir une nouvelle ligne (pas de lien ou facture source annulée)."""
    return not booking_has_blocking_invoice_line(booking)


def round_trip_component_id_sets(
    bookings: list[Any],
    *,
    amount_ht_fn: Callable[[Any], Decimal] | None = None,
) -> list[set[int]]:
    """Composantes connexes (unités A/R) : chaque ensemble a au moins 1 booking."""
    if not bookings:
        return []
    _, roots = build_round_trip_billing_dsu(bookings, amount_ht_fn=amount_ht_fn)
    by_root: dict[int, set[int]] = defaultdict(set)
    for b in bookings:
        try:
            bid = int(b.id)
        except Exception:
            continue
        r = roots.get(bid, bid)
        by_root[r].add(bid)
    return list(by_root.values())


def peer_blocked_booking_ids(
    bookings: list[Any],
    *,
    amount_ht_fn: Callable[[Any], Decimal] | None = None,
) -> set[int]:
    """IDs a traiter comme deja couverts par la facturation d'un autre segment du groupe."""
    _, roots = build_round_trip_billing_dsu(bookings, amount_ht_fn=amount_ht_fn)
    by_root: dict[int, list[Any]] = defaultdict(list)
    for b in bookings:
        try:
            bid = int(b.id)
        except Exception:
            continue
        r = roots.get(bid, bid)
        by_root[r].append(b)

    blocked: set[int] = set()
    for _root, members in by_root.items():
        if any(booking_has_blocking_invoice_line(m) for m in members):
            for m in members:
                try:
                    blocked.add(int(m.id))
                except Exception:
                    continue
    return blocked


def filter_bookings_open_for_new_invoice_line(
    bookings: list[Any],
    *,
    amount_ht_fn: Callable[[Any], Decimal] | None = None,
) -> list[Any]:
    """Garde les réservations encore facturables.

    - propre ``invoice_line_id`` non bloquant (ou facture source annulée) ;
    - non revendiquées explicitement par une InvoiceLine active
      (``covered_booking_ids``), même si la FK Booking est NULL.

    ``amount_ht_fn`` est conservé pour compatibilité d'appel ; le DSU pair
    n'est plus utilisé ici (évite faux blocage après single-leg volontaire).
    """
    _ = amount_ht_fn  # compat signature ; non utilisé pour la décision claim
    if not bookings:
        return []
    from application.invoices.active_invoice_claim import (
        filter_bookings_open_and_unclaimed,
    )

    return filter_bookings_open_and_unclaimed(bookings)
