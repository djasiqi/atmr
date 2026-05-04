"""Détection des paires aller-retour pour bookings.

- Explicite : ``parent_booking_id``
- Inverse géographique : A→B et B→A
- Retour au hub : …→hub et hub→… (pas inverse strict)
- **Chaîne** : A→B puis B→C (dépose du 1er = prise du 2e), ex. clinique→foyer puis foyer→domicile
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

_AMOUNT_TOLERANCE_CHF = Decimal("5.00")
_MAX_ROUND_TRIP_TIME_WINDOW_HOURS = 12
_MIN_ROUND_TRIP_PAIR_GROUP = 2


def _normalize_address_for_comparison(address: str) -> str:
    if not address:
        return ""
    normalized = address.lower().strip()
    try:
        normalized = unicodedata.normalize("NFD", normalized)
        normalized = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    except Exception:
        pass
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r",+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _is_booking_cancelled(booking: Any) -> bool:
    if not booking:
        return False
    status_raw = getattr(booking, "status", None)
    if status_raw is None:
        return False
    status_str = getattr(status_raw, "value", None) or str(status_raw) or ""
    return status_str.upper().strip() in {"CANCELED", "CANCELLED"}


def find_round_trip_merge_booking_pairs(
    bookings: list[Any],
    *,
    amount_ht_fn: Callable[[Any], Decimal] | None = None,
) -> list[tuple[int, int]]:
    """Paires ``(booking_id_principal, booking_id_secondaire)`` à afficher comme une ligne A/R.

    - Explicite : ``parent_booking_id`` + parent présent dans la liste.
    - Heuristique : même patient (``client_id``), même jour, montants proches, fenêtre horaire.
    - Inverse strict : A→B et B→A.
    - Retour au hub : A→B et C→A (ex. domicile→activité puis clinique→domicile).

    ``principal`` = aller explicite ou segment chronologiquement premier pour la description.

    ``amount_ht_fn`` : si fourni (ex. montant HT d'apercu periode), utilise pour comparer les montants
    au lieu de ``booking.amount`` / ``estimated_amount`` — aligné PDF / lignes facture.
    """
    if len(bookings) < _MIN_ROUND_TRIP_PAIR_GROUP:
        return []
    by_id: dict[int, Any] = {}
    for b in bookings:
        try:
            by_id[int(b.id)] = b
        except Exception:
            continue
    used_ids: set[int] = set()
    pairs_out: list[tuple[int, int]] = []

    # --- 1) Liens explicites parent / retour ---
    explicit_children: dict[int, list[int]] = defaultdict(list)
    for b in bookings:
        pid = getattr(b, "parent_booking_id", None)
        if pid is not None and int(pid) in by_id:
            explicit_children[int(pid)].append(int(b.id))

    for parent_id, children in explicit_children.items():
        if len(children) != 1:
            continue
        rid = children[0]
        if parent_id in used_ids or rid in used_ids:
            continue
        pb, rb = by_id.get(parent_id), by_id.get(rid)
        if not pb or not rb:
            continue
        if _is_booking_cancelled(pb) or _is_booking_cancelled(rb):
            continue
        pairs_out.append((parent_id, rid))
        used_ids.add(parent_id)
        used_ids.add(rid)

    # --- 2) Groupes (jour + patient) : ``client_id`` si présent, sinon ``user_id`` (bookings sans client lié).
    groups: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for b in bookings:
        bid = int(b.id)
        if bid in used_ids:
            continue
        if _is_booking_cancelled(b):
            continue
        st = getattr(b, "scheduled_time", None)
        if st is None:
            continue
        if not isinstance(st, datetime):
            continue
        date_key = st.strftime("%Y-%m-%d")
        cid = getattr(b, "client_id", None)
        if cid is not None:
            groups[("c", int(cid), date_key)].append(b)
            continue
        uid = getattr(b, "user_id", None)
        if uid is not None:
            groups[("u", int(uid), date_key)].append(b)

    for _key, group_bookings in groups.items():
        if len(group_bookings) < _MIN_ROUND_TRIP_PAIR_GROUP:
            continue
        active = [b for b in group_bookings if int(b.id) not in used_ids]
        if len(active) < _MIN_ROUND_TRIP_PAIR_GROUP:
            continue

        normalized_pairs: list[dict[str, Any]] = []
        for b in active:
            pickup = getattr(b, "pickup_location", "") or ""
            dropoff = getattr(b, "dropoff_location", "") or ""
            if not pickup or not dropoff:
                continue
            if amount_ht_fn is not None:
                try:
                    amount_dec = amount_ht_fn(b)
                except Exception:
                    amount_dec = Decimal("0")
            else:
                amt = getattr(b, "amount", None) or getattr(b, "estimated_amount", None)
                try:
                    amount_dec = Decimal(str(amt or 0))
                except Exception:
                    amount_dec = Decimal("0")
            normalized_pairs.append(
                {
                    "booking": b,
                    "bid": int(b.id),
                    "pickup_norm": _normalize_address_for_comparison(pickup),
                    "dropoff_norm": _normalize_address_for_comparison(dropoff),
                    "pickup_orig": pickup,
                    "dropoff_orig": dropoff,
                    "amount": amount_dec,
                    "date": getattr(b, "scheduled_time", None),
                }
            )

        if len(normalized_pairs) < _MIN_ROUND_TRIP_PAIR_GROUP:
            continue

        matched_pairs: list[tuple[int, int]] = []
        used_idx: set[int] = set()

        by_route: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx_route, pr in enumerate(normalized_pairs):
            by_route[(pr["pickup_norm"], pr["dropoff_norm"])].append(idx_route)

        candidate_pairs: list[dict[str, Any]] = []
        for i, pair1 in enumerate(normalized_pairs):
            if i in used_idx:
                continue
            rev_key = (pair1["dropoff_norm"], pair1["pickup_norm"])
            for j in by_route.get(rev_key, []):
                if j <= i or j in used_idx:
                    continue
                pair2 = normalized_pairs[j]
                item1 = pair1
                item2 = pair2
                date1 = item1.get("date")
                date2 = item2.get("date")
                delta_seconds = float("inf")
                if (
                    date1
                    and date2
                    and isinstance(date1, datetime)
                    and isinstance(date2, datetime)
                ):
                    delta_seconds = abs((date2 - date1).total_seconds())
                candidate_pairs.append(
                    {
                        "idx1": i,
                        "idx2": j,
                        "pair1": pair1,
                        "pair2": pair2,
                        "delta_seconds": delta_seconds,
                    }
                )

        candidate_pairs.sort(key=lambda c: c["delta_seconds"])

        for candidate in candidate_pairs:
            idx1, idx2 = candidate["idx1"], candidate["idx2"]
            pair1, pair2 = candidate["pair1"], candidate["pair2"]
            amount1 = Decimal(str(pair1.get("amount", 0)))
            amount2 = Decimal(str(pair2.get("amount", 0)))
            if abs(amount1 - amount2) > _AMOUNT_TOLERANCE_CHF:
                continue
            date1 = pair1.get("date")
            date2 = pair2.get("date")
            if (
                date1
                and date2
                and isinstance(date1, datetime)
                and isinstance(date2, datetime)
                and abs((date2 - date1).total_seconds() / 3600)
                > _MAX_ROUND_TRIP_TIME_WINDOW_HOURS
            ):
                continue
            pickup1_norm = pair1["pickup_norm"]
            dropoff1_norm = pair1["dropoff_norm"]
            possible_returns = 0
            for other_pair in normalized_pairs:
                if (
                    other_pair["pickup_norm"] == dropoff1_norm
                    and other_pair["dropoff_norm"] == pickup1_norm
                ):
                    possible_returns += 1
            if possible_returns > 1:
                continue
            if idx1 in used_idx or idx2 in used_idx:
                continue
            matched_pairs.append((idx1, idx2))
            used_idx.add(idx1)
            used_idx.add(idx2)

        # Retour au hub (non couvert par l'inverse strict)
        unmatched_for_hub = [
            i for i in range(len(normalized_pairs)) if i not in used_idx
        ]
        hub_candidates: list[dict[str, Any]] = []
        for ii in range(len(unmatched_for_hub)):
            for jj in range(ii + 1, len(unmatched_for_hub)):
                ia = unmatched_for_hub[ii]
                ib = unmatched_for_hub[jj]
                pa = normalized_pairs[ia]
                pb = normalized_pairs[ib]
                for a_idx, b_idx, par_a, par_b in (
                    (ia, ib, pa, pb),
                    (ib, ia, pb, pa),
                ):
                    if (
                        par_b["dropoff_norm"] == par_a["pickup_norm"]
                        and par_b["pickup_norm"] != par_a["dropoff_norm"]
                    ):
                        date_a = par_a.get("date")
                        date_b = par_b.get("date")
                        if (
                            date_a
                            and date_b
                            and isinstance(date_a, datetime)
                            and isinstance(date_b, datetime)
                            and abs((date_b - date_a).total_seconds() / 3600)
                            > _MAX_ROUND_TRIP_TIME_WINDOW_HOURS
                        ):
                            continue
                        amount_a = Decimal(str(par_a.get("amount", 0)))
                        amount_b = Decimal(str(par_b.get("amount", 0)))
                        if abs(amount_a - amount_b) > _AMOUNT_TOLERANCE_CHF:
                            continue
                        d1 = par_a.get("date")
                        d2 = par_b.get("date")
                        delta_seconds = float("inf")
                        if (
                            d1
                            and d2
                            and isinstance(d1, datetime)
                            and isinstance(d2, datetime)
                        ):
                            delta_seconds = abs((d2 - d1).total_seconds())
                        hub_candidates.append(
                            {
                                "a_idx": a_idx,
                                "b_idx": b_idx,
                                "delta_seconds": delta_seconds,
                            }
                        )

        hub_candidates.sort(key=lambda c: c["delta_seconds"])
        for cand in hub_candidates:
            ai, bi = cand["a_idx"], cand["b_idx"]
            if ai in used_idx or bi in used_idx:
                continue
            matched_pairs.append((ai, bi))
            used_idx.add(ai)
            used_idx.add(bi)

        # Chaine : trajet 1 se termine la ou commence le trajet 2 (pas B->A inverse).
        chain_unmatched = [i for i in range(len(normalized_pairs)) if i not in used_idx]
        if len(chain_unmatched) >= _MIN_ROUND_TRIP_PAIR_GROUP:
            chain_candidates: list[dict[str, Any]] = []
            for ii in range(len(chain_unmatched)):
                for jj in range(ii + 1, len(chain_unmatched)):
                    ia = chain_unmatched[ii]
                    ib = chain_unmatched[jj]
                    pa = normalized_pairs[ia]
                    pb = normalized_pairs[ib]
                    if pa["dropoff_norm"] != pb["pickup_norm"]:
                        continue
                    if (
                        pa["pickup_norm"] == pb["dropoff_norm"]
                        and pa["dropoff_norm"] == pb["pickup_norm"]
                    ):
                        continue
                    amt_a = Decimal(str(pa.get("amount", 0)))
                    amt_b = Decimal(str(pb.get("amount", 0)))
                    if abs(amt_a - amt_b) > _AMOUNT_TOLERANCE_CHF:
                        continue
                    da = pa.get("date")
                    db = pb.get("date")
                    if (
                        da
                        and db
                        and isinstance(da, datetime)
                        and isinstance(db, datetime)
                        and abs((db - da).total_seconds() / 3600)
                        > _MAX_ROUND_TRIP_TIME_WINDOW_HOURS
                    ):
                        continue
                    delta_seconds = float("inf")
                    if (
                        da
                        and db
                        and isinstance(da, datetime)
                        and isinstance(db, datetime)
                    ):
                        delta_seconds = abs((db - da).total_seconds())
                    chain_candidates.append(
                        {
                            "ia": ia,
                            "ib": ib,
                            "delta_seconds": delta_seconds,
                        }
                    )
            chain_candidates.sort(key=lambda c: c["delta_seconds"])
            for cand in chain_candidates:
                ia, ib = cand["ia"], cand["ib"]
                if ia in used_idx or ib in used_idx:
                    continue
                matched_pairs.append((ia, ib))
                used_idx.add(ia)
                used_idx.add(ib)

        for idx1, idx2 in matched_pairs:
            p1 = normalized_pairs[idx1]
            p2 = normalized_pairs[idx2]
            id1 = p1["bid"]
            id2 = p2["bid"]
            if id1 in used_ids or id2 in used_ids:
                continue
            d1 = p1.get("date")
            d2 = p2.get("date")
            if d1 and d2 and isinstance(d1, datetime) and isinstance(d2, datetime):
                if d1 <= d2:
                    pairs_out.append((id1, id2))
                else:
                    pairs_out.append((id2, id1))
            else:
                pairs_out.append((id1, id2))
            used_ids.add(id1)
            used_ids.add(id2)

    return pairs_out


# Alias public pour le verrouillage d'eligibilite (meme normalisation que la fusion A/R).
normalize_address_for_round_trip_comparison = _normalize_address_for_comparison
