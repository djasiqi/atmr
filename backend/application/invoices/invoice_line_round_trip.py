"""Structure A/R d'une ligne de facture — source de vérité unique (HTML + PDF).

Deux notions distinctes, à ne jamais confondre :

1. INFORMATION — la réservation a pu être créée en aller-retour
   (``billing_unit``, ``transport_type``, ``booking.is_round_trip``).
   Sert au badge éditeur, jamais au tag client ``[A/R]``.

2. STRUCTURE — cette ligne facture réellement aller + retour
   (réservations rattachées, ou paire deux lignes dont l'autre jambe est présente).
   Seule cette notion autorise ``[A/R]`` sur l'aperçu HTML et le PDF officiel.

Le montant (45 / 90 CHF, etc.) n'intervient dans aucune décision.
"""

from __future__ import annotations

import json
from typing import Any

STRUCTURE_SINGLE = "single"
STRUCTURE_MERGED_BOTH_LEGS = "merged_both_legs"
STRUCTURE_PAIR_PRIMARY = "pair_primary"
STRUCTURE_PAIR_RETURN = "pair_return"

VALID_STRUCTURES = frozenset(
    {
        STRUCTURE_SINGLE,
        STRUCTURE_MERGED_BOTH_LEGS,
        STRUCTURE_PAIR_PRIMARY,
        STRUCTURE_PAIR_RETURN,
    }
)


def parse_line_meta(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(raw, dict):
        return raw
    return None


def get_invoice_line_meta(line: Any) -> dict[str, Any] | None:
    if isinstance(line, dict):
        return parse_line_meta(line.get("line_meta"))
    return parse_line_meta(getattr(line, "line_meta", None))


def _line_reservation_id(line: Any) -> int | None:
    raw = (
        line.get("reservation_id")
        if isinstance(line, dict)
        else getattr(line, "reservation_id", None)
    )
    return _finite_id(raw)


def _finite_id(value: Any) -> int | None:
    if value is None or value is False:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _add_finite_id(target: set[int], value: Any) -> None:
    parsed = _finite_id(value)
    if parsed is not None:
        target.add(parsed)


def attached_round_trip_booking_ids(line: Any) -> list[int]:
    """Réservations distinctes réellement rattachées à CETTE ligne (méta persistée).

    Même résolution que le frontend ``attachedRoundTripBookingIds`` et que
    ``_split_single_merged_round_trip_line`` : ``booking_ids`` (≥ 2) sinon
    ``reservation_id`` + réservations secondaires. Le partenaire d'une paire
    deux lignes n'est pas compté : il vit sur une autre ligne.
    """
    meta = get_invoice_line_meta(line)
    if not meta:
        return []
    from_booking_ids: set[int] = set()
    raw_ids = meta.get("booking_ids")
    if isinstance(raw_ids, list):
        for item in raw_ids:
            _add_finite_id(from_booking_ids, item)
    if len(from_booking_ids) >= 2:
        return list(from_booking_ids)
    reservation_id = _line_reservation_id(line)
    if reservation_id is None:
        return list(from_booking_ids)
    with_secondary: set[int] = set()
    with_secondary.add(reservation_id)
    secondary = meta.get("round_trip_secondary_reservation_ids")
    if isinstance(secondary, list):
        for item in secondary:
            _add_finite_id(with_secondary, item)
    else:
        _add_finite_id(with_secondary, meta.get("round_trip_secondary_reservation_id"))
    return list(with_secondary) if len(with_secondary) >= 2 else list(from_booking_ids)


def invoice_line_has_both_round_trip_legs(line: Any) -> bool:
    """Cette ligne contient effectivement aller + retour (≥ 2 réservations rattachées)."""
    meta = get_invoice_line_meta(line)
    if not meta or meta.get("period_preview_single_leg"):
        return False
    return len(attached_round_trip_booking_ids(line)) >= 2


def find_invoice_line_by_reservation_id(
    lines: list[Any] | None, reservation_id: Any
) -> Any | None:
    wanted = _finite_id(reservation_id)
    if wanted is None or not lines:
        return None
    for line in lines:
        if _line_reservation_id(line) == wanted:
            return line
    return None


def get_round_trip_partner_line(line: Any, all_lines: list[Any] | None) -> Any | None:
    meta = get_invoice_line_meta(line)
    if not meta:
        return None
    partner = meta.get("round_trip_merge_partner_reservation_id")
    if partner is not None:
        return find_invoice_line_by_reservation_id(all_lines, partner)
    primary = meta.get("round_trip_merge_primary_reservation_id")
    if primary is not None:
        return find_invoice_line_by_reservation_id(all_lines, primary)
    return None


def round_trip_line_structure(line: Any, all_lines: list[Any] | None = None) -> str:
    """Nature structurelle d'une ligne vis-à-vis de l'aller-retour.

    Si ``all_lines`` est fourni, une paire deux lignes n'est reconnue que si
    l'autre jambe est réellement présente (méta de paire obsolète → ``single``).
    """
    meta = get_invoice_line_meta(line)
    if not meta:
        return STRUCTURE_SINGLE
    if meta.get("period_preview_single_leg"):
        return STRUCTURE_SINGLE
    if invoice_line_has_both_round_trip_legs(line):
        return STRUCTURE_MERGED_BOTH_LEGS
    partner_present = (
        all_lines is None or get_round_trip_partner_line(line, all_lines) is not None
    )
    if meta.get("round_trip_merge_partner_reservation_id") is not None:
        return STRUCTURE_PAIR_PRIMARY if partner_present else STRUCTURE_SINGLE
    if meta.get("preview_hide_merged_round_trip") is True:
        return STRUCTURE_PAIR_RETURN if partner_present else STRUCTURE_SINGLE
    return STRUCTURE_SINGLE


def invoice_line_represents_full_round_trip(
    line: Any, all_lines: list[Any] | None = None
) -> bool:
    """``[A/R]`` client : la ligne facture réellement une prestation aller-retour.

    Vrai pour ``merged_both_legs`` et ``pair_primary`` (partenaire présent).
    Faux pour ``single`` (y compris ``is_round_trip`` historique / ``billing_unit``)
    et pour ``pair_return`` (jambe masquée, le tag est sur la primaire).
    """
    structure = round_trip_line_structure(line, all_lines)
    return structure in (STRUCTURE_MERGED_BOTH_LEGS, STRUCTURE_PAIR_PRIMARY)


def enrich_line_dict_round_trip_structure(
    line_dicts: list[dict[str, Any]],
) -> None:
    """Pose les champs canoniques sur chaque payload ligne (après enrichissement méta)."""
    for payload in line_dicts:
        structure = round_trip_line_structure(payload, line_dicts)
        payload["invoice_line_round_trip_structure"] = structure
        payload["invoice_line_represents_full_round_trip"] = structure in (
            STRUCTURE_MERGED_BOTH_LEGS,
            STRUCTURE_PAIR_PRIMARY,
        )
