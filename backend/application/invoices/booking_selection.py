"""Sélection déterministe des réservations à facturer (P7-L01).

Formule : éligibles ∩ retenues − exclues.
En mode strict, tout identifiant demandé non éligible provoque un refus explicite.
Sans strict, le comportement historique (filtres silencieux / expansion A/R) reste possible.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence


class BookingSelectionRefusal(ValueError):
    """Refus de sélection. Le message est lisible côté API."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        message = detail or code
        super().__init__(message)


def normalize_id_list(raw: Sequence[Any] | None) -> list[int]:
    """IDs positifs uniques, ordre d'apparition conservé."""
    if not raw:
        return []
    seen: set[int] = set()
    out: list[int] = []
    for item in raw:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def selection_fingerprint(booking_ids: Iterable[int]) -> str:
    """Empreinte stable des IDs retenus (contrôle prévisualisation ↔ émission)."""
    ordered = sorted({int(x) for x in booking_ids})
    payload = json.dumps(ordered, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_reservation_ids_from_client_map(
    client_reservations: Any,
    client_id: int | None,
) -> list[int] | None:
    """Extrait la whitelist patient depuis client_reservations si présente."""
    if not isinstance(client_reservations, dict) or client_id is None:
        return None
    keys = (client_id, str(client_id))
    for key in keys:
        if key in client_reservations:
            return normalize_id_list(client_reservations.get(key))
    return None


def apply_excluded(
    reservations: Sequence[Any],
    excluded_booking_ids: Sequence[int] | None,
) -> list[Any]:
    """Retire les IDs exclus. Applicable à tous les parcours."""
    excluded = set(normalize_id_list(excluded_booking_ids))
    if not excluded:
        return list(reservations)
    return [
        row
        for row in reservations
        if int(getattr(row, "id", 0) or 0) not in excluded
    ]


def enforce_strict_selection(
    *,
    requested_ids: Sequence[int],
    eligible_ids: Sequence[int],
    excluded_ids: Sequence[int] | None,
    final_ids: Sequence[int],
    confirmed_fingerprint: str | None,
) -> None:
    """Refuse toute sélection stricte incohérente (pas de facture partielle silencieuse)."""
    requested = set(normalize_id_list(requested_ids))
    if not requested:
        raise BookingSelectionRefusal(
            "selection_stricte_vide",
            "Sélection stricte vide : aucun transport retenu.",
        )
    excluded = set(normalize_id_list(excluded_ids))
    expected = requested - excluded
    if not expected:
        raise BookingSelectionRefusal(
            "selection_stricte_vide",
            "Sélection stricte vide après exclusions.",
        )
    eligible = {int(x) for x in eligible_ids}
    missing = sorted(expected - eligible)
    if missing:
        raise BookingSelectionRefusal(
            "reservation_non_eligible",
            (
                "Réservation(s) non éligible(s) ou déjà facturée(s) dans la sélection "
                f"stricte : {', '.join(str(i) for i in missing)}."
            ),
        )
    final = {int(x) for x in final_ids}
    extra = sorted(final - expected)
    if extra:
        raise BookingSelectionRefusal(
            "selection_elargie",
            (
                "La facture aurait inclus des transports hors sélection stricte : "
                f"{', '.join(str(i) for i in extra)}."
            ),
        )
    dropped = sorted(expected - final)
    if dropped:
        raise BookingSelectionRefusal(
            "selection_incomplete",
            (
                "Transports retenus absents de la facture après filtres : "
                f"{', '.join(str(i) for i in dropped)}."
            ),
        )
    if confirmed_fingerprint:
        actual = selection_fingerprint(final)
        if actual != str(confirmed_fingerprint).strip().lower():
            raise BookingSelectionRefusal(
                "previsualisation_perimee",
                (
                    "La sélection a changé depuis la prévisualisation. "
                    "Relancez l'aperçu avant d'émettre."
                ),
            )
