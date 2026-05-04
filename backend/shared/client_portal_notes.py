"""Notes « transporteur » pour les demandes créées via le portail client.

Récapitulatif des champs côté modèle ``Booking`` (pour éviter les confusions) :

- ``client_note`` (API, max :attr:`CLIENT_PORTAL_FREE_NOTE_MAX_LENGTH`) : texte libre
  saisi par le client ; est fusionné dans ``notes_medical`` avec d'éventuels préfixes
  métier (occurrences, récurrence).
- ``medical_facility`` / ``doctor_name`` : colonnes dédiées (200 car. chacune) ;
  ne sont **pas** recopiées dans ``notes_medical`` par ce module.
- ``pickup_access_notes`` / ``dropoff_access_notes`` : réservés aux flux société /
  institution (saisie manuelle, offres) ; **absents** du schéma portail client.
- Profil client : ``access_notes`` (domicile) est distinct et n'est pas envoyé ici.

Voir :meth:`compose_client_portal_notes_medical` pour l'assemblage persisté.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Aligné sur ``BookingCreateSchema.client_note`` (Marshmallow).
CLIENT_PORTAL_FREE_NOTE_MAX_LENGTH = 500

# Texte assemblé (préfixes + message client) stocké dans ``Booking.notes_medical``.
# La colonne SQL est ``Text`` ; cette borne limite les exports / affichages aberrants.
NOTES_MEDICAL_ASSEMBLED_PORTAL_MAX_LENGTH = 4000


def _truncate_with_ellipsis(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len == 1:
        return "…"
    return f"{text[: max_len - 1].rstrip()}…"


def compose_client_portal_notes_medical(validated_data: dict[str, Any]) -> str | None:
    """Construit ``notes_medical`` à partir du portail client (occurrences, récurrence, note).

    Les lignes « métier » sont préservées ; seul le corps ``client_note`` est raccourci
    si le total dépasse :attr:`NOTES_MEDICAL_ASSEMBLED_PORTAL_MAX_LENGTH`.
    """
    meta_lines: list[str] = []
    try:
        occ = int(validated_data.get("occurrences") or 1)
    except (TypeError, ValueError):
        occ = 1
    if occ > 1:
        meta_lines.append(f"Occurrences demandées (même trajet) : {occ}")
    if validated_data.get("is_recurring"):
        rtype = (validated_data.get("recurrence_type") or "").strip()
        rlen = validated_data.get("recurrence_series_length")
        rend = (validated_data.get("recurrence_end_date") or "").strip()
        rdays = validated_data.get("recurrence_days") or []
        line = (
            f"Récurrence demandée (portail client) : type={rtype or '?'}, "
            f"répétitions prévues={rlen}"
        )
        if rend:
            line += f", jusqu'au {rend}"
        if rtype == "custom" and rdays:
            line += f", jours 0=lun..6=dim : {','.join(str(int(d)) for d in rdays)}"
        line += (
            " — une réservation est créée par cette demande ; "
            "série à confirmer / reproduire côté transporteur."
        )
        meta_lines.append(line)

    client_note = (validated_data.get("client_note") or "").strip()
    if not meta_lines and not client_note:
        return None

    max_total = NOTES_MEDICAL_ASSEMBLED_PORTAL_MAX_LENGTH
    meta_block = "\n".join(meta_lines)

    if not meta_block:
        out = _truncate_with_ellipsis(client_note, max_total)
        return out or None

    if len(meta_block) >= max_total:
        logger.warning(
            "client_portal_notes: meta_block alone exceeds max (%s > %s), hard truncate",
            len(meta_block),
            max_total,
        )
        return meta_block[:max_total]

    sep = "\n"
    overhead = len(meta_block) + (len(sep) if client_note else 0)
    room_for_note = max_total - overhead
    trimmed_note = (
        _truncate_with_ellipsis(client_note, room_for_note) if client_note else ""
    )

    if trimmed_note and len(client_note) > len(trimmed_note):
        logger.info(
            "client_portal_notes: client_note truncated for assembled notes_medical (%s -> %s chars)",
            len(client_note),
            len(trimmed_note),
        )

    if trimmed_note:
        return f"{meta_block}{sep}{trimmed_note}"
    return meta_block
