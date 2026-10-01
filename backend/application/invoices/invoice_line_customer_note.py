"""Note de ligne facture visible par le client — source unique HTML / PDF.

Champ canonique : ``InvoiceLine.adjustment_note`` (déjà exposé à
``InvoiceLivePreview`` via ``line.adjustment_note``). On ne crée pas de
``customer_visible_note`` parallèle : la sémantique actuelle est celle du
destinataire (éditeur brouillon, aperçu, facture).

Ne jamais imprimer depuis ce module :
``Booking.notes_medical``, motifs internes d'annulation, ``Invoice.notes``
(niveau facture), commentaires chauffeur / institution.
"""

from __future__ import annotations

from typing import Any

CANONICAL_CLIENT_VISIBLE_FIELD = "adjustment_note"


def normalize_customer_visible_note(raw: Any) -> str | None:
    """``None`` si vide ou uniquement des espaces ; texte persisté sinon (sans trim interne)."""
    if raw is None:
        return None
    text = str(raw)
    if not text.strip():
        return None
    return text.strip()


def collect_customer_visible_notes(lines: list[Any]) -> list[str]:
    """Notes client-visibles distinctes, ordre des lignes sources, doublons exacts exclus.

    Plusieurs notes différentes sont toutes conservées. Pas de concaténation
    « première seulement ». Le montant de la ligne n'intervient pas.
    """
    notes: list[str] = []
    seen: set[str] = set()
    for line in lines:
        raw = (
            line.get(CANONICAL_CLIENT_VISIBLE_FIELD)
            if isinstance(line, dict)
            else getattr(line, CANONICAL_CLIENT_VISIBLE_FIELD, None)
        )
        note = normalize_customer_visible_note(raw)
        if note is None or note in seen:
            continue
        seen.add(note)
        notes.append(note)
    return notes


def source_lines_from_consolidated_item(item: dict[str, Any]) -> list[Any]:
    """Lignes d'origine d'un item PDF : primaire (`line1`) puis retour (`line2`) puis `line`.

    Un item A/R fusionné n'a souvent que ``line`` malgré ``is_round_trip=True``.
    Un aller simple n'a que ``line``. On ne se fie pas au flag ``is_round_trip``.
    """
    ordered: list[Any] = []
    seen_ids: set[int] = set()
    seen_objs: list[int] = []

    def _add(line: Any) -> None:
        if line is None:
            return
        line_id = getattr(line, "id", None)
        if isinstance(line, dict):
            line_id = line.get("id")
        if line_id is not None:
            try:
                key = int(line_id)
            except (TypeError, ValueError):
                key = None
            if key is not None:
                if key in seen_ids:
                    return
                seen_ids.add(key)
                ordered.append(line)
                return
        obj_key = id(line)
        if obj_key in seen_objs:
            return
        seen_objs.append(obj_key)
        ordered.append(line)

    _add(item.get("line1"))
    _add(item.get("line2"))
    _add(item.get("line"))
    return ordered


def collect_notes_from_consolidated_item(item: dict[str, Any]) -> list[str]:
    """Notes client-visibles d'une prestation consolidée, contrat déterministe."""
    return collect_customer_visible_notes(source_lines_from_consolidated_item(item))


def partner_line_for_customer_note(
    line: Any, all_lines: list[Any] | None
) -> Any | None:
    """Jambe partenaire (paire deux lignes) si le méta la pointe — hors logique A/R client."""
    meta = None
    if isinstance(line, dict):
        meta = line.get("line_meta")
    else:
        meta = getattr(line, "line_meta", None)
    if isinstance(meta, str):
        import json

        try:
            parsed = json.loads(meta)
        except (TypeError, ValueError):
            parsed = None
        meta = parsed if isinstance(parsed, dict) else None
    if not isinstance(meta, dict):
        return None
    partner_rid = meta.get("round_trip_merge_partner_reservation_id")
    if partner_rid is None:
        return None
    try:
        partner_id = int(partner_rid)
    except (TypeError, ValueError):
        return None
    for other in all_lines or []:
        raw_rid = (
            other.get("reservation_id")
            if isinstance(other, dict)
            else getattr(other, "reservation_id", None)
        )
        try:
            if raw_rid is not None and int(raw_rid) == partner_id:
                return other
        except (TypeError, ValueError):
            continue
    return None


def collect_customer_visible_notes_for_preview(
    line: Any, all_lines: list[Any] | None = None
) -> list[str]:
    """Notes à afficher sous une ligne d'aperçu (ligne + jambe partenaire si présente)."""
    sources = [line]
    partner = partner_line_for_customer_note(line, all_lines)
    if partner is not None and partner is not line:
        sources.append(partner)
    return collect_customer_visible_notes(sources)
