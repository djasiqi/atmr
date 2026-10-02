"""Contrat de colonnes du tableau de prestations (client, clinique, partenaire).

Une seule définition : le PDF institutionnel et le PDF partenaire lisent ces libellés.
Le mode ``pickup`` ajoute uniquement « Prise en charge ».
"""

from __future__ import annotations

LINE_TIME_NONE = "none"
LINE_TIME_PICKUP = "pickup"

HEADER_DATE = "Date"
HEADER_PICKUP = "Prise en charge"
HEADER_DESCRIPTION = "Description"
HEADER_AMOUNT = "Montant"

_MODES = frozenset({LINE_TIME_NONE, LINE_TIME_PICKUP})


def normalize_line_time_mode(value: object | None) -> str:
    """Mode explicite. Toute valeur inconnue reste le rendu historique sans heure."""
    text = str(value or "").strip()
    if text in _MODES:
        return text
    return LINE_TIME_NONE


def detail_headers(*, show_date: bool, line_time_mode: str) -> tuple[str, ...]:
    """Colonnes du détail. ``pickup`` insère une seule colonne entre Date et Description."""
    mode = normalize_line_time_mode(line_time_mode)
    headers: list[str] = []
    if show_date:
        headers.append(HEADER_DATE)
    if mode == LINE_TIME_PICKUP:
        headers.append(HEADER_PICKUP)
    headers.append(HEADER_DESCRIPTION)
    headers.append(HEADER_AMOUNT)
    return tuple(headers)
