"""Normalisation linéaire des rues (espaces autour des tirets ASCII)."""


def collapse_hyphen_spaces(value: str) -> str:
    """Retire les espaces autour de « - » : Ernest- Pictet → Ernest-Pictet.

    Ne compacte pas les autres espaces. Complexité linéaire.
    """
    text = (value or "").strip()
    if not text:
        return ""
    return "-".join(part.strip() for part in text.split("-"))


def normalize_street_name(value: str) -> str:
    """Tirets collés + plages d'espaces compactées."""
    collapsed = collapse_hyphen_spaces(value)
    if not collapsed:
        return ""
    return " ".join(collapsed.split())
