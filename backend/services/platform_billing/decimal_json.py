"""Sérialisation monétaire JSON en chaînes décimales (jamais float)."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


def decimal_to_str(value: Decimal | None, places: int = 2) -> str | None:
    if value is None:
        return None
    q = Decimal("1").scaleb(-places)
    return str(value.quantize(q))


def decimal_to_str_trim(value: Decimal | None, places: int = 4) -> str | None:
    """Comme decimal_to_str, sans zéros décimaux inutiles (1.0000 → 1)."""
    text = decimal_to_str(value, places=places)
    if text is None:
        return None
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def parse_decimal(
    raw: Any,
    *,
    field: str,
    min_value: Decimal | None = None,
    max_value: Decimal | None = None,
    allow_none: bool = True,
) -> Decimal | None:
    if raw is None or raw == "":
        if allow_none:
            return None
        raise ValueError(f"{field} est requis")
    try:
        d = Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError) as e:
        raise ValueError(f"{field} invalide") from e
    if min_value is not None and d < min_value:
        raise ValueError(f"{field} doit être >= {min_value}")
    if max_value is not None and d > max_value:
        raise ValueError(f"{field} doit être <= {max_value}")
    return d
