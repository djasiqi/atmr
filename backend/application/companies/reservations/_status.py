from __future__ import annotations

from typing import Any


def status_value(x: Any) -> str:
    """Convertit un statut (Enum SQLAlchemy / str) vers une string comparable."""
    if x is None:
        return ""
    v = getattr(x, "value", None)
    if isinstance(v, str):
        return v
    return str(x)


def set_status(obj: Any, attr: str, status_name: str) -> None:
    """Assigne un statut sans dépendre d'un Enum ORM dans la couche Application."""
    current = getattr(obj, attr, None)
    enum_cls = getattr(current, "__class__", None)
    candidate = status_name.upper()
    if enum_cls is not None and hasattr(enum_cls, candidate):
        setattr(obj, attr, getattr(enum_cls, candidate))
        return
    setattr(obj, attr, candidate.lower())
