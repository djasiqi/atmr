from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


def _enum_value(x: Any) -> str:
    if x is None:
        return ""
    v = getattr(x, "value", None)
    if isinstance(v, str):
        return v
    return str(x)


def _set_enum_value(obj: Any, attr: str, value_str: str) -> None:
    current = getattr(obj, attr, None)
    enum_cls = getattr(current, "__class__", None)
    candidate_name = value_str.upper()
    if enum_cls is not None and hasattr(enum_cls, candidate_name):
        setattr(obj, attr, getattr(enum_cls, candidate_name))
        return
    setattr(obj, attr, value_str.lower())


class _DriverLike(Protocol):
    id: int | None
    driver_type: Any


@dataclass(frozen=True, slots=True)
class ToggleDriverTypeResult:
    ok: bool
    should_trigger_dispatch: bool = False


class ToggleDriverTypeUseCase:
    """Use-case Application: bascule REGULAR <-> EMERGENCY."""

    def execute(self, driver: _DriverLike) -> ToggleDriverTypeResult:
        cur = _enum_value(getattr(driver, "driver_type", None)).lower()
        if cur == "regular":
            _set_enum_value(driver, "driver_type", "EMERGENCY")
        else:
            _set_enum_value(driver, "driver_type", "REGULAR")
        return ToggleDriverTypeResult(ok=True, should_trigger_dispatch=True)
