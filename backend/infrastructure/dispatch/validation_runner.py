from __future__ import annotations

from typing import Any


def validate_dispatch_assignments(
    assignments: list[Any], *, strict: bool = False
) -> dict[str, Any]:
    """Adapter Infrastructure autour de
    `services.unified_dispatch.validation.validate_assignments`."""
    from services.unified_dispatch.validation.constraints import validate_assignments

    return validate_assignments(assignments, strict=strict)
