from __future__ import annotations

from typing import Any


def validate_assignments(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.validation.constraints import (
        validate_assignments as _fn,
    )

    return _fn(*args, **kwargs)


def check_existing_assignment_conflict(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.validation.constraints import (
        check_existing_assignment_conflict as _fn,
    )

    return _fn(*args, **kwargs)
