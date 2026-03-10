from __future__ import annotations


def compute_priority(payload: dict[str, object]) -> str:
    category = (payload.get("category") or "").strip()

    if category == "support" and payload.get("urgency") == "priority":
        return "high"

    if category == "demo" and payload.get("timing") == "immediate":
        return "high"

    if category == "institution" and payload.get("integration_required") == "yes":
        return "high"

    if category in ("demo", "institution", "transport"):
        return "medium"

    return "standard"
