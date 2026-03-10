from __future__ import annotations

HIGH_PRIORITY_THRESHOLD = 70
MEDIUM_PRIORITY_THRESHOLD = 40


def compute_demo_score(payload: dict[str, str | None]) -> int:
    score = 0

    timing = payload.get("timing")
    if timing == "immediate":
        score += 40
    elif timing == "one_three_months":
        score += 25
    elif timing == "three_plus_months":
        score += 10
    elif timing == "exploration":
        score -= 20

    volume = payload.get("volume_range")
    if volume == "20_100":
        score += 20
    elif volume == "100_plus":
        score += 30
    elif volume == "5_20":
        score += 10

    org_type = payload.get("organization_type")
    if org_type in {"transport_company", "hospital", "clinic"}:
        score += 20
    elif org_type in {"ems", "curatorship"}:
        score += 10

    if payload.get("integration_required") == "yes":
        score += 10

    return max(score, 0)


def derive_demo_priority(score: int) -> str:
    if score >= HIGH_PRIORITY_THRESHOLD:
        return "high"
    if score >= MEDIUM_PRIORITY_THRESHOLD:
        return "medium"
    return "standard"
