"""Legacy thread_id helpers (transition period)."""

from __future__ import annotations

THREAD_DISPATCH = "dispatch"
THREAD_TEAM = "team"
THREAD_SUPPORT = "support"
THREAD_PREFIX_MISSION = "mission:"
DIRECT_PREFIX = "direct:"


def mission_thread_id(booking_id: int) -> str:
    return f"{THREAD_PREFIX_MISSION}{booking_id}"


def parse_mission_thread(thread_id: str) -> int | None:
    if not thread_id.startswith(THREAD_PREFIX_MISSION):
        return None
    try:
        return int(thread_id.split(":", 1)[1])
    except (IndexError, ValueError):
        return None


def company_group_legacy_thread_id() -> str:
    return THREAD_TEAM


def company_dispatch_legacy_thread_id() -> str:
    return THREAD_DISPATCH


def company_driver_channel_legacy_thread_id(driver_id: int) -> str:
    """Canal privé exploitation ↔ un chauffeur (distinct du dispatch partagé)."""
    return f"company_driver:{int(driver_id)}"


def parse_direct_thread(thread_id: str) -> int | None:
    """Peer user id depuis l'UI mobile ``direct:{peer_user_id}``."""
    if not thread_id.startswith(DIRECT_PREFIX):
        return None
    parts = thread_id.split(":")
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def direct_pair_legacy_thread_id(user_id_a: int, user_id_b: int) -> str:
    lo, hi = sorted((int(user_id_a), int(user_id_b)))
    return f"{DIRECT_PREFIX}{lo}:{hi}"


def parse_direct_pair_legacy_thread(thread_id: str) -> tuple[int, int] | None:
    if not thread_id.startswith(DIRECT_PREFIX):
        return None
    parts = thread_id.split(":")
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def conversation_id_to_legacy_thread(conversation) -> str | None:
    from models.messaging_enums import ConversationContext, ConversationType

    ctype = str(getattr(conversation, "conversation_type", "") or "")
    ctx = str(getattr(conversation, "context_type", "") or "")
    cid = getattr(conversation, "context_id", None)
    legacy = getattr(conversation, "legacy_thread_id", None)
    if legacy:
        return str(legacy)
    if ctype == ConversationType.MISSION.value and cid:
        return mission_thread_id(int(cid))
    if (
        ctype == ConversationType.COMPANY.value
        and ctx == ConversationContext.COMPANY.value
    ):
        company_id = getattr(conversation, "company_id", None)
        if cid is not None and company_id is not None and int(cid) == int(company_id):
            return THREAD_DISPATCH
        if cid is not None:
            return company_driver_channel_legacy_thread_id(int(cid))
        return THREAD_DISPATCH
    if ctype == ConversationType.GROUP.value:
        return THREAD_TEAM
    return None
