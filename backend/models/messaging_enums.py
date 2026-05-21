"""Enums for the multi-context messaging engine."""

from __future__ import annotations

from enum import Enum as PyEnum


class ConversationType(str, PyEnum):
    MISSION = "MISSION"
    COMPANY = "COMPANY"
    GROUP = "GROUP"
    SYSTEM = "SYSTEM"
    # V2+
    DIRECT = "DIRECT"
    # V3+
    INSTITUTION = "INSTITUTION"
    CLIENT_SELF = "CLIENT_SELF"


class ConversationContext(str, PyEnum):
    MISSION = "MISSION"
    COMPANY = "COMPANY"
    SUPERVISION = "SUPERVISION"
    BOOKING = "BOOKING"


class ParticipantRole(str, PyEnum):
    DRIVER = "DRIVER"
    DISPATCH = "DISPATCH"
    COMPANY = "COMPANY"
    INSTITUTION = "INSTITUTION"
    CLIENT = "CLIENT"


DEFAULT_VISIBILITY_SCOPE: dict = {
    "default": "all_participants",
    "rules": [],
}

DEFAULT_MESSAGE_VISIBILITY_TAGS = ["operational"]
