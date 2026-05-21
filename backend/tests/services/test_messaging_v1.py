"""Tests messaging V1 — enums, legacy threads, permissions surface."""

from models.messaging_enums import (
    ConversationContext,
    ConversationType,
    DEFAULT_VISIBILITY_SCOPE,
    ParticipantRole,
)
from services.messaging.legacy_thread import (
    mission_thread_id,
    parse_mission_thread,
    company_dispatch_legacy_thread_id,
)
from services.messaging.permission_service import MessagingPermissionService
from services.messaging.system_message_emitter import SYSTEM_LABELS


def test_mission_thread_roundtrip():
    assert mission_thread_id(42) == "mission:42"
    assert parse_mission_thread("mission:42") == 42
    assert parse_mission_thread("dispatch") is None


def test_conversation_enums():
    assert ConversationType.MISSION.value == "MISSION"
    assert ConversationContext.SUPERVISION.value == "SUPERVISION"
    assert ParticipantRole.DRIVER.value == "DRIVER"


def test_default_visibility_scope():
    assert DEFAULT_VISIBILITY_SCOPE["default"] == "all_participants"


def test_system_labels_cover_mission_flow():
    assert "assigned" in SYSTEM_LABELS
    assert "arrived" in SYSTEM_LABELS
    assert "in_progress" in SYSTEM_LABELS
    assert "completed" in SYSTEM_LABELS


def test_direct_enabled_for_driver_role():
    class _U:
        role = type("R", (), {"value": "driver"})()
        driver = object()

    assert MessagingPermissionService.can_create_direct(_U()) is True


def test_company_role_uppercase_can_manage_dispatch():
    class _Company:
        id = 1

    class _U:
        role = type("R", (), {"value": "COMPANY"})()
        company = _Company()

    class _Conv:
        company_id = 1

    assert MessagingPermissionService.can_read_conversation(_U(), _Conv()) is True
    assert MessagingPermissionService.can_write_conversation(_U(), _Conv()) is True


def test_direct_pair_legacy_thread_id_stable():
    from services.messaging.legacy_thread import direct_pair_legacy_thread_id

    assert direct_pair_legacy_thread_id(10, 20) == "direct:10:20"
    assert direct_pair_legacy_thread_id(20, 10) == "direct:10:20"


def test_dispatch_legacy_thread():
    assert company_dispatch_legacy_thread_id() == "dispatch"
