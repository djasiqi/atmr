"""Tests hub messages chauffeur."""

from services.messages.hub_service import (
    EMERGENCY_LABELS,
    THREAD_DISPATCH,
    THREAD_TEAM,
    count_company_team_members,
    direct_thread_id,
    list_driver_colleagues,
    mission_thread_id,
    parse_direct_thread,
    parse_mission_thread,
)


def test_mission_thread_id_roundtrip():
    tid = mission_thread_id(30775)
    assert tid == "mission:30775"
    assert parse_mission_thread(tid) == 30775
    assert parse_mission_thread(THREAD_DISPATCH) is None


def test_direct_thread_id_roundtrip():
    tid = direct_thread_id(99)
    assert tid == "direct:99"
    assert parse_direct_thread(tid) == 99


def test_team_thread_constant():
    assert THREAD_TEAM == "team"


def test_emergency_labels_complete():
    assert "patient_absent" in EMERGENCY_LABELS
    assert "besoin_assistance" in EMERGENCY_LABELS


def test_list_driver_colleagues_is_callable():
    assert callable(list_driver_colleagues)


def test_count_company_team_members_is_callable():
    assert callable(count_company_team_members)
