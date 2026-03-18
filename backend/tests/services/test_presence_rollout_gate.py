from services.monitoring.presence_rollout_gate import (
    PresenceRolloutGateInput,
    evaluate_presence_rollout_gate,
)


def test_gate_blocks_pilot_on_non_canonical_signals():
    result = evaluate_presence_rollout_gate(
        PresenceRolloutGateInput(
            non_canonical_fanout_rate=0.01,
            non_canonical_db_write_rate=0.0,
            availability_socket_rate=0.0,
            cross_tenant_mismatch_count=0,
            mission_missing_id_rate=0.0,
        )
    )
    assert result.can_pilot is False
    assert "non_canonical_fanout_detected" in result.reasons


def test_gate_allows_pilot_and_rollout_when_all_green():
    result = evaluate_presence_rollout_gate(
        PresenceRolloutGateInput(
            non_canonical_fanout_rate=0.0,
            non_canonical_db_write_rate=0.0,
            availability_socket_rate=0.0,
            cross_tenant_mismatch_count=0,
            mission_missing_id_rate=0.005,
        )
    )
    assert result.can_pilot is True
    assert result.can_rollout_large is True
