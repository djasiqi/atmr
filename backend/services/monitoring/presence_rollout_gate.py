from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PresenceRolloutGateInput:
    non_canonical_fanout_rate: float
    non_canonical_db_write_rate: float
    availability_socket_rate: float
    cross_tenant_mismatch_count: int
    mission_missing_id_rate: float


@dataclass(frozen=True)
class PresenceRolloutGateResult:
    can_pilot: bool
    can_rollout_large: bool
    reasons: list[str]


def evaluate_presence_rollout_gate(values: PresenceRolloutGateInput) -> PresenceRolloutGateResult:
    reasons: list[str] = []
    if values.non_canonical_fanout_rate > 0:
        reasons.append("non_canonical_fanout_detected")
    if values.non_canonical_db_write_rate > 0:
        reasons.append("non_canonical_db_write_detected")
    if values.availability_socket_rate > 0:
        reasons.append("availability_socket_traffic_detected")
    if values.cross_tenant_mismatch_count > 0:
        reasons.append("cross_tenant_mismatch_detected")

    can_pilot = len(reasons) == 0
    can_rollout_large = can_pilot and values.mission_missing_id_rate <= 0.01
    if can_pilot and not can_rollout_large:
        reasons.append("mission_missing_id_rate_too_high")
    return PresenceRolloutGateResult(
        can_pilot=can_pilot,
        can_rollout_large=can_rollout_large,
        reasons=reasons,
    )
