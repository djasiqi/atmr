"""Rémunération d'une course terminée ou d'un temps ajouté."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from domain.work_time.transport_time import TransportWorkTime

CALCULATED = "calculated"
PENDING = "pending"
REVIEW = "requires_review"


@dataclass(frozen=True)
class PolicyView:
    """Version de règle applicable à une date."""

    policy_id: int | None
    driver_id: int | None
    effective_from: date
    effective_until: date | None
    one_way_minutes: int
    round_trip_minutes: int
    intermediate_stop_minutes: int | None
    transport_flat_minutes: int
    max_reasonable_minutes: int | None = None
    overlap_threshold_minutes: int = 1
    work_type_rules: dict[str, Any] = field(default_factory=dict)
    mode: str = "flat_per_trip"


@dataclass
class CompensationLine:
    """Ligne au format ledger (figeable à la clôture de période)."""

    journey_key: str
    driver_id: int | None
    accounting_date: str | None
    journey_class: str | None
    journey_status: str
    compensation_status: str
    rule_type: str | None
    base_minutes: int
    intermediate_stop_count: int
    intermediate_stop_minutes: int | None
    compensated_minutes: int
    policy_id: int | None
    classification_source: str
    attached_to_booking_id: int | None
    line_key: str = ""
    flat_minutes: int | None = None
    manual_entry_id: int | None = None
    anomalies: list[str] = field(default_factory=list)


def resolve_policy(
    policies: list[PolicyView], *, driver_id: int | None, on_date: date
) -> PolicyView | None:
    """Priorité : surcharge chauffeur, puis règle entreprise. Fenêtre semi-ouverte."""

    def matches(policy: PolicyView) -> bool:
        if on_date < policy.effective_from:
            return False
        return not (
            policy.effective_until is not None and on_date >= policy.effective_until
        )

    driver_rules = [
        policy
        for policy in policies
        if policy.driver_id is not None
        and policy.driver_id == driver_id
        and matches(policy)
    ]
    if driver_rules:
        return sorted(driver_rules, key=lambda item: item.effective_from)[-1]
    company_rules = [
        policy for policy in policies if policy.driver_id is None and matches(policy)
    ]
    if not company_rules:
        return None
    return sorted(company_rules, key=lambda item: item.effective_from)[-1]


def compensate_transport(
    segment,
    work: TransportWorkTime,
    *,
    policies: list[PolicyView],
    payable_minutes: dict[int, int],
    journey_key: str,
    journey_class: str | None = None,
) -> CompensationLine:
    """Une course terminée, une ligne. Le forfait ne dépend pas du journey."""
    accounting = work.classification_date_zurich
    on_date = datetime.fromisoformat(accounting).date() if accounting else None
    policy = (
        resolve_policy(policies, driver_id=segment.driver_id, on_date=on_date)
        if on_date is not None
        else None
    )
    line_key = f"bk:{int(segment.booking_id)}"
    flat = int(policy.transport_flat_minutes) if policy is not None else None

    def make(
        *,
        status: str,
        minutes: int,
        rule_type: str | None,
        extra: list[str],
    ) -> CompensationLine:
        return CompensationLine(
            journey_key=journey_key,
            line_key=line_key,
            driver_id=segment.driver_id,
            accounting_date=accounting,
            journey_class=journey_class,
            journey_status="complete",
            compensation_status=status,
            rule_type=rule_type,
            base_minutes=flat or 0,
            intermediate_stop_count=0,
            intermediate_stop_minutes=0,
            compensated_minutes=minutes,
            policy_id=policy.policy_id if policy is not None else None,
            classification_source="transport",
            attached_to_booking_id=int(segment.booking_id),
            flat_minutes=flat,
            anomalies=list(work.anomalies) + extra,
        )

    if policy is None:
        return make(
            status=REVIEW,
            minutes=0,
            rule_type=None,
            extra=["compensation_rule_missing"],
        )
    if policy.mode in {"validated_work_time", "real_time"}:
        accepted = int(segment.booking_id) in payable_minutes
        if not accepted:
            return make(
                status=PENDING, minutes=0, rule_type="validated_work_time", extra=[]
            )
        return make(
            status=CALCULATED,
            minutes=int(payable_minutes[int(segment.booking_id)]),
            rule_type="validated_work_time",
            extra=[],
        )
    return make(
        status=CALCULATED,
        minutes=int(policy.transport_flat_minutes),
        rule_type="transport_flat",
        extra=[],
    )


def compensate_manual(
    *,
    entry_id: int,
    driver_id: int,
    work_type: str,
    duration_minutes: int,
    work_date: str,
    policies: list[PolicyView],
) -> CompensationLine:
    """Rémunération d'un temps manuel selon la règle du type, sinon revue."""
    on_date = datetime.fromisoformat(work_date).date()
    policy = resolve_policy(policies, driver_id=driver_id, on_date=on_date)
    key = f"manual:{entry_id}"
    if policy is None:
        return CompensationLine(
            journey_key=key,
            line_key=key,
            manual_entry_id=entry_id,
            driver_id=driver_id,
            accounting_date=work_date,
            journey_class=None,
            journey_status="complete",
            compensation_status=REVIEW,
            rule_type=None,
            base_minutes=0,
            intermediate_stop_count=0,
            intermediate_stop_minutes=None,
            compensated_minutes=0,
            policy_id=None,
            classification_source="manual",
            attached_to_booking_id=None,
            anomalies=["manual_compensation_rule_missing"],
        )
    rule = (policy.work_type_rules or {}).get(work_type) or {}
    mode = str(rule.get("mode") or "")
    if mode == "flat":
        minutes = int(rule.get("minutes") or 0)
        return CompensationLine(
            journey_key=key,
            line_key=key,
            manual_entry_id=entry_id,
            driver_id=driver_id,
            accounting_date=work_date,
            journey_class=None,
            journey_status="complete",
            compensation_status=CALCULATED,
            rule_type=f"manual_flat:{work_type}",
            base_minutes=minutes,
            intermediate_stop_count=0,
            intermediate_stop_minutes=0,
            compensated_minutes=minutes,
            policy_id=policy.policy_id,
            classification_source="manual",
            attached_to_booking_id=None,
            flat_minutes=minutes,
            anomalies=[],
        )
    if mode == "real_time":
        return CompensationLine(
            journey_key=key,
            line_key=key,
            manual_entry_id=entry_id,
            driver_id=driver_id,
            accounting_date=work_date,
            journey_class=None,
            journey_status="complete",
            compensation_status=CALCULATED,
            rule_type=f"manual_real:{work_type}",
            base_minutes=duration_minutes,
            intermediate_stop_count=0,
            intermediate_stop_minutes=0,
            compensated_minutes=duration_minutes,
            policy_id=policy.policy_id,
            classification_source="manual",
            attached_to_booking_id=None,
            flat_minutes=duration_minutes,
            anomalies=[],
        )
    return CompensationLine(
        journey_key=key,
        line_key=key,
        manual_entry_id=entry_id,
        driver_id=driver_id,
        accounting_date=work_date,
        journey_class=None,
        journey_status="complete",
        compensation_status=REVIEW,
        rule_type=None,
        base_minutes=0,
        intermediate_stop_count=0,
        intermediate_stop_minutes=None,
        compensated_minutes=0,
        policy_id=policy.policy_id,
        classification_source="manual",
        attached_to_booking_id=None,
        flat_minutes=None,
        anomalies=["manual_compensation_rule_missing"],
    )


def policy_for_day(
    policies: list[PolicyView], driver_id: int | None, day: str | None
) -> PolicyView | None:
    """Résout la politique d'un jour civil, pour le seuil de durée."""
    if not day:
        return None
    return resolve_policy(
        policies, driver_id=driver_id, on_date=datetime.fromisoformat(day).date()
    )
