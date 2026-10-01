"""Assemblage du rapport : le résumé est la somme du détail."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from domain.work_time.anomalies import mark_overlaps
from domain.work_time.clock import as_utc, scheduled_zurich_date
from domain.work_time.compensation import (
    CompensationLine,
    PolicyView,
    compensate_manual,
    compensate_transport,
    policy_for_day,
)
from domain.work_time.journeys import SegmentInput, build_journeys
from domain.work_time.periods import date_in_period
from domain.work_time.transport_time import (
    AdjustmentView,
    TransportInput,
    TransportWorkTime,
    compute_transport_work_time,
)
from domain.work_time.transports import counts_as_transport


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    aware = as_utc(value)
    if aware is None:
        return None
    return aware.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class DurationDecisionView:
    """Décision administrative sur une durée, sans inventer d'horodatage."""

    validated_worked_minutes: int
    source: str
    proposed_worked_minutes: int | None = None
    route_minutes: int | None = None
    margin_minutes: int | None = None
    route_provider: str | None = None
    reason: str | None = None


def proposal_for_segment(
    segment: SegmentInput, *, margin_minutes: int
) -> dict[str, Any] | None:
    """Temps de trajet estimé + minutes offertes au chauffeur. Aucune distance à vol d'oiseau."""
    if margin_minutes < 0 or segment.routing_status != "ok":
        return None
    if segment.route_provider != "osrm" or segment.routing_profile != "driving":
        return None
    seconds = segment.route_duration_seconds
    distance = segment.route_distance_m
    if seconds is None or seconds <= 0 or distance is None or distance < 0:
        return None
    route_minutes = round(seconds / 60)
    if route_minutes <= 0:
        return None
    return {
        "route_minutes": route_minutes,
        "margin_minutes": margin_minutes,
        "proposed_worked_minutes": route_minutes + margin_minutes,
        "route_provider": "osrm",
        "routing_profile": "driving",
        "route_distance_m": int(distance),
        "route_duration_seconds": int(seconds),
        "routing_status": "ok",
        "calculated_at": segment.route_calculated_at,
    }


def _status_for(quality: str) -> str:
    if quality == "verified":
        return "verified"
    if quality == "adjusted":
        return "adjusted"
    if quality == "estimated_historical":
        return "estimated_historical"
    return "incomplete"


def _payable_minutes(
    segments: list[SegmentInput],
    work_by_booking: dict[int, TransportWorkTime],
    decisions: dict[int, DurationDecisionView],
) -> dict[int, int]:
    """Minutes retenues pour la paie : horloge réelle ou durée validée."""
    payable: dict[int, int] = {}
    for segment in segments:
        work = work_by_booking.get(segment.booking_id)
        if work is None:
            continue
        decision = decisions.get(segment.booking_id)
        if decision is not None and work.quality == "incomplete":
            payable[segment.booking_id] = int(decision.validated_worked_minutes)
        elif work.quality != "incomplete" and work.worked_minutes is not None:
            payable[segment.booking_id] = int(work.worked_minutes)
    return payable


def _line_dict(line: CompensationLine, *, minutes: int | None = None) -> dict[str, Any]:
    return {
        "journey_key": line.journey_key,
        "line_key": line.line_key,
        "driver_id": line.driver_id,
        "accounting_date": line.accounting_date,
        "journey_class": line.journey_class,
        "journey_status": line.journey_status,
        "compensation_status": line.compensation_status,
        "rule_type": line.rule_type,
        "base_minutes": line.base_minutes,
        "intermediate_stop_count": line.intermediate_stop_count,
        "intermediate_stop_minutes": line.intermediate_stop_minutes,
        "compensated_minutes": line.compensated_minutes if minutes is None else minutes,
        "policy_id": line.policy_id,
        "classification_source": line.classification_source,
        "attached_to_booking_id": line.attached_to_booking_id,
        "flat_minutes": line.flat_minutes,
        "manual_entry_id": line.manual_entry_id,
    }


def _empty_driver(driver_id: int) -> dict[str, Any]:
    return {
        "driver_id": driver_id,
        "completed_segments_count": 0,
        "incomplete_segments_count": 0,
        "completion_time_missing_count": 0,
        "compensated_journeys_count": 0,
        "pending_journeys_count": 0,
        "review_journeys_count": 0,
        "worked_verified_minutes": 0,
        "worked_estimated_minutes": 0,
        "worked_adjusted_minutes": 0,
        "manual_minutes": 0,
        "worked_validated_minutes": 0,
        "pending_validation_minutes": 0,
        "total_worked_minutes": 0,
        "compensated_minutes": 0,
        "transport_count": 0,
        "real_transport_minutes": 0,
        "flat_transport_minutes": 0,
        "real_added_minutes": 0,
        "flat_added_minutes": 0,
        "review_count": 0,
        "review_count_real": 0,
        "review_count_flat": 0,
        "days_worked": 0,
        "anomalies_count": 0,
        "display_name": None,
    }


def _ledger_entry(
    report: dict[str, Any], row: dict[str, Any], legacy: bool
) -> dict[str, Any] | None:
    owner = row.get("driver_id")
    if owner is None:
        return None
    key = (
        str(row.get("journey_key") or "") if legacy else str(row.get("line_key") or "")
    )
    if not key:
        return None
    for day in report.get("days_by_driver", {}).get(str(int(owner)), []):
        for entry in day.get("entries") or []:
            entry_key = (
                str(entry.get("journey_key") or "")
                if legacy
                else str(entry.get("line_key") or "")
            )
            if entry_key == key:
                return entry
    return None


def _remember_review(
    items: list[dict[str, Any]], entry: dict[str, Any], *, real: bool, flat: bool
) -> None:
    """Mémorise l'élément au moment où le compteur augmente, pour la même source."""
    modes: list[str] = []
    if real:
        modes.append("real")
    if flat:
        modes.append("flat")
    if not modes:
        return
    current = entry.setdefault("review_modes", [])
    for mode in modes:
        if mode not in current:
            current.append(mode)
    reasons = [str(code) for code in (entry.get("anomalies") or []) if code]
    items.append(
        {
            "driver_id": int(entry["driver_id"]),
            "date": entry.get("date"),
            "kind": entry.get("kind"),
            "booking_id": entry.get("booking_id"),
            "entry_id": entry.get("entry_id"),
            "line_key": entry.get("line_key"),
            "pickup_label": entry.get("pickup_label") or "",
            "dropoff_label": entry.get("dropoff_label") or "",
            "description": entry.get("description") or "",
            "work_type": entry.get("work_type"),
            "modes": list(modes),
            "reasons": reasons,
        }
    )


def build_work_time_report(
    *,
    segments: list[SegmentInput],
    adjustments: dict[int, AdjustmentView],
    manuals: list[dict[str, Any]],
    policies: list[PolicyView],
    cutover: datetime,
    period_from: str,
    period_to: str,
    duration_decisions: dict[int, DurationDecisionView] | None = None,
    route_margin_minutes: int = 5,
    route_estimates_enabled: bool = True,
) -> dict[str, Any]:
    """Produit résumé, détail par jour et lignes de rémunération.

    ``segments`` inclut les tronçons frères hors période (structure du trajet).
    Seules les tranches dont le jour civil tombe dans la période sont comptées.
    """
    journeys = build_journeys(segments)
    work_by_booking: dict[int, TransportWorkTime] = {}
    for segment in segments:
        if not counts_as_transport(segment):
            continue
        work_by_booking[segment.booking_id] = compute_transport_work_time(
            TransportInput(
                booking_id=segment.booking_id,
                arrived_at=segment.arrived_at,
                boarded_at=segment.boarded_at,
                completed_at=segment.completed_at,
                scheduled_time=segment.scheduled_time,
                status=segment.status_key,
            ),
            adjustments.get(segment.booking_id),
            cutover,
        )
        if segment.is_cancelled:
            work = work_by_booking[segment.booking_id]
            work.anomalies = [
                code for code in work.anomalies if code != "segment_not_completed"
            ]
            if "cancelled_after_arrival" not in work.anomalies:
                work.anomalies.append("cancelled_after_arrival")
            if work.effective_completed_at is None:
                work.worked_minutes = None
                work.worked_minutes_source = None
                work.quality = "incomplete"
                work.day_slices = []

    decisions = duration_decisions or {}
    payable = _payable_minutes(segments, work_by_booking, decisions)
    journey_by_booking: dict[int, str] = {}
    journey_class_by_booking: dict[int, str | None] = {}
    for journey in journeys:
        for segment in journey.segments:
            journey_by_booking[segment.booking_id] = journey.journey_key
            journey_class_by_booking[segment.booking_id] = journey.journey_class
    lines = []
    for segment in segments:
        if not counts_as_transport(segment) or segment.driver_id is None:
            continue
        work = work_by_booking.get(segment.booking_id)
        if work is None:
            continue
        if (
            work.classification_date_zurich
            and not date_in_period(
                work.classification_date_zurich, period_from, period_to
            )
            and not any(
                date_in_period(day, period_from, period_to)
                for day, _minutes in work.day_slices
            )
        ):
            continue
        lines.append(
            compensate_transport(
                segment,
                work,
                policies=policies,
                payable_minutes=payable,
                journey_key=journey_by_booking.get(
                    segment.booking_id, f"bk:{segment.booking_id}"
                ),
                journey_class=journey_class_by_booking.get(segment.booking_id),
            )
        )
    line_by_booking = {
        int(line.attached_to_booking_id): line
        for line in lines
        if line.attached_to_booking_id is not None
    }

    entries: list[dict[str, Any]] = []
    for segment in segments:
        if not counts_as_transport(segment) or segment.driver_id is None:
            continue
        work = work_by_booking[segment.booking_id]
        key = journey_by_booking.get(segment.booking_id)
        line = line_by_booking.get(segment.booking_id)
        slices = [
            (day, minutes)
            for day, minutes in work.day_slices
            if date_in_period(day, period_from, period_to) and minutes > 0
        ]
        if (
            not slices
            and work.classification_date_zurich
            and date_in_period(work.classification_date_zurich, period_from, period_to)
        ):
            slices = [(work.classification_date_zurich, 0)]

        for day, slice_minutes in slices:
            pay_here = (
                line is not None
                and line.accounting_date == day
                and date_in_period(line.accounting_date, period_from, period_to)
            )
            shown_minutes = line.compensated_minutes if pay_here and line else 0
            anomalies = list(work.anomalies)
            if line is not None:
                for code in line.anomalies:
                    if code not in anomalies and (
                        pay_here
                        or code
                        in {"journey_incomplete", "journey_split_across_drivers"}
                    ):
                        anomalies.append(code)
            policy = policy_for_day(policies, segment.driver_id, day)
            if (
                policy is not None
                and policy.max_reasonable_minutes is not None
                and work.worked_minutes is not None
                and work.worked_minutes > policy.max_reasonable_minutes
                and "duration_exceeds_threshold" not in anomalies
            ):
                anomalies.append("duration_exceeds_threshold")
            worked_out: int | None
            if work.quality == "incomplete" or (
                slice_minutes == 0 and not work.day_slices
            ):
                worked_out = None
            else:
                worked_out = slice_minutes
            proposal = None
            routing_status = None
            if (
                worked_out is None
                and anomalies
                and route_estimates_enabled
                and "cancelled_after_arrival" not in anomalies
            ):
                proposal = proposal_for_segment(
                    segment, margin_minutes=route_margin_minutes
                )
                routing_status = "ok" if proposal is not None else "unavailable"
            decision = decisions.get(segment.booking_id)
            source = work.worked_minutes_source
            status = _status_for(work.quality)
            quality_notes: list[str] = []
            if decision is not None and work.quality == "incomplete":
                worked_out = int(decision.validated_worked_minutes)
                source = decision.source
                status = (
                    "adjusted"
                    if decision.source == "admin_adjustment"
                    else "validated_estimate"
                )
                if "arrival_not_recorded" in anomalies:
                    anomalies = [
                        code for code in anomalies if code != "arrival_not_recorded"
                    ]
                    quality_notes.append("arrival_not_recorded")
            elif proposal is not None:
                worked_out = None
                source = "route_estimate"
                status = "pending_validation"
            entries.append(
                {
                    "kind": "transport",
                    "is_manual": False,
                    "booking_id": segment.booking_id,
                    "entry_id": None,
                    "driver_id": segment.driver_id,
                    "date": day,
                    "route_group_id": segment.route_group_id,
                    "journey_key": key,
                    "line_key": line.line_key
                    if line is not None
                    else f"bk:{segment.booking_id}",
                    "pickup_label": segment.pickup_location,
                    "dropoff_label": segment.dropoff_location,
                    "effective_arrived_at": _iso(work.effective_arrived_at),
                    "effective_completed_at": _iso(work.effective_completed_at),
                    "worked_minutes": worked_out,
                    "proposed_worked_minutes": (
                        decision.proposed_worked_minutes
                        if decision is not None and decision.proposed_worked_minutes
                        else (proposal or {}).get("proposed_worked_minutes")
                    ),
                    "route_minutes": (
                        decision.route_minutes
                        if decision is not None and decision.route_minutes is not None
                        else (proposal or {}).get("route_minutes")
                    ),
                    "margin_minutes": (
                        decision.margin_minutes
                        if decision is not None and decision.margin_minutes is not None
                        else (proposal or {}).get("margin_minutes")
                    ),
                    "route_provider": (
                        decision.route_provider
                        if decision is not None and decision.route_provider
                        else (proposal or {}).get("route_provider")
                    ),
                    "routing_profile": (proposal or {}).get("routing_profile"),
                    "route_distance_m": (proposal or {}).get("route_distance_m"),
                    "route_duration_seconds": (proposal or {}).get(
                        "route_duration_seconds"
                    ),
                    "routing_status": routing_status,
                    "route_calculated_at": (proposal or {}).get("calculated_at"),
                    "work_time_status": status,
                    "quality_notes": quality_notes,
                    "worked_minutes_source": source,
                    "quality": work.quality,
                    "compensated_minutes": shown_minutes,
                    "real_minutes": None
                    if status == "pending_validation"
                    else worked_out,
                    "flat_minutes": line.flat_minutes
                    if pay_here and line is not None
                    else None,
                    "flat_status": (
                        "calculated"
                        if pay_here
                        and line is not None
                        and line.flat_minutes is not None
                        else "requires_review"
                        if pay_here
                        else None
                    ),
                    "compensation": _line_dict(line, minutes=shown_minutes)
                    if line is not None
                    else None,
                    "is_manually_adjusted": segment.booking_id in adjustments,
                    "anomalies": anomalies,
                    "work_type": None,
                    "description": None,
                    "_start": work.effective_arrived_at,
                    "_end": work.effective_completed_at,
                    "_segment_quality": work.quality,
                    "_count_incomplete": (
                        work.quality == "incomplete"
                        and status
                        not in {"pending_validation", "validated_estimate", "adjusted"}
                    ),
                }
            )

    open_statuses = {"PENDING", "ACCEPTED", "ASSIGNED", "EN_ROUTE", "IN_PROGRESS"}
    seen_open: set[int] = set()
    for segment in segments:
        if segment.is_completed or segment.is_cancelled or segment.driver_id is None:
            continue
        if segment.status_key not in open_statuses or segment.booking_id in seen_open:
            continue
        day = scheduled_zurich_date(segment.scheduled_time)
        if not day or not date_in_period(day, period_from, period_to):
            continue
        seen_open.add(segment.booking_id)
        entries.append(
            {
                "kind": "open",
                "is_manual": False,
                "booking_id": segment.booking_id,
                "entry_id": None,
                "driver_id": segment.driver_id,
                "date": day,
                "route_group_id": segment.route_group_id,
                "journey_key": journey_by_booking.get(segment.booking_id),
                "line_key": None,
                "pickup_label": segment.pickup_location,
                "dropoff_label": segment.dropoff_location,
                "effective_arrived_at": None,
                "effective_completed_at": None,
                "worked_minutes": None,
                "real_minutes": None,
                "flat_minutes": None,
                "flat_status": "requires_review",
                "work_time_status": "requires_review",
                "quality": "incomplete",
                "compensated_minutes": 0,
                "compensation": None,
                "is_manually_adjusted": False,
                "anomalies": ["transport_not_completed"],
                "work_type": None,
                "description": None,
                "_start": None,
                "_end": None,
            }
        )

    manual_lines: list[CompensationLine] = []
    for manual in manuals:
        work_date = str(manual["work_date"])
        if not date_in_period(work_date, period_from, period_to):
            continue
        if manual.get("cancelled"):
            continue
        compensated = compensate_manual(
            entry_id=int(manual["entry_id"]),
            driver_id=int(manual["driver_id"]),
            work_type=str(manual["work_type"]),
            duration_minutes=int(manual["duration_minutes"]),
            work_date=work_date,
            policies=policies,
        )
        manual_lines.append(compensated)
        entries.append(
            {
                "kind": "manual",
                "is_manual": True,
                "booking_id": manual.get("booking_id"),
                "entry_id": int(manual["entry_id"]),
                "driver_id": int(manual["driver_id"]),
                "date": work_date,
                "route_group_id": None,
                "journey_key": compensated.journey_key,
                "line_key": compensated.line_key,
                "pickup_label": manual.get("pickup_location") or "",
                "dropoff_label": manual.get("dropoff_location") or "",
                "effective_arrived_at": _iso(manual.get("started_at")),
                "effective_completed_at": _iso(manual.get("ended_at")),
                "worked_minutes": int(manual["duration_minutes"]),
                "worked_minutes_source": "manual_entry",
                "quality": "manual",
                "compensated_minutes": compensated.compensated_minutes,
                "real_minutes": int(manual["duration_minutes"]),
                "flat_minutes": compensated.flat_minutes,
                "flat_status": (
                    "calculated"
                    if compensated.compensation_status == "calculated"
                    else "requires_review"
                ),
                "compensation": _line_dict(compensated),
                "is_manually_adjusted": False,
                "anomalies": list(compensated.anomalies),
                "work_type": manual["work_type"],
                "description": manual.get("description") or "",
                "_start": manual.get("started_at"),
                "_end": manual.get("ended_at"),
                "_segment_quality": "manual",
            }
        )

    threshold = 1
    if policies:
        company_policies = [item for item in policies if item.driver_id is None]
        chosen = company_policies[-1] if company_policies else policies[-1]
        threshold = chosen.overlap_threshold_minutes
    mark_overlaps(entries, threshold_minutes=threshold)

    drivers: dict[int, dict[str, Any]] = {}
    days: dict[int, dict[str, list[dict[str, Any]]]] = {}
    segment_seen: dict[int, dict[int, dict[str, Any]]] = {}
    worked_days: dict[int, set[str]] = {}
    real_review_seen: set[tuple[int, int]] = set()
    flat_review_seen: set[tuple[int, int]] = set()
    review_items: list[dict[str, Any]] = []

    for entry in entries:
        public = {key: value for key, value in entry.items() if not key.startswith("_")}
        driver_id = int(entry["driver_id"])
        bucket = drivers.setdefault(driver_id, _empty_driver(driver_id))
        days.setdefault(driver_id, {}).setdefault(entry["date"], []).append(public)
        if entry["kind"] == "open":
            bucket["review_count_real"] += 1
            bucket["review_count_flat"] += 1
            _remember_review(review_items, public, real=True, flat=True)
        elif entry["kind"] == "transport":
            segment_seen.setdefault(driver_id, {})[int(entry["booking_id"])] = entry
            minutes = entry["worked_minutes"]
            status = entry.get("work_time_status")
            if isinstance(minutes, int) and minutes > 0:
                quality = entry["quality"]
                if status == "validated_estimate":
                    bucket["worked_validated_minutes"] += minutes
                elif status == "adjusted" or quality == "adjusted":
                    bucket["worked_adjusted_minutes"] += minutes
                elif quality == "verified":
                    bucket["worked_verified_minutes"] += minutes
                elif quality == "estimated_historical":
                    bucket["worked_estimated_minutes"] += minutes
                bucket["total_worked_minutes"] += minutes
                worked_days.setdefault(driver_id, set()).add(entry["date"])
            elif status == "pending_validation":
                bucket["pending_validation_minutes"] += int(
                    entry.get("proposed_worked_minutes") or 0
                )
            booking_key = (driver_id, int(entry["booking_id"]))
            real = entry.get("real_minutes")
            if isinstance(real, int):
                bucket["real_transport_minutes"] += real
            elif booking_key not in real_review_seen:
                real_review_seen.add(booking_key)
                bucket["review_count_real"] += 1
                _remember_review(review_items, public, real=True, flat=False)
            if (
                entry.get("flat_status") == "calculated"
                and entry.get("flat_minutes") is not None
            ):
                bucket["flat_transport_minutes"] += int(entry["flat_minutes"])
            elif (
                entry.get("flat_status") == "requires_review"
                and booking_key not in flat_review_seen
            ):
                flat_review_seen.add(booking_key)
                bucket["review_count_flat"] += 1
                _remember_review(review_items, public, real=False, flat=True)
        else:
            minutes = int(entry["worked_minutes"] or 0)
            bucket["manual_minutes"] += minutes
            bucket["total_worked_minutes"] += minutes
            bucket["real_added_minutes"] += int(entry.get("real_minutes") or 0)
            if entry.get("flat_status") == "calculated":
                bucket["flat_added_minutes"] += int(entry.get("flat_minutes") or 0)
            else:
                bucket["review_count_flat"] += 1
                _remember_review(review_items, public, real=False, flat=True)
            if minutes > 0:
                worked_days.setdefault(driver_id, set()).add(entry["date"])
        if public["anomalies"]:
            bucket["anomalies_count"] += 1

    for driver_id, seen in segment_seen.items():
        bucket = drivers[driver_id]
        bucket["transport_count"] = len(seen)
        for entry in seen.values():
            if entry.get("_count_incomplete"):
                bucket["incomplete_segments_count"] += 1
            elif entry.get("work_time_status") != "pending_validation":
                bucket["completed_segments_count"] += 1
            if "completion_time_missing" in entry["anomalies"]:
                bucket["completion_time_missing_count"] += 1

    for line in [*lines, *manual_lines]:
        owner = line.driver_id
        if owner is None or not line.accounting_date:
            continue
        if not date_in_period(line.accounting_date, period_from, period_to):
            continue
        bucket = drivers.setdefault(owner, _empty_driver(owner))
        if line.compensation_status == "calculated":
            bucket["compensated_journeys_count"] += 1
            bucket["compensated_minutes"] += int(line.compensated_minutes)
        elif line.compensation_status == "pending":
            bucket["pending_journeys_count"] += 1
        else:
            bucket["review_journeys_count"] += 1

    for driver_id, bucket in drivers.items():
        bucket["days_worked"] = len(worked_days.get(driver_id, set()))
        bucket["review_count"] = int(bucket["review_count_real"])

    day_payload: dict[str, list[dict[str, Any]]] = {}
    for driver_id, by_day in days.items():
        grouped = []
        for day in sorted(by_day):
            rows = by_day[day]
            counted = [row for row in rows if row.get("kind") != "open"]
            grouped.append(
                {
                    "date": day,
                    "totals": {
                        "worked_minutes": _known_duration_sum(
                            _entry_known_worked(row) for row in counted
                        ),
                        "pending_minutes": sum(
                            int(row.get("proposed_worked_minutes") or 0)
                            for row in rows
                            if row.get("work_time_status") == "pending_validation"
                        ),
                        "compensated_minutes": sum(
                            int(row["compensated_minutes"] or 0) for row in rows
                        ),
                        "real_minutes": _known_duration_sum(
                            _entry_known_real(row) for row in counted
                        ),
                        "flat_minutes": _known_duration_sum(
                            _entry_known_flat(row) for row in counted
                        ),
                    },
                    "entries": rows,
                }
            )
        day_payload[str(driver_id)] = grouped

    driver_rows = [drivers[key] for key in sorted(drivers)]
    kpis = _sum_kpis(driver_rows)
    compensation_lines = [_line_dict(line) for line in lines] + [
        _line_dict(line) for line in manual_lines
    ]
    _stamp_real_minutes(compensation_lines, entries)
    return {
        "period": {"from": period_from, "to": period_to},
        "cutover_at": _iso(cutover),
        "kpis": kpis,
        "contractual_rules": _contractual_rules(policies, period_from, period_to),
        "drivers": driver_rows,
        "days_by_driver": day_payload,
        "compensation_lines": compensation_lines,
        "compensation_source": "dynamic",
        "review_items": review_items,
    }


def _stamp_real_minutes(
    lines: list[dict[str, Any]], entries: list[dict[str, Any]]
) -> None:
    """Fige la durée réelle connue au moment du calcul, pour la relecture du snapshot.

    ``-1`` veut dire que la durée réelle n'était pas connue. Une validation
    ultérieure ne doit pas la faire apparaître dans une période déjà clôturée.
    """
    totals: dict[str, int] = {}
    unknown: set[str] = set()
    for entry in entries:
        kind = entry.get("kind")
        if kind not in {"transport", "manual"}:
            continue
        key = str(entry.get("line_key") or "")
        if not key:
            continue
        real = entry.get("real_minutes")
        if kind == "manual":
            if isinstance(real, int):
                totals[key] = int(real)
            continue
        if not isinstance(real, int):
            unknown.add(key)
            continue
        totals[key] = totals.get(key, 0) + int(real)
    for line in lines:
        key = str(line.get("line_key") or "")
        if not key:
            continue
        if line.get("classification_source") != "manual" and (
            key in unknown or key not in totals
        ):
            line["real_minutes"] = -1
        elif key in totals:
            line["real_minutes"] = totals[key]


def _known_duration_sum(values) -> int | None:
    """Somme des durées connues. Une valeur inconnue, ou l'absence de ligne, reste null."""
    items = list(values)
    if not items or any(item is None for item in items):
        return None
    return sum(int(item) for item in items)


def _entry_known_worked(row: dict[str, Any]) -> int | None:
    if row.get("kind") == "open" or row.get("work_time_status") == "pending_validation":
        return None
    worked = row.get("worked_minutes")
    if isinstance(worked, int):
        return worked
    real = row.get("real_minutes")
    if isinstance(real, int):
        return real
    return None


def _entry_known_real(row: dict[str, Any]) -> int | None:
    if row.get("kind") == "open" or row.get("work_time_status") == "pending_validation":
        return None
    real = row.get("real_minutes")
    if isinstance(real, int):
        return real
    worked = row.get("worked_minutes")
    if isinstance(worked, int):
        return worked
    return None


def _entry_known_flat(row: dict[str, Any]) -> int | None:
    if row.get("kind") == "open":
        return None
    if row.get("flat_status") == "calculated" and isinstance(
        row.get("flat_minutes"), int
    ):
        return int(row["flat_minutes"])
    if row.get("kind") in {"transport", "manual"}:
        return None
    return None


def _sum_kpis(driver_rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "completed_segments_count",
        "incomplete_segments_count",
        "completion_time_missing_count",
        "compensated_journeys_count",
        "pending_journeys_count",
        "review_journeys_count",
        "worked_verified_minutes",
        "worked_estimated_minutes",
        "worked_adjusted_minutes",
        "worked_validated_minutes",
        "manual_minutes",
        "pending_validation_minutes",
        "total_worked_minutes",
        "compensated_minutes",
        "transport_count",
        "real_transport_minutes",
        "flat_transport_minutes",
        "real_added_minutes",
        "flat_added_minutes",
        "review_count",
        "review_count_real",
        "review_count_flat",
        "anomalies_count",
    ]
    kpis = {key: sum(int(row.get(key) or 0) for row in driver_rows) for key in keys}
    kpis["active_drivers"] = sum(
        1
        for row in driver_rows
        if int(row.get("total_worked_minutes") or 0) > 0
        or int(row.get("pending_validation_minutes") or 0) > 0
        or int(row.get("incomplete_segments_count") or 0) > 0
    )
    kpis["anomalies"] = kpis["anomalies_count"]
    return kpis


def _contractual_rules(
    policies: list[PolicyView], period_from: str, period_to: str
) -> dict[str, Any]:
    """Versions dont la fenêtre coupe la période. La clôture les applique ligne à ligne."""
    versions = []
    for policy in policies:
        start = policy.effective_from.isoformat()
        end = policy.effective_until.isoformat() if policy.effective_until else None
        if start > period_to:
            continue
        if end is not None and end <= period_from:
            continue
        versions.append(
            {
                "policy_id": policy.policy_id,
                "driver_id": policy.driver_id,
                "mode": policy.mode,
                "transport_flat_minutes": int(policy.transport_flat_minutes),
                "effective_from": start,
                "effective_until": end,
            }
        )
    versions.sort(key=lambda item: (item["effective_from"], item["policy_id"] or 0))
    return {"version_count": len(versions), "versions": versions}


def _driver_bucket(report: dict[str, Any], driver_id: int) -> dict[str, Any]:
    for item in report["drivers"]:
        if item["driver_id"] == driver_id:
            return item
    bucket = _empty_driver(driver_id)
    report["drivers"].append(bucket)
    return bucket


DISPLAY_PARITY_KEYS = (
    "transport_count",
    "real_transport_minutes",
    "flat_transport_minutes",
    "real_added_minutes",
    "flat_added_minutes",
    "review_count_real",
    "review_count_flat",
)


def closure_parity_errors(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Écarts entre le rapport ouvert et le même rapport relu depuis le snapshot."""
    errors = _parity_bucket(
        "période", before.get("kpis") or {}, after.get("kpis") or {}
    )
    before_drivers = {
        int(row["driver_id"]): row
        for row in before.get("drivers") or []
        if row.get("driver_id") is not None
    }
    after_drivers = {
        int(row["driver_id"]): row
        for row in after.get("drivers") or []
        if row.get("driver_id") is not None
    }
    for driver_id in sorted(set(before_drivers) | set(after_drivers)):
        errors.extend(
            _parity_bucket(
                f"chauffeur {driver_id}",
                before_drivers.get(driver_id) or {},
                after_drivers.get(driver_id) or {},
            )
        )
    return errors


def _parity_bucket(
    label: str, before: dict[str, Any], after: dict[str, Any]
) -> list[str]:
    errors = []
    for key in DISPLAY_PARITY_KEYS:
        left = int(before.get(key) or 0)
        right = int(after.get(key) or 0)
        if left != right:
            errors.append(f"{label}.{key}: {left} → {right}")
    return errors


def _snapshot_fields(snapshot: dict[str, Any], minutes: int) -> dict[str, Any]:
    return {
        "compensated_minutes": minutes,
        "flat_minutes": snapshot.get("flat_minutes"),
        "compensation_status": snapshot.get("compensation_status"),
        "rule_type": snapshot.get("rule_type"),
        "base_minutes": snapshot.get("base_minutes"),
        "intermediate_stop_count": snapshot.get("intermediate_stop_count"),
        "intermediate_stop_minutes": snapshot.get("intermediate_stop_minutes"),
        "policy_id": snapshot.get("policy_id"),
        "classification_source": snapshot.get("classification_source"),
        "journey_status": snapshot.get("journey_status"),
        "accounting_date": snapshot.get("accounting_date"),
    }


def _freeze_unattached(
    report: dict[str, Any], snapshot: dict[str, Any], minutes: int
) -> None:
    """Garde la minute figée visible même si la course a changé de jour."""
    driver_id = snapshot.get("driver_id")
    day = snapshot.get("accounting_date")
    if driver_id is None or not day:
        return
    entry = {
        "date": day,
        "kind": "ledger_snapshot",
        "driver_id": int(driver_id),
        "booking_id": snapshot.get("attached_to_booking_id"),
        "journey_key": snapshot.get("journey_key"),
        "worked_minutes": 0,
        "compensated_minutes": minutes,
        "quality": "ledger",
        "is_manual": False,
        "is_manually_adjusted": False,
        "anomalies": ["ledger_detached_from_live_entry"],
        "compensation": _snapshot_fields(snapshot, minutes),
    }
    days = report["days_by_driver"].setdefault(str(driver_id), [])
    bucket = next((item for item in days if item["date"] == day), None)
    if bucket is None:
        days.append(
            {
                "date": day,
                "totals": {"worked_minutes": 0, "compensated_minutes": 0},
                "entries": [entry],
            }
        )
        days.sort(key=lambda item: item["date"])
        return
    bucket["entries"].append(entry)


def _synthetic_review_entry(row: dict[str, Any]) -> dict[str, Any]:
    owner = row.get("driver_id")
    return {
        "driver_id": int(owner) if owner is not None else 0,
        "date": row.get("accounting_date"),
        "kind": "ledger_snapshot",
        "booking_id": row.get("booking_id") or row.get("attached_to_booking_id"),
        "entry_id": row.get("manual_entry_id"),
        "line_key": row.get("line_key"),
        "pickup_label": "",
        "dropoff_label": "",
        "description": "",
        "work_type": None,
        "anomalies": [],
    }


def _line_real_display(
    report: dict[str, Any], row: dict[str, Any], legacy: bool
) -> tuple[int, bool, dict[str, Any] | None, bool]:
    """Minutes réelles des entrées de la ligne, et la première entrée sans durée."""
    owner = row.get("driver_id")
    if owner is None:
        return 0, False, None, False
    key = (
        str(row.get("journey_key") or "") if legacy else str(row.get("line_key") or "")
    )
    if not key:
        return 0, False, None, False
    total = 0
    missing: dict[str, Any] | None = None
    found = False
    for day in report.get("days_by_driver", {}).get(str(int(owner)), []):
        for entry in day.get("entries") or []:
            entry_key = (
                str(entry.get("journey_key") or "")
                if legacy
                else str(entry.get("line_key") or "")
            )
            if entry_key != key or entry.get("kind") == "manual":
                continue
            found = True
            real = entry.get("real_minutes")
            if isinstance(real, int):
                total += real
            elif missing is None:
                missing = entry
    if not found:
        return 0, False, None, False
    return total, missing is not None, missing, True


def apply_ledger_snapshot(
    report: dict[str, Any], ledger_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Remplace la rémunération dynamique par le snapshot figé.

    Les totaux rémunérés viennent uniquement des lignes du ledger. Une course
    déplacée après la clôture ne les efface pas : la minute reste sur la date
    comptable figée.
    """
    legacy = bool(ledger_rows) and all(
        str(row.get("line_key") or "").startswith("legacy:")
        or not str(row.get("line_key") or "")
        for row in ledger_rows
    )
    if legacy:
        by_key = {str(row.get("journey_key") or ""): row for row in ledger_rows}
    else:
        by_key = {
            str(row.get("line_key") or row.get("journey_key") or ""): row
            for row in ledger_rows
        }
    report["review_items"] = []
    for days in report.get("days_by_driver", {}).values():
        for day in days:
            for entry in day.get("entries") or []:
                entry.pop("review_modes", None)

    for driver in report["drivers"]:
        driver["compensated_minutes"] = 0
        driver["compensated_journeys_count"] = 0
        driver["pending_journeys_count"] = 0
        driver["review_journeys_count"] = 0
        driver["transport_count"] = 0
        driver["real_transport_minutes"] = 0
        driver["flat_transport_minutes"] = 0
        driver["real_added_minutes"] = 0
        driver["flat_added_minutes"] = 0
        driver["review_count"] = 0
        driver["review_count_real"] = 0
        driver["review_count_flat"] = 0

    attached: set[str] = set()
    for days in report["days_by_driver"].values():
        for day in days:
            for entry in day["entries"]:
                key = (
                    str(entry.get("journey_key") or "")
                    if legacy
                    else str(entry.get("line_key") or "")
                )
                snapshot = by_key.get(key) if key else None
                if snapshot is None:
                    entry["compensated_minutes"] = 0
                    if entry.get("compensation"):
                        entry["compensation"]["compensated_minutes"] = 0
                        entry["compensation"]["compensation_status"] = "requires_review"
                    if "outside_finalized_snapshot" not in entry["anomalies"]:
                        entry["anomalies"].append("outside_finalized_snapshot")
                    continue
                same_day = snapshot.get("accounting_date") == entry["date"]
                if legacy:
                    pay = same_day and (
                        entry.get("kind") == "manual"
                        or snapshot.get("attached_to_booking_id")
                        == entry.get("booking_id")
                    )
                else:
                    pay = same_day
                minutes = int(snapshot.get("compensated_minutes") or 0) if pay else 0
                frozen_flat = snapshot.get("flat_minutes")
                if (
                    pay
                    and snapshot.get("classification_source") != "manual"
                    and frozen_flat is not None
                    and snapshot.get("compensation_status") != "pending"
                ):
                    minutes = max(minutes, int(frozen_flat))
                    entry["flat_minutes"] = int(frozen_flat)
                    entry["flat_status"] = "calculated"
                entry["compensated_minutes"] = minutes
                if entry.get("compensation"):
                    entry["compensation"].update(_snapshot_fields(snapshot, minutes))
                if pay:
                    attached.add(key)

    for row in ledger_rows:
        key = str(row.get("journey_key") if legacy else row.get("line_key") or "")
        owner = row.get("driver_id")
        if owner is None:
            continue
        bucket = _driver_bucket(report, int(owner))
        minutes = int(row.get("compensated_minutes") or 0)
        status = row.get("compensation_status")
        source = row.get("classification_source")
        flat_amount = row.get("flat_minutes")
        paid_flat = (
            source != "manual" and flat_amount is not None and status != "pending"
        )
        if status == "calculated":
            bucket["compensated_journeys_count"] += 1
            bucket["compensated_minutes"] += minutes
        elif status == "pending":
            bucket["pending_journeys_count"] += 1
        elif paid_flat:
            bucket["compensated_journeys_count"] += 1
            bucket["compensated_minutes"] += int(flat_amount)
        else:
            bucket["review_journeys_count"] += 1
        if source == "manual" and status == "calculated":
            recorded = row.get("real_minutes")
            if not isinstance(recorded, int) or recorded < 0:
                matched = _ledger_entry(report, row, legacy)
                live_real = matched.get("real_minutes") if matched else None
                recorded = live_real if isinstance(live_real, int) else minutes
            bucket["real_added_minutes"] += int(recorded)
            bucket["flat_added_minutes"] += int(
                flat_amount if flat_amount is not None else minutes
            )
        elif source == "manual" and status not in {"calculated", "pending"}:
            matched = _ledger_entry(report, row, legacy) or _synthetic_review_entry(row)
            bucket["review_count_flat"] += 1
            _remember_review(report["review_items"], matched, real=False, flat=True)
        elif source != "manual":
            bucket["transport_count"] += 1
            rule = str(row.get("rule_type") or "")
            if status == "calculated" and rule == "validated_work_time":
                bucket["real_transport_minutes"] += minutes
            elif paid_flat:
                bucket["flat_transport_minutes"] += int(flat_amount)
            elif status == "calculated":
                bucket["flat_transport_minutes"] += minutes
            elif status != "pending":
                matched = _ledger_entry(report, row, legacy) or _synthetic_review_entry(
                    row
                )
                bucket["review_count_flat"] += 1
                _remember_review(report["review_items"], matched, real=False, flat=True)
            if status != "pending" and rule != "validated_work_time":
                stored_real = row.get("real_minutes")
                if isinstance(stored_real, int) and stored_real >= 0:
                    bucket["real_transport_minutes"] += stored_real
                elif isinstance(stored_real, int) and stored_real < 0:
                    target = _ledger_entry(
                        report, row, legacy
                    ) or _synthetic_review_entry(row)
                    bucket["review_count_real"] += 1
                    _remember_review(
                        report["review_items"], target, real=True, flat=False
                    )
                elif str(row.get("line_key") or "").startswith("bk:"):
                    real_total, real_missing, missing_entry, found = _line_real_display(
                        report, row, legacy
                    )
                    if found:
                        bucket["real_transport_minutes"] += real_total
                    if real_missing:
                        target = missing_entry or _synthetic_review_entry(row)
                        bucket["review_count_real"] += 1
                        _remember_review(
                            report["review_items"], target, real=True, flat=False
                        )
        if status == "calculated" and key not in attached and minutes:
            _freeze_unattached(report, row, minutes)

    for days in report["days_by_driver"].values():
        for day in days:
            for entry in day.get("entries") or []:
                if entry.get("kind") != "open":
                    continue
                owner = entry.get("driver_id")
                if owner is None:
                    continue
                bucket = _driver_bucket(report, int(owner))
                bucket["review_count_real"] += 1
                bucket["review_count_flat"] += 1
                _remember_review(report["review_items"], entry, real=True, flat=True)

    for driver in report["drivers"]:
        driver["review_count"] = int(driver.get("review_count_real") or 0)

    for days in report["days_by_driver"].values():
        for day in days:
            day["totals"]["compensated_minutes"] = sum(
                int(entry["compensated_minutes"] or 0) for entry in day["entries"]
            )

    report["kpis"] = _sum_kpis(report["drivers"])
    report["compensation_lines"] = list(ledger_rows)
    report["compensation_source"] = "ledger"
    report["period_finalized"] = True
    return report
