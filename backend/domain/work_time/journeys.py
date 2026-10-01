"""Regroupement des tronçons en unités de rémunération (journeys)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

COMPLETED = {"COMPLETED", "RETURN_COMPLETED"}
CANCELLED = {"CANCELED", "CANCELLED"}


def normalize_address(address: str | None) -> str:
    """Même normalisation que les paires aller-retour facture (comparaison seule)."""
    if not address:
        return ""
    normalized = address.lower().strip()
    normalized = unicodedata.normalize("NFD", normalized)
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


@dataclass
class SegmentInput:
    """Tronçon connu, terminé ou non, servant à la structure du trajet."""

    booking_id: int
    driver_id: int | None
    status: str
    is_return: bool = False
    is_round_trip: bool = False
    parent_booking_id: int | None = None
    route_group_id: str | None = None
    route_sequence_number: int | None = None
    pickup_location: str = ""
    dropoff_location: str = ""
    scheduled_time: datetime | None = None
    arrived_at: datetime | None = None
    boarded_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: int | None = None
    pickup_lat: float | None = None
    pickup_lon: float | None = None
    dropoff_lat: float | None = None
    dropoff_lon: float | None = None
    route_provider: str | None = None
    route_duration_seconds: int | None = None
    route_distance_m: int | None = None
    routing_profile: str | None = None
    routing_status: str | None = None
    route_calculated_at: str | None = None
    assignment_status: str | None = None

    @property
    def status_key(self) -> str:
        return str(self.status or "").upper()

    @property
    def is_cancelled(self) -> bool:
        return self.status_key in CANCELLED

    @property
    def is_completed(self) -> bool:
        return self.status_key in COMPLETED


@dataclass
class Journey:
    """Unité de rémunération construite sur la structure canonique connue."""

    journey_key: str
    segments: list[SegmentInput]
    active_segments: list[SegmentInput]
    journey_class: str | None
    journey_status: str
    classification_source: str
    return_booking_ids: set[int] = field(default_factory=set)
    anomalies: list[str] = field(default_factory=list)

    @property
    def driver_ids(self) -> set[int]:
        return {
            segment.driver_id
            for segment in self.active_segments
            if segment.driver_id is not None
        }

    @property
    def outbound_segments(self) -> list[SegmentInput]:
        return [
            segment
            for segment in self.active_segments
            if segment.booking_id not in self.return_booking_ids
        ]

    @property
    def intermediate_stop_count(self) -> int:
        if self.journey_class != "round_trip":
            return 0
        return max(0, len(self.outbound_segments) - 1)

    def carrier(self) -> SegmentInput | None:
        """Tronçon porteur du forfait : le retour s'il existe, sinon l'unique aller."""
        if self.journey_status != "complete":
            return None
        for segment in reversed(self.active_segments):
            if segment.booking_id in self.return_booking_ids:
                return segment
        if self.active_segments:
            return self.active_segments[-1]
        return None


def _sort_key(segment: SegmentInput) -> tuple[int, int]:
    sequence = (
        segment.route_sequence_number
        if segment.route_sequence_number is not None
        else 10**9
    )
    return sequence, segment.booking_id


def _journey_key(segment: SegmentInput, child_parent_ids: set[int]) -> str:
    if segment.route_group_id:
        return f"rg:{segment.route_group_id}"
    if segment.parent_booking_id:
        return f"bk:{segment.parent_booking_id}"
    if segment.booking_id in child_parent_ids:
        return f"bk:{segment.booking_id}"
    return f"bk:{segment.booking_id}"


def _addresses_close_the_loop(ordered: list[SegmentInput]) -> bool:
    if len(ordered) < 2:
        return False
    start = normalize_address(ordered[0].pickup_location)
    end = normalize_address(ordered[-1].dropoff_location)
    return bool(start) and start == end


def _classify(key: str, segments: list[SegmentInput]) -> Journey:
    ordered_all = sorted(segments, key=_sort_key)
    active = [segment for segment in ordered_all if not segment.is_cancelled]
    cancelled_returns = [
        segment for segment in ordered_all if segment.is_cancelled and segment.is_return
    ]
    anomalies: list[str] = []
    if not active:
        return Journey(
            journey_key=key,
            segments=ordered_all,
            active_segments=[],
            journey_class=None,
            journey_status="undetermined",
            classification_source="no_active_segment",
            anomalies=["journey_structure_undetermined"],
        )

    has_return = any(segment.is_return for segment in active)
    linked = any(
        segment.parent_booking_id
        and any(other.booking_id == segment.parent_booking_id for other in active)
        for segment in active
    )
    all_done = all(segment.is_completed for segment in active)
    status = "complete" if all_done else "incomplete"
    return_ids: set[int] = set()

    if has_return or linked:
        journey_class = "round_trip"
        source = "is_return_flag" if has_return else "parent_booking"
        return_ids = {segment.booking_id for segment in active if segment.is_return}
        if not return_ids and linked:
            return_ids = {
                segment.booking_id
                for segment in active
                if segment.parent_booking_id is not None
            }
    elif len(active) >= 2 and _addresses_close_the_loop(active):
        journey_class = "round_trip"
        source = "address_match"
        return_ids = {active[-1].booking_id}
    elif len(active) >= 2:
        journey_class = None
        status = "undetermined"
        source = "undetermined"
        anomalies.append("journey_structure_undetermined")
    elif cancelled_returns:
        journey_class = "one_way"
        source = "cancelled_return_reevaluated"
        status = "complete" if active[0].is_completed else "incomplete"
    elif active[0].is_round_trip:
        # A/R connu (ancre) mais le retour n'existe pas encore : pas un aller simple.
        journey_class = "round_trip"
        status = "incomplete"
        source = "is_round_trip_flag"
    else:
        journey_class = "one_way"
        source = "single_segment"
        status = "complete" if active[0].is_completed else "incomplete"

    if status == "incomplete":
        anomalies.append("journey_incomplete")
    if len({segment.driver_id for segment in active if segment.driver_id}) > 1:
        anomalies.append("journey_split_across_drivers")

    return Journey(
        journey_key=key,
        segments=ordered_all,
        active_segments=active,
        journey_class=journey_class,
        journey_status=status,
        classification_source=source,
        return_booking_ids=return_ids,
        anomalies=anomalies,
    )


def build_journeys(segments: list[SegmentInput]) -> list[Journey]:
    """Regroupe par ``route_group_id``, sinon par ancre ``parent_booking_id``."""
    child_parent_ids = {
        segment.parent_booking_id
        for segment in segments
        if segment.parent_booking_id is not None
    }
    grouped: dict[str, list[SegmentInput]] = {}
    for segment in segments:
        key = _journey_key(segment, child_parent_ids)
        grouped.setdefault(key, []).append(segment)
    return [_classify(key, members) for key, members in grouped.items()]
