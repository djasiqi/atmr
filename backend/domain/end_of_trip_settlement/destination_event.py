"""Arrivée à destination, à l'étude. Le parcours réel ne l'utilise pas.

L'événement serait horodaté par le serveur. Il ne terminerait pas la
mission, ne créerait pas de paiement et ne prouverait pas l'immobilisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

DESTINATION_ARRIVAL_EVENT_ENABLED = False


@dataclass(frozen=True, slots=True)
class DestinationArrival:
    booking_id: int
    driver_id: int
    recorded_at: datetime | None
    applied: bool
    reason: str
    completes_mission: bool
    creates_payment: bool
    geo_proof: bool
    immobilization_proof: bool
    idempotent_replay: bool


def record_destination_arrival(
    existing: DestinationArrival | None,
    *,
    enabled: bool,
    booking_id: int,
    driver_id: int,
    assigned_driver_id: int,
    server_time: datetime,
) -> DestinationArrival:
    """Décrit l'événement. Le drapeau produit reste faux."""
    if not enabled:
        return DestinationArrival(
            booking_id=booking_id,
            driver_id=driver_id,
            recorded_at=None,
            applied=False,
            reason="not_activated",
            completes_mission=False,
            creates_payment=False,
            geo_proof=False,
            immobilization_proof=False,
            idempotent_replay=False,
        )
    if server_time.tzinfo is None or driver_id < 1 or booking_id < 1:
        raise ValueError("Arrivée à destination incomplète.")
    if driver_id != assigned_driver_id:
        return DestinationArrival(
            booking_id=booking_id,
            driver_id=driver_id,
            recorded_at=None,
            applied=False,
            reason="driver_not_assigned",
            completes_mission=False,
            creates_payment=False,
            geo_proof=False,
            immobilization_proof=False,
            idempotent_replay=False,
        )
    if existing is not None and existing.booking_id == booking_id and existing.applied:
        return DestinationArrival(
            booking_id=existing.booking_id,
            driver_id=existing.driver_id,
            recorded_at=existing.recorded_at,
            applied=True,
            reason="replay",
            completes_mission=False,
            creates_payment=False,
            geo_proof=False,
            immobilization_proof=False,
            idempotent_replay=True,
        )
    return DestinationArrival(
        booking_id=booking_id,
        driver_id=driver_id,
        recorded_at=server_time,
        applied=True,
        reason="recorded_for_study",
        completes_mission=False,
        creates_payment=False,
        geo_proof=False,
        immobilization_proof=False,
        idempotent_replay=False,
    )


def propose_destination_arrival(
    existing: DestinationArrival | None,
    *,
    booking_id: int,
    driver_id: int,
    assigned_driver_id: int,
    server_time: datetime,
) -> DestinationArrival:
    """Point d'entrée produit. Il reste inactif."""
    return record_destination_arrival(
        existing,
        enabled=DESTINATION_ARRIVAL_EVENT_ENABLED,
        booking_id=booking_id,
        driver_id=driver_id,
        assigned_driver_id=assigned_driver_id,
        server_time=server_time,
    )
