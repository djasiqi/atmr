from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


def _status_value(status: Any) -> str:
    """Normalise un statut (Enum SQLAlchemy, str, etc.) en str."""
    if status is None:
        return ""
    v = getattr(status, "value", None)
    if isinstance(v, str):
        return v
    return str(status)


def _set_status(obj: Any, status_str: str) -> None:
    """Assigne un statut sans dépendre de l'Enum ORM.

    - Si `obj.status` est un Enum (ex: BookingStatus), on tente
      `BookingStatus.PENDING`, etc.
    - Sinon, on assigne directement la string.
    """
    current = getattr(obj, "status", None)
    enum_cls = getattr(current, "__class__", None)
    candidate_name = status_str.upper()
    if enum_cls is not None and hasattr(enum_cls, candidate_name):
        obj.status = getattr(enum_cls, candidate_name)
        return
    obj.status = status_str


class _BookingLike(Protocol):
    id: int | None
    status: Any

    pickup_location: str | None
    dropoff_location: str | None
    scheduled_time: Any
    amount: Any
    medical_facility: str | None
    doctor_name: str | None
    notes_medical: str | None

    pickup_lat: Any
    pickup_lon: Any
    dropoff_lat: Any
    dropoff_lon: Any


@dataclass(frozen=True, slots=True)
class UpdatePendingBookingResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    addresses_changed: bool = False
    old_pickup: str | None = None
    old_dropoff: str | None = None
    new_pickup: str | None = None
    new_dropoff: str | None = None


class UpdatePendingBookingUseCase:
    """Use-case Application: mise à jour d'une réservation **PENDING**.

    La couche Application ne commit pas et ne déclenche pas d'effets de bord:
    - la route décide quand commit/rollback
    - l'invalidation cache / dispatch est pilotée à l'extérieur via le résultat.
    """

    def execute(
        self, booking: _BookingLike, *, validated_data: dict[str, Any]
    ) -> UpdatePendingBookingResult:
        if _status_value(getattr(booking, "status", None)) != "pending":
            return UpdatePendingBookingResult(
                ok=False,
                error={
                    "error": "Seules les réservations en attente peuvent être modifiées"
                },
                status_code=400,
            )

        old_pickup = getattr(booking, "pickup_location", None)
        old_dropoff = getattr(booking, "dropoff_location", None)
        addresses_changed = False

        if "pickup_location" in validated_data:
            new_pickup = validated_data.get("pickup_location")
            booking.pickup_location = new_pickup
            if old_pickup != new_pickup:
                addresses_changed = True

        if "dropoff_location" in validated_data:
            new_dropoff = validated_data.get("dropoff_location")
            booking.dropoff_location = new_dropoff
            if old_dropoff != new_dropoff:
                addresses_changed = True

        if "scheduled_time" in validated_data:
            booking.scheduled_time = validated_data.get("scheduled_time")

        if "amount" in validated_data:
            booking.amount = validated_data.get("amount")

        if "medical_facility" in validated_data:
            booking.medical_facility = validated_data.get("medical_facility")

        if "doctor_name" in validated_data:
            booking.doctor_name = validated_data.get("doctor_name")

        if "notes_medical" in validated_data:
            booking.notes_medical = validated_data.get("notes_medical")

        # Par sécurité: empêche un update de statut via cette route
        # (si payload contient "status").
        # La transition de statut est gérée par d'autres flows.
        if "status" in validated_data:
            # ignorer silencieusement
            pass

        # Réaffirme le statut pending (en cas de payload incohérent côté client).
        _set_status(booking, "pending")

        return UpdatePendingBookingResult(
            ok=True,
            addresses_changed=addresses_changed,
            old_pickup=old_pickup,
            old_dropoff=old_dropoff,
            new_pickup=getattr(booking, "pickup_location", None),
            new_dropoff=getattr(booking, "dropoff_location", None),
        )
