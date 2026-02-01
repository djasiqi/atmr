from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from shared.time_utils import parse_local_naive

from ._status import status_value


class _BookingLike(Protocol):
    id: int | None
    status: Any
    pickup_location: Any
    dropoff_location: Any
    pickup_lat: Any
    pickup_lon: Any
    dropoff_lat: Any
    dropoff_lon: Any
    scheduled_time: Any
    medical_facility: Any
    doctor_name: Any
    notes_medical: Any
    amount: Any
    mission_type: Any
    delivery_description: Any


@dataclass(frozen=True, slots=True)
class UpdateCompanyReservationResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    updated_fields: list[str] | None = None
    should_trigger_dispatch: bool = False
    trigger_reason: str | None = None


class UpdateCompanyReservationUseCase:
    """Use-case Application: mise à jour d'une réservation (company-side).

    Autorisé uniquement si statut ∈ {PENDING, ACCEPTED, ASSIGNED}.
    """

    _ALLOWED: ClassVar[set[str]] = {"pending", "accepted", "assigned"}

    def execute(
        self,
        booking: _BookingLike,
        *,
        validated_data: dict[str, Any],
    ) -> UpdateCompanyReservationResult:
        st = status_value(getattr(booking, "status", None)).lower()
        if st not in self._ALLOWED:
            return UpdateCompanyReservationResult(
                ok=False,
                error={
                    "error": (
                        "Impossible de modifier une réservation avec le statut "
                        + f"'{status_value(getattr(booking, 'status', None))}'."
                    )
                },
                status_code=400,
            )

        updated_fields: list[str] = []

        if "pickup_location" in validated_data:
            booking.pickup_location = validated_data["pickup_location"]
            updated_fields.append("pickup_location")
        if "dropoff_location" in validated_data:
            booking.dropoff_location = validated_data["dropoff_location"]
            updated_fields.append("dropoff_location")
        if "pickup_lat" in validated_data:
            booking.pickup_lat = validated_data["pickup_lat"]
            updated_fields.append("pickup_lat")
        if "pickup_lon" in validated_data:
            booking.pickup_lon = validated_data["pickup_lon"]
            updated_fields.append("pickup_lon")
        if "dropoff_lat" in validated_data:
            booking.dropoff_lat = validated_data["dropoff_lat"]
            updated_fields.append("dropoff_lat")
        if "dropoff_lon" in validated_data:
            booking.dropoff_lon = validated_data["dropoff_lon"]
            updated_fields.append("dropoff_lon")

        if "scheduled_time" in validated_data:
            try:
                booking.scheduled_time = parse_local_naive(
                    validated_data["scheduled_time"]
                )
            except Exception as e:
                return UpdateCompanyReservationResult(
                    ok=False,
                    error={"error": f"Format de date invalide: {e}"},
                    status_code=400,
                )
            updated_fields.append("scheduled_time")

        if "medical_facility" in validated_data:
            booking.medical_facility = validated_data["medical_facility"]
            updated_fields.append("medical_facility")
        if "doctor_name" in validated_data:
            booking.doctor_name = validated_data["doctor_name"]
            updated_fields.append("doctor_name")
        if "notes_medical" in validated_data:
            booking.notes_medical = validated_data["notes_medical"]
            updated_fields.append("notes_medical")
        if "amount" in validated_data:
            booking.amount = validated_data["amount"]
            updated_fields.append("amount")
        # ✅ Livraison matériel : permettre de corriger mission_type et delivery_description
        if "mission_type" in validated_data:
            booking.mission_type = validated_data["mission_type"]
            updated_fields.append("mission_type")
        if "delivery_description" in validated_data:
            booking.delivery_description = validated_data["delivery_description"]
            updated_fields.append("delivery_description")

        if not updated_fields:
            return UpdateCompanyReservationResult(
                ok=False,
                error={"error": "Aucun champ à mettre à jour fourni."},
                status_code=400,
            )

        return UpdateCompanyReservationResult(
            ok=True,
            updated_fields=updated_fields,
            should_trigger_dispatch=True,
            trigger_reason="update",
        )
