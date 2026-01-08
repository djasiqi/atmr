from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError("Date invalide")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class _VehicleLike(Protocol):
    id: int | None
    brand: str | None
    model: str | None
    license_plate: str | None
    color: str | None
    year: int | None
    seats: int | None
    is_wheelchair_accessible: bool
    wheelchair_accessible: bool  # Compat ancien champ
    is_active: bool
    notes: str | None
    insurance_expires_at: datetime | None
    inspection_expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class UpdateCompanyVehicleResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None


class UpdateCompanyVehicleUseCase:
    """Use-case Application: mise à jour d'un véhicule.

    Le parsing des dates legacy (insurance_expires_at /
    inspection_expires_at) est géré ici.
    """

    def execute(
        self,
        vehicle: _VehicleLike,
        *,
        validated_data: dict[str, Any],
        raw_data: dict[str, Any],
    ) -> UpdateCompanyVehicleResult:
        # Champs validés (schema)
        if "brand" in validated_data:
            vehicle.brand = validated_data["brand"]
        if "model" in validated_data:
            vehicle.model = validated_data["model"]
        if "license_plate" in validated_data:
            vehicle.license_plate = validated_data["license_plate"]
        if "color" in validated_data:
            vehicle.color = validated_data["color"]
        if "year" in validated_data:
            vehicle.year = validated_data["year"]
        if "seats" in validated_data:
            vehicle.seats = validated_data["seats"]
        if "is_wheelchair_accessible" in validated_data:
            vehicle.is_wheelchair_accessible = validated_data[
                "is_wheelchair_accessible"
            ]
            # compat ancien champ
            if hasattr(vehicle, "wheelchair_accessible"):
                vehicle.wheelchair_accessible = validated_data[
                    "is_wheelchair_accessible"
                ]
        if "is_active" in validated_data:
            vehicle.is_active = validated_data["is_active"]
        if "notes" in validated_data:
            vehicle.notes = validated_data["notes"]

        # Dates legacy (pas dans le schema)
        try:
            if "insurance_expires_at" in raw_data:
                vehicle.insurance_expires_at = _parse_dt(
                    raw_data.get("insurance_expires_at")
                )
            if "inspection_expires_at" in raw_data:
                vehicle.inspection_expires_at = _parse_dt(
                    raw_data.get("inspection_expires_at")
                )
        except Exception as e:
            return UpdateCompanyVehicleResult(
                ok=False, error={"error": str(e)}, status_code=400
            )

        return UpdateCompanyVehicleResult(ok=True)
