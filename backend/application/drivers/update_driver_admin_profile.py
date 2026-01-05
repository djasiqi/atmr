from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class UpdateDriverAdminProfileResult:
    response: dict[str, Any]
    status_code: int
    should_commit: bool = False


class UpdateDriverAdminProfileUseCase:
    """Use-case Application: mise à jour profil driver (endpoint admin/ops legacy)."""

    def execute(
        self, *, driver: Any, payload: dict[str, Any] | Any
    ) -> UpdateDriverAdminProfileResult:
        if not isinstance(payload, dict):
            return UpdateDriverAdminProfileResult(
                response={"error": "Missing JSON payload"},
                status_code=400,
                should_commit=False,
            )

        if "vehicle_assigned" in payload:
            driver.vehicle_assigned = payload.get(
                "vehicle_assigned", driver.vehicle_assigned
            )
        if "brand" in payload:
            driver.brand = payload.get("brand", driver.brand)
        if "license_plate" in payload:
            driver.license_plate = payload.get("license_plate", driver.license_plate)
        if "photo" in payload:
            driver.driver_photo = payload.get("photo", driver.driver_photo)

        if getattr(driver, "user", None) and "phone" in payload:
            driver.user.phone = payload.get("phone", driver.user.phone)

        return UpdateDriverAdminProfileResult(
            response={"message": "Profil mis à jour avec succès."},
            status_code=200,
            should_commit=True,
        )
