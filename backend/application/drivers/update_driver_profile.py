from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class UpdateDriverProfileResult:
    response: dict[str, Any]
    status_code: int
    should_commit: bool = False


class UpdateDriverProfileUseCase:
    """Use-case Application: appliquer les champs validés sur le profil chauffeur."""

    def execute(
        self, *, driver: Any, validated_data: dict[str, Any]
    ) -> UpdateDriverProfileResult:
        if not getattr(driver, "user", None):
            return UpdateDriverProfileResult(
                response={"error": "Aucun utilisateur associé au driver"},
                status_code=500,
                should_commit=False,
            )

        user = driver.user

        # Champs user
        first_name = validated_data.get("first_name")
        if first_name:
            user.first_name = first_name
        last_name = validated_data.get("last_name")
        if last_name:
            user.last_name = last_name
        phone = validated_data.get("phone")
        if phone:
            user.phone = phone

        # Statut (UI legacy)
        status = validated_data.get("status")
        if status:
            try:
                status_val = str(status).strip().lower()
            except Exception:
                status_val = ""
            if status_val == "disponible":
                driver.is_active = True
            elif status_val == "hors service":
                driver.is_active = False

        # HR fields
        contract_type = validated_data.get("contract_type")
        if contract_type:
            driver.contract_type = str(contract_type).upper()
        weekly_hours = validated_data.get("weekly_hours")
        if weekly_hours is not None:
            driver.weekly_hours = weekly_hours
        hourly_rate_cents = validated_data.get("hourly_rate_cents")
        if hourly_rate_cents is not None:
            driver.hourly_rate_cents = hourly_rate_cents

        # Dates (déjà validées côté schema)
        employment_start_date = validated_data.get("employment_start_date")
        if employment_start_date:
            driver.employment_start_date = employment_start_date
        employment_end_date = validated_data.get("employment_end_date")
        if employment_end_date:
            driver.employment_end_date = employment_end_date
        license_valid_until = validated_data.get("license_valid_until")
        if license_valid_until:
            driver.license_valid_until = license_valid_until
        medical_valid_until = validated_data.get("medical_valid_until")
        if medical_valid_until:
            driver.medical_valid_until = medical_valid_until

        # Listes
        license_categories = validated_data.get("license_categories")
        if license_categories:
            driver.license_categories = [str(cat) for cat in license_categories]
        trainings = validated_data.get("trainings")
        if trainings:
            driver.trainings = trainings

        return UpdateDriverProfileResult(
            response={
                "profile": driver.serialize,
                "message": "Profil mis à jour avec succès",
            },
            status_code=200,
            should_commit=True,
        )
