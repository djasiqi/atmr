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


class _VehicleWriterPort(Protocol):
    def create_vehicle(self, *, company_id: int, attrs: dict[str, Any]) -> Any: ...


@dataclass(frozen=True, slots=True)
class CreateCompanyVehicleResult:
    ok: bool
    vehicle: Any | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class CreateCompanyVehicleUseCase:
    """Use-case Application: créer un véhicule pour une company."""

    def __init__(self, *, vehicle_writer: _VehicleWriterPort) -> None:
        super().__init__()
        self._vehicle_writer = vehicle_writer

    def execute(
        self, *, company_id: int, data: dict[str, Any]
    ) -> CreateCompanyVehicleResult:
        def _fail(msg: str, code: int = 400) -> CreateCompanyVehicleResult:
            return CreateCompanyVehicleResult(
                ok=False, error={"error": msg}, status_code=code
            )

        model = (
            (data.get("model") or "").strip()
            if isinstance(data.get("model"), str)
            else data.get("model")
        )
        license_plate = (
            (data.get("license_plate") or "").strip()
            if isinstance(data.get("license_plate"), str)
            else data.get("license_plate")
        )

        if not model:
            return _fail("Le modèle est requis")
        if not license_plate:
            return _fail("La plaque d'immatriculation est requise")

        try:
            # Conversions optionnelles
            year_val = data.get("year")
            if year_val is None or (isinstance(year_val, str) and not year_val.strip()):
                year: int | None = None
            else:
                year = int(year_val)

            seats_val = data.get("seats")
            if seats_val is None or (
                isinstance(seats_val, str) and not str(seats_val).strip()
            ):
                seats: int | None = None
            else:
                seats = int(seats_val)

            vin_val = data.get("vin")
            vin = str(vin_val).strip() if vin_val and str(vin_val).strip() else None

            insurance_company_name_val = data.get("insurance_company_name")
            insurance_company_name = (
                str(insurance_company_name_val).strip()
                if insurance_company_name_val and str(insurance_company_name_val).strip()
                else None
            )

            wheelchair_accessible = bool(data.get("wheelchair_accessible", False))
            is_active = bool(data.get("is_active", True))

            insurance_expires_at = _parse_dt(data.get("insurance_expires_at"))
            inspection_expires_at = _parse_dt(data.get("inspection_expires_at"))
            tachograph_expires_at = _parse_dt(data.get("tachograph_expires_at"))
        except Exception as e:
            return _fail(str(e))

        attrs: dict[str, Any] = {
            "model": model,
            "license_plate": license_plate,
            "year": year,
            "vin": vin,
            "seats": seats,
            "wheelchair_accessible": wheelchair_accessible,
            # compat champ alternatif
            "is_wheelchair_accessible": wheelchair_accessible,
            "is_active": is_active,
            "insurance_company_name": insurance_company_name,
            "insurance_expires_at": insurance_expires_at,
            "inspection_expires_at": inspection_expires_at,
            "tachograph_expires_at": tachograph_expires_at,
        }

        try:
            vehicle = self._vehicle_writer.create_vehicle(
                company_id=company_id, attrs=attrs
            )
        except ValueError as e:
            return _fail(str(e))

        return CreateCompanyVehicleResult(ok=True, vehicle=vehicle)
