from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


def _set_enum_value(obj: Any, attr: str, value_str: str) -> None:
    """Assigne un champ Enum sans dépendre de l'Enum ORM."""
    current = getattr(obj, attr, None)
    enum_cls = getattr(current, "__class__", None)
    candidate_name = value_str.upper()
    if enum_cls is not None and hasattr(enum_cls, candidate_name):
        setattr(obj, attr, getattr(enum_cls, candidate_name))
        return
    setattr(obj, attr, value_str.lower())


class _UserLike(Protocol):
    id: int
    first_name: Any
    last_name: Any
    email: Any
    address: Any


class _DriverLike(Protocol):
    id: int | None
    user: _UserLike | None
    is_active: Any
    driver_type: Any
    vehicle_id: Any


class _UserRepo(Protocol):
    def find_by_email_excluding_user(
        self, *, email: str, exclude_user_id: int
    ) -> Any | None: ...


class _DriverRepo(Protocol):
    def find_models_by_company_id(self, company_id: int) -> list[Any]: ...


class _VehicleRepo(Protocol):
    def find_by_id_and_company(
        self, vehicle_id: int, company_id: int
    ) -> Any | None: ...


@dataclass(frozen=True, slots=True)
class UpdateCompanyDriverResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    should_trigger_dispatch: bool = False


class UpdateCompanyDriverUseCase:
    """Use-case Application: mise à jour d'un chauffeur (user + driver + vehicle)."""

    def __init__(
        self,
        *,
        user_repo: _UserRepo,
        vehicle_repo: _VehicleRepo,
        driver_repo: _DriverRepo | None = None,
    ) -> None:
        super().__init__()
        self._user_repo = user_repo
        self._vehicle_repo = vehicle_repo
        self._driver_repo = driver_repo

    def execute(
        self,
        *,
        driver: _DriverLike,
        company_id: int,
        data: dict[str, Any],
    ) -> UpdateCompanyDriverResult:
        validation_error: dict[str, str] | None = None
        validation_status: int | None = None

        user = getattr(driver, "user", None)
        if user:
            if "first_name" in data:
                user.first_name = (
                    str(data["first_name"]).strip() if data["first_name"] else None
                )
            if "last_name" in data:
                user.last_name = (
                    str(data["last_name"]).strip() if data["last_name"] else None
                )
            if "email" in data:
                email = str(data["email"]).strip() if data["email"] else None
                if email:
                    existing = self._user_repo.find_by_email_excluding_user(
                        email=email,
                        exclude_user_id=int(user.id),
                    )
                    if existing:
                        validation_error = {
                            "error": (
                                "Cet email est déjà utilisé par un autre utilisateur"
                            )
                        }
                        validation_status = 400
                if not validation_error:
                    user.email = email
            if "address" in data:
                user.address = str(data["address"]).strip() if data["address"] else None
            if "phone" in data:
                user.phone = str(data["phone"]).strip() if data["phone"] else None
            if "birth_date" in data:
                bd_raw = data["birth_date"]
                if bd_raw:
                    from datetime import date as _date

                    if isinstance(bd_raw, str):
                        try:
                            user.birth_date = _date.fromisoformat(bd_raw)
                        except ValueError:
                            validation_error = {"error": "Format de date invalide (AAAA-MM-JJ)"}
                            validation_status = 400
                    elif isinstance(bd_raw, _date):
                        user.birth_date = bd_raw
                else:
                    user.birth_date = None

        # Champs identite / urgence (sur le driver)
        if not validation_error:
            _str_fields = [
                ("avs_number", 16),
                ("nationality", 60),
                ("emergency_contact_name", 120),
                ("emergency_contact_phone", 30),
            ]
            for field_name, max_len in _str_fields:
                if field_name in data:
                    val = data[field_name]
                    val = str(val).strip()[:max_len] if val else None
                    setattr(driver, field_name, val)

        # Champs HR
        if not validation_error:
            if "contract_type" in data:
                driver.contract_type = str(data["contract_type"]).strip() if data["contract_type"] else "CDI"
            if "weekly_hours" in data:
                wh = data["weekly_hours"]
                driver.weekly_hours = int(wh) if wh else None
            if "hourly_rate_cents" in data:
                hrc = data["hourly_rate_cents"]
                driver.hourly_rate_cents = int(hrc) if hrc else None
            _date_fields = [
                "employment_start_date",
                "employment_end_date",
                "license_valid_until",
                "medical_valid_until",
            ]
            for df in _date_fields:
                if df in data:
                    import contextlib
                    from datetime import date as _date

                    dv = data[df]
                    if dv:
                        if isinstance(dv, str):
                            with contextlib.suppress(ValueError):
                                setattr(driver, df, _date.fromisoformat(dv))
                        elif isinstance(dv, _date):
                            setattr(driver, df, dv)
                    else:
                        setattr(driver, df, None)
            if "license_categories" in data:
                lc = data["license_categories"]
                driver.license_categories = lc if isinstance(lc, list) else []

        if not validation_error and "is_active" in data:
            driver.is_active = bool(data["is_active"])

        if not validation_error and "driver_type" in data:
            try:
                _set_enum_value(driver, "driver_type", str(data["driver_type"]))
            except Exception:
                validation_error = {
                    "error": "Type de chauffeur invalide: REGULAR | EMERGENCY"
                }
                validation_status = 400

        if not validation_error and "vehicle_id" in data:
            vehicle_id = data["vehicle_id"]
            if vehicle_id is None or vehicle_id == "":
                driver.vehicle_id = None
                driver.vehicle_assigned = None
                driver.brand = None
                driver.license_plate = None
            else:
                try:
                    vehicle_id_int = int(vehicle_id)
                except (ValueError, TypeError):
                    validation_error = {
                        "error": "vehicle_id doit être un nombre entier valide"
                    }
                    validation_status = 400
                else:
                    vehicle = self._vehicle_repo.find_by_id_and_company(
                        vehicle_id_int, company_id
                    )
                    if vehicle:
                        if self._driver_repo:
                            all_drivers = self._driver_repo.find_models_by_company_id(
                                company_id
                            )
                            for d in all_drivers:
                                if (
                                    getattr(d, "vehicle_id", None) == vehicle_id_int
                                    and getattr(d, "id", None) != getattr(driver, "id", None)
                                ):
                                    d.vehicle_id = None
                                    d.vehicle_assigned = None
                                    d.brand = None
                                    d.license_plate = None
                        driver.vehicle_id = vehicle_id_int
                        driver.vehicle_assigned = getattr(vehicle, "model", None)
                        driver.license_plate = getattr(vehicle, "license_plate", None)
                    else:
                        validation_error = {
                            "error": (
                                f"Véhicule {vehicle_id_int} non trouvé ou "
                                f"n'appartient pas à cette entreprise"
                            )
                        }
                        validation_status = 400

        if validation_error:
            return UpdateCompanyDriverResult(
                ok=False, error=validation_error, status_code=validation_status or 400
            )

        return UpdateCompanyDriverResult(ok=True, should_trigger_dispatch=True)
