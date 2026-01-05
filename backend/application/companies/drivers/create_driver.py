from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class _UserRepo(Protocol):
    def find_by_email(self, *, email: str) -> Any | None: ...

    def find_by_username(self, *, username: str) -> Any | None: ...


class _DriverWriterPort(Protocol):
    def create_driver_for_company(
        self,
        *,
        company_id: int,
        user_attrs: dict[str, Any],
        driver_attrs: dict[str, Any],
    ) -> tuple[Any, Any]: ...


@dataclass(frozen=True, slots=True)
class CreateCompanyDriverResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    driver: Any | None = None
    user: Any | None = None
    should_trigger_dispatch: bool = False


class CreateCompanyDriverUseCase:
    """Use-case Application: créer un user+driver pour une company.

    La route reste responsable du commit, et des side-effects (audit logging / metrics).
    """

    def __init__(
        self,
        *,
        user_repo: _UserRepo,
        driver_writer: _DriverWriterPort,
        password_validator_fn: Callable[[str], bool],
        make_public_id_fn: Callable[[], str],
    ) -> None:
        super().__init__()
        self._user_repo = user_repo
        self._driver_writer = driver_writer
        self._validate_password = password_validator_fn
        self._make_public_id = make_public_id_fn

    def execute(
        self, *, company_id: int, validated_data: dict[str, Any]
    ) -> CreateCompanyDriverResult:
        email = str(validated_data["email"]).strip()
        username = str(validated_data["username"]).strip()

        existing_email = self._user_repo.find_by_email(email=email)
        existing_username = self._user_repo.find_by_username(username=username)
        if existing_email or existing_username:
            errors: list[str] = []
            if existing_email:
                errors.append("Cette adresse email est déjà utilisée.")
            if existing_username:
                errors.append("Ce nom d'utilisateur est déjà utilisé.")
            return CreateCompanyDriverResult(
                ok=False,
                error={"error": " ".join(errors)},
                status_code=409,
            )

        password = str(validated_data["password"])
        if not self._validate_password(password):
            return CreateCompanyDriverResult(
                ok=False,
                error={
                    "error": (
                        "Le mot de passe doit contenir au moins 8 caractères, "
                        "une majuscule, une minuscule et un chiffre."
                    )
                },
                status_code=400,
            )

        user_attrs = {
            "username": username,
            "first_name": validated_data["first_name"],
            "last_name": validated_data["last_name"],
            "email": email,
            "role": "driver",
            "public_id": self._make_public_id(),
            "password": password,
        }
        driver_attrs = {
            "vehicle_assigned": validated_data.get("vehicle_assigned"),
            "brand": validated_data.get("brand"),
            "license_plate": validated_data.get("license_plate"),
            "is_active": True,
            "is_available": True,
        }

        user, driver = self._driver_writer.create_driver_for_company(
            company_id=company_id,
            user_attrs=user_attrs,
            driver_attrs=driver_attrs,
        )

        return CreateCompanyDriverResult(
            ok=True,
            user=user,
            driver=driver,
            should_trigger_dispatch=True,
        )
