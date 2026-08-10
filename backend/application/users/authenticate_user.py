"""Use-case: authentifier un utilisateur (login)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AuthenticateUserInput:
    """Input pour authentifier un utilisateur."""

    email: str  # email ou identifiant institution (slug/username)
    password: str


@dataclass(frozen=True, slots=True)
class AuthenticateUserOutput:
    success: bool
    access_token: str | None = None
    refresh_token: str | None = None
    user: Any | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class AuthenticateUserUseCase:
    """Use-case Application: authentifier un utilisateur (login)."""

    def execute(self, input_data: AuthenticateUserInput) -> AuthenticateUserOutput:
        validation_error = self._validate_input(input_data)
        if validation_error:
            return AuthenticateUserOutput(
                success=False,
                error=validation_error,
                status_code=400,
            )

        from repositories.user_repository import UserRepository

        user_repo = UserRepository()
        identifier = input_data.email.strip()
        user = self._resolve_user(user_repo, identifier)

        if not user:
            return AuthenticateUserOutput(
                success=False,
                error={"error": "invalid_credentials"},
                status_code=401,
            )

        inactive_error = self._inactive_profile_error(user)
        if inactive_error is not None:
            return inactive_error

        if not user.check_password(input_data.password):
            return AuthenticateUserOutput(
                success=False,
                error={"error": "invalid_password"},
                status_code=401,
            )

        return AuthenticateUserOutput(success=True, user=user)

    def _resolve_user(self, user_repo, identifier: str):
        lowered = identifier.strip().lower()
        return user_repo.find_model_by_email(
            lowered
        ) or user_repo.find_model_by_username(lowered)

    def _inactive_profile_error(self, user) -> AuthenticateUserOutput | None:
        """Refuse la connexion si le profil métier associé est désactivé."""
        from models.enums import UserRole

        if getattr(user, "account_status", None) == "pending_activation":
            return AuthenticateUserOutput(
                success=False,
                error={"error": "account_pending_activation"},
                status_code=403,
            )

        if user.role == UserRole.DRIVER:
            driver = getattr(user, "driver", None)
            if driver is not None and not getattr(driver, "is_active", True):
                return AuthenticateUserOutput(
                    success=False,
                    error={"error": "account_disabled"},
                    status_code=403,
                )

        if user.role == UserRole.CLIENT:
            from models import Client

            client_rows = Client.query.filter_by(user_id=user.id).all()
            if client_rows and not any(c.is_active for c in client_rows):
                return AuthenticateUserOutput(
                    success=False,
                    error={"error": "account_disabled"},
                    status_code=403,
                )

        return None

    def _validate_input(
        self, input_data: AuthenticateUserInput
    ) -> dict[str, str] | None:
        errors: dict[str, str] = {}

        if not input_data.email or len(input_data.email.strip()) == 0:
            errors["email"] = "L'identifiant est requis"

        if not input_data.password or len(input_data.password.strip()) == 0:
            errors["password"] = "Le mot de passe est requis"

        return errors if errors else None
