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

    def _validate_input(
        self, input_data: AuthenticateUserInput
    ) -> dict[str, str] | None:
        errors: dict[str, str] = {}

        if not input_data.email or len(input_data.email.strip()) == 0:
            errors["email"] = "L'identifiant est requis"

        if not input_data.password or len(input_data.password.strip()) == 0:
            errors["password"] = "Le mot de passe est requis"

        return errors if errors else None
