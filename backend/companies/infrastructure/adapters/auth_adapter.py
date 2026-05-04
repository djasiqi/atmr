"""Adapter pour AuthService (compatibilité temporaire)."""

from __future__ import annotations

from typing import Any

from companies.application.use_cases.get_current_company import GetCurrentUserPort
from shared.application.use_cases.get_current_user import (
    GetJwtIdentityPort,
    UserRepositoryPort,
)


class AuthServiceAdapter(GetCurrentUserPort):
    """Adapter qui adapte GetCurrentUserUseCase vers GetCurrentUserPort.

    Permet d'utiliser GetCurrentUserUseCase avec les nouveaux use-cases.
    """

    def __init__(self, get_current_user_fn: Any | None = None) -> None:  # pyright: ignore[reportMissingSuperCall]
        """Initialise l'adapter.

        Args:
            get_current_user_fn: Fonction pour obtenir l'utilisateur courant
                (par défaut: get_current_user_via_use_case)
        """
        if get_current_user_fn is None:
            from shared.infrastructure.adapters.auth_adapter import (
                get_current_user_via_use_case,
            )

            self.get_current_user_fn = get_current_user_via_use_case
        else:
            self.get_current_user_fn = get_current_user_fn

    def get_current_user(self) -> Any | None:  # pyright: ignore[reportImplicitOverride]
        """Récupère l'utilisateur courant via GetCurrentUserUseCase."""
        return self.get_current_user_fn()


class JwtIdentityAdapter(GetJwtIdentityPort):
    """Adapter pour récupérer l'identité JWT depuis flask_jwt_extended."""

    def get_jwt_identity(self) -> str | None:  # pyright: ignore[reportImplicitOverride]
        """Récupère l'identité depuis le token JWT."""
        from flask_jwt_extended import get_jwt_identity

        return get_jwt_identity()


class UserRepositoryAdapter(UserRepositoryPort):
    """Adapter qui adapte UserRepository vers UserRepositoryPort."""

    def __init__(self, user_repo: Any) -> None:  # pyright: ignore[reportMissingSuperCall]
        """Initialise l'adapter.

        Args:
            user_repo: Instance de UserRepository.
        """
        self.user_repo = user_repo

    def find_by_public_id(self, public_id: str) -> Any | None:  # pyright: ignore[reportImplicitOverride]
        """Trouve un utilisateur par public_id."""
        return self.user_repo.find_by_public_id(public_id)
