"""Use-case: récupérer l'utilisateur courant depuis le token JWT."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class UserRepositoryPort(Protocol):
    """Port pour récupérer un utilisateur."""

    def find_by_public_id(self, public_id: str) -> Any | None:
        """Trouve un utilisateur par public_id."""
        ...


class GetJwtIdentityPort(Protocol):
    """Port pour récupérer l'identité JWT."""

    def get_jwt_identity(self) -> str | None:
        """Récupère l'identité depuis le token JWT."""
        ...


@dataclass(frozen=True, slots=True)
class GetCurrentUserResult:
    """Résultat du use-case GetCurrentUser."""

    user: Any | None
    error: dict[str, str] | None
    status_code: int | None


class GetCurrentUserUseCase:
    """Use-case Application: récupérer l'utilisateur courant depuis le token JWT.

    Remplace l'appel direct à AuthService.get_current_user() dans les routes.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        *,
        get_jwt_identity_port: GetJwtIdentityPort,
        user_repo: UserRepositoryPort,
    ) -> None:
        """Initialise le use-case.

        Args:
            get_jwt_identity_port: Port pour récupérer l'identité JWT.
            user_repo: Repository pour les utilisateurs.
        """
        self.get_jwt_identity_port = get_jwt_identity_port
        self.user_repo = user_repo

    def execute(self) -> GetCurrentUserResult:
        """Exécute la récupération de l'utilisateur courant.

        Returns:
            GetCurrentUserResult avec l'utilisateur si trouvé, ou erreur.
        """
        public_id = self.get_jwt_identity_port.get_jwt_identity()
        if not public_id:
            return GetCurrentUserResult(
                user=None,
                error={"error": "Token JWT invalide ou manquant"},
                status_code=401,
            )

        user = self.user_repo.find_by_public_id(public_id)
        if not user:
            return GetCurrentUserResult(
                user=None,
                error={"error": "Utilisateur non trouvé"},
                status_code=404,
            )

        return GetCurrentUserResult(user=user, error=None, status_code=None)
