"""Use-case: récupérer l'utilisateur courant.

Ce use case est un wrapper autour de GetCurrentUserUseCase dans shared/application/use_cases/
pour maintenir la cohérence avec les autres use cases dans application/users/.
"""

from __future__ import annotations  # noqa: I001

from dataclasses import dataclass
from typing import Any

from repositories.user_repository import UserRepository
from shared.application.use_cases.get_current_user import (
    GetCurrentUserResult as SharedGetCurrentUserResult,
    GetCurrentUserUseCase as SharedGetCurrentUserUseCase,
)
from shared.infrastructure.adapters.jwt_adapter import JwtIdentityAdapter
from shared.infrastructure.adapters.user_repository_adapter import (
    UserRepositoryAdapter,
)


@dataclass(frozen=True, slots=True)
class GetCurrentUserResult:
    """Résultat du use-case GetCurrentUser."""

    found: bool
    user: Any | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class GetCurrentUserUseCase:
    """Use-case Application: récupérer l'utilisateur courant depuis le token JWT.

    Wrapper autour de GetCurrentUserUseCase dans shared/ pour maintenir la cohérence.
    """

    def __init__(self) -> None:
        super().__init__()
        jwt_adapter = JwtIdentityAdapter()
        user_repo_adapter = UserRepositoryAdapter(UserRepository())
        self._shared_use_case = SharedGetCurrentUserUseCase(
            get_jwt_identity_port=jwt_adapter,
            user_repo=user_repo_adapter,
        )

    def execute(self) -> GetCurrentUserResult:
        """Exécute la récupération de l'utilisateur courant.

        Returns:
            GetCurrentUserResult avec l'utilisateur si trouvé
        """
        result: SharedGetCurrentUserResult = self._shared_use_case.execute()

        if result.error or result.status_code:
            return GetCurrentUserResult(
                found=False,
                error=result.error,
                status_code=result.status_code,
            )

        return GetCurrentUserResult(found=True, user=result.user)
