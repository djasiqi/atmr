"""Adapter pour GetCurrentUserUseCase (compatibilité avec routes)."""

from __future__ import annotations

from typing import Any

from repositories.user_repository import UserRepository
from shared.application.use_cases.get_current_user import (
    GetCurrentUserResult,
    GetCurrentUserUseCase,
)
from shared.infrastructure.adapters.jwt_adapter import JwtIdentityAdapter
from shared.infrastructure.adapters.user_repository_adapter import (
    UserRepositoryAdapter,
)


def get_current_user_via_use_case() -> Any | None:
    """Helper function pour récupérer l'utilisateur courant via use-case.

    Returns:
        User model si trouvé, None sinon.

    Raises:
        Peut lever des exceptions si l'authentification échoue.
    """
    jwt_adapter = JwtIdentityAdapter()
    user_repo_adapter = UserRepositoryAdapter(UserRepository())

    use_case = GetCurrentUserUseCase(
        get_jwt_identity_port=jwt_adapter,
        user_repo=user_repo_adapter,
    )

    result: GetCurrentUserResult = use_case.execute()
    if result.error or result.status_code:
        return None

    return result.user
