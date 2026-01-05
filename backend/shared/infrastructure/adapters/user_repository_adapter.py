"""Adapter qui adapte UserRepository vers UserRepositoryPort."""

from __future__ import annotations

from typing import Any

from shared.application.use_cases.get_current_user import UserRepositoryPort


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
