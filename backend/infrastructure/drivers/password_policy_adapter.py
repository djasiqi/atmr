from __future__ import annotations

from dataclasses import dataclass

from security.password_policy import PasswordPolicyService


@dataclass(frozen=True, slots=True)
class PasswordPolicyAdapter:
    """Adapter Infrastructure: proxy vers `security.password_policy.PasswordPolicyService`."""

    def validate_password(
        self, *, password: str, user_id: int, check_history: bool
    ) -> None:
        PasswordPolicyService.validate_password(
            password,
            user_id=user_id,
            check_history=check_history,
        )
