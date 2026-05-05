from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _UserLike(Protocol):
    id: int
    force_password_change: bool

    def set_password(self, password: str) -> None: ...


class _PasswordPolicyPort(Protocol):
    def validate_password(
        self, *, password: str, user_id: int, check_history: bool
    ) -> None: ...


def _generate_strong_password(*, length: int = 16) -> str:
    # Standard lib only (ok en couche Application)
    import secrets
    import string

    MIN_PASSWORD_LEN = 12
    length = max(length, MIN_PASSWORD_LEN)

    upper = secrets.choice(string.ascii_uppercase)
    lower = secrets.choice(string.ascii_lowercase)
    digit = secrets.choice(string.digits)
    special = secrets.choice("!@#$%^&*()-_=+[]{};:,.<>?")

    remaining = [
        secrets.choice(
            string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{};:,.<>?"
        )
        for _ in range(length - 4)
    ]
    chars = [upper, lower, digit, special, *remaining]
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars)


@dataclass(frozen=True, slots=True)
class ResetDriverPasswordResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    new_password: str | None = None
    force_password_change: bool = False


class ResetDriverPasswordUseCase:
    """Use-case Application: rÃ©initialiser le mot de passe d'un chauffeur."""

    def __init__(self, *, password_policy: _PasswordPolicyPort) -> None:
        super().__init__()
        self._policy = password_policy

    def execute(self, user: _UserLike) -> ResetDriverPasswordResult:
        # On tente quelques gÃ©nÃ©rations au cas oÃ¹ une contrainte externe
        # (HIBP/historique) rejette.
        last_error: str | None = None
        for _ in range(5):
            pwd = _generate_strong_password()
            try:
                self._policy.validate_password(
                    password=pwd, user_id=int(user.id), check_history=True
                )
            except Exception as e:
                last_error = str(e)
                continue

            user.set_password(  # nosemgrep: python.django.security.audit.unvalidated-password.unvalidated-password
                pwd
            )
            user.force_password_change = True
            return ResetDriverPasswordResult(
                ok=True,
                new_password=pwd,
                force_password_change=True,
            )

        return ResetDriverPasswordResult(
            ok=False,
            error={
                "error": last_error or "Impossible de gÃ©nÃ©rer un mot de passe valide"
            },
            status_code=400,
        )
