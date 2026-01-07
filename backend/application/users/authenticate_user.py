"""Use-case: authentifier un utilisateur (login).

⚠️ TODO: Ce use case encapsule temporairement la logique d'authentification dans routes/auth.py
pour permettre une migration progressive. La logique métier devrait être migrée progressivement
vers ce use case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AuthenticateUserInput:
    """Input pour authentifier un utilisateur.

    Attributes:
        email: Email de l'utilisateur
        password: Mot de passe de l'utilisateur
    """

    email: str
    password: str


@dataclass(frozen=True, slots=True)
class AuthenticateUserOutput:
    """Output pour authentifier un utilisateur.

    Attributes:
        success: True si l'authentification a réussi
        access_token: Token d'accès (si succès)
        refresh_token: Token de rafraîchissement (si succès)
        user: Utilisateur authentifié (si succès)
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    access_token: str | None = None
    refresh_token: str | None = None
    user: Any | None = None  # User model
    error: dict[str, str] | None = None
    status_code: int | None = None


class AuthenticateUserUseCase:
    """Use-case Application: authentifier un utilisateur (login).

    ⚠️ TODO: Migration progressive - Ce use case encapsule la logique d'authentification.
    La logique métier devrait être migrée progressivement ici.
    """

    def execute(self, input_data: AuthenticateUserInput) -> AuthenticateUserOutput:
        """Authentifie un utilisateur avec email et mot de passe.

        Args:
            input_data: Input avec email et password

        Returns:
            AuthenticateUserOutput avec les tokens et l'utilisateur si authentification réussie

        Note:
            La génération des tokens, audit logging, et métriques sont gérés dans les routes
            pour l'instant. Ces aspects seront migrés progressivement vers ce use case.
        """
        # Validation
        validation_error = self._validate_input(input_data)
        if validation_error:
            return AuthenticateUserOutput(
                success=False,
                error=validation_error,
                status_code=400,
            )

        from repositories.user_repository import UserRepository

        # #region agent log
        log_path = Path(r"c:\Users\jasiq\atmr\.cursor\debug.log")
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "location": (
                                "authenticate_user.py:AuthenticateUserUseCase.execute"
                            ),
                            "message": "before UserRepository.find_model_by_email",
                            "data": {
                                "email": (
                                    input_data.email[:10] + "***"
                                    if input_data.email
                                    else None
                                ),
                            },
                            "timestamp": datetime.now(UTC).isoformat(),
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "C",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        user_repo = UserRepository()
        user = user_repo.find_model_by_email(input_data.email)
        # #region agent log
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "location": (
                                "authenticate_user.py:AuthenticateUserUseCase.execute"
                            ),
                            "message": "after UserRepository.find_model_by_email",
                            "data": {
                                "user_found": user is not None,
                                "user_id": user.id if user else None,
                            },
                            "timestamp": datetime.now(UTC).isoformat(),
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "C",
                        }
                    )
                    + "\n"
                )
        except Exception as log_err:
            # Log l'erreur de logging elle-même
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "location": (
                                    "authenticate_user.py:AuthenticateUserUseCase.execute"
                                ),
                                "message": "ERROR logging after find_model_by_email",
                                "data": {"log_error": str(log_err)},
                                "timestamp": datetime.now(UTC).isoformat(),
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "C",
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
        # #endregion

        if not user or not user.check_password(input_data.password):
            return AuthenticateUserOutput(
                success=False,
                error={"error": "Email ou mot de passe invalide"},
                status_code=401,
            )

        # ⚠️ TODO: La génération des tokens devrait être dans ce use case
        # Pour l'instant, les routes gèrent la génération des tokens
        return AuthenticateUserOutput(success=True, user=user)

    def _validate_input(
        self, input_data: AuthenticateUserInput
    ) -> dict[str, str] | None:
        """Valide les inputs du use case.

        Args:
            input_data: Input à valider

        Returns:
            None si valide, dict d'erreurs sinon
        """
        errors: dict[str, str] = {}

        if not input_data.email or len(input_data.email.strip()) == 0:
            errors["email"] = "L'email est requis"

        if not input_data.password or len(input_data.password.strip()) == 0:
            errors["password"] = "Le mot de passe est requis"

        return errors if errors else None
