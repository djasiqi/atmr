"""Adapter pour récupérer l'identité JWT depuis flask_jwt_extended."""

from __future__ import annotations

from shared.application.use_cases.get_current_user import GetJwtIdentityPort


class JwtIdentityAdapter(GetJwtIdentityPort):
    """Adapter pour récupérer l'identité JWT depuis flask_jwt_extended."""

    def get_jwt_identity(self) -> str | None:  # pyright: ignore[reportImplicitOverride]
        """Récupère l'identité depuis le token JWT."""
        from flask_jwt_extended import (  # pyright: ignore[reportMissingImports]
            get_jwt_identity,
        )

        return get_jwt_identity()
