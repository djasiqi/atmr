"""Erreurs métier structurées — facturation plateforme LIRIE."""

from __future__ import annotations

from typing import Any


class BillingInvariantError(ValueError):
    """Conflit d'état / invariant financier (réponse API 409 typiquement)."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 409,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def to_response(self) -> tuple[dict[str, Any], int]:
        body: dict[str, Any] = {
            "error": self.code,
            "message": self.message,
        }
        if self.details:
            body["details"] = self.details
        return body, self.status_code
