"""Exceptions métier facturation (distinctes des ValueError génériques)."""

from __future__ import annotations


class BillingValidationError(ValueError):
    """État de facturation incomplet ou incohérent (réponse API 422)."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field
