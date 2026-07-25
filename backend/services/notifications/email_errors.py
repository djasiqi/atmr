"""Exceptions typées pour l'envoi d'emails (retries Celery)."""

from __future__ import annotations


class EmailSendError(Exception):
    """Erreur d'envoi email de base."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class EmailRetryableError(EmailSendError):
    """Timeout, réseau, HTTP 429 / 5xx — déclenche self.retry()."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message, status_code=status_code, retryable=True)


class EmailPermanentError(EmailSendError):
    """HTTP 400/401/403, config absente — pas de retry."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message, status_code=status_code, retryable=False)
