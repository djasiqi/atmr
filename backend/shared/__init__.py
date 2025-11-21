"""Module shared - Utilitaires partagés pour l'application ATMR."""

from shared.error_handling import safe_execute
from shared.timeouts import TimeoutError, timeout  # noqa: A004

__all__ = ["TimeoutError", "safe_execute", "timeout"]
