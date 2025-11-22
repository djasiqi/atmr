"""Module de gestion d'erreurs centralisé pour l'application ATMR.

Fournit des décorateurs et utilitaires pour gérer les exceptions de manière
sécurisée dans les fonctions critiques.
"""

import functools
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def safe_execute(
    default_return: Any = None, log_error: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Décorateur pour exécuter une fonction de manière sécurisée.

    Capture toutes les exceptions et retourne une valeur par défaut au lieu
    de laisser l'exception se propager. Utile pour protéger les fonctions
    critiques contre les erreurs inattendues.

    Args:
        default_return: Valeur de retour par défaut en cas d'erreur (par défaut: None)
        log_error: Si True, log l'erreur avec la trace complète (par défaut: True)

    Returns:
        Décorateur qui enveloppe la fonction avec gestion d'erreurs

    Examples:
        >>> @safe_execute(default_return=[], log_error=True)
        ... def get_drivers(company_id: int) -> list[dict]:
        ...     # Code qui peut lever une exception
        ...     return drivers_list
        ...
        >>> # Si une exception survient, retourne [] au lieu de crasher
        >>> drivers = get_drivers(123)  # Retourne [] en cas d'erreur

        >>> @safe_execute(default_return={}, log_error=False)
        ... def get_booking_stats(booking_id: int) -> dict:
        ...     # Code qui peut lever une exception
        ...     return stats_dict
        ...
        >>> # Si une exception survient, retourne {} sans logger
        >>> stats = get_booking_stats(456)  # Retourne {} en cas d'erreur
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.exception(
                        "[safe_execute] Erreur dans %s: %s", func.__name__, e
                    )
                return default_return

        return wrapper

    return decorator
