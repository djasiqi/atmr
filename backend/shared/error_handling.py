"""Module de gestion d'erreurs centralisé pour l'application ATMR.

Fournit des décorateurs, context managers et utilitaires pour gérer les exceptions
de manière sécurisée dans les fonctions critiques.

✅ P1: Amélioration pour réduire les exceptions trop larges (Recommandation 1.1 audit).
"""

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

# Import optionnel pour éviter dépendance circulaire
try:
    from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError
except ImportError:
    # En cas d'import échoué (tests, etc.)
    DBAPIError = Exception
    IntegrityError = Exception
    OperationalError = Exception

logger = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def handle_db_errors(context: str = ""):
    """Context manager pour gérer les erreurs DB avec logging approprié.

    ✅ P1: Remplace les except Exception: pour erreurs DB.

    Args:
        context: Contexte de l'opération (pour logging, ex: "loading booking 123")

    Examples:
        >>> with handle_db_errors("loading booking"):
        ...     booking = Booking.query.filter_by(id=1).first()
    """
    try:
        yield
    except (OperationalError, DBAPIError) as e:
        logger.error("Erreur DB transitoire %s: %s", context, e)
        raise  # Re-lever pour retry
    except IntegrityError as e:
        logger.error("Erreur d'intégrité DB %s: %s", context, e)
        raise
    except Exception:
        logger.exception("Erreur DB inattendue %s", context)
        raise  # Ne pas masquer les erreurs inattendues


@contextmanager
def handle_validation_errors(context: str = ""):
    """Context manager pour gérer les erreurs de validation.

    ✅ P1: Remplace les except Exception: pour erreurs de validation.

    Args:
        context: Contexte de l'opération (pour logging)

    Examples:
        >>> with handle_validation_errors("parsing date"):
        ...     date = datetime.fromisoformat(date_str)
    """
    try:
        yield
    except (ValueError, TypeError) as e:
        logger.warning("Erreur de validation %s: %s", context, e)
        raise
    except Exception:
        logger.exception("Erreur inattendue %s", context)
        raise


@contextmanager
def handle_network_errors(context: str = ""):
    """Context manager pour gérer les erreurs réseau.

    ✅ P1: Remplace les except Exception: pour erreurs réseau.

    Args:
        context: Contexte de l'opération (pour logging)

    Examples:
        >>> with handle_network_errors("calling OSRM"):
        ...     response = requests.get(osrm_url)
    """
    try:
        yield
    except (ConnectionError, TimeoutError, OSError) as e:
        logger.error("Erreur réseau %s: %s", context, e)
        raise
    except Exception:
        logger.exception("Erreur inattendue %s", context)
        raise


def safe_execute(
    default_return: Any = None, log_error: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Décorateur pour exécuter une fonction de manière sécurisée.

    ⚠️ ATTENTION: Ce décorateur capture TOUTES les exceptions (y compris Exception).
    À utiliser uniquement pour des cas spécifiques où on veut vraiment ignorer
    toutes les erreurs (ex: cleanup, logging, etc.).

    Pour la plupart des cas, préférer handle_db_errors(),
    handle_validation_errors(), etc.

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
