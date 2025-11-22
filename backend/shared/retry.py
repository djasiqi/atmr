"""✅ Utilitaire uniformisé pour retry avec exponential backoff + jitter.

Ce module fournit un mécanisme de retry standardisé pour l'application,
remplaçant les implémentations ad-hoc dans différents services.

Caractéristiques:
- Exponential backoff avec jitter (évite thundering herd)
- Support exceptions retryables personnalisées
- Compatible sync (async ready)
- Configurable (max_retries, delays, jitter)

Usage:
    from shared.retry import retry_with_backoff

    # Exemple 1: Retry simple
    result = retry_with_backoff(
        lambda: requests.get("http://api.example.com/data"),
        max_retries=3
    )

    # Exemple 2: Retry avec exceptions spécifiques
    result = retry_with_backoff(
        lambda: db.session.query(User).first(),
        max_retries=5,
        retryable_exceptions=(OperationalError, TimeoutError)
    )

    # Exemple 3: Décorateur
    @retry_with_backoff(max_retries=3, base_delay_ms=250)
    def fetch_osrm_route(origin, dest):
        return requests.get(...)
"""

from __future__ import annotations

import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Tuple, Type, TypeVar, cast

logger = logging.getLogger(__name__)

# Type variable pour les fonctions retryables
T = TypeVar("T")

# Constantes par défaut
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY_MS = 250
DEFAULT_MAX_DELAY_MS = 10000  # 10 secondes max
DEFAULT_JITTER = True

# Exceptions retryables par défaut (erreurs réseau/transient)
DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
)


def calculate_backoff_delay(
    attempt: int,
    base_delay_ms: int = DEFAULT_BASE_DELAY_MS,
    max_delay_ms: int = DEFAULT_MAX_DELAY_MS,
    use_jitter: bool = True,
) -> float:
    """Calcule le délai de backoff avec exponential backoff et jitter optionnel.

    Formule: delay = min(base_delay_ms * (2 ** attempt), max_delay_ms)
    Avec jitter: delay = delay * (0.5 + random() * 0.5)  # ±50%

    Args:
        attempt: Numéro de la tentative (0 = première retry)
        base_delay_ms: Délai de base en millisecondes
        max_delay_ms: Délai maximum en millisecondes
        use_jitter: Activer le jitter (évite thundering herd)

    Returns:
        Délai en secondes (float)

    Examples:
        >>> calculate_backoff_delay(0, base_delay_ms=250)
        0.125...  # ~250ms avec jitter 50%
        >>> calculate_backoff_delay(1, base_delay_ms=250)
        0.25...   # ~500ms avec jitter
        >>> calculate_backoff_delay(2, base_delay_ms=250)
        0.5...    # ~1000ms avec jitter
    """
    # Exponential backoff: base * (2 ** attempt)
    delay_ms = base_delay_ms * (2**attempt)

    # Limiter au maximum
    delay_ms = min(delay_ms, max_delay_ms)

    # Appliquer jitter si demandé (évite synchronisation des retries)
    if use_jitter:
        # Jitter: multiplie par un facteur aléatoire entre 0.5 et 1.5
        jitter_factor = 0.5 + random.random()  # [0.5, 1.5)
        delay_ms = delay_ms * jitter_factor

    # Convertir en secondes
    return delay_ms / 1000.0


def is_retryable_exception(
    exception: Exception,
    retryable_exceptions: Tuple[Type[Exception], ...] | None = None,
) -> bool:
    """Détermine si une exception est retryable.

    Args:
        exception: Exception à vérifier
        retryable_exceptions: Tuple de classes d'exceptions retryables
            (défaut: exceptions réseau/timeout)

    Returns:
        True si l'exception est retryable
    """
    if retryable_exceptions is None:
        retryable_exceptions = DEFAULT_RETRYABLE_EXCEPTIONS

    # Vérifier si l'exception est une sous-classe d'une exception retryable
    return any(
        isinstance(exception, retryable_type) for retryable_type in retryable_exceptions
    )


def retry_with_backoff(
    func: Callable[[], T] | None = None,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_ms: int = DEFAULT_BASE_DELAY_MS,
    max_delay_ms: int = DEFAULT_MAX_DELAY_MS,
    use_jitter: bool = DEFAULT_JITTER,
    retryable_exceptions: Tuple[Type[Exception], ...] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    logger_instance: logging.Logger | None = None,
):
    """Exécute une fonction avec retry et exponential backoff.

    Peut être utilisé comme décorateur ou fonction directe.

    Args:
        func: Fonction à exécuter (si None, retourne un décorateur)
        max_retries: Nombre maximum de tentatives (total = max_retries + 1)
        base_delay_ms: Délai de base en millisecondes pour le backoff
        max_delay_ms: Délai maximum en millisecondes
        use_jitter: Activer le jitter pour éviter la synchronisation
        retryable_exceptions: Exceptions qui déclenchent un retry
            (défaut: TimeoutError, ConnectionError, OSError)
        on_retry: Callback appelé avant chaque retry
            (attempt: int, exception: Exception, delay: float)
        logger_instance: Logger à utiliser (défaut: logger module)

    Returns:
        Résultat de la fonction

    Raises:
        Exception: Dernière exception si toutes les tentatives échouent

    Examples:
        # Utilisation comme fonction
        result = retry_with_backoff(
            lambda: requests.get("http://api.com"),
            max_retries=3
        )

        # Utilisation comme décorateur
        @retry_with_backoff(max_retries=5)
        def fetch_data():
            return requests.get("http://api.com")
    """
    log = logger_instance or logger

    # Si func est None, on retourne un décorateur
    if func is None:

        def decorator(f: Callable[..., T]) -> Callable[..., T]:
            @wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                return cast(
                    T,
                    retry_with_backoff(
                        lambda: f(*args, **kwargs),
                        max_retries=max_retries,
                        base_delay_ms=base_delay_ms,
                        max_delay_ms=max_delay_ms,
                        use_jitter=use_jitter,
                        retryable_exceptions=retryable_exceptions,
                        on_retry=on_retry,
                        logger_instance=logger_instance,
                    ),
                )

            return wrapper

        return decorator

    # Sinon, on exécute directement (func n'est pas None ici)
    assert func is not None  # Pour le type checker
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e

            # Vérifier si l'exception est retryable
            if not is_retryable_exception(e, retryable_exceptions):
                log.debug(
                    "[Retry] Exception non retryable: %s (type: %s)",
                    e,
                    type(e).__name__,
                )
                raise

            # Vérifier si on a encore des tentatives
            if attempt >= max_retries:
                log.warning(
                    "[Retry] Toutes les tentatives épuisées (%d/%d). Dernière erreur: %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                raise

            # Calculer délai de backoff
            delay = calculate_backoff_delay(
                attempt=attempt,
                base_delay_ms=base_delay_ms,
                max_delay_ms=max_delay_ms,
                use_jitter=use_jitter,
            )

            # Appeler callback si fourni
            if on_retry:
                from contextlib import suppress

                with suppress(Exception):
                    # Ne pas faire échouer le retry si callback échoue
                    on_retry(attempt + 1, e, delay)

            log.info(
                "[Retry] Tentative %d/%d échouée: %s. Retry dans %.3fs",
                attempt + 1,
                max_retries + 1,
                type(e).__name__,
                delay,
            )

            # Attendre avant retry
            time.sleep(delay)

    # Ne devrait jamais arriver, mais pour type safety
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry exhausted without exception")


def retry_http_request(
    func: Callable[[], Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_ms: int = DEFAULT_BASE_DELAY_MS,
    retryable_status_codes: set[int] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> Any:
    """Helper spécialisé pour retry de requêtes HTTP.

    Retry automatique pour:
    - Exceptions réseau (TimeoutError, ConnectionError)
    - Codes HTTP retryables (500, 502, 503, 504, 429)

    Args:
        func: Fonction qui retourne un objet requests.Response
        max_retries: Nombre maximum de retries
        base_delay_ms: Délai de base pour backoff
        retryable_status_codes: Codes HTTP qui déclenchent un retry
            (défaut: {429, 500, 502, 503, 504})
        on_retry: Callback appelé avant chaque retry

    Returns:
        Response HTTP (si succès)

    Raises:
        requests.RequestException: Si toutes les tentatives échouent

    Example:
        response = retry_http_request(
            lambda: requests.get("http://api.com/data"),
            max_retries=5
        )
    """
    import requests

    if retryable_status_codes is None:
        retryable_status_codes = {429, 500, 502, 503, 504}

    # Retryable exceptions = exceptions réseau + HTTP retryables
    custom_retryable: Tuple[Type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
        requests.RequestException,
        requests.Timeout,
        requests.ConnectionError,
    )

    def wrapper() -> Any:
        """Wrapper qui vérifie aussi les codes HTTP."""
        response = func()

        # Vérifier code HTTP
        if (
            hasattr(response, "status_code")
            and response.status_code in retryable_status_codes
        ):
            # Créer une exception HTTP pour déclencher le retry
            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                raise e

        return response

    return cast(
        Any,
        retry_with_backoff(
            wrapper,
            max_retries=max_retries,
            base_delay_ms=base_delay_ms,
            retryable_exceptions=custom_retryable,
            on_retry=on_retry,
        ),
    )


def retry_db_operation(
    func: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay_ms: int = 100,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """Helper spécialisé pour retry d'opérations base de données.

    Retry automatique pour:
    - OperationalError (transient)
    - DBAPIError (connexion invalidée)

    Args:
        func: Fonction qui exécute une opération DB
        max_retries: Nombre maximum de retries
        base_delay_ms: Délai de base pour backoff
        on_retry: Callback appelé avant chaque retry

    Returns:
        Résultat de l'opération DB

    Raises:
        SQLAlchemyError: Si toutes les tentatives échouent

    Example:
        user = retry_db_operation(
            lambda: User.query.filter_by(id=123).first()
        )
    """
    try:
        from sqlalchemy.exc import DBAPIError, OperationalError

        db_retryable: Tuple[Type[Exception], ...] = (
            OperationalError,
            DBAPIError,
        )

        return cast(
            T,
            retry_with_backoff(
                func,
                max_retries=max_retries,
                base_delay_ms=base_delay_ms,
                retryable_exceptions=db_retryable,
                on_retry=on_retry,
            ),
        )
    except ImportError:
        # SQLAlchemy non disponible, utiliser retry standard
        return cast(
            T,
            retry_with_backoff(
                func,
                max_retries=max_retries,
                base_delay_ms=base_delay_ms,
                on_retry=on_retry,
            ),
        )
