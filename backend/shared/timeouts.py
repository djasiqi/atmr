"""Module de gestion des timeouts uniformisés pour l'application ATMR.

Fournit un décorateur `timeout` pour standardiser la gestion des timeouts
dans l'application, améliorant ainsi la robustesse et la prévisibilité
des opérations longues.

Caractéristiques:
- Cross-platform (utilise threading au lieu de signal.SIGALRM)
- Compatible avec Windows, Linux et macOS
- Gestion propre des exceptions et nettoyage des ressources
- Configurable via paramètre `seconds`

Usage:
    from shared.timeouts import timeout, TimeoutError

    # Exemple 1: Décorateur simple
    @timeout(seconds=30)
    def long_running_function():
        # Code qui peut prendre du temps
        return result

    # Exemple 2: Timeout court pour opérations critiques
    @timeout(seconds=5)
    def critical_operation():
        # Doit se terminer rapidement
        return result

    # Exemple 3: Gestion d'exception
    try:
        result = long_running_function()
    except TimeoutError as e:
        logger.error(f"Operation timed out: {e}")
        # Fallback ou retry
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TimeoutError(Exception):  # noqa: A001
    """Exception levée en cas de timeout d'une fonction.

    Différente de builtins.TimeoutError pour éviter les conflits
    et permettre une gestion spécifique des timeouts de l'application.
    Note: Le nom TimeoutError est intentionnel pour correspondre au plan d'audit.
    """


def timeout(seconds: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Décorateur pour ajouter un timeout à une fonction.

    Utilise threading pour exécuter la fonction dans un thread séparé
    et un Timer pour détecter les timeouts. Cross-platform (Windows, Linux, macOS).

    Note: Cette implémentation ne peut pas forcer l'arrêt d'une fonction bloquante,
    mais elle détecte le timeout et lève une exception. Pour une vraie interruption,
    considérer l'utilisation de multiprocessing.

    Args:
        seconds: Délai maximum en secondes avant de lever TimeoutError

    Returns:
        Décorateur qui enveloppe la fonction avec gestion de timeout

    Raises:
        TimeoutError: Si la fonction dépasse le délai spécifié

    Examples:
        >>> @timeout(seconds=30)
        ... def fetch_data():
        ...     # Code qui peut prendre du temps
        ...     return data
        ...
        >>> # Si la fonction prend plus de 30 secondes, TimeoutError est levée
        >>> try:
        ...     data = fetch_data()
        ... except TimeoutError:
        ...     # Gérer le timeout
        ...     pass
    """
    if seconds <= 0:
        raise ValueError("Timeout seconds must be positive")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Variables partagées pour la communication entre threads
            result_container: list[Any] = []
            exception_container: list[Exception] = []
            timeout_occurred = threading.Event()
            execution_done = threading.Event()

            def target() -> None:
                """Fonction cible exécutée dans un thread séparé."""
                try:
                    result = func(*args, **kwargs)
                    if not timeout_occurred.is_set():
                        result_container.append(result)
                except Exception as e:
                    if not timeout_occurred.is_set():
                        exception_container.append(e)
                finally:
                    execution_done.set()

            def timeout_handler() -> None:
                """Handler appelé par le timer en cas de timeout."""
                if not execution_done.is_set():
                    timeout_occurred.set()
                    logger.warning(
                        "[timeout] Function %s timed out after %.2f seconds",
                        func.__name__,
                        seconds,
                    )

            # Créer et démarrer le thread d'exécution
            thread = threading.Thread(target=target, daemon=True)
            thread.start()

            # Créer et démarrer le timer
            timer = threading.Timer(seconds, timeout_handler)
            timer.daemon = True
            timer.start()

            try:
                # Attendre que le thread se termine ou que le timeout se produise
                thread.join(
                    timeout=seconds + 1
                )  # +1 pour laisser le temps au timer de se déclencher

                # Arrêter le timer
                timer.cancel()

                # Vérifier si un timeout s'est produit
                if timeout_occurred.is_set():
                    raise TimeoutError(
                        f"Function {func.__name__} timed out after {seconds}s"
                    )

                # Vérifier s'il y a une exception à propager
                if exception_container:
                    raise exception_container[0]

                # Retourner le résultat
                if result_container:
                    return result_container[0]

                # Cas où la fonction ne retourne rien (None)
                return None  # type: ignore[return-value]

            finally:
                # S'assurer que le timer est toujours annulé
                timer.cancel()

        return wrapper

    return decorator
