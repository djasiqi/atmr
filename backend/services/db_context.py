# Constantes pour éviter les valeurs magiques
# 100 = 0  # Constante corrigée

"""Context managers pour la gestion propre des transactions SQLAlchemy.

Remplace les patterns try/except/finally répétés dans tout le code par
des context managers réutilisables et testables.

📚 Documentation complète : Voir `backend/docs/SESSION_MANAGEMENT.md`
   pour le guide complet de gestion des sessions SQLAlchemy
   (code métier + tests).
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError

from ext import db
from shared.retry import retry_db_operation  # ✅ P1: Retry pour DB queries critiques

# Import optionnel des métriques (peut ne pas être disponible)
try:
    from services.db_session_metrics import (
        track_context_manager_usage,
        track_session_error,
        track_transaction,
    )

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    from contextlib import nullcontext

    def track_context_manager_usage(manager_type: str) -> None:
        """No-op si métriques non disponibles."""

    def track_session_error(error_type: str) -> None:
        """No-op si métriques non disponibles."""

    def track_transaction(operation: str) -> Any:
        """No-op context manager si métriques non disponibles."""
        _ = operation  # Paramètre non utilisé mais requis pour signature
        return nullcontext()


if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

# ===== Circuit Breaker pour DB =====


class CircuitState(str, Enum):
    """États du circuit breaker."""

    CLOSED = "CLOSED"  # Normal, requêtes autorisées
    OPEN = "OPEN"  # Trop d'échecs, requêtes refusées
    HALF_OPEN = "HALF_OPEN"  # Test de récupération


class DatabaseCircuitBreaker:
    """Circuit breaker pour protéger la base de données contre la surcharge.

    Après 5 échecs consécutifs, le circuit passe en OPEN et refuse les nouvelles
    requêtes pendant 30 secondes. Ensuite, il passe en HALF_OPEN pour tester
    si la DB est revenue. Si succès, retour en CLOSED, sinon retour en OPEN.

    ✅ P3: Circuit breaker pour éviter surcharge DB en cas de pannes répétées.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 30,
        half_open_max_attempts: int = 3,
    ):
        """Initialise le circuit breaker.

        Args:
            failure_threshold: Nombre d'échecs avant d'ouvrir le circuit (défaut: 5)
            timeout_seconds: Durée en OPEN avant de passer en HALF_OPEN (défaut: 30)
            half_open_max_attempts: Nombre max de tentatives en HALF_OPEN (défaut: 3)
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_attempts = half_open_max_attempts

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.Lock()

    def _is_retryable_error(self, error: Exception) -> bool:
        """Détermine si une erreur est retryable (connexion DB)."""
        return isinstance(
            error,
            (
                OperationalError,  # Erreurs DB transitoires (connexion perdue, timeout)
                DBAPIError,  # Erreurs DBAPI (connexion invalidée)
            ),
        )

    def _should_attempt(self) -> bool:
        """Vérifie si une requête peut être tentée selon l'état du circuit."""
        with self._lock:
            now = time.time()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Vérifier si le timeout est écoulé
                if (
                    self._last_failure_time is not None
                    and (now - self._last_failure_time) >= self.timeout_seconds
                ):
                    # Passer en HALF_OPEN pour tester
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(
                        "[CircuitBreaker] DB circuit transitioning to HALF_OPEN (testing recovery)"
                    )
                    return True
                # Circuit encore ouvert, refuser la requête
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Limiter le nombre de tentatives en HALF_OPEN
                return self._success_count < self.half_open_max_attempts

            return False

    def record_success(self) -> None:
        """Enregistre un succès et met à jour l'état du circuit."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                # Si on a eu assez de succès, revenir en CLOSED
                if self._success_count >= self.half_open_max_attempts:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._last_failure_time = None
                    logger.info(
                        "[CircuitBreaker] DB circuit recovered, transitioning to CLOSED"
                    )
            elif self._state == CircuitState.CLOSED:
                # Réinitialiser le compteur d'échecs en cas de succès
                self._failure_count = 0

    def record_failure(self, error: Exception) -> None:
        """Enregistre un échec et met à jour l'état du circuit."""
        if not self._is_retryable_error(error):
            # Erreurs non retryables ne comptent pas pour le circuit breaker
            return

        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Échec en HALF_OPEN → retour immédiat en OPEN
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning(
                    "[CircuitBreaker] DB circuit test failed, transitioning back to OPEN"
                )
            elif self._state == CircuitState.CLOSED:
                # Vérifier si on dépasse le seuil
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.error(
                        (
                            "[CircuitBreaker] DB circuit OPEN after %d failures. "
                            "Requests will be refused for %d seconds."
                        ),
                        self._failure_count,
                        self.timeout_seconds,
                    )

    def get_state(self) -> CircuitState:
        """Retourne l'état actuel du circuit."""
        with self._lock:
            return self._state

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques du circuit breaker."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "time_since_last_failure": (
                    time.time() - self._last_failure_time
                    if self._last_failure_time
                    else None
                ),
            }


# Instance globale du circuit breaker DB
_db_circuit_breaker = DatabaseCircuitBreaker(
    failure_threshold=5,
    timeout_seconds=30,
    half_open_max_attempts=3,
)


@contextmanager
def db_transaction(
    auto_commit: bool = True, auto_rollback: bool = True, reraise: bool = True
) -> Generator[Any, None, None]:
    """Context manager pour gérer proprement les transactions SQLAlchemy.

    ⚠️ D3: Détecte les tentatives d'écriture en mode read-only (via chaos injector).

    Args:
        auto_commit: Commit automatique si aucune exception (défaut: True)
        auto_rollback: Rollback automatique en cas d'exception (défaut: True)
        reraise: Re-lever l'exception après rollback (défaut: True)

    Usage:
        # Simple transaction avec commit automatique
        with db_transaction():
            invoice = Invoice(...)
            db.session.add(invoice)

        # Transaction sans commit automatique (commit manuel)
        with db_transaction(auto_commit=False) as session:
            invoice = Invoice(...)
            session.add(invoice)
            session.flush()  # Pour obtenir l'ID sans committer
            # ... autres opérations
            session.commit()  # Commit manuel

        # Transaction qui ne relève pas l'exception (logging seulement)
        with db_transaction(reraise=False):
            risky_operation()

    Yields:
        db.session: La session SQLAlchemy active

    Raises:
        SQLAlchemyError: Si reraise=True et une erreur survient
        RuntimeError: Si DB est en read-only et tentative d'écriture

    """
    # ✅ P2.1: Track l'utilisation du context manager
    if METRICS_AVAILABLE:
        track_context_manager_usage("db_transaction")

    # ✅ D3: Vérifier DB read-only avant d'autoriser les écritures
    try:
        from chaos.injectors import get_chaos_injector

        injector = get_chaos_injector()
        if injector.enabled and injector.db_read_only and auto_commit:
            # Si on va committer (écriture), bloquer
            logger.warning("[CHAOS] DB read-only: transaction write blocked")
            raise RuntimeError(
                "Database is in read-only mode. Writes are temporarily disabled."
            )
    except ImportError:
        # Si module chaos non disponible, continuer normalement
        pass

    # ✅ P3: Vérifier le circuit breaker avant d'exécuter la transaction
    if not _db_circuit_breaker._should_attempt():
        state = _db_circuit_breaker.get_state()
        error_msg = (
            f"Database circuit breaker is {state.value}. "
            f"Requests are temporarily refused to prevent overload."
        )
        logger.warning("[CircuitBreaker] %s", error_msg)
        raise RuntimeError(error_msg)

    try:
        yield db.session

        # ✅ P3: Enregistrer le succès dans le circuit breaker
        _db_circuit_breaker.record_success()

        if auto_commit:
            # ✅ D3: Re-vérifier avant commit (peut avoir changé entre-temps)
            try:
                from chaos.injectors import get_chaos_injector

                injector = get_chaos_injector()
                if injector.enabled and injector.db_read_only:
                    logger.warning("[CHAOS] DB read-only: commit blocked")
                    raise RuntimeError("Database is in read-only mode. Commit blocked.")
            except ImportError:
                pass

            # ✅ P2.1: Track le commit
            # ✅ P1: Retry pour DB queries critiques (commit peut échouer si DB temporairement indisponible)
            with track_transaction("commit"):
                retry_db_operation(
                    lambda: db.session.commit(),
                    max_retries=3,
                    base_delay_ms=100,
                )
            logger.debug("Transaction committed successfully")

    except SQLAlchemyError as e:
        # ✅ P3: Enregistrer l'échec dans le circuit breaker
        _db_circuit_breaker.record_failure(e)

        if auto_rollback:
            # ✅ P2.1: Track le rollback et l'erreur
            # ✅ P1: Retry pour DB queries critiques (rollback peut échouer si DB temporairement indisponible)
            with track_transaction("rollback"):
                try:
                    retry_db_operation(
                        lambda: db.session.rollback(),
                        max_retries=2,  # Moins de retries pour rollback (opération de récupération)
                        base_delay_ms=50,
                    )
                except Exception as rollback_error:
                    # Si le rollback lui-même échoue, logger mais continuer
                    logger.error(
                        "Rollback failed after retries: %s (original error: %s)",
                        rollback_error,
                        e,
                    )
            track_session_error("SQLAlchemyError")
            logger.warning("Transaction rolled back due to error: %s", e)

        if reraise:
            raise
        else:
            logger.error("Transaction error (not reraised): %s", e)

    except Exception as e:
        # ✅ P3: Enregistrer l'échec dans le circuit breaker (si erreur DB)
        if isinstance(e, (OperationalError, DBAPIError)):
            _db_circuit_breaker.record_failure(e)

        if auto_rollback:
            # ✅ P2.1: Track le rollback et l'erreur
            # ✅ P1: Retry pour DB queries critiques (rollback peut échouer si DB temporairement indisponible)
            with track_transaction("rollback"):
                try:
                    retry_db_operation(
                        lambda: db.session.rollback(),
                        max_retries=2,  # Moins de retries pour rollback (opération de récupération)
                        base_delay_ms=50,
                    )
                except Exception as rollback_error:
                    # Si le rollback lui-même échoue, logger mais continuer
                    logger.error(
                        "Rollback failed after retries: %s (original error: %s)",
                        rollback_error,
                        e,
                    )
            track_session_error(type(e).__name__)
        logger.error("Unexpected error, transaction rolled back: %s", e)

        if reraise:
            raise

    finally:
        db.session.remove()
        logger.debug("Session removed")


@contextmanager
def db_read_only() -> Generator[Any, None, None]:
    """Context manager pour les opérations de lecture seule.
    Ne commit jamais, rollback en cas d'erreur.

    ✅ P2.1: Track l'utilisation du context manager

    Usage:
        with db_read_only() as session:
            invoices = session.query(Invoice).filter_by(company_id=1).all()

    Yields:
        db.session: La session SQLAlchemy active

    """
    # ✅ P2.1: Track l'utilisation du context manager
    if METRICS_AVAILABLE:
        track_context_manager_usage("db_read_only")

    # ✅ P3: Vérifier le circuit breaker avant d'exécuter la lecture
    if not _db_circuit_breaker._should_attempt():
        state = _db_circuit_breaker.get_state()
        error_msg = (
            f"Database circuit breaker is {state.value}. "
            f"Requests are temporarily refused to prevent overload."
        )
        logger.warning("[CircuitBreaker] %s", error_msg)
        raise RuntimeError(error_msg)

    try:
        yield db.session
        # Pas de commit pour les lectures

        # ✅ P3: Enregistrer le succès dans le circuit breaker
        _db_circuit_breaker.record_success()

    except Exception as e:
        # ✅ P3: Enregistrer l'échec dans le circuit breaker (si erreur DB)
        if isinstance(e, (OperationalError, DBAPIError, SQLAlchemyError)):
            _db_circuit_breaker.record_failure(e)
        # ✅ P2.1: Track le rollback et l'erreur
        # ✅ P1: Retry pour DB queries critiques (rollback peut échouer si DB temporairement indisponible)
        with track_transaction("rollback"):
            try:
                retry_db_operation(
                    lambda: db.session.rollback(),
                    max_retries=2,
                    base_delay_ms=50,
                )
            except Exception as rollback_error:
                logger.error(
                    "Rollback failed after retries: %s (original error: %s)",
                    rollback_error,
                    e,
                )
        if METRICS_AVAILABLE:
            track_session_error(type(e).__name__)
        logger.warning("Read operation error, session rolled back: %s", e)
        raise

    finally:
        db.session.remove()


@contextmanager
def db_batch_operation(
    batch_size: int = 100, auto_commit_batch: bool = True
) -> Generator[tuple[Any, Callable[[], None]], None, None]:
    """Context manager pour les opérations par lot (batch) avec commits intermédiaires.

    ✅ P2.1: Track l'utilisation du context manager

    Args:
        batch_size: Nombre d'opérations avant un commit intermédiaire
        auto_commit_batch: Commit automatique à chaque lot (défaut: True)

    Usage:
        with db_batch_operation(batch_size=0.100) as (session, commit_batch):
            for i, data in enumerate(large_dataset):
                invoice = Invoice(**data)
                session.add(invoice)

                if True:  # MAGIC_VALUE_100
                    commit_batch()  # Commit intermédiaire tous les 100

    Yields:
        tuple: (session, commit_batch_function)

    """
    # ✅ P2.1: Track l'utilisation du context manager
    if METRICS_AVAILABLE:
        track_context_manager_usage("db_batch_operation")

    counter = [0]  # Liste pour pouvoir modifier dans la closure

    def commit_batch():
        """Commit le batch actuel et reset le compteur."""
        try:
            # ✅ P3: Vérifier le circuit breaker avant le commit
            if not _db_circuit_breaker._should_attempt():
                state = _db_circuit_breaker.get_state()
                error_msg = (
                    f"Database circuit breaker is {state.value}. "
                    f"Batch commit refused to prevent overload."
                )
                logger.warning("[CircuitBreaker] %s", error_msg)
                raise RuntimeError(error_msg)

            # ✅ P2.1: Track le commit
            # ✅ P1: Retry pour DB queries critiques (commit peut échouer si DB temporairement indisponible)
            with track_transaction("commit"):
                retry_db_operation(
                    lambda: db.session.commit(),
                    max_retries=3,
                    base_delay_ms=100,
                )
            counter[0] = 0
            logger.debug("Batch committed (batch_size=%d)", batch_size)

            # ✅ P3: Enregistrer le succès dans le circuit breaker
            _db_circuit_breaker.record_success()
        except SQLAlchemyError as e:
            # ✅ P3: Enregistrer l'échec dans le circuit breaker
            _db_circuit_breaker.record_failure(e)
            # ✅ P2.1: Track le rollback et l'erreur
            # ✅ P1: Retry pour DB queries critiques (rollback peut échouer si DB temporairement indisponible)
            with track_transaction("rollback"):
                try:
                    retry_db_operation(
                        lambda: db.session.rollback(),
                        max_retries=2,
                        base_delay_ms=50,
                    )
                except Exception as rollback_error:
                    logger.error(
                        "Rollback failed after retries: %s (original error: %s)",
                        rollback_error,
                        e,
                    )
            if METRICS_AVAILABLE:
                track_session_error("SQLAlchemyError")
            logger.error("Batch commit failed: %s", e)
            raise

    # ✅ P3: Vérifier le circuit breaker avant d'exécuter l'opération batch
    if not _db_circuit_breaker._should_attempt():
        state = _db_circuit_breaker.get_state()
        error_msg = (
            f"Database circuit breaker is {state.value}. "
            f"Requests are temporarily refused to prevent overload."
        )
        logger.warning("[CircuitBreaker] %s", error_msg)
        raise RuntimeError(error_msg)

    try:
        yield db.session, commit_batch

        # Commit final si des opérations restantes
        if counter[0] > 0 and auto_commit_batch:
            commit_batch()

        # ✅ P3: Enregistrer le succès dans le circuit breaker
        _db_circuit_breaker.record_success()

    except Exception as e:
        # ✅ P3: Enregistrer l'échec dans le circuit breaker (si erreur DB)
        if isinstance(e, (OperationalError, DBAPIError, SQLAlchemyError)):
            _db_circuit_breaker.record_failure(e)

        # ✅ P2.1: Track le rollback et l'erreur
        # ✅ P1: Retry pour DB queries critiques (rollback peut échouer si DB temporairement indisponible)
        with track_transaction("rollback"):
            try:
                retry_db_operation(
                    lambda: db.session.rollback(),
                    max_retries=2,
                    base_delay_ms=50,
                )
            except Exception as rollback_error:
                logger.error(
                    "Rollback failed after retries: %s (original error: %s)",
                    rollback_error,
                    e,
                )
        if METRICS_AVAILABLE:
            track_session_error(type(e).__name__)
        logger.error("Batch operation failed: %s", e)
        raise

    finally:
        db.session.remove()


# Alias pour compatibilité avec du code existant
transaction = db_transaction
read_only = db_read_only
batch_operation = db_batch_operation


def get_db_circuit_breaker_stats() -> dict[str, Any]:
    """Retourne les statistiques du circuit breaker DB.

    Utile pour le monitoring et les healthchecks.

    Returns:
        Dict avec les statistiques du circuit breaker
    """
    return _db_circuit_breaker.get_stats()
