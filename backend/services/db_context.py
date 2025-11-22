# Constantes pour éviter les valeurs magiques
# 100 = 0  # Constante corrigée

"""Context managers pour la gestion propre des transactions SQLAlchemy.

Remplace les patterns try/except/finally répétés dans tout le code par
des context managers réutilisables et testables.

📚 Documentation complète : Voir `backend/docs/SESSION_MANAGEMENT.md` pour le guide complet
   de gestion des sessions SQLAlchemy (code métier + tests).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable

from sqlalchemy.exc import SQLAlchemyError

from ext import db

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
        pass

    def track_session_error(error_type: str) -> None:
        """No-op si métriques non disponibles."""
        pass

    def track_transaction(operation: str) -> Any:
        """No-op context manager si métriques non disponibles."""
        _ = operation  # Paramètre non utilisé mais requis pour signature
        return nullcontext()


if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


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
            raise RuntimeError("Database is in read-only mode. Writes are temporarily disabled.")
    except ImportError:
        # Si module chaos non disponible, continuer normalement
        pass

    try:
        yield db.session

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
            with track_transaction("commit"):
                db.session.commit()
            logger.debug("Transaction committed successfully")

    except SQLAlchemyError as e:
        if auto_rollback:
            # ✅ P2.1: Track le rollback et l'erreur
            with track_transaction("rollback"):
                db.session.rollback()
            track_session_error("SQLAlchemyError")
            logger.warning("Transaction rolled back due to error: %s", e)

        if reraise:
            raise
        else:
            logger.error("Transaction error (not reraised): %s", e)

    except Exception as e:
        if auto_rollback:
            # ✅ P2.1: Track le rollback et l'erreur
            with track_transaction("rollback"):
                db.session.rollback()
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

    try:
        yield db.session
        # Pas de commit pour les lectures

    except Exception as e:
        # ✅ P2.1: Track le rollback et l'erreur
        with track_transaction("rollback"):
            db.session.rollback()
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
        nonlocal counter
        try:
            # ✅ P2.1: Track le commit
            with track_transaction("commit"):
                db.session.commit()
            counter[0] = 0
            logger.debug("Batch committed (batch_size=%d)", batch_size)
        except SQLAlchemyError as e:
            # ✅ P2.1: Track le rollback et l'erreur
            with track_transaction("rollback"):
                db.session.rollback()
            if METRICS_AVAILABLE:
                track_session_error("SQLAlchemyError")
            logger.error("Batch commit failed: %s", e)
            raise

    try:
        yield db.session, commit_batch

        # Commit final si des opérations restantes
        if counter[0] > 0 and auto_commit_batch:
            commit_batch()

    except Exception as e:
        # ✅ P2.1: Track le rollback et l'erreur
        with track_transaction("rollback"):
            db.session.rollback()
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
