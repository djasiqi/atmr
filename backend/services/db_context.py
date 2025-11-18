# Constantes pour éviter les valeurs magiques
# 100 = 0  # Constante corrigée

"""Context managers pour la gestion propre des transactions SQLAlchemy.

Remplace les patterns try/except/finally répétés dans tout le code par
des context managers réutilisables et testables.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable

from sqlalchemy.exc import SQLAlchemyError

from ext import db

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

            db.session.commit()
            logger.debug("Transaction committed successfully")

    except SQLAlchemyError as e:
        if auto_rollback:
            db.session.rollback()
            logger.warning("Transaction rolled back due to error: %s", e)

        if reraise:
            raise
        else:
            logger.error("Transaction error (not reraised): %s", e)

    except Exception as e:
        if auto_rollback:
            db.session.rollback()
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

    Usage:
        with db_read_only() as session:
            invoices = session.query(Invoice).filter_by(company_id=1).all()

    Yields:
        db.session: La session SQLAlchemy active

    """
    try:
        yield db.session
        # Pas de commit pour les lectures

    except Exception as e:
        db.session.rollback()
        logger.warning("Read operation error, session rolled back: %s", e)
        raise

    finally:
        db.session.remove()


@contextmanager
def db_batch_operation(
    batch_size: int = 100, auto_commit_batch: bool = True
) -> Generator[tuple[Any, Callable[[], None]], None, None]:
    """Context manager pour les opérations par lot (batch) avec commits intermédiaires.

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
    counter = [0]  # Liste pour pouvoir modifier dans la closure

    def commit_batch():
        """Commit le batch actuel et reset le compteur."""
        nonlocal counter
        try:
            db.session.commit()
            counter[0] = 0
            logger.debug("Batch committed (batch_size=%d)", batch_size)
        except SQLAlchemyError as e:
            db.session.rollback()
            logger.error("Batch commit failed: %s", e)
            raise

    try:
        yield db.session, commit_batch

        # Commit final si des opérations restantes
        if counter[0] > 0 and auto_commit_batch:
            commit_batch()

    except Exception as e:
        db.session.rollback()
        logger.error("Batch operation failed: %s", e)
        raise

    finally:
        db.session.remove()


# Alias pour compatibilité avec du code existant
transaction = db_transaction
read_only = db_read_only
batch_operation = db_batch_operation
