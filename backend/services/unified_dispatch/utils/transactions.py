# backend/services/unified_dispatch/transaction_helpers.py
"""Helpers de gestion de transactions DB pour éviter les cycles d'import.

Ce module centralise les helpers de transaction utilisés par engine.py et apply.py.
"""

from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import text

from ext import db

__all__ = ["_begin_tx", "_in_tx"]  # Exported functions used by engine.py and apply.py


def _in_tx() -> bool:
    """Détecte de façon fiable si une transaction est déjà ouverte sur la session.

    IMPORTANT: SQLAlchemy 2.x peut ouvrir une transaction implicitement (autobegin)
    dès qu'une requête est exécutée. Cette fonction doit détecter ce cas.

    Dans le contexte de pytest, la fixture `db` crée souvent un savepoint, donc
    il y a toujours une transaction active. Cette fonction doit la détecter.

    Returns:
        True si une transaction est active, False sinon
    """
    try:
        # ✅ SQLAlchemy 2.x : méthode la plus fiable
        if db.session.in_transaction():
            return True
    except Exception:
        pass

    # ✅ Fallback : vérifier get_transaction() qui est plus fiable pour autobegin
    try:
        if db.session.get_transaction() is not None:
            return True
    except Exception:
        pass

    # ✅ Dernier recours : vérifier is_active (moins fiable mais peut aider)
    try:
        if getattr(db.session, "is_active", False):
            return True
    except Exception:
        pass

    return False


@contextmanager
def _begin_tx():
    """Démarre une transaction en s'adaptant à l'état courant de la Session.

    - Si une transaction existe déjà (y compris autobegin) -> SAVEPOINT (begin_nested)
    - Sinon -> transaction racine (begin)

    ✅ FIX: Gère les transactions PostgreSQL en échec en faisant un rollback
    avant d'essayer de créer un savepoint.

    ✅ FIX: Approche défensive : essaie toujours begin_nested() d'abord car
    dans pytest il y a toujours une transaction active. Si ça échoue avec
    une erreur spécifique indiquant qu'aucune transaction n'existe, alors utilise begin().

    Usage:
        with _begin_tx():
            # Code exécuté dans une transaction ou savepoint
            pass
    """
    from sqlalchemy.exc import InvalidRequestError

    # ✅ Approche défensive : essayer begin_nested() d'abord
    # Dans pytest, il y a toujours une transaction active (savepoint de la fixture db)
    # En production, si une transaction existe déjà, begin_nested() fonctionnera
    try:
        # Tester si la transaction est valide en essayant une opération simple
        # Si elle est en échec, PostgreSQL lèvera une exception
        try:
            db.session.execute(text("SELECT 1"))
        except Exception:
            # Transaction en échec, rollback avant de créer un savepoint
            try:
                db.session.rollback()
            except Exception:
                # Si le rollback échoue aussi, fermer la session
                db.session.close()

        # Essayer de créer un savepoint (fonctionne si transaction existe)
        with db.session.begin_nested():
            yield
        return
    except InvalidRequestError as e:
        # ✅ Si begin_nested() échoue car aucune transaction n'existe,
        # alors créer une transaction racine
        error_msg = str(e).lower()
        if "no transaction" in error_msg or "not within a transaction" in error_msg:
            # Aucune transaction active, créer une transaction racine
            with db.session.begin():
                yield
            return
        # Autre erreur InvalidRequestError (ex: "already begun"), re-lancer
        raise
    except Exception:
        # Autre erreur, re-lancer
        raise
