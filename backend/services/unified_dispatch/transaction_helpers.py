# backend/services/unified_dispatch/transaction_helpers.py
"""Helpers de gestion de transactions DB pour éviter les cycles d'import.

Ce module centralise les helpers de transaction utilisés par engine.py et apply.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

from ext import db

__all__ = ["_begin_tx", "_in_tx"]  # Exported functions used by engine.py and apply.py


def _in_tx() -> bool:
    """Détecte proprement une transaction active sur la session SQLAlchemy,
    sans dépendre d'un stub précis (Pylance-friendly).

    Returns:
        True si une transaction est active, False sinon
    """
    try:
        meth: Callable[[], Any] | None = getattr(db.session, "in_transaction", None)
        if callable(meth):
            return bool(meth())
        get_tx: Callable[[], Any] | None = getattr(db.session, "get_transaction", None)
        if callable(get_tx):
            return get_tx() is not None
    except Exception:
        pass
    # Fallback raisonnable
    return bool(getattr(db.session, "is_active", False))


@contextmanager
def _begin_tx():
    """Ouvre une transaction en s'adaptant à l'état courant de la Session.
    - Si une transaction est déjà ouverte (implicitement ou non), on utilise un savepoint (begin_nested).
    - Sinon, on ouvre une transaction normale (begin).

    Usage:
        with _begin_tx():
            # Code exécuté dans une transaction ou savepoint
            pass
    """
    cm = db.session.begin_nested() if _in_tx() else db.session.begin()
    with cm:
        yield
