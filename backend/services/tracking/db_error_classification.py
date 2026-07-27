"""Classification des erreurs SQLAlchemy pour le consumer tracking ingest.

Matrice P0 (fail-stop durcie) :
- OperationalError / InterfaceError / DisconnectionError / Timeout → infra (retry puis fail-stop)
- ProgrammingError → fail-stop immédiat (schéma / SQL cassé)
- DataError exacte connue attribuable au payload → DLQ
- Toute IntegrityError → fail-stop (zéro DLQ ; duplicate nominal via ON CONFLICT seulement)
- Erreur inconnue (hors SQLAlchemy classifiée) → None → fail-stop côté consumer
"""

from __future__ import annotations

from enum import Enum

from sqlalchemy.exc import (
    DataError,
    DisconnectionError,
    IntegrityError,
    InterfaceError,
    OperationalError,
    ProgrammingError,
    TimeoutError as SQLAlchemyTimeoutError,
)


class DbErrorAction(str, Enum):
    """Action à prendre après classification d'une erreur DB."""

    INFRASTRUCTURE_RETRY = "infrastructure_retry"
    FAIL_STOP = "fail_stop"
    DLQ = "dlq"


_INFRASTRUCTURE_DB_ERRORS = (
    OperationalError,
    InterfaceError,
    DisconnectionError,
    SQLAlchemyTimeoutError,
)


def _iter_exception_chain(exc: BaseException):
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _message_of(exc: BaseException) -> str:
    return str(exc).lower()


def _is_message_attributable_data_error(exc: DataError) -> bool:
    """DataError typiquement dû à un payload (type, longueur, encoding)."""
    msg = _message_of(exc)
    attributable = (
        "invalid input",
        "value too long",
        "out of range",
        "invalid byte sequence",
        "date/time field",
        "numeric value",
        "character varying",
    )
    return any(token in msg for token in attributable)


def classify_db_error(exc: BaseException) -> DbErrorAction | None:
    """Classe une exception (et sa chaîne) selon la matrice P0.

    Retourne None si ce n'est pas une erreur SQLAlchemy classifiée.
    """
    for current in _iter_exception_chain(exc):
        if isinstance(current, ProgrammingError):
            return DbErrorAction.FAIL_STOP

        if isinstance(current, IntegrityError):
            return DbErrorAction.FAIL_STOP

        if isinstance(current, DataError):
            if _is_message_attributable_data_error(current):
                return DbErrorAction.DLQ
            return DbErrorAction.FAIL_STOP

        if isinstance(current, _INFRASTRUCTURE_DB_ERRORS):
            return DbErrorAction.INFRASTRUCTURE_RETRY

    return None


def is_infrastructure_db_error(exc: BaseException) -> bool:
    return classify_db_error(exc) == DbErrorAction.INFRASTRUCTURE_RETRY
