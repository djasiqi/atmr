"""Classification des erreurs SQLAlchemy pour le consumer tracking ingest.

Matrice P0 :
- OperationalError / InterfaceError / DisconnectionError / Timeout → infra (retry puis fail-stop)
- ProgrammingError → fail-stop immédiat (schéma / SQL cassé)
- DataError attribuable au message → DLQ
- IntegrityError unique location_event_id → duplicate idempotent
- IntegrityError connue (données invalides) → DLQ
- IntegrityError inconnue → fail-stop
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
    IDEMPOTENT_DUPLICATE = "idempotent_duplicate"


_INFRASTRUCTURE_DB_ERRORS = (
    OperationalError,
    InterfaceError,
    DisconnectionError,
    SQLAlchemyTimeoutError,
)

# Contraintes / messages IntegrityError considérés comme données message invalides.
_KNOWN_INVALID_DATA_INTEGRITY_TOKENS = (
    "check constraint",
    "not-null constraint",
    "foreign key constraint",
    "violates check constraint",
    "violates not-null constraint",
    "violates foreign key constraint",
)

_LOCATION_EVENT_ID_UNIQUE_TOKENS = (
    "location_event_id",
    "uq_dle",
    "driver_location_events",
    "ix_dle_driver_event",
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


def _is_location_event_id_unique_violation(exc: IntegrityError) -> bool:
    msg = _message_of(exc)
    if "unique" not in msg and "duplicate" not in msg:
        return False
    return any(token in msg for token in _LOCATION_EVENT_ID_UNIQUE_TOKENS)


def _is_known_invalid_data_integrity(exc: IntegrityError) -> bool:
    msg = _message_of(exc)
    return any(token in msg for token in _KNOWN_INVALID_DATA_INTEGRITY_TOKENS)


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
    # Priorité : parcourir la chaîne ; les types « plus spécifiques » d'abord
    # via l'ordre des isinstance dans chaque nœud.
    for current in _iter_exception_chain(exc):
        if isinstance(current, ProgrammingError):
            return DbErrorAction.FAIL_STOP

        if isinstance(current, IntegrityError):
            if _is_location_event_id_unique_violation(current):
                return DbErrorAction.IDEMPOTENT_DUPLICATE
            if _is_known_invalid_data_integrity(current):
                return DbErrorAction.DLQ
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
