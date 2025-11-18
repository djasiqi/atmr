# backend/shared/dispatch_logging.py
"""✅ Logger centralisé avec contexte dispatch_run_id et corrélation OpenTelemetry.

Usage:
    from shared.dispatch_logging import get_dispatch_logger

    logger = get_dispatch_logger(dispatch_run_id=123)
    logger.info("Message")  # Inclut automatiquement dispatch_run_id dans le contexte
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import override
else:

    def override(func):
        return func


# Context variable pour dispatch_run_id (thread-safe)
_dispatch_run_id: ContextVar[int | None] = ContextVar("dispatch_run_id", default=None)
_company_id: ContextVar[int | None] = ContextVar("company_id", default=None)

# Cache des loggers par nom
_loggers_cache: dict[str, logging.Logger] = {}


class DispatchLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    """Adapter de logger qui ajoute automatiquement dispatch_run_id et company_id au contexte."""

    @override
    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:
        """Ajoute dispatch_run_id et company_id au contexte des logs."""
        # Récupérer les valeurs depuis context vars
        dispatch_run_id = _dispatch_run_id.get()
        company_id = _company_id.get()

        # Ajouter au dictionnaire 'extra' pour structuré logging
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        extra = kwargs["extra"]

        # Ajouter dispatch_run_id si disponible
        if dispatch_run_id is not None:
            extra["dispatch_run_id"] = dispatch_run_id

        # Ajouter company_id si disponible
        if company_id is not None:
            extra["company_id"] = company_id

        # Ajouter trace_id OpenTelemetry si disponible
        try:
            from opentelemetry import trace  # type: ignore[import-untyped]

            span = trace.get_current_span()
            if span and span.is_recording():
                span_context = span.get_span_context()
                if span_context.is_valid:
                    trace_id = format(span_context.trace_id, "032x")
                    extra["trace_id"] = trace_id
                    extra["span_id"] = format(span_context.span_id, "016x")
        except ImportError:
            pass  # OpenTelemetry non disponible
        except Exception:
            pass  # Ne pas bloquer si OpenTelemetry échoue

        # Format du message avec contexte
        if dispatch_run_id is not None:
            msg = f"[dispatch_run_id={dispatch_run_id}] {msg}"

        return msg, kwargs


def set_dispatch_context(dispatch_run_id: int | None = None, company_id: int | None = None) -> None:
    """Définit le contexte dispatch pour les logs suivants.

    Args:
        dispatch_run_id: ID du dispatch run
        company_id: ID de l'entreprise
    """
    if dispatch_run_id is not None:
        _dispatch_run_id.set(dispatch_run_id)
    if company_id is not None:
        _company_id.set(company_id)


def clear_dispatch_context() -> None:
    """Efface le contexte dispatch."""
    _dispatch_run_id.set(None)
    _company_id.set(None)


def get_dispatch_logger(
    name: str | None = None,
    dispatch_run_id: int | None = None,
    company_id: int | None = None,
) -> logging.LoggerAdapter[logging.Logger]:
    """Retourne un logger avec contexte dispatch_run_id.

    Args:
        name: Nom du logger (défaut: nom du module appelant)
        dispatch_run_id: ID du dispatch run (optionnel, peut être défini via set_dispatch_context)
        company_id: ID de l'entreprise (optionnel, peut être défini via set_dispatch_context)

    Returns:
        LoggerAdapter avec contexte dispatch_run_id
    """
    if name is None:
        import inspect

        frame = inspect.currentframe()
        name = frame.f_back.f_globals.get("__name__", "root") if frame and frame.f_back else "root"

    # À ce point, name est toujours une string (pas None)
    assert name is not None, "name should never be None at this point"
    logger_name = name

    # Récupérer ou créer le logger
    if logger_name not in _loggers_cache:
        base_logger = logging.getLogger(logger_name)
        _loggers_cache[logger_name] = base_logger

    base_logger = _loggers_cache[logger_name]

    # Si dispatch_run_id ou company_id fournis, les définir dans le contexte
    if dispatch_run_id is not None:
        set_dispatch_context(dispatch_run_id=dispatch_run_id, company_id=company_id)
    elif company_id is not None:
        set_dispatch_context(company_id=company_id)

    return DispatchLoggerAdapter(base_logger, {})


@contextmanager
def dispatch_logging_context(
    dispatch_run_id: int | None = None,
    company_id: int | None = None,
):
    """Context manager pour définir le contexte dispatch pour les logs.

    Usage:
        with dispatch_logging_context(dispatch_run_id=123, company_id=1):
            logger.info("Message")  # Inclut dispatch_run_id=123 dans le contexte
    """
    # Sauvegarder les valeurs précédentes
    old_dispatch_run_id = _dispatch_run_id.get()
    old_company_id = _company_id.get()

    # Définir le nouveau contexte
    set_dispatch_context(dispatch_run_id=dispatch_run_id, company_id=company_id)

    try:
        yield
    finally:
        # Restaurer le contexte précédent
        _dispatch_run_id.set(old_dispatch_run_id)
        _company_id.set(old_company_id)
