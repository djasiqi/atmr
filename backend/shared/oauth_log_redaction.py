"""Masque les codes OAuth dans les journaux HTTP.

Le callback reçoit le code dans l'URL. Le traitement du paramètre
ne change pas : seule la ligne écrite dans le journal est filtrée.
"""

from __future__ import annotations

import logging
import re

_OAUTH_QUERY = re.compile(
    r"(?i)([?&\s]|^)((?:code|id_token|access_token|refresh_token|"
    r"code_verifier|client_secret)=)([^&\s\"']+)"
)
_INSTALLED = "_oauth_log_redaction_installed"


def redact_oauth_query(value: str) -> str:
    """Remplace la valeur d'un code ou d'un jeton par un marqueur fixe."""
    return _OAUTH_QUERY.sub(r"\1\2[filtre]", value)


class OAuthLogFilter(logging.Filter):
    """Filtre les lignes d'accès Gunicorn sans convertir leurs arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, dict):
            for key, item in list(args.items()):
                if isinstance(item, str):
                    args[key] = redact_oauth_query(item)
        elif isinstance(args, tuple):
            record.args = tuple(
                redact_oauth_query(item) if isinstance(item, str) else item
                for item in args
            )
        if isinstance(record.msg, str):
            record.msg = redact_oauth_query(record.msg)
        return True


def install_oauth_log_redaction() -> None:
    """Branche le filtre sur les journaux qui voient l'URL de callback."""
    names = ("gunicorn.access", "gunicorn.error", "werkzeug", "")
    redactor = OAuthLogFilter()
    for name in names:
        logger = logging.getLogger(name)
        if getattr(logger, _INSTALLED, False):
            continue
        logger.addFilter(redactor)
        setattr(logger, _INSTALLED, True)
