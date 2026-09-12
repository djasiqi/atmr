"""Politique unique des URLs de logo persistées (company / institution).

Contrat :
- persisté : chemin ``/uploads/...`` ou URL ``https://``
- refusé : http, data, blob, javascript, file, vbscript, protocol-relative
- ``blob:`` n'est jamais une valeur persistée
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

LOGO_URL_DENIED_MESSAGE = "URL de logo non autorisée"
MAX_LOGO_URL_LENGTH = 500
_UPLOADS_PATH_RE = re.compile(r"^/uploads/[A-Za-z0-9._/-]+$")


class InvalidLogoUrl(ValueError):
    """URL de logo hors politique."""


def normalize_persisted_logo_url(raw: str | None) -> str | None:
    """Valide et normalise une URL de logo destinée à la base.

    Returns:
        None si vide, sinon le chemin ``/uploads/...`` ou une URL https reconstruite.

    Raises:
        InvalidLogoUrl: si la valeur n'est pas autorisée.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise InvalidLogoUrl(LOGO_URL_DENIED_MESSAGE)

    value = raw.strip()
    if not value:
        return None
    if len(value) > MAX_LOGO_URL_LENGTH:
        raise InvalidLogoUrl(LOGO_URL_DENIED_MESSAGE)
    if any(ch in value for ch in ("\r", "\n", "\x00")):
        raise InvalidLogoUrl(LOGO_URL_DENIED_MESSAGE)
    if value.startswith("//"):
        raise InvalidLogoUrl(LOGO_URL_DENIED_MESSAGE)

    if value.startswith("/"):
        path = value.split("?", 1)[0].split("#", 1)[0]
        if ".." in path or "//" in path:
            raise InvalidLogoUrl(LOGO_URL_DENIED_MESSAGE)
        if not _UPLOADS_PATH_RE.fullmatch(path):
            raise InvalidLogoUrl(LOGO_URL_DENIED_MESSAGE)
        return path

    parsed = urlparse(value)
    scheme = (parsed.scheme or "").lower()
    if scheme != "https":
        raise InvalidLogoUrl(LOGO_URL_DENIED_MESSAGE)
    if parsed.username or parsed.password:
        raise InvalidLogoUrl(LOGO_URL_DENIED_MESSAGE)
    hostname = parsed.hostname
    if not hostname:
        raise InvalidLogoUrl(LOGO_URL_DENIED_MESSAGE)

    host = hostname
    if parsed.port:
        host = f"{hostname}:{parsed.port}"
    path = parsed.path or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"https://{host}{path}{query}"
