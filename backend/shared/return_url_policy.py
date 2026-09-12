"""Politique commune des return URL http(s) de paiement.

Compare scheme, hostname et port effectif, puis le chemin sous la base.
Pas un validateur d'URL générique : uniquement le contrat redirect PSP.
"""

from __future__ import annotations

from urllib.parse import urlsplit


def strip_return_url_base(url: str) -> str:
    return url.strip().rstrip("/")


def _effective_http_port(parsed) -> int | None:
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    return None


def _normalize_hostname(host: str | None) -> str:
    return (host or "").lower().rstrip(".")


def http_return_matches_allowed_base(url: str, base: str) -> bool:
    """True si ``url`` est same-origin de ``base`` et sous son chemin."""
    candidate = urlsplit(url.strip())
    allowed = urlsplit(strip_return_url_base(base))
    if candidate.scheme not in {"http", "https"}:
        return False
    if allowed.scheme not in {"http", "https"}:
        return False
    if candidate.scheme != allowed.scheme:
        return False
    if candidate.username or candidate.password:
        return False
    if _normalize_hostname(candidate.hostname) != _normalize_hostname(allowed.hostname):
        return False
    if not candidate.hostname or not allowed.hostname:
        return False
    if _effective_http_port(candidate) != _effective_http_port(allowed):
        return False
    allowed_path = (allowed.path or "").rstrip("/")
    candidate_path = candidate.path or ""
    if not allowed_path:
        return True
    return candidate_path == allowed_path or candidate_path.startswith(
        allowed_path + "/"
    )
