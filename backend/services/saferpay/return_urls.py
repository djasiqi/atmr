"""URLs de retour Saferpay Payment Page (open redirect mitigé comme Worldline)."""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from services.saferpay.config import ensure_saferpay_loopback_env_from_repo


def _strip_slash(url: str) -> str:
    return url.strip().rstrip("/")


def allowed_saferpay_return_prefixes() -> list[str]:
    prefixes: list[str] = []
    for key in (
        "SAFERPAY_CHECKOUT_PUBLIC_BASE_URL",
        "WORLDLINE_CHECKOUT_PUBLIC_BASE_URL",
        "CLIENT_WEB_BASE_URL",
        "PUBLIC_BASE_URL",
    ):
        v = (os.getenv(key) or "").strip()
        if v:
            prefixes.append(_strip_slash(v))
    raw = (os.getenv("SAFERPAY_ALLOWED_RETURN_URL_PREFIXES") or "").strip()
    if raw:
        for part in raw.split(","):
            p = _strip_slash(part)
            if p:
                prefixes.append(p)
    # Deep link app mobile — retour Saferpay parcours invité.
    # - lirie: build (scheme app.json)
    # - exp / exps: Expo Go (dev), souvent requis en local (createURL n'utilise pas lirie://)
    # - expo+lirie: certains dev clients / builds Expo
    prefixes.append("lirie://")
    prefixes.append("exp://")
    prefixes.append("exps://")
    prefixes.append("expo+lirie://")
    seen: set[str] = set()
    out: list[str] = []
    for p in prefixes:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


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


def _http_return_matches_allowed_base(url: str, base: str) -> bool:
    """Même origine (scheme + host + port) et chemin sous le base Saferpay."""
    candidate = urlsplit(url)
    allowed = urlsplit(base)
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


def _custom_scheme_return_matches(url: str, prefix: str) -> bool:
    """Politique des deep links app (lirie / exp / …), hors http(s)."""
    allowed = urlsplit(prefix.strip())
    candidate = urlsplit(url.strip())
    if not allowed.scheme or allowed.scheme in {"http", "https"}:
        return False
    if candidate.scheme.lower() != allowed.scheme.lower():
        return False
    if not allowed.netloc and not allowed.path:
        return True
    if allowed.netloc.lower() != (candidate.netloc or "").lower():
        return False
    allowed_path = (allowed.path or "").rstrip("/")
    candidate_path = candidate.path or ""
    if not allowed_path:
        return True
    return candidate_path == allowed_path or candidate_path.startswith(
        allowed_path + "/"
    )


def _url_matches_prefix(url: str, prefix: str) -> bool:
    raw_url = url.strip()
    raw_prefix = prefix.strip()
    parsed_prefix = urlsplit(raw_prefix)
    if parsed_prefix.scheme and parsed_prefix.scheme not in {"http", "https"}:
        return _custom_scheme_return_matches(raw_url, raw_prefix)
    if parsed_prefix.scheme in {"http", "https"}:
        return _http_return_matches_allowed_base(raw_url, _strip_slash(raw_prefix))
    return False


def validate_return_url_override(url: str) -> str:
    u = (url or "").strip()
    if not u:
        raise ValueError("return_url ne peut pas être vide")
    parsed = urlsplit(u)
    if not parsed.scheme:
        raise ValueError("return_url doit définir un schéma")
    if parsed.scheme in {"javascript", "data", "file"}:
        raise ValueError("return_url utilise un schéma interdit")
    if parsed.scheme in ("http", "https"):
        if not parsed.netloc:
            raise ValueError("return_url doit être une URL absolue (http/https)")
    elif not (parsed.netloc or parsed.path):
        raise ValueError("return_url doit être une URL absolue")
    prefixes = allowed_saferpay_return_prefixes()
    if not prefixes:
        raise ValueError(
            "return_url personnalisee interdite: definir CLIENT_WEB_BASE_URL, "
            + "PUBLIC_BASE_URL ou SAFERPAY_CHECKOUT_PUBLIC_BASE_URL"
        )
    if not any(_url_matches_prefix(u, pr) for pr in prefixes):
        raise ValueError("return_url non autorisée par la configuration serveur")
    return u


def _is_loopback_http_url(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    return host.endswith(".localhost")


def _localhost_return_allowed_for_saferpay() -> bool:
    """True si retour localhost autorise (env explicite ou mode developpement Flask)."""
    raw = (os.getenv("SAFERPAY_ALLOW_LOCALHOST_RETURN") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    env = (os.getenv("FLASK_ENV") or os.getenv("FLASK_CONFIG") or "").strip().lower()
    debug_on = (os.getenv("FLASK_DEBUG") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    return env in ("development", "dev") or debug_on


def default_saferpay_return_urls(
    *,
    booking_id: int,
    payment_id: int,
) -> tuple[str, str, str]:
    """Success / Fail / Abort — mêmes paramètres GET pour corréler côté client."""
    ensure_saferpay_loopback_env_from_repo()
    base = (os.getenv("SAFERPAY_CHECKOUT_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if not base:
        base = (os.getenv("CLIENT_WEB_BASE_URL") or "").strip().rstrip("/")
    if not base:
        base = (
            (os.getenv("PUBLIC_BASE_URL") or "http://localhost:3000")
            .strip()
            .rstrip("/")
        )
    q = f"bookingId={booking_id}&paymentId={payment_id}"
    success = f"{base}/client/payment/saferpay/return?{q}&outcome=success"
    fail = f"{base}/client/payment/saferpay/return?{q}&outcome=fail"
    abort = f"{base}/client/payment/saferpay/return?{q}&outcome=abort"
    for u in (success, fail, abort):
        if _is_loopback_http_url(u) and not _localhost_return_allowed_for_saferpay():
            raise ValueError(
                "Saferpay: URL de retour en localhost souvent refusée. "
                + "Définissez SAFERPAY_CHECKOUT_PUBLIC_BASE_URL (HTTPS public) "
                + "ou SAFERPAY_ALLOW_LOCALHOST_RETURN=1."
            )
    return success, fail, abort
