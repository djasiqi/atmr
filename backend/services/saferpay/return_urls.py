"""URLs de retour Saferpay Payment Page (open redirect mitigé comme Worldline)."""

from __future__ import annotations

import os
from urllib.parse import urlparse

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


def _url_matches_prefix(url: str, prefix: str) -> bool:
    u = _strip_slash(url)
    p = _strip_slash(prefix)
    # Schémas personnalisés (ex. lirie://) : préfixe exact sur l'URL complète.
    if p.endswith("://"):
        return u.startswith(p)
    return u == p or u.startswith(p + "/") or u.startswith(p + "?")


def validate_return_url_override(url: str) -> str:
    u = (url or "").strip()
    if not u:
        raise ValueError("return_url ne peut pas être vide")
    parsed = urlparse(u)
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
    parsed = urlparse(url)
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
