"""Variables d'environnement Saferpay (Backoffice → JSON API)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

_TEST_API_HOST_MARKERS = ("test.saferpay.com", "test.saferpay")

_SAFERPAY_REQUIRED_KEYS = (
    "SAFERPAY_CUSTOMER_ID",
    "SAFERPAY_TERMINAL_ID",
    "SAFERPAY_API_USERNAME",
    "SAFERPAY_API_PASSWORD",
)

# Cles additionnelles a relire depuis le .env racine si absentes / vides (launch.json, etc.).
_SAFERPAY_OPTIONAL_MERGE_KEYS = (
    "SAFERPAY_ALLOW_LOCALHOST_RETURN",
    "SAFERPAY_CHECKOUT_PUBLIC_BASE_URL",
    "SAFERPAY_ALLOWED_RETURN_URL_PREFIXES",
    "SAFERPAY_API_BASE_URL",
    "SAFERPAY_NOTIFY_API_BASE_URL",
    "SAFERPAY_SPEC_VERSION",
    "SAFERPAY_ALLOW_TEST_API_IN_PRODUCTION",
    "SAFERPAY_PAYER_LANGUAGE",
)

_SAFERPAY_MERGE_KEYS = _SAFERPAY_REQUIRED_KEYS + _SAFERPAY_OPTIONAL_MERGE_KEYS

# Mutable pour eviter ``global`` (Ruff PLW0603) tout en gardant une passe unique.
_saferpay_repo_merge_state: dict[str, bool] = {"done": False}


def _repo_root_env_path() -> Path:
    """``atmr/.env`` lorsque ce fichier est sous ``atmr/backend/services/saferpay/``."""
    return Path(__file__).resolve().parents[3] / ".env"


def _ensure_saferpay_env_from_repo_root_once() -> None:
    """Complete les cles SAFERPAY (API + URLs / localhost) si absentes ou vides.

    ``load_dotenv(..., override=False)`` ne remplace pas une variable deja
    definie (y compris a la chaine vide), ce qui arrive avec des launch.json
    ou des templates. On relit alors uniquement le ``.env`` racine du depot.
    """
    if _saferpay_repo_merge_state["done"]:
        return
    _saferpay_repo_merge_state["done"] = True
    root = _repo_root_env_path()
    if not root.is_file():
        return
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    vals = dotenv_values(root)
    for key in _SAFERPAY_MERGE_KEYS:
        if (os.getenv(key) or "").strip():
            continue
        raw = vals.get(key)
        if raw is None:
            continue
        v = str(raw).strip()
        if v:
            os.environ[key] = v


def ensure_saferpay_loopback_env_from_repo() -> None:
    """Complemente depuis le .env racine a chaque init des URLs de retour Saferpay.

    La fusion ``_ensure_saferpay_env_from_repo_root_once`` ne s'execute qu'une fois
    au premier ``saferpay_configured()`` ; ici on re-lit les cles liees au
    localhost / base publique si elles sont encore absentes ou vides (evite
    ``backend/.env`` sans ces lignes ou ``ALLOW`` vide sous Windows).
    """
    root = _repo_root_env_path()
    if not root.is_file():
        return
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    vals = dotenv_values(root)
    for key in (
        "SAFERPAY_ALLOW_LOCALHOST_RETURN",
        "SAFERPAY_CHECKOUT_PUBLIC_BASE_URL",
    ):
        if (os.getenv(key) or "").strip():
            continue
        raw = vals.get(key)
        if raw is None:
            continue
        v = str(raw).strip()
        if v:
            os.environ[key] = v


def saferpay_api_url_looks_like_test_host(url: str) -> bool:
    u = (url or "").lower()
    return any(m in u for m in _TEST_API_HOST_MARKERS)


def warn_saferpay_test_api_url_in_production(
    logger: Any | None = None,
    *,
    config_name: str,
) -> None:
    """En production, journalise une erreur si l'API JSON pointe vers un hôte de test."""
    if (config_name or "").strip().lower() != "production":
        return
    if (os.getenv("SAFERPAY_ALLOW_TEST_API_IN_PRODUCTION") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    log = logger or logging.getLogger(__name__)
    base = saferpay_api_base_url()
    if saferpay_api_url_looks_like_test_host(base):
        msg = (
            "SAFERPAY_API_BASE_URL semble pointer vers un environnement de test (%s) "
            + "alors que FLASK_CONFIG=production. Corriger l'URL ou definir "
            + "SAFERPAY_ALLOW_TEST_API_IN_PRODUCTION=1 pour une exception explicite."
        )
        log.error(msg, base)


def saferpay_configured() -> bool:
    _ensure_saferpay_env_from_repo_root_once()
    return bool(
        (os.getenv("SAFERPAY_CUSTOMER_ID") or "").strip()
        and (os.getenv("SAFERPAY_TERMINAL_ID") or "").strip()
        and (os.getenv("SAFERPAY_API_USERNAME") or "").strip()
        and (os.getenv("SAFERPAY_API_PASSWORD") or "").strip()
    )


def saferpay_api_base_url() -> str:
    return (
        (os.getenv("SAFERPAY_API_BASE_URL") or "").strip().rstrip("/")
        or "https://test.saferpay.com/api"
    )


def saferpay_spec_version() -> str:
    return (os.getenv("SAFERPAY_SPEC_VERSION") or "1.30").strip()


def saferpay_public_api_prefix() -> str:
    """Base publique de l'API ATMR pour les Notification URLs (ex. https://api.example.com/api/v1)."""
    return (os.getenv("SAFERPAY_NOTIFY_API_BASE_URL") or "").strip().rstrip("/")
