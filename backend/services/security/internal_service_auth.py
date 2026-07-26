"""Authentification M2M fail-closed pour les appels internes (F-01).

Sans dépendance aux blueprints routes — boot-safe depuis ``app.validate_required_env_vars``.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Mapping

logger = logging.getLogger(__name__)

_DEFAULT_AUDIENCE = "ws-service"
_MIN_TOKEN_LEN = 32
_RATE_LIMIT_PRINCIPAL = "ws-service"

# TTL idempotence (secondes) — figés F-01
_DEFAULT_PENDING_TTL_SEC = 60
_DEFAULT_DONE_TTL_SEC = 86400  # 24 h
_PENDING_TTL_MIN = 5
_PENDING_TTL_MAX = 300
_DONE_TTL_MIN = 3600  # 1 h
_DONE_TTL_MAX = 604800  # 7 j


def ingest_enabled() -> bool:
    return os.getenv("INTERNAL_TRACKING_INGEST_ENABLED", "true").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def get_internal_service_tokens() -> tuple[str, str]:
    """Retourne (CURRENT, NEXT) lus à chaque appel (pas de freeze à l'import)."""
    current = (os.getenv("INTERNAL_SERVICE_TOKEN") or "").strip()
    next_token = (os.getenv("INTERNAL_SERVICE_TOKEN_NEXT") or "").strip()
    return current, next_token


def get_expected_audience() -> str:
    raw = (os.getenv("INTERNAL_SERVICE_AUDIENCE") or _DEFAULT_AUDIENCE).strip()
    return raw or _DEFAULT_AUDIENCE


def rate_limit_principal() -> str:
    """Clé rate-limit normalisée — jamais la valeur brute du header client."""
    return _RATE_LIMIT_PRINCIPAL


def get_idempotency_ttls() -> tuple[int, int]:
    """Retourne (ttl_pending, ttl_done) en secondes."""
    pending = int(
        os.getenv(
            "INTERNAL_TRACKING_IDEMPOTENCY_PENDING_TTL_SEC",
            str(_DEFAULT_PENDING_TTL_SEC),
        )
    )
    done = int(
        os.getenv(
            "INTERNAL_TRACKING_IDEMPOTENCY_DONE_TTL_SEC",
            str(_DEFAULT_DONE_TTL_SEC),
        )
    )
    return pending, done


def validate_internal_service_token_for_boot(*, config_name: str) -> None:
    """Échoue au démarrage en production si secret / TTL invalides (ingest actif)."""
    if config_name != "production":
        return
    if not ingest_enabled():
        return

    current, next_token = get_internal_service_tokens()
    if not current:
        raise RuntimeError(
            "INTERNAL_SERVICE_TOKEN est requis en production lorsque "
            "INTERNAL_TRACKING_INGEST_ENABLED est actif (F-01 fail-closed)."
        )
    if len(current) < _MIN_TOKEN_LEN:
        raise RuntimeError(
            f"INTERNAL_SERVICE_TOKEN trop court en production "
            f"(min {_MIN_TOKEN_LEN} caractères, F-01)."
        )
    if next_token:
        if len(next_token) < _MIN_TOKEN_LEN:
            raise RuntimeError(
                f"INTERNAL_SERVICE_TOKEN_NEXT trop court en production "
                f"(min {_MIN_TOKEN_LEN} caractères, F-01)."
            )
        if hmac.compare_digest(next_token, current):
            raise RuntimeError(
                "INTERNAL_SERVICE_TOKEN_NEXT doit être différent de "
                "INTERNAL_SERVICE_TOKEN (F-01)."
            )

    audience = get_expected_audience()
    if audience != _DEFAULT_AUDIENCE:
        # Défaut obligatoire ws-service ; surcharge non standard refusée en prod.
        raise RuntimeError(
            "INTERNAL_SERVICE_AUDIENCE doit valoir 'ws-service' en production (F-01)."
        )

    try:
        pending, done = get_idempotency_ttls()
    except ValueError as exc:
        raise RuntimeError(
            f"TTL idempotence F-01 invalides (non entier) : {exc}"
        ) from exc

    if not (_PENDING_TTL_MIN <= pending <= _PENDING_TTL_MAX):
        raise RuntimeError(
            f"INTERNAL_TRACKING_IDEMPOTENCY_PENDING_TTL_SEC hors bornes "
            f"[{_PENDING_TTL_MIN}, {_PENDING_TTL_MAX}] (F-01)."
        )
    if not (_DONE_TTL_MIN <= done <= _DONE_TTL_MAX):
        raise RuntimeError(
            f"INTERNAL_TRACKING_IDEMPOTENCY_DONE_TTL_SEC hors bornes "
            f"[{_DONE_TTL_MIN}, {_DONE_TTL_MAX}] (F-01)."
        )

    # ttl_pending doit rester > timeout producteur Kafka (+ marge).
    kafka_timeout = float(os.getenv("KAFKA_PRODUCE_TIMEOUT_S", "1.5"))
    kafka_max_block_s = float(os.getenv("KAFKA_MAX_BLOCK_MS", "1000")) / 1000.0
    min_pending = kafka_timeout + kafka_max_block_s + 5.0
    if pending <= min_pending:
        raise RuntimeError(
            f"INTERNAL_TRACKING_IDEMPOTENCY_PENDING_TTL_SEC ({pending}s) doit être "
            f"> timeout Kafka producteur + marge ({min_pending:.1f}s, F-01)."
        )


def authorize_internal_request(
    headers: Mapping[str, str] | None,
) -> tuple[bool, str | None]:
    """Auth service fail-closed.

    Returns:
        (ok, error_code) — missing_token / invalid_token / invalid_audience.
    """
    current, next_token = get_internal_service_tokens()
    if not current and not next_token:
        return False, "missing_token"

    hdrs = headers or {}
    provided = (hdrs.get("X-Internal-Token") or hdrs.get("x-internal-token") or "").strip()
    if not provided:
        return False, "invalid_token"

    token_ok = False
    if current and hmac.compare_digest(provided, current):
        token_ok = True
    elif next_token and hmac.compare_digest(provided, next_token):
        token_ok = True
    if not token_ok:
        return False, "invalid_token"

    audience_expected = get_expected_audience()
    audience_provided = (
        hdrs.get("X-Internal-Service") or hdrs.get("x-internal-service") or ""
    ).strip()
    if not audience_provided or not hmac.compare_digest(
        audience_provided, audience_expected
    ):
        return False, "invalid_audience"

    return True, None
