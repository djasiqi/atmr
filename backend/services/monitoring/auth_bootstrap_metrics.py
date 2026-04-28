"""Métriques Prometheus — bootstrap GET /auth/me (optionnel si prometheus_client absent)."""

from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram

    _AUTH_ME_TOTAL = Counter(
        "auth_me_http_total",
        "Réponses GET /api/v1/auth/me par code HTTP",
        ["status"],
    )
    _AUTH_ME_FORBIDDEN = Counter(
        "auth_me_forbidden_total",
        "403 GET /auth/me par access_denied_code",
        ["access_denied_code"],
    )
    _AUTH_ME_DRIVER_NO_ROW = Counter(
        "auth_me_driver_role_without_driver_row_total",
        "Succès 200 avec rôle driver mais aucune ligne Driver (héritage métier)",
    )
    _AUTH_ME_DURATION = Histogram(
        "auth_me_duration_seconds",
        "Durée traitement GET /auth/me (use case)",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )
    _AUTH_ME_PAYLOAD_BYTES = Histogram(
        "auth_me_response_bytes",
        "Taille approximative du JSON GET /auth/me",
        buckets=(128, 256, 512, 1024, 2048, 4096, 8192),
    )
except Exception:  # noqa: BLE001
    _AUTH_ME_TOTAL = None
    _AUTH_ME_FORBIDDEN = None
    _AUTH_ME_DRIVER_NO_ROW = None
    _AUTH_ME_DURATION = None
    _AUTH_ME_PAYLOAD_BYTES = None


def observe_auth_me(status_code: int, duration_s: float, payload_bytes: int) -> None:
    if _AUTH_ME_TOTAL is not None:
        _AUTH_ME_TOTAL.labels(status=str(status_code)).inc()
    if _AUTH_ME_DURATION is not None:
        _AUTH_ME_DURATION.observe(max(0.0, duration_s))
    if _AUTH_ME_PAYLOAD_BYTES is not None:
        _AUTH_ME_PAYLOAD_BYTES.observe(max(0.0, float(payload_bytes)))


def observe_auth_me_forbidden(access_denied_code: str) -> None:
    if _AUTH_ME_FORBIDDEN is not None:
        _AUTH_ME_FORBIDDEN.labels(access_denied_code=access_denied_code).inc()


def inc_driver_role_without_driver_row() -> None:
    if _AUTH_ME_DRIVER_NO_ROW is not None:
        _AUTH_ME_DRIVER_NO_ROW.inc()
