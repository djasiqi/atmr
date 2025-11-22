# backend/services/unified_dispatch/error_metrics.py
"""Métriques Prometheus pour les erreurs de dispatch.

Ce module fournit des métriques pour surveiller les erreurs critiques :
- CompanyNotFoundError
- ForeignKey violations
- Autres erreurs de dispatch
"""

from __future__ import annotations

import logging

# Import optionnel prometheus_client (peut ne pas être installé en dev)
try:
    from prometheus_client import Counter

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None

logger = logging.getLogger(__name__)

# ==================== Prometheus Metrics ====================

# Métriques Prometheus pour erreurs de dispatch (créées uniquement si prometheus_client disponible)
if PROMETHEUS_AVAILABLE and Counter is not None:
    DISPATCH_ERRORS_TOTAL = Counter(
        "dispatch_errors_total",
        "Nombre total d'erreurs de dispatch",
        [
            "error_type",
            "company_id",
        ],  # error_type: "company_not_found", "fk_violation", etc.
    )

    DISPATCH_COMPANY_NOT_FOUND_TOTAL = Counter(
        "dispatch_company_not_found_total",
        "Nombre total de CompanyNotFoundError",
        ["company_id"],
    )

    DISPATCH_FK_VIOLATION_TOTAL = Counter(
        "dispatch_fk_violation_total",
        "Nombre total de violations de contrainte ForeignKey",
        [
            "fk_constraint",
            "company_id",
        ],  # fk_constraint: "company_id", "driver_id", etc.
    )

    DISPATCH_INTEGRITY_ERROR_TOTAL = Counter(
        "dispatch_integrity_error_total",
        "Nombre total d'erreurs d'intégrité DB (IntegrityError)",
        [
            "error_code",
            "company_id",
        ],  # error_code: "23503" (FK), "23505" (unique), etc.
    )

    # ✅ FIX: Initialiser les métriques avec 0.0 pour qu'elles apparaissent même si jamais incrémentées
    try:
        DISPATCH_ERRORS_TOTAL.labels(
            error_type="company_not_found", company_id="0"
        ).inc(0)
        DISPATCH_ERRORS_TOTAL.labels(error_type="fk_violation", company_id="0").inc(0)
        DISPATCH_COMPANY_NOT_FOUND_TOTAL.labels(company_id="0").inc(0)
        DISPATCH_FK_VIOLATION_TOTAL.labels(
            fk_constraint="company_id", company_id="0"
        ).inc(0)
        DISPATCH_INTEGRITY_ERROR_TOTAL.labels(error_code="23503", company_id="0").inc(0)
    except Exception as e:
        # Ignorer les erreurs d'initialisation (peut échouer si Prometheus non configuré)
        logger.debug(
            "[Dispatch Error Metrics] Failed to initialize metrics with 0.0: %s", e
        )
else:
    DISPATCH_ERRORS_TOTAL = None
    DISPATCH_COMPANY_NOT_FOUND_TOTAL = None
    DISPATCH_FK_VIOLATION_TOTAL = None
    DISPATCH_INTEGRITY_ERROR_TOTAL = None


def track_company_not_found(
    company_id: int, dispatch_run_id: int | None = None
) -> None:
    """Track une erreur CompanyNotFoundError.

    Args:
        company_id: ID de la company introuvable
        dispatch_run_id: ID du dispatch_run (optionnel, pour corrélation)
    """
    if DISPATCH_COMPANY_NOT_FOUND_TOTAL is not None:
        DISPATCH_COMPANY_NOT_FOUND_TOTAL.labels(company_id=str(company_id)).inc()

    if DISPATCH_ERRORS_TOTAL is not None:
        DISPATCH_ERRORS_TOTAL.labels(
            error_type="company_not_found", company_id=str(company_id)
        ).inc()

    logger.debug(
        "[Dispatch Error Metrics] CompanyNotFoundError tracked: company_id=%s, dispatch_run_id=%s",
        company_id,
        dispatch_run_id,
    )


def track_fk_violation(
    fk_constraint: str,
    company_id: int | None = None,
    dispatch_run_id: int | None = None,
) -> None:
    """Track une violation de contrainte ForeignKey.

    Args:
        fk_constraint: Nom de la contrainte FK (ex: "company_id", "driver_id")
        company_id: ID de la company (optionnel)
        dispatch_run_id: ID du dispatch_run (optionnel, pour corrélation)
    """
    company_id_str = str(company_id) if company_id is not None else "unknown"

    if DISPATCH_FK_VIOLATION_TOTAL is not None:
        DISPATCH_FK_VIOLATION_TOTAL.labels(
            fk_constraint=fk_constraint, company_id=company_id_str
        ).inc()

    if DISPATCH_ERRORS_TOTAL is not None:
        DISPATCH_ERRORS_TOTAL.labels(
            error_type="fk_violation", company_id=company_id_str
        ).inc()

    logger.debug(
        "[Dispatch Error Metrics] FK violation tracked: fk_constraint=%s, company_id=%s, dispatch_run_id=%s",
        fk_constraint,
        company_id,
        dispatch_run_id,
    )


def track_integrity_error(
    error_code: str,
    company_id: int | None = None,
    dispatch_run_id: int | None = None,
) -> None:
    """Track une erreur d'intégrité DB (IntegrityError).

    Args:
        error_code: Code d'erreur PostgreSQL (ex: "23503" pour FK, "23505" pour unique)
        company_id: ID de la company (optionnel)
        dispatch_run_id: ID du dispatch_run (optionnel, pour corrélation)
    """
    company_id_str = str(company_id) if company_id is not None else "unknown"

    if DISPATCH_INTEGRITY_ERROR_TOTAL is not None:
        DISPATCH_INTEGRITY_ERROR_TOTAL.labels(
            error_code=error_code, company_id=company_id_str
        ).inc()

    # Détecter le type d'erreur basé sur le code
    error_type = "unknown"
    if error_code == "23503":
        error_type = "fk_violation"
        track_fk_violation("unknown", company_id, dispatch_run_id)
    elif error_code == "23505":
        error_type = "unique_violation"
    elif error_code == "23502":
        error_type = "not_null_violation"

    if DISPATCH_ERRORS_TOTAL is not None:
        DISPATCH_ERRORS_TOTAL.labels(
            error_type=error_type, company_id=company_id_str
        ).inc()

    logger.debug(
        "[Dispatch Error Metrics] IntegrityError tracked: error_code=%s, company_id=%s, dispatch_run_id=%s",
        error_code,
        company_id,
        dispatch_run_id,
    )


def track_dispatch_error(
    error_type: str,
    company_id: int | None = None,
    dispatch_run_id: int | None = None,
) -> None:
    """Track une erreur de dispatch générique.

    Args:
        error_type: Type d'erreur (ex: "timeout", "validation_error", etc.)
        company_id: ID de la company (optionnel)
        dispatch_run_id: ID du dispatch_run (optionnel, pour corrélation)
    """
    company_id_str = str(company_id) if company_id is not None else "unknown"

    if DISPATCH_ERRORS_TOTAL is not None:
        DISPATCH_ERRORS_TOTAL.labels(
            error_type=error_type, company_id=company_id_str
        ).inc()

    logger.debug(
        "[Dispatch Error Metrics] Dispatch error tracked: error_type=%s, company_id=%s, dispatch_run_id=%s",
        error_type,
        company_id,
        dispatch_run_id,
    )
