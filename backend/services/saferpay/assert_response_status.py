"""Valeurs canoniques de ``status`` pour la réponse assert Saferpay (§11.2 plan LIRIE).

Toute évolution de cette liste doit être synchronisée avec le frontend et les tests.
"""

from __future__ import annotations

# Réponse métier 200 — clé JSON ``status`` (snake_case uniquement)
SAFERPAY_FINALIZE_ALREADY_COMPLETED = "already_completed"
SAFERPAY_FINALIZE_COMPLETED = "completed"
SAFERPAY_FINALIZE_PAYMENT_FAILED = "payment_failed"
SAFERPAY_FINALIZE_ASSERT_FAILED = "assert_failed"
SAFERPAY_FINALIZE_UNEXPECTED_TX_STATUS = "unexpected_tx_status"
SAFERPAY_FINALIZE_ASSERT_TRANSIENT = "assert_transient"
SAFERPAY_FINALIZE_CAPTURE_FAILED = "capture_failed"

SAFERPAY_FINALIZE_RESPONSE_STATUSES: tuple[str, ...] = (
    SAFERPAY_FINALIZE_ALREADY_COMPLETED,
    SAFERPAY_FINALIZE_COMPLETED,
    SAFERPAY_FINALIZE_PAYMENT_FAILED,
    SAFERPAY_FINALIZE_ASSERT_FAILED,
    SAFERPAY_FINALIZE_UNEXPECTED_TX_STATUS,
    SAFERPAY_FINALIZE_ASSERT_TRANSIENT,
    SAFERPAY_FINALIZE_CAPTURE_FAILED,
)
