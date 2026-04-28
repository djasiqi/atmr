"""Assert + Capture Saferpay à partir du seul token de session (sans ligne Payment)."""

from __future__ import annotations

import logging
import os
import time
import uuid
from http import HTTPStatus
from typing import Any

from services.saferpay.assert_response_status import (
    SAFERPAY_FINALIZE_ASSERT_FAILED,
    SAFERPAY_FINALIZE_ASSERT_TRANSIENT,
    SAFERPAY_FINALIZE_CAPTURE_FAILED,
    SAFERPAY_FINALIZE_COMPLETED,
    SAFERPAY_FINALIZE_PAYMENT_FAILED,
    SAFERPAY_FINALIZE_UNEXPECTED_TX_STATUS,
)
from services.saferpay.config import saferpay_spec_version
from services.saferpay.http_client import saferpay_post_json

logger = logging.getLogger(__name__)


def _request_header(customer_id: str) -> dict[str, Any]:
    return {
        "SpecVersion": saferpay_spec_version(),
        "CustomerId": customer_id,
        "RequestId": uuid.uuid4().hex,
        "RetryIndicator": 0,
    }


def _assert_http_is_transient(st_code: int) -> bool:
    return (
        st_code == 0
        or st_code >= HTTPStatus.INTERNAL_SERVER_ERROR
        or st_code
        in (HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.TOO_MANY_REQUESTS)
    )


def _saferpay_400_is_transient_retryable(assert_data: dict[str, Any]) -> bool:
    """Erreurs Assert typiques enchaînées (assert trop tôt) — réessai côté client possible."""
    err = (assert_data.get("ErrorName") or "").strip().upper()
    if err in {
        "TRANSACTION_IN_WRONG_STATE",
        "TRANSACTION_NOT_STARTED",
        "INVALID_ACTION",
    }:
        return True
    msg = (assert_data.get("ErrorMessage") or "").lower()
    deets = assert_data.get("ErrorDetail")
    blob = " ".join(
        [msg, *([str(d) for d in deets] if isinstance(deets, list) else [])]
    ).lower()
    return (
        "not started" in blob
        or "wrong state" in blob
        or "still in progress" in blob
        or "invalid action" in blob
    )


# Délai entre reprises (en parallele des reprises cote app) quand l'autorisation existe deja
# dans l'e-banking mais l'API Assert n'a pas encore le bon etat (NOT_STARTED, WRONG_STATE, ...).
_SAFERPAY_ASSERT_MAX_ATTEMPTS = 6
_SAFERPAY_ASSERT_BACKOFF_MAX_SEC = 3.0


def run_saferpay_paymentpage_assert_capture(session_token: str) -> dict[str, Any]:  # noqa: PLR0911
    """Exécute PaymentPage Assert puis Transaction Capture si AUTHORIZED.

    Ne touche pas à la base. Retourne un dict avec au minimum ``status`` (constantes
    SAFERPAY_FINALIZE_* sauf ``already_completed``) et ``tx_id`` / ``tx_status`` si pertinent.
    """
    token = (session_token or "").strip()
    if not token:
        return {
            "status": SAFERPAY_FINALIZE_ASSERT_FAILED,
            "detail": "Token de session Saferpay vide",
        }

    customer_id = os.environ["SAFERPAY_CUSTOMER_ID"].strip()
    assert_body: dict[str, Any] = {
        "RequestHeader": _request_header(customer_id),
        "Token": token,
    }
    st_code: int = 0
    assert_data: dict[str, Any] | None = None
    raw = ""

    for attempt in range(_SAFERPAY_ASSERT_MAX_ATTEMPTS):
        st_code, assert_data, raw = saferpay_post_json(
            "Payment/v1/PaymentPage/Assert",
            assert_body,
        )
        if st_code == HTTPStatus.OK and assert_data is not None:
            break
        is_trans_400 = (
            st_code == HTTPStatus.BAD_REQUEST
            and isinstance(assert_data, dict)
            and _saferpay_400_is_transient_retryable(assert_data)
        )
        if is_trans_400:
            if attempt < _SAFERPAY_ASSERT_MAX_ATTEMPTS - 1:
                delay = min(0.35 * (2**attempt), _SAFERPAY_ASSERT_BACKOFF_MAX_SEC)
                logger.info(
                    "Saferpay Assert 400 transitoire (tent. %s/%s), attente %.2fs — %s",
                    attempt + 1,
                    _SAFERPAY_ASSERT_MAX_ATTEMPTS,
                    delay,
                    (assert_data or {}).get("ErrorName", ""),
                )
                time.sleep(delay)
                continue
            return {
                "status": SAFERPAY_FINALIZE_ASSERT_TRANSIENT,
                "http_status": st_code,
                "detail": (
                    "Le prestataire n’a pas encore finalisé côté API. "
                    "L’autorisation peut déjà s’afficher dans l’e-banking : "
                    "réessayez dans un instant ou via « Payer en ligne »."
                ),
            }
        if _assert_http_is_transient(st_code):
            logger.warning(
                "Saferpay Assert transitoire (guest) http=%s",
                st_code,
            )
            return {
                "status": SAFERPAY_FINALIZE_ASSERT_TRANSIENT,
                "http_status": st_code,
                "detail": (raw or "")[:500] if raw else None,
            }
        detail_text: str
        if isinstance(assert_data, dict):
            msg = (assert_data.get("ErrorMessage") or "").strip()
            deets = assert_data.get("ErrorDetail")
            if not msg and isinstance(deets, list) and deets:
                first = deets[0]
                msg = str(first)[:300] if first is not None else ""
            detail_text = msg or (raw or "")
        else:
            detail_text = (raw or "") if raw else (str(assert_data) if assert_data is not None else "")
        return {
            "status": SAFERPAY_FINALIZE_ASSERT_FAILED,
            "http_status": st_code,
            "detail": detail_text[:500] if detail_text else None,
        }

    if assert_data is None:
        return {
            "status": SAFERPAY_FINALIZE_ASSERT_FAILED,
            "detail": "Reponse Assert Saferpay vide ou invalide",
        }
    tx = assert_data.get("Transaction") or {}
    tx_id = (tx.get("Id") or "").strip()
    tx_status = (tx.get("Status") or "").strip().upper()

    if tx_status in {"FAILED", "CANCELED", "VOIDED"}:
        return {
            "status": SAFERPAY_FINALIZE_PAYMENT_FAILED,
            "tx_id": tx_id or None,
            "tx_status": tx_status,
        }

    if tx_status == "AUTHORIZED":
        if not tx_id:
            return {
                "status": SAFERPAY_FINALIZE_UNEXPECTED_TX_STATUS,
                "tx_status": tx_status,
                "detail": "Transaction Id manquant",
            }
        cap_body = {
            "RequestHeader": _request_header(customer_id),
            "TransactionReference": {"TransactionId": tx_id},
        }
        cap_st, cap_data, cap_raw = saferpay_post_json(
            "Payment/v1/Transaction/Capture",
            cap_body,
        )
        if cap_st != HTTPStatus.OK or not cap_data:
            logger.error(
                "Saferpay Capture échoué (guest) tx=%s: %s",
                tx_id,
                (cap_raw or "")[:500],
            )
            return {
                "status": SAFERPAY_FINALIZE_CAPTURE_FAILED,
                "tx_id": tx_id,
                "http_status": cap_st,
                "detail": (cap_raw or "")[:500] if cap_raw else None,
            }
        return {"status": SAFERPAY_FINALIZE_COMPLETED, "tx_id": tx_id}

    if tx_status == "CAPTURED":
        return {
            "status": SAFERPAY_FINALIZE_COMPLETED,
            "tx_id": tx_id or None,
        }

    logger.warning(
        "Saferpay Assert statut inattendu (guest) status=%s",
        tx_status,
    )
    return {
        "status": SAFERPAY_FINALIZE_UNEXPECTED_TX_STATUS,
        "tx_id": tx_id or None,
        "tx_status": tx_status,
    }
