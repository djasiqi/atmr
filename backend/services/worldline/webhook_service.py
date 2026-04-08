"""Traitement des webhooks Worldline Connect v1."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Sequence

from ext import db
from models.enums import PaymentStatus
from models.payment import Payment
from models.worldline_webhook_event import WorldlineWebhookEvent

logger = logging.getLogger(__name__)


def _webhook_secret_key_store() -> Any:
    from worldline.connect.sdk.webhooks.in_memory_secret_key_store import (
        InMemorySecretKeyStore,
    )

    store = InMemorySecretKeyStore()
    raw = (os.getenv("WORLDLINE_WEBHOOK_KEYS_JSON") or "").strip()
    if raw:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("WORLDLINE_WEBHOOK_KEYS_JSON doit être un objet JSON")
        for kid, sec in data.items():
            store.store_secret_key(str(kid), str(sec))
    else:
        kid = (os.getenv("WORLDLINE_WEBHOOK_KEY_ID") or "").strip()
        sec = (os.getenv("WORLDLINE_WEBHOOK_SECRET") or "").strip()
        if kid and sec:
            store.store_secret_key(kid, sec)
    return store


def webhook_helper_configured() -> bool:
    raw = (os.getenv("WORLDLINE_WEBHOOK_KEYS_JSON") or "").strip()
    if raw:
        try:
            data = json.loads(raw)
            return isinstance(data, dict) and len(data) > 0
        except json.JSONDecodeError:
            return False
    return bool(
        (os.getenv("WORLDLINE_WEBHOOK_KEY_ID") or "").strip()
        and (os.getenv("WORLDLINE_WEBHOOK_SECRET") or "").strip()
    )


def create_webhooks_helper() -> Any:
    if not webhook_helper_configured():
        msg = (
            "Webhooks Worldline: définir WORLDLINE_WEBHOOK_KEYS_JSON "
            "ou WORLDLINE_WEBHOOK_KEY_ID + WORLDLINE_WEBHOOK_SECRET"
        )
        raise RuntimeError(msg)
    from worldline.connect.sdk.v1.webhooks.v1_webhooks_factory import V1WebhooksFactory

    return V1WebhooksFactory.create_helper(_webhook_secret_key_store())


def _headers_to_sdk(headers: Any) -> Sequence[Any]:
    from worldline.connect.sdk.communication.request_header import RequestHeader

    out: list[RequestHeader] = []
    for name, value in headers:
        if value is None:
            continue
        out.append(RequestHeader(str(name), str(value)))
    return out


def _success_statuses() -> frozenset[str]:
    return frozenset(
        {
            "CAPTURED",
            "PAID",
            "PENDING_CAPTURE",
        }
    )


def _failure_statuses() -> frozenset[str]:
    return frozenset(
        {
            "REJECTED",
            "CANCELLED",
            "REVERSED",
            "REJECTED_CAPTURE",
        }
    )


def process_webhook_request(*, body: bytes, wsgi_headers: Any) -> None:
    """Valide la signature, déduplique par event.id, met à jour Payment si trouvé."""
    helper = create_webhooks_helper()
    hdrs = _headers_to_sdk(wsgi_headers)
    event = helper.unmarshal(body, hdrs)

    eid = event.id
    if not eid:
        logger.warning("Worldline webhook sans id")
        return

    if WorldlineWebhookEvent.query.get(eid):
        return

    row = WorldlineWebhookEvent(event_id=eid, event_type=event.type)
    db.session.add(row)

    pay = event.payment
    if pay is None:
        db.session.commit()
        return

    wl_payment_id = pay.id
    hosted_id = None
    if pay.hosted_checkout_specific_output is not None:
        hosted_id = pay.hosted_checkout_specific_output.hosted_checkout_id

    payment_row: Payment | None = None
    if hosted_id:
        payment_row = Payment.query.filter_by(
            worldline_hosted_checkout_id=hosted_id
        ).first()
    if payment_row is None and wl_payment_id:
        payment_row = Payment.query.filter_by(
            worldline_payment_id=wl_payment_id
        ).first()

    status_wl = (pay.status or "").strip().upper()
    payment_updated = False
    if payment_row is not None:
        if wl_payment_id:
            payment_row.worldline_payment_id = wl_payment_id
        if status_wl in _success_statuses():
            payment_row.status = PaymentStatus.COMPLETED
            payment_updated = True
        elif status_wl in _failure_statuses():
            payment_row.status = PaymentStatus.FAILED
            payment_updated = True

    db.session.commit()

    logger.info(
        "Worldline webhook traité",
        extra={
            "worldline_event_id": eid,
            "worldline_event_type": event.type,
            "worldline_payment_status": status_wl or None,
            "hosted_checkout_id": hosted_id,
            "worldline_payment_id": wl_payment_id,
            "local_payment_id": payment_row.id if payment_row else None,
            "payment_row_updated": payment_updated,
        },
    )
