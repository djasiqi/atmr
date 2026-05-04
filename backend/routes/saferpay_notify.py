"""Notifications HTTP GET Saferpay (sans JWT — corrélation paymentId + clé opaque)."""

from __future__ import annotations

import logging

from flask import Blueprint, Response

from services.guest_saferpay import try_finalize_guest_saferpay_notify
from services.saferpay.payment_page import try_finalize_saferpay_by_payment_id

logger = logging.getLogger(__name__)

saferpay_notify_bp = Blueprint("saferpay_notify", __name__)


def _payment_id_arg() -> int | None:
    from flask import request

    raw = (
        request.args.get("paymentId") or request.args.get("payment_id") or ""
    ).strip()
    if not raw.isdigit():
        return None
    return int(raw)


@saferpay_notify_bp.route("/payments/saferpay/notify/success", methods=["GET"])
def saferpay_notify_success() -> Response | tuple[str, int]:
    from flask import request

    pid = _payment_id_arg()
    if pid is None:
        return "", 400
    key = (request.args.get("k") or "").strip()
    out = try_finalize_saferpay_by_payment_id(payment_id=pid, notify_key=key or None)
    if not out.get("ok"):
        logger.info("Saferpay notify success ignoré: %s", out)
    return "", 200


def _guest_booking_id_arg() -> str | None:
    from flask import request

    raw = (
        request.args.get("guestBookingId") or request.args.get("guest_booking_id") or ""
    ).strip()
    return raw or None


def _redis_guest_cache_ops():
    from ext import redis_client
    from routes.auth import _public_cache_get, _public_cache_setex

    def redis_get(key: str):
        if redis_client:
            raw = redis_client.get(key)
            if isinstance(raw, bytes):
                return raw.decode("utf-8")
            return raw
        return _public_cache_get(key)

    def redis_setex(key: str, ttl: int, value: str):
        if redis_client:
            redis_client.setex(key, ttl, value)
            return
        _public_cache_setex(key, ttl, value)

    return redis_get, redis_setex


@saferpay_notify_bp.route("/payments/saferpay/notify/guest-success", methods=["GET"])
def saferpay_notify_guest_success() -> Response | tuple[str, int]:
    from flask import request

    gid = _guest_booking_id_arg()
    if not gid:
        return "", 400
    key = (request.args.get("k") or "").strip()
    redis_get, redis_setex = _redis_guest_cache_ops()
    out = try_finalize_guest_saferpay_notify(
        guest_booking_id=gid,
        notify_key=key or None,
        redis_get=redis_get,
        redis_setex=redis_setex,
    )
    if not out.get("ok"):
        logger.info(
            "Saferpay guest notify success ignoré",
            extra={
                "guest_booking_id": gid,
                "notify_result": out,
                "outcome": "guest_notify_ignored",
            },
        )
    return "", 200


@saferpay_notify_bp.route("/payments/saferpay/notify/guest-fail", methods=["GET"])
def saferpay_notify_guest_fail() -> Response | tuple[str, int]:
    from flask import request

    gid = _guest_booking_id_arg()
    if not gid:
        return "", 400
    key = (request.args.get("k") or "").strip()
    redis_get, redis_setex = _redis_guest_cache_ops()
    out = try_finalize_guest_saferpay_notify(
        guest_booking_id=gid,
        notify_key=key or None,
        redis_get=redis_get,
        redis_setex=redis_setex,
    )
    if not out.get("ok"):
        logger.info(
            "Saferpay guest notify fail ignoré",
            extra={
                "guest_booking_id": gid,
                "notify_result": out,
                "outcome": "guest_notify_ignored",
            },
        )
    return "", 200


@saferpay_notify_bp.route("/payments/saferpay/notify/fail", methods=["GET"])
def saferpay_notify_fail() -> Response | tuple[str, int]:
    """Saferpay n'envoie pas le detail ; on tente un Assert pour statuer (echec ou deja paye)."""
    from flask import request

    pid = _payment_id_arg()
    if pid is None:
        return "", 400
    key = (request.args.get("k") or "").strip()
    out = try_finalize_saferpay_by_payment_id(payment_id=pid, notify_key=key or None)
    if not out.get("ok"):
        logger.info("Saferpay notify fail ignoré: %s", out)
    return "", 200
