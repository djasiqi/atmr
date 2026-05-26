# backend/services/notifications/company_push.py
"""Envoi synchrone de push aux entreprises (multi-appareils DeviceToken)."""

from __future__ import annotations

import uuid
from typing import Any, Dict

from ext import app_logger, db
from services.notifications.push import send_push_message


def _ensure_event_ids(data: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(data)
    if not out.get("event_id"):
        out["event_id"] = str(uuid.uuid4())
    if not out.get("dedupe_key"):
        out["dedupe_key"] = f"company:{out.get('event_id')}"
    out.setdefault("recipient_role", "company")
    return out


def send_push_to_company_sync(
    company_id: int,
    title: str,
    body: str,
    data: Dict[str, Any],
    *,
    timeout: int = 5,
) -> bool:
    """Envoie une push à tous les DeviceToken actifs de l'entreprise."""
    success = False
    try:
        data = _ensure_event_ids(data)

        actor_role = data.get("actor_role")
        actor_id = data.get("actor_id")
        if (
            actor_role == "company"
            and actor_id is not None
            and str(actor_id) == str(company_id)
        ):
            app_logger.info(
                "[company_push] GUARD: self-notification blocked (company %s is actor)",
                company_id,
            )
            return True

        from models import Company, DeviceToken

        company = Company.query.get(company_id)
        if not company:
            app_logger.debug(
                "[company_push] Company %s not found, push skipped", company_id
            )
            return False

        device_tokens = (
            DeviceToken.query.filter_by(company_id=company_id, is_active=True)
            .order_by(DeviceToken.updated_at.desc())
            .all()
        )

        if not device_tokens:
            app_logger.warning(
                "PUSH_COMPANY_TOKEN_MISSING company_id=%s company_user_id_target=%s",
                company_id,
                company.user_id,
            )
            # Legacy fallback (1 release)
            from models import User

            company_user = User.query.get(company.user_id)
            legacy_token = getattr(company_user, "push_token", None) if company_user else None
            if legacy_token:
                result = send_push_message(
                    token=legacy_token,
                    title=title,
                    body=body,
                    data=data,
                    timeout=timeout,
                )
                return bool(result.get("ok"))
            return False

        from services.notifications.token_audit import (
            _token_hash,
            log_push_recipient_proof,
        )

        token_hashes = [_token_hash(dt.token) for dt in device_tokens if dt.token]
        log_push_recipient_proof(
            trace_id=data.get("trace_id"),
            booking_id=data.get("booking_id"),
            status=data.get("status"),
            recipient_role="company",
            recipient_id=company.user_id or company_id,
            token_count=len(device_tokens),
            token_hashes=token_hashes,
            collapse_key=data.get("collapse_key"),
            dedupe_key=data.get("dedupe_key"),
            routing_version=data.get("routing_version"),
            routing_decision=data.get("routing_decision"),
            source=data.get("source"),
            actor_role=data.get("actor_role"),
            actor_id=data.get("actor_id"),
        )

        driver_id_from_booking = data.get("driver_id")
        if driver_id_from_booking and device_tokens:
            from services.notifications.token_audit import check_token_collision

            driver_tokens = [
                dt.token
                for dt in DeviceToken.query.filter_by(
                    driver_id=int(driver_id_from_booking),
                    is_active=True,
                ).all()
                if dt.token
            ]
            first_company_token = device_tokens[0].token
            if first_company_token:
                check_token_collision(
                    driver_tokens=driver_tokens,
                    company_token=first_company_token,
                    driver_id=int(driver_id_from_booking),
                    company_user_id=company.user_id,
                    trace_id=data.get("trace_id"),
                )

        success_count = 0
        for device_token in device_tokens:
            result = send_push_message(
                token=device_token.token,
                title=title,
                body=body,
                data={**data, "company_id": company_id},
                timeout=timeout,
                provider=getattr(device_token, "provider", None),
                platform=getattr(device_token, "platform", None),
                device_token_id=device_token.id,
            )
            if result.get("ok"):
                success_count += 1
                app_logger.debug(
                    "[company_push] Push sent to company %s (device %s)",
                    company_id,
                    device_token.id,
                )
            else:
                app_logger.warning(
                    "[company_push] Push failed for company %s (device %s): %s",
                    company_id,
                    device_token.id,
                    result.get("error", "Unknown error"),
                )

        try:
            db.session.commit()
        except Exception:
            app_logger.exception("[company_push] commit after push lifecycle")

        success = success_count > 0
        if success:
            app_logger.info(
                "[company_push] Push sent to company %s (%d/%d devices)",
                company_id,
                success_count,
                len(device_tokens),
            )
    except (ValueError, TypeError, AttributeError) as e:
        app_logger.error(
            "[company_push] Push failed (validation error: %s): %s",
            type(e).__name__,
            e,
        )
    except (ConnectionError, OSError, TimeoutError):
        raise
    except Exception:
        app_logger.exception("[company_push] Push failed")

    return success
