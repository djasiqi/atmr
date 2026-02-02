# backend/services/notifications/company_push.py
"""Envoi synchrone de push aux entreprises (dispatcher).

Module séparé pour éviter le cycle d'import fanout ↔ notification_tasks.
"""

from __future__ import annotations

from typing import Any, Dict

from ext import app_logger
from services.notifications.push import send_push_message


def send_push_to_company_sync(
    company_id: int,
    title: str,
    body: str,
    data: Dict[str, Any],
    *,
    timeout: int = 5,
) -> bool:
    """Envoie une push notification à une entreprise (dispatcher).

    Args:
        company_id: ID de l'entreprise
        title: Titre de la notification
        body: Corps du message
        data: Données additionnelles (pour deep linking, etc.)
        timeout: Timeout en secondes (défaut: 5)

    Returns:
        True si la push a été envoyée avec succès, False sinon
    """
    success = False
    try:
        from models import Company, User

        company = Company.query.get(company_id)
        if not company:
            app_logger.debug(
                "[event_fanout] Company %s not found, push skipped", company_id
            )
        else:
            company_user = User.query.get(company.user_id)
            if not company_user:
                app_logger.debug(
                    "[event_fanout] Company user %s not found, push skipped",
                    company.user_id,
                )
            else:
                push_token = getattr(company_user, "push_token", None)
                if not push_token:
                    app_logger.warning(
                        "PUSH_COMPANY_TOKEN_MISSING company_id=%s company_user_id_target=%s",
                        company_id,
                        company.user_id,
                    )
                    app_logger.debug(
                        "[event_fanout] Company user %s has no push_token, push skipped",
                        company.user_id,
                    )
                else:
                    # P0.4: Recipient proof logging (DEBUG_NOTIF_ROUTING)
                    from services.notifications.token_audit import (
                        _token_hash,
                        check_token_collision,
                        log_push_recipient_proof,
                    )

                    token_hash = _token_hash(push_token)
                    log_push_recipient_proof(
                        trace_id=data.get("trace_id"),
                        booking_id=data.get("booking_id"),
                        status=data.get("status"),
                        recipient_role="company",
                        recipient_id=company.user_id or company_id,
                        token_count=1,
                        token_hashes=[token_hash],
                        collapse_key=data.get("collapse_key"),
                        dedupe_key=data.get("dedupe_key"),
                        routing_version=data.get("routing_version"),
                        routing_decision=data.get("routing_decision"),
                        source=data.get("source"),
                        actor_role=data.get("actor_role"),
                        actor_id=data.get("actor_id"),
                    )
                    # P0.4: Détection collision token (driver vs company)
                    driver_id_from_booking = data.get("driver_id")
                    if driver_id_from_booking:
                        from models import DeviceToken

                        driver_tokens = [
                            dt.token
                            for dt in DeviceToken.query.filter_by(
                                driver_id=int(driver_id_from_booking),
                                is_active=True,
                            ).all()
                            if dt.token
                        ]
                        check_token_collision(
                            driver_tokens=driver_tokens,
                            company_token=push_token,
                            driver_id=int(driver_id_from_booking),
                            company_user_id=company.user_id,
                            trace_id=data.get("trace_id"),
                        )

                    result = send_push_message(
                        token=push_token,
                        title=title,
                        body=body,
                        data=data,
                        timeout=timeout,
                    )

                    if result.get("ok"):
                        app_logger.info(
                            "[event_fanout] Push sent to company %s", company_id
                        )
                        success = True
                    else:
                        app_logger.warning(
                            "[event_fanout] Push failed for company %s: %s",
                            company_id,
                            result.get("error", "Unknown error"),
                        )
    except (ValueError, TypeError, AttributeError) as e:
        app_logger.error(
            "[event_fanout] Push failed (validation error: %s): %s",
            type(e).__name__,
            e,
        )
    except (ConnectionError, OSError, TimeoutError) as e:
        app_logger.error(
            "[event_fanout] Push failed (network error: %s): %s",
            type(e).__name__,
            e,
        )
        raise  # P1: Propager pour que Celery task puisse retry
    except Exception:
        app_logger.exception("[event_fanout] Push failed")

    return success
