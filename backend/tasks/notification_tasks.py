# backend/tasks/notification_tasks.py
"""Tâches Celery pour les notifications push avec fallback SMS/Email.

Features:
- Queue persistante Redis via Celery
- Retry automatique avec exponential backoff
- Fallback SMS après échecs répétés
- Fallback Email en dernier recours
- Dead Letter Queue pour échecs définitifs
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import suppress
from typing import Any, ClassVar, Dict

from celery import Task
from typing_extensions import override

from celery_app import celery

logger = logging.getLogger(__name__)

# Configuration fallback
MAX_PUSH_RETRIES = 3  # Nombre de tentatives push avant fallback SMS
MAX_SMS_RETRIES = 2  # Nombre de tentatives SMS avant fallback Email
RETRY_BACKOFF_BASE = 60  # Délai de base en secondes (1 min, 2 min, 4 min, ...)


class NotificationTask(Task):
    """Task Celery personnalisée pour les notifications avec gestion d'erreurs avancée."""

    autoretry_for: ClassVar[tuple[type[Exception], ...]] = (
        ConnectionError,
        TimeoutError,
    )
    retry_kwargs: ClassVar[dict[str, int]] = {"max_retries": MAX_PUSH_RETRIES}
    retry_backoff: ClassVar[bool] = True
    retry_backoff_max: ClassVar[int] = 600  # 10 minutes max
    retry_jitter: ClassVar[bool] = True

    @override
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Appelé quand la task échoue après tous les retries.

        Envoie la notification en DLQ pour traitement manuel.
        """
        logger.error(
            "[notification_task] Task %s failed after retries: %s",
            task_id,
            exc,
            exc_info=einfo,
        )

        # Enregistrer l'échec définitif pour monitoring
        driver_id = kwargs.get("driver_id")
        notification_type = kwargs.get("notification_type", "unknown")

        if driver_id:
            logger.critical(
                "[notification_task] Échec définitif pour driver %s (type: %s) - notification perdue ou nécessite intervention manuelle",
                driver_id,
                notification_type,
            )


@celery.task(
    name="tasks.notification_tasks.send_push_notification",
    bind=True,
    base=NotificationTask,
    acks_late=True,
    task_time_limit=20,
    task_soft_time_limit=10,
    max_retries=MAX_PUSH_RETRIES,
)
def send_push_notification_task(
    self,
    driver_id: int,
    title: str,
    body: str,
    data: Dict[str, Any] | None = None,
    *,
    notification_type: str = "unknown",
    bypass_rate_limit: bool = False,
    fallback_to_sms: bool = True,
    fallback_to_email: bool = True,
) -> Dict[str, Any]:
    """Envoie une notification push via Celery avec fallback SMS/Email.

    Args:
        self: Task instance (bind=True)
        driver_id: ID du driver
        title: Titre de la notification
        body: Corps de la notification
        data: Données additionnelles pour deep linking
        notification_type: Type de notification (booking, message, etc.)
        bypass_rate_limit: Si True, contourne le rate limiting
        fallback_to_sms: Si True, fallback SMS après échecs push
        fallback_to_email: Si True, fallback Email après échecs SMS

    Returns:
        Dict avec status de l'envoi
    """
    from celery_app import get_flask_app
    from ext import db
    from models import Driver
    from services.notifications.push import send_push_message

    # ✅ CRITIQUE: Créer un contexte d'application Flask pour utiliser SQLAlchemy
    app = get_flask_app()
    with app.app_context():
        try:
            from services.notifications.push_pipeline_log import log_driver_push_stage

            data = dict(data or {})
            push_dispatch_id = str(data.get("push_dispatch_id") or uuid.uuid4())
            data["push_dispatch_id"] = push_dispatch_id
            _booking_id = data.get("booking_id")
            log_driver_push_stage(
                "driver_push.task_start",
                event_id=(data or {}).get("event_id"),
                correlation_id=(data or {}).get("correlation_id"),
                booking_id=_booking_id,
                driver_id=driver_id,
                notification_type=notification_type,
                celery_task_id=getattr(self.request, "id", None),
            )

            logger.warning(
                "[notification_task] send_push_notification_task started: driver_id=%s notification_type=%s",
                driver_id,
                notification_type,
            )

            if (
                os.getenv("NOTIFICATIONS_KAFKA_PUBLISH_ENABLED", "false").lower()
                == "true"
            ):
                from services.monitoring.prometheus import (
                    inc_notification_kafka_enqueue,
                    observe_notification_kafka_enqueue_latency,
                )
                from services.notifications.kafka_producer import send_push_via_kafka

                publish_ok = False
                fail_reason = "returned_false"
                t0 = time.perf_counter()
                try:
                    publish_ok = bool(
                        send_push_via_kafka(
                            driver_id,
                            title,
                            body,
                            data=data,
                            notification_type=notification_type,
                            bypass_rate_limit=bypass_rate_limit,
                        )
                    )
                except Exception as enqueue_exc:
                    fail_reason = "exception"
                    _elapsed = time.perf_counter() - t0
                    with suppress(Exception):
                        observe_notification_kafka_enqueue_latency(
                            status="exception", seconds=_elapsed
                        )
                    with suppress(Exception):
                        inc_notification_kafka_enqueue(status="exception")
                    logger.warning(
                        "notification_kafka_enqueue_failed fallback=direct reason=%s err=%s",
                        fail_reason,
                        enqueue_exc,
                    )
                else:
                    _elapsed = time.perf_counter() - t0
                    with suppress(Exception):
                        observe_notification_kafka_enqueue_latency(
                            status="success" if publish_ok else "fallback",
                            seconds=_elapsed,
                        )

                if publish_ok:
                    with suppress(Exception):
                        inc_notification_kafka_enqueue(status="success")
                    return {
                        "ok": True,
                        "channel": "kafka",
                        "notification_type": notification_type,
                    }

                if fail_reason != "exception":
                    logger.warning(
                        "notification_kafka_enqueue_failed fallback=direct reason=%s",
                        fail_reason,
                    )
                    with suppress(Exception):
                        inc_notification_kafka_enqueue(status="fallback")

            # Récupérer le driver
            driver = db.session.get(Driver, driver_id)
            if not driver:
                logger.error("[notification_task] Driver %s not found", driver_id)
                return {"ok": False, "error": "Driver not found", "channel": "none"}

            # ✅ CORRECTIF #3: Utiliser DeviceToken pour support multi-device
            from models import DeviceToken

            device_tokens_raw = DeviceToken.query.filter_by(
                driver_id=driver_id,
                is_active=True,
            ).all()

            # ✅ FIX: Extraire les données AVANT de fermer la session DB
            # pour éviter les connexions stale pendant les opérations longues (push)
            #
            # P1 stabilisation notifs : dédup par token + priorité FCM Android par device_id.
            from services.notifications.push_device_selection import (
                prepare_driver_push_targets,
            )

            device_tokens_data = prepare_driver_push_targets(
                device_tokens_raw,
                driver_id=driver_id,
            )
            if len(device_tokens_data) < len(device_tokens_raw):
                logger.info(
                    "[notification_task] push targets driver=%s raw=%s selected=%s",
                    driver_id,
                    len(device_tokens_raw),
                    len(device_tokens_data),
                )

            # ✅ FIX: Fermer la session DB AVANT les opérations longues (push avec retries)
            # Cela évite les erreurs "server closed the connection unexpectedly"
            import contextlib

            with contextlib.suppress(Exception):
                db.session.commit()  # Commit pour s'assurer que tout est persisté

            if not device_tokens_data:
                logger.warning(
                    "[notification_task] No active push tokens for driver %s, using fallback",
                    driver_id,
                )
            else:
                logger.warning(
                    "[notification_task] push_dispatch_id=%s booking_id=%s driver_id=%s "
                    "selected_devices=%s attempt=%d/%d",
                    push_dispatch_id,
                    _booking_id,
                    driver_id,
                    len(device_tokens_data),
                    self.request.retries + 1,
                    MAX_PUSH_RETRIES,
                )

                # P0.4: Recipient proof logging (DEBUG_NOTIF_ROUTING)
                try:
                    from services.notifications.token_audit import (
                        _token_hash,
                        log_push_recipient_proof,
                    )

                    token_hashes = [
                        _token_hash(dt["token"]) for dt in device_tokens_data
                    ]
                    log_push_recipient_proof(
                        trace_id=data.get("trace_id") if data else None,
                        booking_id=data.get("booking_id") if data else None,
                        status=data.get("status") if data else None,
                        recipient_role="driver",
                        recipient_id=driver_id,
                        token_count=len(device_tokens_data),
                        token_hashes=token_hashes,
                        collapse_key=data.get("collapse_key") if data else None,
                        dedupe_key=data.get("dedupe_key") if data else None,
                        routing_version=data.get("routing_version") if data else None,
                        routing_decision=data.get("routing_decision") if data else None,
                        source=data.get("source") if data else None,
                        actor_role=data.get("actor_role") if data else None,
                        actor_id=data.get("actor_id") if data else None,
                    )
                except Exception:
                    pass

                # Envoyer à tous les devices actifs
                # ✅ FIX: Utiliser les données extraites (pas les objets ORM)
                from services.notifications.device_token_lifecycle import (
                    is_push_device_token_lifecycle_enabled,
                )

                success_count = 0
                token_invalid_count = 0
                invalid_token_ids_flag_off: list[int] = []
                last_result: Dict[str, Any] | None = None
                for device_token in device_tokens_data:
                    result = send_push_message(
                        token=device_token["token"],
                        title=title,
                        body=body,
                        data=data,
                        driver_id=driver_id,
                        bypass_rate_limit=bypass_rate_limit,
                        provider=device_token.get("provider"),
                        platform=device_token.get("platform"),
                        device_token_id=device_token["id"],
                    )
                    last_result = (
                        result  # Garder le dernier résultat pour logging/retry
                    )

                    if result.get("ok"):
                        success_count += 1
                        logger.info(
                            "push_sent provider=%s device_token_id=%s push_dispatch_id=%s "
                            "booking_id=%s driver_id=%s",
                            device_token.get("provider") or "unknown",
                            device_token["id"],
                            push_dispatch_id,
                            _booking_id,
                            driver_id,
                        )
                    else:
                        error = result.get("error", "Unknown error")
                        logger.warning(
                            "[notification_task] Push failed for driver %s (device %s): %s",
                            driver_id,
                            device_token["id"],
                            error,
                        )

                        if result.get("token_invalid"):
                            token_invalid_count += 1
                            if not is_push_device_token_lifecycle_enabled():
                                invalid_token_ids_flag_off.append(device_token["id"])

                if invalid_token_ids_flag_off:
                    try:
                        DeviceToken.query.filter(
                            DeviceToken.id.in_(invalid_token_ids_flag_off)
                        ).update({"is_active": False}, synchronize_session=False)
                    except Exception as e:
                        logger.warning(
                            "[notification_task] bulk invalidate (flag off): %s",
                            str(e)[:200],
                        )

                try:
                    db.session.commit()
                except Exception as e:
                    logger.warning(
                        "[notification_task] commit after lifecycle: %s",
                        str(e)[:200],
                    )

                logger.warning(
                    "push_sent_summary push_dispatch_id=%s push_sent_count=%s "
                    "devices_total=%s driver_id=%s booking_id=%s",
                    push_dispatch_id,
                    success_count,
                    len(device_tokens_data),
                    driver_id,
                    _booking_id,
                )

                # Si au moins un envoi a réussi, considérer comme succès
                if success_count > 0:
                    logger.warning(
                        "[notification_task] Push sent successfully to driver %s (%d/%d devices)",
                        driver_id,
                        success_count,
                        len(device_tokens_data),
                    )
                    result_payload = {
                        "ok": True,
                        "channel": "push",
                        "attempts": self.request.retries + 1,
                        "devices_sent": success_count,
                        "devices_total": len(device_tokens_data),
                    }
                    log_driver_push_stage(
                        "driver_push.task_done",
                        event_id=(data or {}).get("event_id"),
                        correlation_id=(data or {}).get("correlation_id"),
                        booking_id=_booking_id,
                        driver_id=driver_id,
                        notification_type=notification_type,
                        celery_task_id=getattr(self.request, "id", None),
                        ok=True,
                        devices_sent=success_count,
                    )
                    return result_payload

                # Tous les envois ont échoué
                # Si tous les tokens sont invalides, passer directement au fallback
                if token_invalid_count == len(device_tokens_data):
                    logger.warning(
                        "[notification_task] Tous les tokens invalides pour driver %s, skip retry",
                        driver_id,
                    )
                    # Passer directement au fallback sans retry
                    raise self.retry(countdown=0, max_retries=0)

                # Au moins un token valide mais échec réseau → retry
                # Utiliser le dernier résultat pour déterminer le type d'erreur
                # Note: last_result est toujours défini car device_tokens_data n'est pas vide et la boucle s'est exécutée
                if last_result:
                    error = last_result.get("error", "Unknown error")

                    # Si circuit breaker ouvert, attendre plus longtemps avant retry
                    if last_result.get("circuit_breaker_open"):
                        logger.warning(
                            "[notification_task] Circuit breaker open, retry in 2 minutes"
                        )
                        raise self.retry(countdown=120, exc=ConnectionError(error))

                    # Retry avec backoff exponentiel
                    logger.warning(
                        "[notification_task] Push failed for driver %s: %s (retry in %ds)",
                        driver_id,
                        error,
                        RETRY_BACKOFF_BASE * (2**self.request.retries),
                    )
                    raise self.retry(
                        exc=ConnectionError(error),
                        countdown=RETRY_BACKOFF_BASE * (2**self.request.retries),
                    )

                # Cas de secours (ne devrait jamais arriver)
                error = "Push failed for all devices"
                logger.warning(
                    "[notification_task] Push failed for driver %s: %s (retry in %ds)",
                    driver_id,
                    error,
                    RETRY_BACKOFF_BASE * (2**self.request.retries),
                )
                raise self.retry(
                    exc=ConnectionError(error),
                    countdown=RETRY_BACKOFF_BASE * (2**self.request.retries),
                )

            # Pas de push token → passer directement au fallback
            logger.warning(
                "[notification_task] No push token for driver %s, using fallback",
                driver_id,
            )
            if fallback_to_sms:
                return _send_sms_fallback(driver, title, body, notification_type)

            if fallback_to_email:
                return _send_email_fallback(driver, title, body, notification_type)

            return {
                "ok": False,
                "error": "No push token and fallback disabled",
                "channel": "none",
            }

        except self.MaxRetriesExceededError:
            # Tous les retries push épuisés → Fallback SMS/Email
            logger.warning(
                "[notification_task] Max push retries exceeded for driver %s, using fallback",
                driver_id,
            )

            if fallback_to_sms:
                try:
                    driver = db.session.get(Driver, driver_id)
                    if driver:
                        return _send_sms_fallback(
                            driver, title, body, notification_type
                        )
                except Exception as e:
                    logger.exception("[notification_task] SMS fallback failed: %s", e)

            if fallback_to_email:
                try:
                    driver = db.session.get(Driver, driver_id)
                    if driver:
                        return _send_email_fallback(
                            driver, title, body, notification_type
                        )
                except Exception as e:
                    logger.exception("[notification_task] Email fallback failed: %s", e)

            # Tous les fallbacks ont échoué
            return {
                "ok": False,
                "error": "All notification channels failed",
                "channel": "none",
                "attempts": self.request.retries + 1,
            }
        except Exception as e:
            # Erreur inattendue
            logger.exception(
                "[notification_task] Unexpected error in send_push_notification_task: %s",
                e,
            )
            return {
                "ok": False,
                "error": str(e),
                "channel": "none",
            }


@celery.task(
    name="tasks.notification_tasks.send_push_company_notification",
    bind=True,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_kwargs={"max_retries": MAX_PUSH_RETRIES, "countdown": 5},
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    acks_late=True,
    task_time_limit=20,
    task_soft_time_limit=10,
)
def send_push_company_notification_task(
    _self,
    company_id: int,
    title: str,
    body: str,
    data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Envoie une notification push à l'entreprise via Celery (P1: async comme driver).

    Args:
        company_id: ID de l'entreprise
        title: Titre de la notification
        body: Corps du message
        data: Données additionnelles (recipient_role, trace_id, etc.)

    Returns:
        Dict avec ok (bool) et error (str) si échec
    """
    import os

    from celery_app import get_flask_app

    if os.environ.get("DEBUG_NOTIF_ROUTING", "").lower() in ("1", "true", "yes"):
        logger.info(
            "[notification_task] push_company_task_started company_id=%s trace_id=%s routing_decision=%s",
            company_id,
            (data or {}).get("trace_id"),
            (data or {}).get("routing_decision"),
        )

    # Guard centralise : bloquer self-notification (company est l'acteur)
    actor_role = data.get("actor_role") if data else None
    actor_id = data.get("actor_id") if data else None
    if (
        actor_role == "company"
        and actor_id is not None
        and str(actor_id) == str(company_id)
    ):
        logger.info(
            "[notification_task] GUARD: self-notification blocked (company %s is actor)",
            company_id,
        )
        return {"ok": True, "skipped": "self-notification blocked"}

    app = get_flask_app()
    with app.app_context():
        try:
            from services.notifications.company_push import (
                send_push_to_company_sync,
            )

            success = send_push_to_company_sync(
                company_id=company_id,
                title=title,
                body=body,
                data=data or {},
            )
            return {"ok": success, "company_id": company_id}
        except (ConnectionError, TimeoutError):
            raise
        except Exception as e:
            logger.exception(
                "[notification_task] send_push_company_notification_task failed: %s",
                e,
            )
            if (data or {}).get("type") == "new_request":
                try:
                    from services.metrics.institution_metrics import (
                        track_company_push_new_request_delivery_failed,
                    )

                    track_company_push_new_request_delivery_failed(
                        company_id=company_id,
                        reason="task_error",
                    )
                except Exception:
                    pass
            return {"ok": False, "error": str(e), "company_id": company_id}


@celery.task(
    name="tasks.notification_tasks.send_activation_email",
    bind=True,
    acks_late=True,
    # soft 20s > BREVO_HTTP_TIMEOUT_SECONDS (8s) pour permettre EmailRetryableError
    # avant SoftTimeLimitExceeded. max_retries=2 ⇒ 3 tentatives au total.
    task_time_limit=30,
    task_soft_time_limit=20,
    max_retries=2,
    default_retry_delay=2,
)
def send_activation_email_task(
    self,
    *,
    activation_session_id: str,
    email_delivery_id: str,
) -> Dict[str, Any]:
    """Envoie l'email d'activation (args non sensibles : session + delivery_id).

    max_retries=2 ⇒ 3 tentatives au total. Le jeton Redis est stable par delivery_id.
    """
    import random

    from celery_app import get_flask_app

    app = get_flask_app()
    with app.app_context():
        from datetime import UTC, datetime

        from flask import render_template

        from ext import db
        from models import ActivationSession, User
        from services.notifications.activation_email_delivery import (
            cas_claim_sending,
            mark_delivery_failed,
            mark_delivery_sent,
            resolve_activation_token_for_delivery,
            sanitize_email_error,
        )
        from services.notifications.email import send_email_notification
        from services.notifications.email_errors import (
            EmailPermanentError,
            EmailRetryableError,
        )

        def _build_link(token: str) -> str:
            environment = str(app.config.get("ENVIRONMENT", "")).strip().lower()
            default_frontend_url = (
                "http://localhost:3000"
                if environment in {"development", "testing"}
                else "https://www.lirie.ch"
            )
            import os

            frontend_url = (
                os.getenv("FRONTEND_URL")
                or os.getenv("PUBLIC_FRONTEND_URL")
                or os.getenv("PUBLIC_APP_URL")
                or default_frontend_url
            ).rstrip("/")
            return f"{frontend_url}/activate-account?token={token}"

        session = ActivationSession.query.filter_by(
            activation_session_id=activation_session_id
        ).first()
        if not session:
            logger.warning(
                "[notification_task] activation session introuvable sid=%s",
                activation_session_id,
            )
            return {"ok": False, "error": "session_not_found"}

        claim = cas_claim_sending(session, email_delivery_id)
        if claim == "ignore":
            logger.info(
                "[notification_task] ignore stale/duplicate delivery_id=%s sid=%s",
                email_delivery_id,
                activation_session_id,
            )
            return {"ok": True, "ignored": True}

        token = resolve_activation_token_for_delivery(email_delivery_id)
        if not token:
            mark_delivery_failed(
                session,
                "activation_token_derive_failed",
                email_delivery_id=email_delivery_id,
            )
            db.session.commit()
            return {"ok": False, "error": "token_missing"}

        user = User.query.get(session.user_id)
        if not user or not user.email:
            mark_delivery_failed(
                session, "user_or_email_missing", email_delivery_id=email_delivery_id
            )
            db.session.commit()
            return {"ok": False, "error": "user_missing"}

        verification_link = _build_link(token)
        from services.notifications.lirie_email_brand import (
            build_lirie_logo_email_assets,
        )

        logo_src, logo_attachments = build_lirie_logo_email_assets()
        html_body = ""
        try:
            html_body = render_template(
                "emails/activation_email.html",
                activation_link=verification_link,
                logo_src=logo_src,
                first_name=(user.first_name or "").strip() or None,
                product_name="LIRIE",
                company_name="LIRIE",
                current_year=datetime.now(UTC).year,
            )
        except Exception:
            html_body = ""

        text_body = (
            "Bienvenue sur LIRIE.\n\n"
            "Cliquez sur ce lien pour confirmer votre email:\n"
            f"{verification_link}\n\n"
            "Ce lien expire rapidement. Si vous n'etes pas a l'origine de cette action, "
            "ignorez cet email."
        )

        import os

        from_email = os.getenv("ACTIVATION_EMAIL_FROM", "noreply@lirie.ch")
        from_name = os.getenv("ACTIVATION_EMAIL_FROM_NAME", "LIRIE")
        reply_to = os.getenv("ACTIVATION_EMAIL_REPLY_TO", "support@lirie.ch")

        try:
            send_result = send_email_notification(
                email=str(user.email).strip(),
                subject="Activation de votre compte LIRIE",
                body=html_body or text_body,
                html=bool(html_body),
                notification_type="activation_signup",
                from_email=from_email,
                from_name=from_name,
                reply_to=reply_to,
                raise_on_error=True,
                # Lot 1 : corrélation webhook Brevo (X-Mailin-custom)
                headers={"X-Mailin-custom": email_delivery_id},
                attachments=logo_attachments or None,
            )
        except EmailRetryableError as e:
            logger.warning(
                "[notification_task] activation retryable delivery_id=%s: %s",
                email_delivery_id,
                sanitize_email_error(str(e)),
            )
            # Backoff exponentiel + jitter ; même jeton HMAC / delivery_id.
            if self.request.retries >= self.max_retries:
                mark_delivery_failed(
                    session, str(e), email_delivery_id=email_delivery_id
                )
                db.session.commit()
                return {
                    "ok": False,
                    "error": "max_retries",
                    "delivery_id": email_delivery_id,
                }
            countdown = int((2**self.request.retries) + random.uniform(0, 1))
            raise self.retry(exc=e, countdown=countdown) from e
        except EmailPermanentError as e:
            logger.warning(
                "[notification_task] activation permanent fail delivery_id=%s: %s",
                email_delivery_id,
                sanitize_email_error(str(e)),
            )
            mark_delivery_failed(session, str(e), email_delivery_id=email_delivery_id)
            db.session.commit()
            return {"ok": False, "error": "permanent", "delivery_id": email_delivery_id}

        mark_delivery_sent(
            session,
            email_delivery_id=email_delivery_id,
            message_id=str(send_result.get("message_id") or "") or None,
        )
        # finalize_after_provider_accepted commit déjà (F-03)
        logger.info(
            "[notification_task] Activation email sent sid=%s delivery_id=%s task_id=%s",
            activation_session_id,
            email_delivery_id,
            getattr(self.request, "id", None),
        )
        return {
            "ok": True,
            "channel": "email",
            "activation_session_id": activation_session_id,
            "email_delivery_id": email_delivery_id,
        }


def _send_sms_fallback(
    driver, title: str, body: str, notification_type: str
) -> Dict[str, Any]:
    """Envoie une notification SMS en fallback.

    Args:
        driver: Instance Driver
        title: Titre de la notification
        body: Corps de la notification
        notification_type: Type de notification

    Returns:
        Dict avec status de l'envoi
    """
    try:
        from services.notifications.sms import send_sms_notification

        # Récupérer le numéro de téléphone du driver
        phone = None
        if hasattr(driver, "user") and driver.user:
            phone = getattr(driver.user, "phone", None)

        if not phone:
            logger.warning(
                "[notification_task] No phone number for driver %s, skip SMS",
                driver.id,
            )
            return {"ok": False, "error": "No phone number", "channel": "sms"}

        # Envoyer SMS
        logger.info(
            "[notification_task] Sending SMS fallback to driver %s",
            driver.id,
        )

        # Format SMS: [ATMR] Titre - Corps
        sms_content = f"[ATMR] {title}: {body[:100]}"  # Limiter à 160 caractères

        result = send_sms_notification(
            phone=phone,
            message=sms_content,
            notification_type=notification_type,
        )

        if result.get("ok"):
            logger.info(
                "[notification_task] SMS sent successfully to driver %s",
                driver.id,
            )
            return {"ok": True, "channel": "sms"}

        # Canal SMS désactivé par configuration : état attendu, pas une panne.
        # On évite de polluer Sentry (niveau error) avec un simple cas de config.
        if result.get("disabled"):
            logger.info(
                "[notification_task] SMS non envoyé à driver %s (canal désactivé)",
                driver.id,
            )
            return {
                "ok": False,
                "error": result.get("error"),
                "channel": "sms",
                "skipped": True,
            }

        logger.error(
            "[notification_task] SMS failed for driver %s: %s",
            driver.id,
            result.get("error"),
        )
        return {"ok": False, "error": result.get("error"), "channel": "sms"}

    except Exception as e:
        logger.exception("[notification_task] SMS fallback exception: %s", e)
        return {"ok": False, "error": str(e), "channel": "sms"}


def _send_email_fallback(
    driver, title: str, body: str, notification_type: str
) -> Dict[str, Any]:
    """Envoie une notification Email en dernier fallback.

    Args:
        driver: Instance Driver
        title: Titre de la notification
        body: Corps de la notification
        notification_type: Type de notification

    Returns:
        Dict avec status de l'envoi
    """
    try:
        from services.notifications.email import send_email_notification

        # Récupérer l'email du driver
        email = None
        if hasattr(driver, "user") and driver.user:
            email = getattr(driver.user, "email", None)

        if not email:
            logger.warning(
                "[notification_task] No email for driver %s, skip Email",
                driver.id,
            )
            return {"ok": False, "error": "No email", "channel": "email"}

        # Envoyer Email
        logger.info(
            "[notification_task] Sending Email fallback to driver %s",
            driver.id,
        )

        result = send_email_notification(
            email=email,
            subject=f"[ATMR] {title}",
            body=body,
            notification_type=notification_type,
        )

        if result.get("ok"):
            logger.info(
                "[notification_task] Email sent successfully to driver %s",
                driver.id,
            )
            return {"ok": True, "channel": "email"}

        logger.error(
            "[notification_task] Email failed for driver %s: %s",
            driver.id,
            result.get("error"),
        )
        return {"ok": False, "error": result.get("error"), "channel": "email"}

    except Exception as e:
        logger.exception("[notification_task] Email fallback exception: %s", e)
        return {"ok": False, "error": str(e), "channel": "email"}


# ==================== Task pour batch notifications ====================


@celery.task(
    name="tasks.notification_tasks.send_bulk_notifications",
    acks_late=True,
    task_time_limit=300,  # 5 minutes max
)
def send_bulk_notifications_task(notifications: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Envoie plusieurs notifications en batch.

    Args:
        notifications: Liste de dicts avec driver_id, title, body, data, etc.

    Returns:
        Dict avec statistiques d'envoi
    """
    logger.info(
        "[notification_task] Sending %d notifications in batch",
        len(notifications),
    )

    success_count = 0
    failed_count = 0

    for notif in notifications:
        try:
            # Envoyer chaque notification de manière asynchrone
            send_push_notification_task.delay(  # pyright: ignore[reportFunctionMemberAccess]
                driver_id=notif["driver_id"],
                title=notif["title"],
                body=notif["body"],
                data=notif.get("data"),
                notification_type=notif.get("notification_type", "unknown"),
                bypass_rate_limit=notif.get("bypass_rate_limit", False),
                fallback_to_sms=notif.get("fallback_to_sms", True),
                fallback_to_email=notif.get("fallback_to_email", True),
            )
            success_count += 1
        except Exception as e:
            logger.error(
                "[notification_task] Failed to queue notification: %s",
                e,
            )
            failed_count += 1

    return {
        "ok": True,
        "total": len(notifications),
        "success": success_count,
        "failed": failed_count,
    }


@celery.task(name="notifications.deactivate_stale_device_tokens")
def deactivate_stale_device_tokens_task() -> dict[str, int]:
    """Désactive les tokens push zombies (cron quotidien)."""
    from celery_app import get_flask_app
    from ext import db
    from services.monitoring.prometheus import refresh_push_active_owners_gauges
    from services.notifications.device_token_lifecycle import (
        deactivate_stale_device_tokens,
    )

    app = get_flask_app()
    with app.app_context():
        deactivated = deactivate_stale_device_tokens()
        db.session.commit()
        refresh_push_active_owners_gauges()
        logger.info(
            "[notification_task] deactivate_stale_device_tokens deactivated=%s",
            deactivated,
        )
        return {"deactivated": deactivated}


@celery.task(name="notifications.fetch_expo_push_receipts")
def fetch_expo_push_receipts_task() -> dict:
    """Récupère les receipts Expo et met à jour delivery_status / invalidation."""
    from celery_app import get_flask_app
    from services.notifications.expo_receipts import apply_expo_receipts

    app = get_flask_app()
    with app.app_context():
        result = apply_expo_receipts()
        logger.info("[notification_task] fetch_expo_push_receipts %s", result)
        return result


@celery.task(name="notifications.refresh_push_coverage_gauges")
def refresh_push_coverage_gauges_task() -> dict[str, str]:
    """Rafraîchit les gauges couverture push (filet horaire)."""
    from celery_app import get_flask_app
    from services.monitoring.prometheus import refresh_push_active_owners_gauges

    app = get_flask_app()
    with app.app_context():
        refresh_push_active_owners_gauges()
        return {"status": "ok"}
