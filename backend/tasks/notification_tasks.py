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
from typing import Any, ClassVar, Dict

from celery import Task  # type: ignore[import-untyped]

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

    def on_failure(self, exc, task_id, _args, kwargs, einfo):
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
    task_time_limit=30,  # 30 secondes max
    task_soft_time_limit=25,
    max_retries=MAX_PUSH_RETRIES,
)
def send_push_notification_task(  # noqa: PLR0911
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
            logger.warning(
                "[notification_task] send_push_notification_task started: driver_id=%s notification_type=%s",
                driver_id,
                notification_type,
            )
            # Récupérer le driver
            driver = db.session.get(Driver, driver_id)
            if not driver:
                logger.error("[notification_task] Driver %s not found", driver_id)
                return {"ok": False, "error": "Driver not found", "channel": "none"}

            # ✅ CORRECTIF #3: Utiliser DeviceToken pour support multi-device
            from models import DeviceToken

            device_tokens = DeviceToken.query.filter_by(
                driver_id=driver_id,
                is_active=True,
            ).all()

            if not device_tokens:
                logger.warning(
                    "[notification_task] No active push tokens for driver %s, using fallback",
                    driver_id,
                )
            else:
                logger.warning(
                    "[notification_task] Attempt %d/%d: Sending push to driver %s (%d devices)",
                    self.request.retries + 1,
                    MAX_PUSH_RETRIES,
                    driver_id,
                    len(device_tokens),
                )

                # Envoyer à tous les devices actifs
                success_count = 0
                invalid_tokens = []
                last_result: Dict[str, Any] | None = None
                for device_token in device_tokens:
                    result = send_push_message(
                        token=device_token.token,
                        title=title,
                        body=body,
                        data=data,
                        driver_id=driver_id,
                        bypass_rate_limit=bypass_rate_limit,
                    )
                    last_result = (
                        result  # Garder le dernier résultat pour logging/retry
                    )

                    if result.get("ok"):
                        success_count += 1
                        logger.debug(
                            "[notification_task] Push sent to driver %s (device %s)",
                            driver_id,
                            device_token.id,
                        )
                    else:
                        error = result.get("error", "Unknown error")
                        logger.warning(
                            "[notification_task] Push failed for driver %s (device %s): %s",
                            driver_id,
                            device_token.id,
                            error,
                        )

                        # Si token invalide, marquer pour invalidation
                        if result.get("token_invalid"):
                            invalid_tokens.append(device_token)

                # Invalider les tokens invalides
                if invalid_tokens:
                    for device_token in invalid_tokens:
                        device_token.is_active = False
                    db.session.commit()
                    logger.warning(
                        "[notification_task] %d tokens invalidés pour driver %s",
                        len(invalid_tokens),
                        driver_id,
                    )

                # Si au moins un envoi a réussi, considérer comme succès
                if success_count > 0:
                    logger.warning(
                        "[notification_task] Push sent successfully to driver %s (%d/%d devices)",
                        driver_id,
                        success_count,
                        len(device_tokens),
                    )
                    return {
                        "ok": True,
                        "channel": "push",
                        "attempts": self.request.retries + 1,
                        "devices_sent": success_count,
                        "devices_total": len(device_tokens),
                    }

                # Tous les envois ont échoué
                # Si tous les tokens sont invalides, passer directement au fallback
                if len(invalid_tokens) == len(device_tokens):
                    logger.warning(
                        "[notification_task] Tous les tokens invalides pour driver %s, skip retry",
                        driver_id,
                    )
                    # Passer directement au fallback sans retry
                    raise self.retry(countdown=0, max_retries=0)

                # Au moins un token valide mais échec réseau → retry
                # Utiliser le dernier résultat pour déterminer le type d'erreur
                # Note: last_result est toujours défini car device_tokens n'est pas vide et la boucle s'est exécutée
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
