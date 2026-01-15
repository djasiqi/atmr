# backend/services/event_fanout.py
"""Service centralisé pour le fan-out hybride (Socket.IO + Push).

Ce module centralise toute la logique de fan-out pour éviter la duplication de code
et garantir la cohérence entre Socket.IO (foreground) et Push notifications (background).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from ext import app_logger
from schemas.socket_events import EVENT_VERSION, SocketEvent
from services.events.night_mode import should_send_night_notification
from services.notifications.push import send_push_message
from services.realtime.socketio import (
    emit_company_event,
    emit_date_event,
    emit_driver_event,
)

# ✅ Phase 2 - Analytics: Import des métriques
try:
    from services.monitoring.notification_metrics import (
        track_notification_sent,
        track_notification_skipped_night,
    )
except ImportError:
    # Fallback si prometheus_client pas installé
    def track_notification_sent(*args, **kwargs):
        pass

    def track_notification_skipped_night(*args, **kwargs):
        pass


def _get_notification_channel(notification_type: str) -> str:
    """Détermine le canal Android approprié selon le type.

    Correspond aux canaux définis côté mobile:
    - critical: Urgences, accidents
    - missions: Missions, retards
    - messages: Chat, communications
    - info: Stats, informations générales
    """
    channel_mapping = {
        "urgent_alert": "critical",
        "accident": "critical",
        "emergency": "critical",
        "booking": "missions",
        "booking_updated": "missions",
        "booking_cancelled": "missions",
        "delay": "missions",
        "message": "messages",
        "team_chat_message": "messages",
        "dispatch_completed": "info",
        "stats": "info",
        "info": "info",
    }

    return channel_mapping.get(notification_type, "missions")


def _get_notification_category(notification_type: str) -> str | None:
    """Détermine la catégorie d'actions selon le type de notification.

    Phase 2 - Permet les actions directes (Accept/Reject) depuis les notifications.

    Categories disponibles côté mobile:
    - mission_available: Accepter | Refuser | Voir
    - mission_urgent: Appeler | Voir Détails
    - message_received: Répondre | Marquer Lu
    """
    category_mapping = {
        "booking": "mission_available",
        "booking_assigned": "mission_available",
        "urgent_alert": "mission_urgent",
        "accident": "mission_urgent",
        "emergency": "mission_urgent",
        "message": "message_received",
        "team_chat_message": "message_received",
    }

    return category_mapping.get(notification_type)


def _get_notification_thread_id(notification_type: str) -> str | None:
    """Détermine le threadId pour groupement intelligent.

    Phase 2 - Groupement des notifications similaires pour éviter le spam.

    Groupes disponibles:
    - missions: Toutes les missions
    - messages: Tous les messages
    - alerts: Toutes les alertes
    - infos: Toutes les informations
    """
    thread_mapping = {
        "booking": "missions",
        "booking_assigned": "missions",
        "booking_updated": "missions",
        "booking_cancelled": "missions",
        "delay": "missions",
        "urgent_alert": "alerts",
        "accident": "alerts",
        "emergency": "alerts",
        "message": "messages",
        "team_chat_message": "messages",
        "dispatch_completed": "infos",
        "stats": "infos",
        "info": "infos",
    }

    return thread_mapping.get(notification_type)


def _send_push_to_driver(
    driver_id: int,
    title: str,
    body: str,
    data: Dict[str, Any],
    *,
    timeout: int = 5,
    use_celery: bool = True,
) -> bool:
    """Envoie une push notification à un chauffeur via Celery (queue persistante + fallback).

    Args:
        driver_id: ID du chauffeur
        title: Titre de la notification
        body: Corps du message
        data: Données additionnelles (pour deep linking, etc.)
        timeout: Timeout en secondes (défaut: 5) - ignoré si use_celery=True
        use_celery: Si True, utilise Celery avec fallback SMS/Email (recommandé)

    Returns:
        True si la notification a été queued/envoyée avec succès, False sinon
    """
    success = False
    try:
        # ✅ AMÉLIORATION: Vérifier la déduplication avant d'envoyer
        notification_type = data.get("type", "unknown") if data else "unknown"

        # ✅ Phase 1 - Quick Wins: Ajouter le canal Android approprié
        data["channelId"] = _get_notification_channel(notification_type)

        # ✅ Phase 2 - Enrichissement: Ajouter la catégorie pour actions directes
        category = _get_notification_category(notification_type)
        if category:
            data["categoryId"] = category

        # ✅ Phase 2 - Enrichissement: Ajouter threadId pour groupement intelligent
        thread_id = _get_notification_thread_id(notification_type)
        if thread_id:
            data["threadId"] = thread_id
            data["group"] = thread_id

        # ✅ Phase 1 - Quick Wins: Vérifier mode nuit
        if not should_send_night_notification(notification_type, driver_id):
            app_logger.info(
                "[fanout] Notification skipped (night mode): driver=%s, type=%s",
                driver_id,
                notification_type,
            )
            # ✅ Phase 2 - Analytics: Tracker notification bloquée
            reason = "night_mode" if not driver_id else "driver_off_duty"
            track_notification_skipped_night(notification_type, reason)
            # Retourner succès pour ne pas logger comme erreur
            return True

        from services.notifications.push import _check_duplicate_notification

        if _check_duplicate_notification(driver_id, title, body, notification_type):
            app_logger.debug(
                "[event_fanout] Notification dupliquée ignorée pour driver %s",
                driver_id,
            )
            return True  # Considéré comme succès (déjà envoyée)

        # Déterminer si on doit bypasser le rate limit (alertes urgentes)
        is_urgent = notification_type == "urgent_alert"
        bypass_rate_limit = is_urgent

        # ✅ AMÉLIORATION MAJEURE: Utiliser Celery pour queue persistante + fallback SMS/Email
        if use_celery:
            from tasks.notification_tasks import send_push_notification_task

            app_logger.info(
                "[event_fanout] Queueing notification to driver %s via Celery (type: %s)",
                driver_id,
                notification_type,
            )

            # Envoyer la task en asynchrone (non-bloquant)
            send_push_notification_task.delay(  # pyright: ignore[reportFunctionMemberAccess]
                driver_id=driver_id,
                title=title,
                body=body,
                data=data,
                notification_type=notification_type,
                bypass_rate_limit=bypass_rate_limit,
                fallback_to_sms=True,  # Activer fallback SMS si push échoue
                fallback_to_email=True,  # Activer fallback Email en dernier recours
            )

            # ✅ Phase 2 - Analytics: Tracker notification envoyée
            channel = data.get("channelId", "unknown")
            track_notification_sent(notification_type, channel, "queued")

            # Considéré comme succès car la notification est en queue
            return True

        # Mode legacy: envoi direct (sans fallback ni queue persistante)
        # ✅ CORRECTIF #3: Utiliser DeviceToken pour support multi-device
        from ext import db
        from models import DeviceToken

        device_tokens = DeviceToken.query.filter_by(
            driver_id=driver_id,
            is_active=True,
        ).all()

        if not device_tokens:
            app_logger.debug(
                "[event_fanout] Driver %s has no active push tokens, push skipped",
                driver_id,
            )
        else:
            # Envoyer à tous les devices actifs
            success_count = 0
            for device_token in device_tokens:
                result = send_push_message(
                    token=device_token.token,
                    title=title,
                    body=body,
                    data=data,
                    timeout=timeout,
                    driver_id=driver_id,
                    bypass_rate_limit=bypass_rate_limit,
                )

                if result.get("ok"):
                    success_count += 1
                    app_logger.debug(
                        "[event_fanout] Push sent to driver %s (device %s)",
                        driver_id,
                        device_token.id,
                    )
                else:
                    error_msg = result.get("error", "Unknown error")
                    app_logger.warning(
                        "[event_fanout] Push failed for driver %s (device %s): %s",
                        driver_id,
                        device_token.id,
                        error_msg,
                    )

                    # ✅ CORRECTIF #3: Invalider ce token spécifique (pas tous les tokens du driver)
                    if result.get("token_invalid"):
                        device_token.is_active = False
                        db.session.commit()
                        app_logger.info(
                            "[event_fanout] Token invalidé pour driver %s (device %s)",
                            driver_id,
                            device_token.id,
                        )
                        # ✅ INSTRUMENTATION: Métrique Prometheus pour token invalide
                        try:
                            from services.monitoring.prometheus import (
                                track_push_token_invalidated,
                            )

                            track_push_token_invalidated(reason="device_not_registered")
                        except ImportError:
                            pass  # Prometheus non disponible

            success = success_count > 0
            if success:
                app_logger.info(
                    "[event_fanout] Push sent to driver %s (%d/%d devices)",
                    driver_id,
                    success_count,
                    len(device_tokens),
                )
    except (ValueError, TypeError, AttributeError) as e:
        app_logger.error(
            "[event_fanout] Push failed (validation error: %s): %s",
            type(e).__name__,
            e,
        )
    except (ConnectionError, OSError) as e:
        app_logger.error(
            "[event_fanout] Push failed (network error: %s): %s",
            type(e).__name__,
            e,
        )
    except Exception:
        app_logger.exception("[event_fanout] Push failed")

    return success


def _send_push_to_company(
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
                    app_logger.debug(
                        "[event_fanout] Company user %s has no push_token, push skipped",
                        company.user_id,
                    )
                else:
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
    except (ConnectionError, OSError) as e:
        app_logger.error(
            "[event_fanout] Push failed (network error: %s): %s",
            type(e).__name__,
            e,
        )
    except Exception:
        app_logger.exception("[event_fanout] Push failed")

    return success


# ==================== Helper pour création de payloads d'événements ====================


# ✅ Utiliser le schéma centralisé SocketEvent pour garantir la cohérence
def _create_event_payload(data: Dict[str, Any], event_type: str) -> Dict[str, Any]:
    """Crée un payload d'événement Socket.IO enrichi avec event_id, version, timestamp.

    Utilise le schéma centralisé SocketEvent pour garantir la cohérence.

    Args:
        data: Données métier de l'événement (seront fusionnées avec les métadonnées)
        event_type: Type d'événement métier (ex: "booking_assigned", "dispatch_completed")

    Returns:
        Payload enrichi contenant:
        - event_id: UUID v4 unique pour cet événement
        - version: Version du format d'événement (actuellement "1.0")
        - timestamp: Timestamp ISO 8601 UTC de l'émission
        - event_type: Type d'événement métier
        - ...tous les champs de data (fusionnés)
    """
    return SocketEvent.create(
        event_type=event_type, payload=data, version=EVENT_VERSION
    )


# ==================== Fonctions de fan-out par événement métier ====================


def fanout_booking_assigned_to_driver(
    driver_id: int,
    booking_id: int,
    booking_data: Dict[str, Any] | None = None,
) -> None:
    """Fan-out hybride pour une mission assignée à un chauffeur.

    Args:
        driver_id: ID du chauffeur
        booking_id: ID de la mission
        booking_data: Données de la mission (optionnel)
    """
    # 1. Socket.IO (foreground)
    try:
        base_data: Dict[str, Any] = booking_data or {"id": booking_id}
        payload = _create_event_payload(base_data, "booking_assigned")
        emit_driver_event(driver_id, "new_booking", payload)
    except Exception:
        app_logger.exception(
            "[event_fanout] Socket.IO failed for booking_assigned (driver %s)",
            driver_id,
        )

    # 2. Push notification (background)
    pickup_address = (
        booking_data.get("pickup_address", "Nouvelle mission")
        if booking_data
        else "Nouvelle mission"
    )
    _send_push_to_driver(
        driver_id=driver_id,
        title="Nouvelle mission assignée",
        body=f"Mission #{booking_id} - {pickup_address}",
        data={
            "type": "booking",
            "booking_id": booking_id,
            "deepLink": f"atmr://booking/{booking_id}",
        },
    )


def fanout_booking_assigned_to_company(
    company_id: int,
    booking_id: int,
    driver_id: int | None = None,
) -> None:
    """Fan-out hybride pour une mission assignée (notification à l'entreprise).

    Args:
        company_id: ID de l'entreprise
        booking_id: ID de la mission
        driver_id: ID du chauffeur assigné (optionnel)
    """
    # 1. Socket.IO (foreground)
    try:
        base_data: Dict[str, Any] = {
            "booking_id": booking_id,
            "driver_id": driver_id,
        }
        payload = _create_event_payload(base_data, "booking_assigned")
        emit_company_event(company_id, "booking_assigned", payload)
    except Exception:
        app_logger.exception(
            "[event_fanout] Socket.IO failed for booking_assigned (company %s)",
            company_id,
        )

    # 2. Push notification (background)
    _send_push_to_company(
        company_id=company_id,
        title="Mission assignée",
        body=f"Mission #{booking_id} assignée au chauffeur",
        data={
            "type": "booking_assigned",
            "booking_id": booking_id,
            "driver_id": driver_id,
            "deepLink": f"atmr://booking/{booking_id}",
        },
    )


def fanout_booking_updated(
    driver_id: int,
    booking_id: int,
    booking_data: Dict[str, Any] | None = None,
    *,
    send_push: bool = True,
) -> None:
    """Fan-out hybride pour une mise à jour de mission.

    Args:
        driver_id: ID du chauffeur
        booking_id: ID de la mission
        booking_data: Données de la mission (optionnel)
        send_push: Si True, envoie aussi une push notification (défaut: True)
    """
    # 1. Socket.IO (foreground)
    try:
        base_data: Dict[str, Any] = booking_data or {"id": booking_id}
        payload = _create_event_payload(base_data, "booking_updated")
        emit_driver_event(driver_id, "booking_updated", payload)
    except Exception:
        app_logger.exception(
            "[event_fanout] Socket.IO failed for booking_updated (driver %s)",
            driver_id,
        )

    # 2. Push notification (background) - conditionnelle
    if send_push:
        _send_push_to_driver(
            driver_id=driver_id,
            title="Mission mise à jour",
            body=f"Mission #{booking_id} a été mise à jour",
            data={
                "type": "booking_updated",
                "booking_id": booking_id,
                "deepLink": f"atmr://booking/{booking_id}",
            },
        )


def fanout_booking_cancelled(
    driver_id: int,
    booking_id: int,
) -> None:
    """Fan-out hybride pour une mission annulée.

    Args:
        driver_id: ID du chauffeur
        booking_id: ID de la mission
    """
    # 1. Socket.IO (foreground)
    try:
        base_data: Dict[str, Any] = {"booking_id": booking_id}
        payload = _create_event_payload(base_data, "booking_cancelled")
        emit_driver_event(driver_id, "booking_cancelled", payload)
    except Exception:
        app_logger.exception(
            "[event_fanout] Socket.IO failed for booking_cancelled (driver %s)",
            driver_id,
        )

    # 2. Push notification (background)
    _send_push_to_driver(
        driver_id=driver_id,
        title="Mission annulée",
        body=f"Mission #{booking_id} a été annulée",
        data={
            "type": "booking_cancelled",
            "booking_id": booking_id,
            "deepLink": "atmr://bookings",
        },
    )


def fanout_message_new(
    driver_id: int,
    message_id: int,
    sender_name: str,
    message_preview: str,
    company_id: int | None = None,
) -> None:
    """Fan-out hybride pour un nouveau message.

    Args:
        driver_id: ID du chauffeur destinataire
        message_id: ID du message
        sender_name: Nom de l'expéditeur
        message_preview: Aperçu du message
        company_id: ID de l'entreprise (optionnel)
    """
    # Note: Socket.IO est déjà émis dans chat.py, on ne fait que la push ici
    # Si besoin, on pourrait aussi émettre Socket.IO depuis ici

    # Push notification (background)
    _send_push_to_driver(
        driver_id=driver_id,
        title=f"Nouveau message de {sender_name}",
        body=message_preview,
        data={
            "type": "message",
            "message_id": message_id,
            "company_id": company_id,
            "deepLink": f"atmr://chat/message/{message_id}",
        },
    )


def fanout_delay_detected(
    driver_id: int,
    booking_id: int,
    delay_minutes: float,
    assignment_id: str | None = None,
) -> None:
    """Fan-out hybride pour un retard détecté.

    Args:
        driver_id: ID du chauffeur
        booking_id: ID de la mission
        delay_minutes: Retard en minutes
        assignment_id: ID de l'assignation (optionnel)
    """
    # Note: Socket.IO est déjà émis dans socketio_service.py, on ne fait que la push ici
    # Si besoin, on pourrait aussi émettre Socket.IO depuis ici

    # Push notification (background)
    delay_text = f"{int(delay_minutes)} min" if delay_minutes >= 1 else "< 1 min"
    _send_push_to_driver(
        driver_id=driver_id,
        title="Retard détecté",
        body=f"Retard de {delay_text} sur la mission #{booking_id}",
        data={
            "type": "delay",
            "booking_id": booking_id,
            "assignment_id": assignment_id,
            "delay_minutes": float(delay_minutes),
            "deepLink": f"atmr://booking/{booking_id}?alert=delay",
        },
    )


def fanout_dispatch_run_completed(
    company_id: int,
    dispatch_run_id: int | str,
    assignments_count: int,
    date_str: str | None = None,
    *,
    send_push_if_urgent: bool = True,
    urgent_threshold: int = 10,
) -> None:
    """Fan-out hybride pour un dispatch terminé.

    Args:
        company_id: ID de l'entreprise
        dispatch_run_id: ID du dispatch run
        assignments_count: Nombre d'assignations créées
        date_str: Date du dispatch (optionnel)
        send_push_if_urgent: Si True, envoie push seulement si urgent (défaut: True)
        urgent_threshold: Seuil pour considérer comme urgent (défaut: 10)
    """
    # 1. Socket.IO (foreground)
    try:
        base_data: Dict[str, Any] = {
            "dispatch_run_id": str(dispatch_run_id),
            "assignments_count": assignments_count,
        }
        if date_str:
            base_data["date"] = date_str
        payload = _create_event_payload(base_data, "dispatch_run_completed")
        emit_company_event(company_id, "dispatch_run_completed", payload)
        if date_str:
            emit_date_event(date_str, "dispatch_run_completed", payload)
    except Exception:
        app_logger.exception(
            "[event_fanout] Socket.IO failed for dispatch_run_completed (company %s)",
            company_id,
        )

    # 2. Push notification (background) - conditionnelle
    is_urgent = assignments_count > urgent_threshold
    if send_push_if_urgent and is_urgent:
        _send_push_to_company(
            company_id=company_id,
            title="Dispatch terminé",
            body=f"Dispatch #{dispatch_run_id} terminé : {assignments_count} assignations",
            data={
                "type": "dispatch_completed",
                "dispatch_run_id": str(dispatch_run_id),
                "assignments_count": int(assignments_count),
                "date": date_str,
                "deepLink": f"atmr://dispatch/run/{dispatch_run_id}",
            },
        )


def fanout_urgent_alert(
    company_id: int,
    alert_id: int | str,
    alert_type: str,
    message: str,
    severity: str = "high",
    booking_id: int | None = None,
    driver_id: int | None = None,
) -> None:
    """Fan-out hybride pour une alerte urgente.

    Args:
        company_id: ID de l'entreprise
        alert_id: ID unique de l'alerte
        alert_type: Type d'alerte
        message: Message de l'alerte
        severity: Niveau de sévérité (défaut: "high")
        booking_id: ID du booking concerné (optionnel)
        driver_id: ID du chauffeur concerné (optionnel)
    """
    # 1. Socket.IO (foreground)
    try:
        base_data: Dict[str, Any] = {
            "alert_id": str(alert_id),
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
        }
        if booking_id:
            base_data["booking_id"] = booking_id
        if driver_id:
            base_data["driver_id"] = driver_id

        payload = _create_event_payload(base_data, "urgent_alert")
        # ✅ FIX: Standardiser avec '_' au lieu de ':' pour cohérence
        emit_company_event(company_id, "urgent_alert", payload)
        if driver_id:
            emit_driver_event(driver_id, "driver_urgent_alert", payload)
    except Exception:
        app_logger.exception(
            "[event_fanout] Socket.IO failed for urgent_alert (company %s)",
            company_id,
        )

    # 2. Push notification (background) - pour company et driver
    push_data: Dict[str, Any] = {
        "type": "urgent_alert",
        "alert_id": str(alert_id),
        "alert_type": alert_type,
        "severity": severity,
        "booking_id": booking_id,
        "driver_id": driver_id,
        "deepLink": f"atmr://alerts/{alert_id}",
    }

    # Push pour company
    _send_push_to_company(
        company_id=company_id,
        title=f"Alerte urgente: {alert_type}",
        body=message,
        data=push_data,
    )

    # Push pour driver si spécifié
    if driver_id:
        _send_push_to_driver(
            driver_id=driver_id,
            title=f"Alerte urgente: {alert_type}",
            body=message,
            data=push_data,
        )


# ========================================
# Phase 2.6 - Notifications Silencieuses
# ========================================


def send_silent_data_update(
    driver_id: int,
    sync_type: str,
    payload: Dict[str, Any],
    priority: str = "normal",  # noqa: ARG001
) -> bool:
    """Envoie une notification silencieuse pour sync données en arrière-plan.

    Phase 2.6 - Notifications silencieuses pour préchargement et sync données.

    Args:
        driver_id: ID du chauffeur cible
        sync_type: Type de sync ("missions", "profile", "maps", "config")
        payload: Données à synchroniser
        priority: "normal" ou "high" (réservé pour usage futur)

    Returns:
        True si envoyé avec succès
    """
    # Import métriques silent notifications
    try:
        from services.monitoring.notification_metrics import (
            track_silent_notification_sent,
        )
    except ImportError:

        def track_silent_notification_sent(*args, **kwargs):
            pass

    try:
        # ✅ CORRECTIF #3: Utiliser DeviceToken pour support multi-device
        from models import DeviceToken

        device_tokens = DeviceToken.query.filter_by(
            driver_id=driver_id,
            is_active=True,
        ).all()

        if not device_tokens:
            app_logger.warning(
                f"[silent_update] Driver {driver_id} sans tokens actifs, skip"
            )
            track_silent_notification_sent(sync_type, "failed")
            return False

        # ⚠️ IMPORTANT: Pas de title/body pour notification silencieuse
        push_data: Dict[str, Any] = {
            "type": "silent_update",
            "sync_type": sync_type,
            "payload": payload,
            "timestamp": int(datetime.now().timestamp()),
            "content-available": 1,  # iOS background fetch
        }

        # Envoyer à tous les devices actifs
        success_count = 0
        for device_token in device_tokens:
            result = send_push_message(
                token=device_token.token,
                title="",  # Vide pour silent notification
                body="",  # Vide pour silent notification
                data=push_data,
                timeout=5,
                use_retry=False,  # Pas de retry pour silent notifications
                driver_id=driver_id,
                bypass_rate_limit=False,
            )

            if result.get("ok"):
                success_count += 1
            elif result.get("token_invalid"):
                # Invalider ce token spécifique
                from ext import db

                device_token.is_active = False
                db.session.commit()

        success = success_count > 0

        if success:
            app_logger.info(
                f"[silent_update] Envoyé à driver {driver_id}: sync_type={sync_type} ({success_count}/{len(device_tokens)} devices)"
            )
            track_silent_notification_sent(sync_type, "success")
        else:
            app_logger.error(
                f"[silent_update] Échec envoi à driver {driver_id}: sync_type={sync_type}"
            )
            track_silent_notification_sent(sync_type, "failed")

        return success

    except Exception as e:
        app_logger.error(
            f"[silent_update] Exception driver {driver_id}: {e}", exc_info=True
        )
        track_silent_notification_sent(sync_type, "failed")
        return False


def send_missions_preload(driver_id: int, missions: list[Dict[str, Any]]) -> bool:
    """Précharge les missions à venir pour un chauffeur.

    Args:
        driver_id: ID du chauffeur
        missions: Liste des missions à précharger

    Returns:
        True si succès
    """
    app_logger.info(
        f"[missions_preload] Préchargement {len(missions)} missions pour driver {driver_id}"
    )

    return send_silent_data_update(
        driver_id=driver_id,
        sync_type="missions",
        payload={"missions": missions},
        priority="normal",
    )


def send_profile_sync(
    driver_id: int, profile: Dict[str, Any], stats: Dict[str, Any] | None = None
) -> bool:
    """Synchronise le profil et stats du chauffeur.

    Args:
        driver_id: ID du chauffeur
        profile: Données du profil
        stats: Stats optionnelles

    Returns:
        True si succès
    """
    app_logger.info(f"[profile_sync] Sync profil pour driver {driver_id}")

    payload = {"profile": profile}
    if stats:
        payload["stats"] = stats

    return send_silent_data_update(
        driver_id=driver_id,
        sync_type="profile",
        payload=payload,
        priority="normal",
    )


def send_maps_precache(driver_id: int, routes: list[Dict[str, Any]]) -> bool:
    """Précharge les cartes pour itinéraires à venir.

    Args:
        driver_id: ID du chauffeur
        routes: Liste des itinéraires à cacher

    Returns:
        True si succès
    """
    app_logger.info(
        f"[maps_precache] Préchargement {len(routes)} itinéraires pour driver {driver_id}"
    )

    return send_silent_data_update(
        driver_id=driver_id,
        sync_type="maps",
        payload={"routes": routes},
        priority="normal",
    )


def send_config_update(driver_id: int, config: Dict[str, Any]) -> bool:
    """Met à jour la configuration de l'app.

    Args:
        driver_id: ID du chauffeur
        config: Configuration à mettre à jour

    Returns:
        True si succès
    """
    app_logger.info(f"[config_update] Mise à jour config pour driver {driver_id}")

    return send_silent_data_update(
        driver_id=driver_id,
        sync_type="config",
        payload={"config": config},
        priority="normal",
    )


# ========================================
# Phase 3.8 - Critical Alerts iOS
# ========================================


def send_critical_alert_ios(
    driver_id: int,
    title: str,
    message: str,
    alert_type: str,
    data: Dict[str, Any] | None = None,
) -> bool:
    """Envoie une Critical Alert pour iOS (et notification prioritaire Android).

    Phase 3.8 - Critical Alerts pour urgences réelles.

    ⚠️ Note: Sur iOS, utilise interruptionLevel "critical" (iOS 15+).
    Pour vraies Critical Alerts (bypass DnD), nécessite entitlement Apple spécial.

    Args:
        driver_id: ID du chauffeur cible
        title: Titre de l'alerte
        message: Message de l'alerte
        alert_type: Type d'alerte (accident, emergency, security)
        data: Données additionnelles

    Returns:
        True si envoyé avec succès
    """
    try:
        # ✅ CORRECTIF #3: Utiliser DeviceToken pour support multi-device
        from ext import db
        from models import DeviceToken

        device_tokens = DeviceToken.query.filter_by(
            driver_id=driver_id,
            is_active=True,
        ).all()

        if not device_tokens:
            app_logger.warning(
                f"[critical_alert] Driver {driver_id} sans tokens actifs, skip"
            )
            return False

        # Construire le payload
        push_data: Dict[str, Any] = {
            "type": "critical_alert",
            "alert_type": alert_type,
            "timestamp": int(datetime.now().timestamp()),
            "deepLink": f"atmr://alerts/{alert_type}",
        }

        # Ajouter données custom
        if data:
            push_data.update(data)

        # Envoyer à tous les devices actifs
        success_count = 0
        for device_token in device_tokens:
            # ✅ Configuration spécifique iOS Critical Alert
            # Note: Pour vraies Critical Alerts (bypass DnD), nécessite entitlement
            # Actuellement: utilise interruptionLevel "critical" (iOS 15+)
            result = send_push_message(
                token=device_token.token,
                title=f"🚨 {title}",
                body=message,
                data={
                    **push_data,
                    # iOS 15+ : Interruption Level
                    "interruptionLevel": "critical",
                    # Android : Canal critical (déjà configuré)
                    "channelId": "critical",
                    # iOS : Son critique (si entitlement approuvé)
                    "sound": {
                        "critical": True,
                        "name": "default",  # ou "emergency_alert.wav" si custom
                        "volume": 1.0,
                    },
                },
                timeout=10,  # Timeout plus long pour urgences
                use_retry=True,
                driver_id=driver_id,
                bypass_rate_limit=True,  # Pas de rate limit pour urgences
            )

            if result.get("ok"):
                success_count += 1
            elif result.get("token_invalid"):
                # Invalider ce token spécifique
                device_token.is_active = False
                db.session.commit()

        success = success_count > 0

        if success:
            app_logger.info(
                f"[critical_alert] Critical alert envoyée à driver {driver_id}: {alert_type} ({success_count}/{len(device_tokens)} devices)"
            )
            # Tracker métrique
            track_notification_sent("critical_alert", "critical", "success")
        else:
            app_logger.error(
                f"[critical_alert] Échec envoi à driver {driver_id}: {alert_type}"
            )
            track_notification_sent("critical_alert", "critical", "failed")

        return success

    except Exception as e:
        app_logger.error(
            f"[critical_alert] Exception driver {driver_id}: {e}", exc_info=True
        )
        track_notification_sent("critical_alert", "critical", "failed")
        return False


def send_accident_alert(driver_id: int, accident_details: Dict[str, Any]) -> bool:
    """Alerte urgente : Accident chauffeur détecté.

    Args:
        driver_id: ID du chauffeur
        accident_details: Détails de l'accident

    Returns:
        True si succès
    """
    app_logger.warning(f"[accident_alert] 🚨 ACCIDENT détecté driver {driver_id}")

    return send_critical_alert_ios(
        driver_id=driver_id,
        title="ACCIDENT DÉTECTÉ",
        message="Un accident a été détecté. Êtes-vous en sécurité ?",
        alert_type="accident",
        data=accident_details,
    )


def send_medical_emergency_alert(
    driver_id: int, emergency_details: Dict[str, Any]
) -> bool:
    """Alerte urgente : Urgence médicale passager.

    Args:
        driver_id: ID du chauffeur
        emergency_details: Détails de l'urgence

    Returns:
        True si succès
    """
    app_logger.warning(f"[medical_emergency] 🚨 URGENCE MÉDICALE driver {driver_id}")

    return send_critical_alert_ios(
        driver_id=driver_id,
        title="URGENCE MÉDICALE",
        message="Un passager nécessite une assistance médicale immédiate.",
        alert_type="medical_emergency",
        data=emergency_details,
    )


def send_security_zone_alert(driver_id: int, zone_details: Dict[str, Any]) -> bool:
    """Alerte urgente : Zone dangereuse.

    Args:
        driver_id: ID du chauffeur
        zone_details: Détails de la zone

    Returns:
        True si succès
    """
    app_logger.warning(f"[security_alert] 🚨 ZONE DANGEREUSE driver {driver_id}")

    return send_critical_alert_ios(
        driver_id=driver_id,
        title="ALERTE SÉCURITÉ",
        message=f"Vous entrez dans une zone à risque : {zone_details.get('zone_name', 'Non spécifiée')}",
        alert_type="security_zone",
        data=zone_details,
    )
