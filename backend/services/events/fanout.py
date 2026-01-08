# backend/services/event_fanout.py
"""Service centralisé pour le fan-out hybride (Socket.IO + Push).

Ce module centralise toute la logique de fan-out pour éviter la duplication de code
et garantir la cohérence entre Socket.IO (foreground) et Push notifications (background).
"""

from __future__ import annotations

from typing import Any, Dict

from ext import app_logger
from schemas.socket_events import EVENT_VERSION, SocketEvent
from services.notifications.push import send_push_message
from services.realtime.socketio import (
    emit_company_event,
    emit_date_event,
    emit_driver_event,
)


def _send_push_to_driver(
    driver_id: int,
    title: str,
    body: str,
    data: Dict[str, Any],
    *,
    timeout: int = 5,
) -> bool:
    """Envoie une push notification à un chauffeur.

    Args:
        driver_id: ID du chauffeur
        title: Titre de la notification
        body: Corps du message
        data: Données additionnelles (pour deep linking, etc.)
        timeout: Timeout en secondes (défaut: 5)

    Returns:
        True si la push a été envoyée avec succès, False sinon
    """
    success = False
    try:
        from models import Driver

        driver = Driver.query.get(driver_id)
        if not driver:
            app_logger.debug(
                "[event_fanout] Driver %s not found, push skipped", driver_id
            )
        elif not driver.push_token:
            app_logger.debug(
                "[event_fanout] Driver %s has no push_token, push skipped", driver_id
            )
        else:
            # Déterminer si on doit bypasser le rate limit (alertes urgentes)
            is_urgent = data.get("type") == "urgent_alert" if data else False
            bypass_rate_limit = is_urgent

            result = send_push_message(
                token=driver.push_token,
                title=title,
                body=body,
                data=data,
                timeout=timeout,
                driver_id=driver_id,
                bypass_rate_limit=bypass_rate_limit,
            )

            if result.get("ok"):
                app_logger.info("[event_fanout] Push sent to driver %s", driver_id)
                success = True
            else:
                app_logger.warning(
                    "[event_fanout] Push failed for driver %s: %s",
                    driver_id,
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
        emit_company_event(company_id, "urgent_alert", payload)
        if driver_id:
            emit_driver_event(driver_id, "driver:urgent_alert", payload)
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
