# backend/services/event_fanout.py
"""Service centralisé pour le fan-out hybride (Socket.IO + Push).

Ce module centralise toute la logique de fan-out pour éviter la duplication de code
et garantir la cohérence entre Socket.IO (foreground) et Push notifications (background).
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict

from ext import app_logger
from schemas.socket_events import EVENT_VERSION, SocketEvent
from services.events.night_mode import should_send_night_notification
from services.notifications.device_token_lifecycle import (
    is_push_device_token_lifecycle_enabled,
)
from services.notifications.push import send_push_message
from services.notifications.push_message_builder import (
    CHANGE_TYPE_ADDRESS_CHANGE,
    CHANGE_TYPE_CANCEL,
    CHANGE_TYPE_TIME_CHANGE,
    CHAT_TYPE_DIRECT,
    EVENT_ASSIGNED,
    _extract_time_hhmm,
    build_chat_push,
    build_push_for_company_to_driver,
    build_push_for_driver_to_company,
    build_push_message,
)
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
    - missions_v2: H2 — canal debug pour contourner channel legacy (PUSH_PROOF=1)
    - messages: Chat, communications
    - info: Stats, informations générales
    """
    channel_mapping = {
        "urgent_alert": "critical",
        "accident": "critical",
        "emergency": "critical",
        "booking": "missions",
        "booking_assigned": "missions",
        "booking_updated": "missions",
        "booking_cancelled": "missions",
        "booking_reassigned": "missions",
        "delay": "missions",
        "message": "messages",
        "team_chat_message": "messages",
        "dispatch_completed": "info",
        "stats": "info",
        "info": "info",
    }

    base = channel_mapping.get(notification_type, "missions")

    # H2: Channel version bump — missions_v2 pour contourner channel legacy
    # (Android ne met pas à jour un channel déjà créé en DEFAULT/LOW)
    # Fallback auto en dev/staging ; PUSH_PROOF=1 en prod (feature flag)
    if base == "missions":
        push_proof = os.environ.get("PUSH_PROOF", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        flask_env = os.environ.get("FLASK_ENV", "").strip().lower()
        if push_proof or flask_env in ("development", "staging"):
            return "missions_v2"

    return base


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


def _log_push_fanout(
    *,
    event_type: str,
    booking_id: int,
    actor_role: str | None,
    driver_id_target: int,
    company_id_target: int,
) -> None:
    """Log structuré PUSH_FANOUT avant envoi pour traçabilité.

    Args:
        event_type: Type d'événement (booking_assigned, booking_updated, etc.)
        booking_id: ID de la mission
        actor_role: Rôle de l'initiateur (driver, dispatcher, system)
        driver_id_target: ID chauffeur cible (0 si push company uniquement)
        company_id_target: ID entreprise cible (0 si push driver uniquement)
    """
    driver_tokens_count = 0
    company_tokens_count = 0
    company_user_id_target = 0
    try:
        if driver_id_target:
            from models import DeviceToken

            driver_tokens_count = DeviceToken.query.filter_by(
                driver_id=driver_id_target,
                is_active=True,
            ).count()
        if company_id_target:
            from models import Company, User

            company = Company.query.get(company_id_target)
            if company:
                company_user_id_target = int(company.user_id or 0)
                company_user = (
                    User.query.get(company.user_id) if company.user_id else None
                )
                company_tokens_count = (
                    1
                    if company_user and getattr(company_user, "push_token", None)
                    else 0
                )
    except Exception:
        pass  # Ne pas faire échouer le fanout si le log échoue
    app_logger.info(
        "PUSH_FANOUT event_type=%s booking_id=%s actor_role=%s driver_id_target=%s company_user_id_target=%s driver_tokens_count=%s company_tokens_count=%s",
        event_type,
        booking_id,
        actor_role or "",
        driver_id_target,
        company_user_id_target,
        driver_tokens_count,
        company_tokens_count,
    )


def _get_recipient_discreet_mode(
    *,
    company_id: int | None = None,
    driver_id: int | None = None,
) -> bool:
    """True si le destinataire (entreprise ou chauffeur) a activé le mode discret push.

    Lit User.push_privacy_mode via Company.user_id ou Driver.user_id.
    Valeurs : "discreet" => True, "detailed" ou absent => False.
    """
    try:
        from models import Company, Driver, User

        user_id: int | None = None
        if company_id:
            company = Company.query.get(company_id)
            if company:
                user_id = getattr(company, "user_id", None)
        elif driver_id:
            driver = Driver.query.get(driver_id)
            if driver:
                user_id = getattr(driver, "user_id", None)
        if not user_id:
            return False
        user = User.query.get(user_id)
        if not user:
            return False
        mode = getattr(user, "push_privacy_mode", None) or "detailed"
        return str(mode).strip().lower() == "discreet"
    except Exception:
        return False


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
        # ✅ Garde dur "assigned driver only" : empêcher broadcast accidentel
        if not driver_id or driver_id <= 0:
            app_logger.warning(
                "PUSH_FANOUT_ABORT driver_id_missing driver_id=%s (invalid/falsy)",
                driver_id,
            )
            return False

        # Guard centralise : bloquer self-notification (driver est l'acteur)
        actor_role = data.get("actor_role") if data else None
        actor_id = data.get("actor_id") if data else None
        if actor_role == "driver" and actor_id is not None:
            try:
                if int(actor_id) == int(driver_id):
                    app_logger.info(
                        "[fanout] GUARD: self-notification blocked (driver %s is actor)",
                        driver_id,
                    )
                    return True
            except (ValueError, TypeError):
                pass

        notification_type = data.get("type", "unknown") if data else "unknown"
        if data:
            booking_id = data.get("booking_id")
            if booking_id is not None and data.get("mission_id") is None:
                data["mission_id"] = booking_id
            data.setdefault(
                "payload_schema_version",
                "booking_v1" if data.get("booking_id") is not None else "mission_v2",
            )

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

            app_logger.warning(
                "[event_fanout] Queueing notification to driver %s via Celery (type: %s)",
                driver_id,
                notification_type,
            )

            send_push_notification_task.delay(  # pyright: ignore[reportFunctionMemberAccess]
                driver_id=driver_id,
                title=title,
                body=body,
                data=data,
                notification_type=notification_type,
                bypass_rate_limit=bypass_rate_limit,
                fallback_to_sms=True,
                fallback_to_email=True,
            )
            app_logger.info(
                "[push_enqueued] driver_id=%s type=%s priority=%s trace_id=%s",
                driver_id,
                notification_type,
                "high" if is_urgent else "default",
                data.get("trace_id"),
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
                    provider=getattr(device_token, "provider", None),
                    platform=getattr(device_token, "platform", None),
                    device_token_id=device_token.id,
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

                    if result.get("token_invalid"):
                        app_logger.info(
                            "[event_fanout] Token invalidé (lifecycle) pour driver %s (device %s)",
                            driver_id,
                            device_token.id,
                        )
                        try:
                            from services.monitoring.prometheus import (
                                track_push_token_invalidated,
                            )

                            track_push_token_invalidated(reason="device_not_registered")
                        except ImportError:
                            pass  # Prometheus non disponible
                        if not is_push_device_token_lifecycle_enabled():
                            device_token.is_active = False

            try:
                db.session.commit()
            except Exception:
                app_logger.exception("[event_fanout] commit after push lifecycle")

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
    from services.events.booking_socket_payload import (
        maybe_shrink_booking_socket_payload,
    )

    shrunk = maybe_shrink_booking_socket_payload(data, event_type)
    return SocketEvent.create(
        event_type=event_type, payload=shrunk, version=EVENT_VERSION
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
    app_logger.warning(
        "[event_fanout] fanout_booking_assigned_to_driver called: driver_id=%s booking_id=%s",
        driver_id,
        booking_id,
    )

    # 1. Socket.IO (foreground)
    try:
        base_data: Dict[str, Any] = booking_data or {"id": booking_id}
        payload = _create_event_payload(base_data, "booking_assigned")
        emit_driver_event(driver_id, "new_booking", payload)
        app_logger.warning(
            "[event_fanout] Socket.IO emitted for booking_assigned (driver %s)",
            driver_id,
        )
    except Exception:
        app_logger.exception(
            "[event_fanout] Socket.IO failed for booking_assigned (driver %s)",
            driver_id,
        )

    # 2. Push notification (background) — message métier (nom client + contexte), jamais ID-only
    push_ctx: Dict[str, Any] = dict(booking_data or {})
    push_ctx.setdefault("id", booking_id)
    discrete = _get_recipient_discreet_mode(driver_id=driver_id)
    msg = build_push_message(
        EVENT_ASSIGNED,
        push_ctx,
        "driver",
        discrete_mode=discrete,
    )
    app_logger.warning(
        "[event_fanout] Sending push notification to driver %s for booking %s (pickup: %s)",
        driver_id,
        booking_id,
        push_ctx.get("pickup_location") or push_ctx.get("pickup_address") or "-",
    )

    data = dict(msg["data"])
    data["recipient_role"] = "driver"
    data["actor_role"] = "company"

    _log_push_fanout(
        event_type="booking_assigned",
        booking_id=booking_id,
        actor_role="dispatcher",
        driver_id_target=driver_id,
        company_id_target=0,
    )
    result = _send_push_to_driver(
        driver_id=driver_id,
        title=msg["title"],
        body=msg["body"],
        data=data,
    )

    app_logger.warning(
        "[event_fanout] _send_push_to_driver returned: driver_id=%s booking_id=%s result=%s",
        driver_id,
        booking_id,
        result,
    )


def fanout_booking_assigned_to_company(
    company_id: int,
    booking_id: int,
    driver_id: int | None = None,
    *,
    booking_data: Dict[str, Any] | None = None,
) -> None:
    """Fan-out hybride pour une mission assignée (notification à l'entreprise).

    Args:
        company_id: ID de l'entreprise
        booking_id: ID de la mission
        driver_id: ID du chauffeur assigné (optionnel)
        booking_data: Données de la mission (client_name, time_formatted, etc.) pour message métier
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

    # 2. Push notification (background) — toujours via builder (jamais "ID-only")
    push_ctx: Dict[str, Any] = (
        {**booking_data, "id": booking_data.get("id") or booking_id}
        if booking_data
        else {"id": booking_id}
    )
    discrete = _get_recipient_discreet_mode(company_id=company_id, driver_id=None)
    msg = build_push_message(
        EVENT_ASSIGNED,
        push_ctx,
        "company",
        discrete_mode=discrete,
    )
    if os.environ.get("DEBUG_NOTIF_ROUTING", "").lower() in ("1", "true", "yes"):
        app_logger.info(
            "[event_fanout] push_company_enqueued company_id=%s event=booking_assigned",
            company_id,
        )
    from tasks.notification_tasks import send_push_company_notification_task

    data = dict(msg["data"])
    data["recipient_role"] = "company"
    data["actor_role"] = "system"
    send_push_company_notification_task.delay(  # pyright: ignore[reportFunctionMemberAccess]
        company_id=company_id,
        title=msg["title"],
        body=msg["body"],
        data=data,
    )


def fanout_booking_updated(
    driver_id: int,
    booking_id: int,
    booking_data: Dict[str, Any] | None = None,
    *,
    send_push: bool = True,
    exclude_driver_id: int | None = None,
) -> None:
    """Fan-out hybride pour une mise à jour de mission (Company→Driver).

    Templates: assign, time_change, address_change, cancel.
    Anti-spam: collapse_key, dedupe_key, throttle.

    Args:
        exclude_driver_id: Si driver_id == exclude_driver_id, skip socket+push (actor).
    """
    if exclude_driver_id is not None and int(driver_id) == int(exclude_driver_id):
        app_logger.debug(
            "[event_fanout] fanout_booking_updated skipped (exclude_actor driver_id=%s)",
            driver_id,
        )
        return

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

    # 2. Push notification (background) - templates centralisés
    if send_push:
        ctx = booking_data or {"id": booking_id}
        changes = ctx.get("changes")
        status_raw = ctx.get("status")
        status = (
            str(getattr(status_raw, "value", status_raw)).lower()
            if status_raw is not None
            else None
        )

        # Détecter change_type (priorité: Annulée > Horaire > Adresse > Statut)
        # Cas "statut inchangé mais info modifiée": si EN_ROUTE + modif heure/adresse,
        # on utilise time_change/address_change (pas status) → dedupe_key inclut event_type.
        change_type = "status"
        old_time: str | None = None
        new_time: str | None = None
        address_change_type: str | None = None
        new_address_short: str | None = None

        if status in {"canceled", "cancelled"}:
            change_type = CHANGE_TYPE_CANCEL
        elif isinstance(changes, dict):
            if "scheduled_time" in changes:
                ch = changes.get("scheduled_time")
                if isinstance(ch, dict):
                    old_time = _extract_time_hhmm(ch.get("from"))
                    new_time = _extract_time_hhmm(ch.get("to"))
                change_type = CHANGE_TYPE_TIME_CHANGE
            elif "pickup_location" in changes or "dropoff_location" in changes:
                ch_p = changes.get("pickup_location")
                ch_d = changes.get("dropoff_location")
                if isinstance(ch_p, dict) and ch_p.get("to"):
                    address_change_type = "pickup"
                    new_address_short = str(ch_p.get("to", ""))[:40]
                elif isinstance(ch_d, dict) and ch_d.get("to"):
                    address_change_type = "dropoff"
                    new_address_short = str(ch_d.get("to", ""))[:40]
                if not address_change_type and ch_p:
                    address_change_type = "pickup"
                elif not address_change_type and ch_d:
                    address_change_type = "dropoff"
                change_type = CHANGE_TYPE_ADDRESS_CHANGE

        discrete = _get_recipient_discreet_mode(driver_id=driver_id)
        msg = build_push_for_company_to_driver(
            ctx,
            change_type=change_type,
            status=status,
            old_time=old_time,
            new_time=new_time,
            address_change_type=address_change_type,
            new_address_short=new_address_short,
            discrete_mode=discrete,
        )

        # Anti-spam: dedup + throttle
        from services.notifications.dedup_throttle import check_dedup_and_throttle

        skip, reason = check_dedup_and_throttle(
            "driver",
            driver_id,
            msg["dedupe_key"],
            f"booking_{booking_id}",
            msg.get("throttle_seconds", 8),
            1,
        )
        if skip:
            app_logger.info(
                "[event_fanout] Push skipped (driver %s): %s",
                driver_id,
                reason,
            )
            return

        data = dict(msg["data"])
        data["collapse_key"] = msg["collapse_key"]
        data["dedupe_key"] = msg["dedupe_key"]
        data["routing_version"] = 2
        data["routing_decision"] = "driver_only"
        data["recipient_role"] = "driver"
        if ctx.get("trace_id"):
            data["trace_id"] = ctx["trace_id"]
        if ctx.get("actor_role") is not None:
            data["actor_role"] = ctx["actor_role"]
        if ctx.get("actor_id") is not None:
            data["actor_id"] = ctx["actor_id"]

        _log_push_fanout(
            event_type="booking_updated",
            booking_id=booking_id,
            actor_role=ctx.get("actor_role"),
            driver_id_target=driver_id,
            company_id_target=0,
        )
        if os.environ.get("DEBUG_NOTIF_ROUTING", "").lower() in ("1", "true", "yes"):
            app_logger.info(
                "[event_fanout] PUSH_DRIVER collapse_key=%s dedupe_key=%s routing_decision=driver_only",
                msg["collapse_key"],
                msg.get("dedupe_key"),
            )
        _send_push_to_driver(
            driver_id=driver_id,
            title=msg["title"],
            body=msg["body"],
            data=data,
        )


def fanout_booking_updated_to_company(
    company_id: int,
    booking_id: int,
    booking_data: Dict[str, Any] | None = None,
    *,
    send_push: bool = True,
) -> None:
    """Fan-out hybride pour une mise à jour de mission (Driver→Company).

    Template: "Course • {STATUT_LABEL}" + "Chauffeur → Client" + trajet.
    Anti-spam: collapse_key, dedupe_key, throttle.
    """
    # 1. Socket.IO (foreground)
    try:
        base_data: Dict[str, Any] = booking_data or {"booking_id": booking_id}
        payload = _create_event_payload(base_data, "booking_updated")
        emit_company_event(company_id, "booking_updated", payload)
    except Exception:
        app_logger.exception(
            "[event_fanout] Socket.IO failed for booking_updated (company %s)",
            company_id,
        )

    # 2. Push notification (background) - template Driver→Company
    if send_push:
        ctx = booking_data or {"id": booking_id}
        status_raw = ctx.get("status")
        if status_raw is None:
            status = None
        elif hasattr(status_raw, "value"):
            status = str(status_raw.value).lower()
        else:
            status = str(status_raw).lower()

        discrete = _get_recipient_discreet_mode(company_id=company_id)
        msg = build_push_for_driver_to_company(
            ctx,
            status=status,
            discrete_mode=discrete,
        )

        # Anti-spam: dedup + throttle
        # ✅ P0: Scope par status pour permettre en_route, in_progress, completed, etc.
        # (max=1 sur booking_{id} bloquait toutes les pushes après la première)
        from services.notifications.dedup_throttle import check_dedup_and_throttle

        # Normaliser status pour scope Redis : [\s-]+ → _, strip
        raw = (status or "misc").lower()
        status_slug = re.sub(r"[\s\-]+", "_", raw).strip("_") or "misc"
        throttle_scope = f"booking_{booking_id}:status_{status_slug}"
        throttle_window = 60  # 60s pour absorber retries sans bloquer status successifs
        throttle_max = 2  # 2 par status/window (anti-spam burst, laisse passer l'info)

        skip, reason = check_dedup_and_throttle(
            "company",
            company_id,
            msg["dedupe_key"],
            throttle_scope,
            throttle_window,
            throttle_max,
        )
        if skip:
            app_logger.info(
                "[event_fanout] Push skipped (company %s): %s",
                company_id,
                reason,
            )
            return

        data = dict(msg["data"])
        data["collapse_key"] = msg["collapse_key"]
        data["dedupe_key"] = msg["dedupe_key"]
        data["routing_version"] = 2
        data["routing_decision"] = "company_only"
        data["recipient_role"] = "company"
        if ctx.get("trace_id"):
            data["trace_id"] = ctx["trace_id"]
        if ctx.get("actor_role") is not None:
            data["actor_role"] = ctx["actor_role"]
        if ctx.get("actor_id") is not None:
            data["actor_id"] = ctx["actor_id"]
        data["channelId"] = _get_notification_channel(
            data.get("type", "booking_updated")
        )
        thread_id = _get_notification_thread_id(data.get("type", "booking_updated"))
        if thread_id:
            data["threadId"] = thread_id
            data["group"] = thread_id

        _log_push_fanout(
            event_type="booking_updated",
            booking_id=booking_id,
            actor_role=ctx.get("actor_role"),
            driver_id_target=0,
            company_id_target=company_id,
        )
        if os.environ.get("DEBUG_NOTIF_ROUTING", "").lower() in ("1", "true", "yes"):
            app_logger.info(
                "[event_fanout] push_company_enqueued company_id=%s trace_id=%s routing_decision=%s collapse_key=%s dedupe_key=%s",
                company_id,
                data.get("trace_id"),
                data.get("routing_decision"),
                msg.get("collapse_key"),
                msg.get("dedupe_key"),
            )
        from tasks.notification_tasks import send_push_company_notification_task

        send_push_company_notification_task.delay(  # pyright: ignore[reportFunctionMemberAccess]
            company_id=company_id,
            title=msg["title"],
            body=msg["body"],
            data=data,
        )


def fanout_booking_cancelled(
    driver_id: int,
    booking_id: int,
    *,
    booking_data: Dict[str, Any] | None = None,
) -> None:
    """Fan-out hybride pour une mission annulée.

    Template: "Course • Annulée" + "Client • heure" (jamais d'ID visible).
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

    # 2. Push notification (background) - template Company→Driver cancel
    ctx = booking_data or {"id": booking_id}
    discrete = _get_recipient_discreet_mode(driver_id=driver_id)
    msg = build_push_for_company_to_driver(
        ctx,
        change_type=CHANGE_TYPE_CANCEL,
        status="cancelled",
        discrete_mode=discrete,
    )
    data = dict(msg["data"])
    data["collapse_key"] = msg["collapse_key"]
    data["dedupe_key"] = msg["dedupe_key"]
    data["recipient_role"] = "driver"
    data["actor_role"] = "company"
    _send_push_to_driver(
        driver_id=driver_id,
        title=msg["title"],
        body=msg["body"],
        data=data,
    )


def fanout_message_new(
    driver_id: int,
    message_id: int,
    sender_name: str,
    message_preview: str,
    company_id: int | None = None,
    *,
    chat_type: str = CHAT_TYPE_DIRECT,
    thread_id: int | None = None,
) -> None:
    """Fan-out hybride pour un nouveau message (messagerie équipe/direct).

    Templates: Équipe • X (team) ou X (direct). Anti-spam: collapse_key, dedupe_key, throttle.
    """
    tid = thread_id or company_id or 0
    msg = build_chat_push(
        chat_type=chat_type,
        sender_name=sender_name,
        message_preview=message_preview,
        message_id=message_id,
        thread_id=tid,
        company_id=company_id,
        recipient_role="driver",
    )

    from services.notifications.dedup_throttle import check_dedup_and_throttle

    skip, reason = check_dedup_and_throttle(
        "driver",
        driver_id,
        msg["dedupe_key"],
        f"chat_{tid}",
        msg.get("throttle_seconds", 3),
        1,
    )
    if skip:
        app_logger.info(
            "[event_fanout] Chat push skipped (driver %s): %s",
            driver_id,
            reason,
        )
        return

    data = dict(msg["data"])
    data["collapse_key"] = msg["collapse_key"]
    data["dedupe_key"] = msg["dedupe_key"]
    data["channelId"] = _get_notification_channel("message")
    data["recipient_role"] = "driver"
    data["actor_role"] = "company"
    if company_id is not None:
        data["actor_id"] = company_id
    thread_id_val = _get_notification_thread_id("message")
    if thread_id_val:
        data["threadId"] = thread_id_val
        data["group"] = thread_id_val

    _send_push_to_driver(
        driver_id=driver_id,
        title=msg["title"],
        body=msg["body"],
        data=data,
    )


def fanout_driver_booking_reassigned(
    *,
    old_driver_id: int,
    booking_id: int,
    new_driver_id: int | None = None,
) -> None:
    """Fan-out hybride quand une mission est retirée à un chauffeur (réassignée).

    Objectif: informer l'ancien chauffeur en temps réel + notification, et l'inviter
    à rafraîchir ses courses.
    """
    # 1) Socket.IO (foreground)
    try:
        payload = _create_event_payload(
            {
                "booking_id": booking_id,
                "old_driver_id": old_driver_id,
                "new_driver_id": new_driver_id,
                "reason": "reassigned",
            },
            "booking_reassigned",
        )
        emit_driver_event(old_driver_id, "booking_reassigned", payload)
        try:
            from services.monitoring.driver_booking_metrics import (
                inc_booking_reassigned_fanout,
            )

            inc_booking_reassigned_fanout()
        except Exception:
            pass
    except Exception:
        app_logger.exception(
            "[event_fanout] Socket.IO failed for booking_reassigned (old_driver %s)",
            old_driver_id,
        )

    # 2) Push notification (background)
    _send_push_to_driver(
        driver_id=old_driver_id,
        title="Course réassignée",
        body=(
            f"La course #{booking_id} a été réassignée à un autre chauffeur. "
            "Vos courses vont être mises à jour."
        ),
        data={
            "type": "booking_reassigned",
            "booking_id": booking_id,
            "new_driver_id": new_driver_id,
            "deepLink": "atmr://bookings",
            "recipient_role": "driver",
            "actor_role": "company",
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
        title="Risque de retard",
        body=f"Retard estimé : {delay_text} sur la course #{booking_id}.",
        data={
            "type": "delay",
            "booking_id": booking_id,
            "assignment_id": assignment_id,
            "delay_minutes": float(delay_minutes),
            "deepLink": f"atmr://booking/{booking_id}?alert=delay",
            "recipient_role": "driver",
            "actor_role": "system",
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

    # 2. Push notification (background) - conditionnelle (P1: async via Celery)
    is_urgent = assignments_count > urgent_threshold
    if send_push_if_urgent and is_urgent:
        if os.environ.get("DEBUG_NOTIF_ROUTING", "").lower() in ("1", "true", "yes"):
            app_logger.info(
                "[event_fanout] push_company_enqueued company_id=%s event=dispatch_completed",
                company_id,
            )
        from tasks.notification_tasks import send_push_company_notification_task

        send_push_company_notification_task.delay(  # pyright: ignore[reportFunctionMemberAccess]
            company_id=company_id,
            title="Dispatch terminé",
            body=f"Dispatch #{dispatch_run_id} terminé : {assignments_count} assignations",
            data={
                "type": "dispatch_completed",
                "recipient_role": "company",
                "actor_role": "system",
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
    push_data_base: Dict[str, Any] = {
        "type": "urgent_alert",
        "alert_id": str(alert_id),
        "alert_type": alert_type,
        "severity": severity,
        "booking_id": booking_id,
        "driver_id": driver_id,
        "deepLink": f"atmr://alerts/{alert_id}",
        "actor_role": "system",
    }

    # Push pour company (P1: async via Celery)
    if os.environ.get("DEBUG_NOTIF_ROUTING", "").lower() in ("1", "true", "yes"):
        app_logger.info(
            "[event_fanout] push_company_enqueued company_id=%s event=urgent_alert",
            company_id,
        )
    from tasks.notification_tasks import send_push_company_notification_task

    company_push_data = {**push_data_base, "recipient_role": "company"}
    send_push_company_notification_task.delay(  # pyright: ignore[reportFunctionMemberAccess]
        company_id=company_id,
        title=f"Alerte urgente: {alert_type}",
        body=message,
        data=company_push_data,
    )

    # Push pour driver si spécifié
    if driver_id:
        driver_push_data = {**push_data_base, "recipient_role": "driver"}
        _send_push_to_driver(
            driver_id=driver_id,
            title=f"Alerte urgente: {alert_type}",
            body=message,
            data=driver_push_data,
        )


# ========================================
# Fanout executing company (partenaire)
# ========================================


def _send_notification_to_executing_company(  # pyright: ignore[reportUnusedFunction]
    executing_company_id: int,
    title: str,
    body: str,
    data: Dict[str, Any],
    *,
    send_push: bool = True,
    persist: bool = True,
) -> bool:
    """Envoie une notification a l'executing company (partenaire) via socket + push + persistence.

    Le partenaire est une Company ; on reutilise les canaux existants.
    """
    success = False
    try:
        # 1. Socket.IO
        try:
            from services.realtime.socketio import emit_company_event

            event_type = data.get("type", "booking_updated")
            emit_company_event(executing_company_id, event_type, data)
            success = True
        except Exception:
            app_logger.exception(
                "[fanout] Socket.IO failed for executing company %s",
                executing_company_id,
            )

        # 2. Push (via Celery)
        if send_push:
            try:
                from tasks.notification_tasks import (
                    send_push_company_notification_task,
                )

                push_data = dict(data)
                push_data["recipient_role"] = "company"
                push_data["actor_role"] = data.get("actor_role", "system")
                send_push_company_notification_task.delay(  # pyright: ignore[reportFunctionMemberAccess]
                    company_id=executing_company_id,
                    title=title,
                    body=body,
                    data=push_data,
                )
                app_logger.info(
                    "[push_enqueued] company_id=%s type=%s priority=high trace_id=%s",
                    executing_company_id,
                    data.get("type", "unknown"),
                    data.get("trace_id"),
                )
            except Exception:
                app_logger.exception(
                    "[fanout] Push failed for executing company %s",
                    executing_company_id,
                )

        # 3. Persistence (CompanyNotification)
        if persist:
            try:
                from services.events.institution_events import (
                    persist_company_notification,
                )

                persist_company_notification(
                    company_id=executing_company_id,
                    event_type=data.get("type", "booking_updated"),
                    title=title,
                    message=body,
                    metadata=data,
                )
            except Exception:
                app_logger.exception(
                    "[fanout] Persistence failed for executing company %s",
                    executing_company_id,
                )

    except Exception:
        app_logger.exception(
            "[fanout] _send_notification_to_executing_company failed for %s",
            executing_company_id,
        )

    return success


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

        success_count = 0
        for device_token in device_tokens:
            dt_provider = getattr(device_token, "provider", None)
            dt_platform = getattr(device_token, "platform", None)

            if dt_provider == "fcm":
                from services.notifications.firebase_push import send_fcm_silent

                result = send_fcm_silent(
                    token=device_token.token,
                    data=push_data,
                    platform=dt_platform or "android",
                )
                try:
                    from services.notifications.device_token_lifecycle import (
                        apply_push_result_to_device_token,
                    )

                    apply_push_result_to_device_token(device_token.id, result)
                except Exception:
                    pass
            else:
                result = send_push_message(
                    token=device_token.token,
                    title="",
                    body="",
                    data=push_data,
                    timeout=5,
                    use_retry=False,
                    driver_id=driver_id,
                    bypass_rate_limit=False,
                    device_token_id=device_token.id,
                )

            if result.get("ok"):
                success_count += 1
            elif (
                result.get("token_invalid")
                and not is_push_device_token_lifecycle_enabled()
            ):
                device_token.is_active = False

        try:
            from ext import db

            db.session.commit()
        except Exception:
            app_logger.exception("[silent_update] commit after push lifecycle")

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
                    "interruptionLevel": "critical",
                    "channelId": "critical",
                    "sound": {
                        "critical": True,
                        "name": "default",
                        "volume": 1.0,
                    },
                },
                timeout=10,
                use_retry=True,
                driver_id=driver_id,
                bypass_rate_limit=True,
                provider=getattr(device_token, "provider", None),
                platform=getattr(device_token, "platform", None),
                device_token_id=device_token.id,
            )

            if result.get("ok"):
                success_count += 1
            elif (
                result.get("token_invalid")
                and not is_push_device_token_lifecycle_enabled()
            ):
                device_token.is_active = False

        try:
            db.session.commit()
        except Exception:
            app_logger.exception("[critical_alert] commit after push lifecycle")

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
