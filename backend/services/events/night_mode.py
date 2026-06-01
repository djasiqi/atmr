# backend/services/events/night_mode.py

"""Service de gestion du mode nuit pour les notifications.



Règles:

- 22h-6h = Nuit (fuseau configurable, défaut Europe/Paris)

- Urgences: toujours envoyées

- Missions: seulement si chauffeur disponible (is_available=True)

- Messages chat: toujours envoyés (notification opérationnelle)

- Infos: jamais la nuit



Note importante:

- Le mode nuit affecte uniquement les notifications PUSH

- L'API reste toujours accessible (les chauffeurs peuvent voir leurs missions

  même la nuit s'ils ouvrent l'application)

"""



from __future__ import annotations



import os

from datetime import UTC, datetime, time

from typing import Any

from zoneinfo import ZoneInfo



from ext import app_logger



# Configuration

NIGHT_START = time(22, 0)  # 22h00

NIGHT_END = time(6, 0)  # 06h00

DEFAULT_NIGHT_MODE_TZ = os.getenv("NIGHT_MODE_TZ", "Europe/Paris")





def _resolve_timezone(tz_name: str | None = None) -> ZoneInfo:

    name = (tz_name or DEFAULT_NIGHT_MODE_TZ).strip() or DEFAULT_NIGHT_MODE_TZ

    try:

        return ZoneInfo(name)

    except Exception:

        app_logger.warning(

            "[night_mode] Invalid timezone '%s', fallback Europe/Paris", name

        )

        return ZoneInfo("Europe/Paris")





def is_night_time(now: datetime | None = None, tz_name: str | None = None) -> bool:

    """Détermine si on est en période nocturne dans le fuseau configuré.



    Args:

        now: Datetime à tester (UTC-aware ou naive interprété en UTC)

        tz_name: Fuseau IANA (défaut: NIGHT_MODE_TZ / Europe/Paris)



    Returns:

        True si entre 22h et 6h dans le fuseau local

    """

    tz = _resolve_timezone(tz_name)

    if now is None:

        now = datetime.now(UTC)

    elif now.tzinfo is None:

        now = now.replace(tzinfo=UTC)



    local_now = now.astimezone(tz)

    current_time = local_now.time()



    # Cas nuit: start > end (ex: 22h-6h = passe minuit)

    if NIGHT_START > NIGHT_END:

        return current_time >= NIGHT_START or current_time < NIGHT_END



    return NIGHT_START <= current_time < NIGHT_END





def should_send_night_notification(

    notification_type: str,

    driver_id: int | None = None,

    *,

    tz_name: str | None = None,

) -> bool:

    """Détermine si une notification peut être envoyée la nuit.



    Règles:

    1. Urgences (urgent_alert, accident, emergency): TOUJOURS

    2. Missions (booking, booking_updated, delay): SI chauffeur disponible (is_available=True)

    3. Messages chat (message, team_chat_message, chat_message): TOUJOURS

    4. Infos (dispatch_completed, stats): JAMAIS



    Note importante:

    - Cette fonction affecte uniquement les notifications PUSH

    - L'API reste toujours accessible (GET /api/driver/me/bookings retourne toujours les missions)



    Args:

        notification_type: Type de notification (ex: "booking", "urgent_alert")

        driver_id: ID du chauffeur (requis pour vérifier statut de disponibilité)

        tz_name: Fuseau IANA optionnel (défaut global LIRIE)



    Returns:

        True si la notification peut être envoyée, False sinon

    """

    if not is_night_time(tz_name=tz_name):

        return True



    urgent_types = ["urgent_alert", "accident", "emergency", "critical"]

    if notification_type in urgent_types:

        app_logger.info(

            "[night_mode] Notification urgente autorisée la nuit: type=%s",

            notification_type,

        )

        return True



    mission_types = [

        "booking",

        "booking_assigned",

        "booking_updated",

        "booking_cancelled",

        "booking_reassigned",

        "delay",

    ]

    if notification_type in mission_types:

        if driver_id is None:

            app_logger.info(

                "[night_mode] Notification de mission bloquée la nuit (driver_id manquant, type=%s)",

                notification_type,

            )

            return False



        from models import Driver



        driver = Driver.query.get(driver_id)

        is_available = driver and getattr(driver, "is_available", False)



        log_msg = "[night_mode] Notification de mission {} la nuit (chauffeur {} {})"

        if is_available:

            app_logger.info(log_msg.format("autorisée", driver_id, "disponible"))

        else:

            reason = "introuvable" if not driver else "indisponible"

            app_logger.info(

                log_msg.format("bloquée", driver_id, reason)

                + " - Protection du sommeil"

            )



        return bool(is_available)



    chat_types = [

        "message",

        "team_chat_message",

        "chat_message",

        "chat",

    ]

    if notification_type in chat_types:

        app_logger.info(

            "[night_mode] Notification chat autorisée la nuit: type=%s",

            notification_type,

        )

        return True



    blocked_types = [

        "dispatch_completed",

        "stats",

        "info",

    ]

    log_action = "refusée la nuit"



    if notification_type in blocked_types:

        app_logger.info(

            f"[night_mode] Notification type={notification_type} {log_action} (non-urgente)"

        )

    else:

        app_logger.warning(

            f"[night_mode] Type inconnu '{notification_type}', {log_action} par défaut"

        )



    return False





def get_night_mode_status(tz_name: str | None = None) -> dict[str, Any]:

    """Récupère le statut actuel du mode nuit.



    Utile pour monitoring et debugging.



    Returns:

        Dict contenant:

        - is_night: bool

        - current_time: str (format HH:MM)

        - night_start: str (format HH:MM)

        - night_end: str (format HH:MM)

        - timezone: str

    """

    tz = _resolve_timezone(tz_name)

    now = datetime.now(UTC)

    local_now = now.astimezone(tz)



    return {

        "is_night": is_night_time(now, tz_name=tz.key),

        "current_time": local_now.strftime("%H:%M"),

        "night_start": NIGHT_START.strftime("%H:%M"),

        "night_end": NIGHT_END.strftime("%H:%M"),

        "timezone": tz.key,

    }

