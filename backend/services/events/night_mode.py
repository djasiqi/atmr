# backend/services/events/night_mode.py
"""Service de gestion du mode nuit pour les notifications.

Règles:
- 22h-6h = Nuit
- Urgences: toujours envoyées
- Missions: seulement si chauffeur disponible (is_available=True)
- Messages/Infos: jamais la nuit

Note importante:
- Le mode nuit affecte uniquement les notifications PUSH
- L'API reste toujours accessible (les chauffeurs peuvent voir leurs missions
  même la nuit s'ils ouvrent l'application)
"""

from datetime import datetime, time
from typing import Any

from ext import app_logger

# Configuration
NIGHT_START = time(22, 0)  # 22h00
NIGHT_END = time(6, 0)  # 06h00


def is_night_time(now: datetime | None = None) -> bool:
    """Détermine si on est en période nocturne.

    Args:
        now: Datetime à tester (par défaut: datetime.now())

    Returns:
        True si entre 22h et 6h
    """
    if now is None:
        now = datetime.now()

    current_time = now.time()

    # Cas simple: start < end (ex: 10h-18h)
    # Cas nuit: start > end (ex: 22h-6h = passe minuit)
    if NIGHT_START > NIGHT_END:
        # Entre 22h et minuit OU entre minuit et 6h
        return current_time >= NIGHT_START or current_time < NIGHT_END

    # Entre start et end (cas normal)
    return NIGHT_START <= current_time < NIGHT_END


def should_send_night_notification(
    notification_type: str,
    driver_id: int | None = None,
) -> bool:
    """Détermine si une notification peut être envoyée la nuit.

    Règles:
    1. Urgences (urgent_alert, accident, emergency): TOUJOURS
    2. Missions (booking, booking_updated, delay): SI chauffeur disponible (is_available=True)
       - Protection du sommeil : si le chauffeur n'est pas disponible, la notification est bloquée
       - Exception : les chauffeurs disponibles (en service) reçoivent les notifications même la nuit
    3. Messages (message, team_chat_message): JAMAIS
    4. Infos (dispatch_completed, stats): JAMAIS

    Note importante:
    - Cette fonction affecte uniquement les notifications PUSH
    - L'API reste toujours accessible (GET /api/driver/me/bookings retourne toujours les missions)

    Args:
        notification_type: Type de notification (ex: "booking", "urgent_alert")
        driver_id: ID du chauffeur (requis pour vérifier statut de disponibilité)

    Returns:
        True si la notification peut être envoyée, False sinon
    """
    # Si ce n'est pas la nuit, toujours OK
    if not is_night_time():
        return True

    # Nuit : vérifier le type

    # 1. Urgences : toujours envoyées
    urgent_types = ["urgent_alert", "accident", "emergency", "critical"]
    if notification_type in urgent_types:
        app_logger.info(
            "[night_mode] Notification urgente autorisée la nuit: type=%s",
            notification_type,
        )
        return True

    # 2. Missions : bloquer la nuit (protection du sommeil)
    # Les chauffeurs peuvent toujours voir leurs missions via l'API même la nuit
    # mais les notifications push sont bloquées pour préserver le sommeil
    mission_types = [
        "booking",
        "booking_assigned",
        "booking_updated",
        "booking_cancelled",
        "booking_reassigned",
        "delay",
    ]
    if notification_type in mission_types:
        # ✅ Protection du sommeil : bloquer toutes les notifications de missions la nuit
        # Exception : si le chauffeur est disponible (is_available), cela signifie qu'il
        # est probablement en service et souhaite recevoir les notifications
        if driver_id is None:
            app_logger.info(
                "[night_mode] Notification de mission bloquée la nuit (driver_id manquant, type=%s)",
                notification_type,
            )
            return False

        from models import Driver

        driver = Driver.query.get(driver_id)
        # ✅ Utiliser is_available au lieu de is_on_duty (qui n'existe pas dans le modèle)
        # is_available indique si le chauffeur est disponible pour recevoir des missions
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

    # 3. Messages et infos : jamais la nuit (refus par défaut)
    blocked_types = [
        "message",
        "team_chat_message",
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


def get_night_mode_status() -> dict[str, Any]:
    """Récupère le statut actuel du mode nuit.

    Utile pour monitoring et debugging.

    Returns:
        Dict contenant:
        - is_night: bool
        - current_time: str (format HH:MM)
        - night_start: str (format HH:MM)
        - night_end: str (format HH:MM)
    """
    now = datetime.now()

    return {
        "is_night": is_night_time(now),
        "current_time": now.strftime("%H:%M"),
        "night_start": NIGHT_START.strftime("%H:%M"),
        "night_end": NIGHT_END.strftime("%H:%M"),
    }
