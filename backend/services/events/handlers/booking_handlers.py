from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)


def handle_booking_assigned(event: dict[str, Any]) -> None:
    from ext import db
    from models import Booking
    from services.notifications.core import notify_booking_assigned

    booking_id = event.get("booking_id")
    if booking_id is None:
        return

    with suppress(Exception):
        db.session.rollback()

    booking = db.session.get(Booking, int(booking_id))
    if booking:
        notify_booking_assigned(booking)


def handle_booking_created(event: dict[str, Any]) -> None:
    """Handler enrichi pour BookingCreatedEvent.

    Actions:
    - Notifie la company via SocketIO (nouvelle réservation en attente)
    - Collecte des métriques (via metrics_handler si enregistré)
    """
    booking_id = event.get("booking_id")
    company_id = event.get("company_id")

    # Logging existant
    logger.info(
        "[EventBus] BookingCreatedEvent received booking_id=%s company_id=%s",
        booking_id,
        company_id,
    )

    # ✅ NOUVEAU : Notifier la company via SocketIO
    if company_id:
        try:
            from services.realtime.socketio import emit_company_event

            emit_company_event(
                company_id=int(company_id),
                event="new_booking_pending",
                payload={"booking_id": int(booking_id) if booking_id else None},
            )
            logger.debug(
                "[EventBus] Notified company %s about new booking %s",
                company_id,
                booking_id,
            )
        except (ValueError, TypeError) as e:
            # Erreurs de validation attendues : conversion de types
            logger.warning(
                "[EventBus] Failed to notify company about booking created (validation error: %s): %s",
                type(e).__name__,
                e,
            )
        except (ConnectionError, OSError) as e:
            # Erreurs réseau attendues : Socket.IO indisponible
            logger.warning(
                "[EventBus] Failed to notify company about booking created (network error: %s): %s",
                type(e).__name__,
                e,
            )
        except Exception:
            # Handler "safe" : ne pas faire échouer le système si notification échoue
            logger.exception(
                "[EventBus] Failed to notify company about booking created"
            )

    # ✅ NOUVEAU : Métriques collectées via metrics_handler (si enregistré)
    # Le metrics_handler sera appelé automatiquement après ce handler


def handle_booking_updated(event: dict[str, Any]) -> None:
    """Handler pour BookingUpdatedEvent.

    Actions:
    - Notifie le driver via SocketIO (mise à jour de booking) si l'update ne vient
      pas de lui (éviter les "self-notifications")
    - Notifie l'entreprise via SocketIO + Push quand l'update vient du chauffeur
    """
    booking_id = event.get("booking_id")
    driver_id = event.get("driver_id")
    actor_role = event.get("actor_role")
    actor_id = event.get("actor_id")

    if not booking_id or not driver_id:
        logger.warning(
            "[EventBus] BookingUpdatedEvent missing booking_id or driver_id: %s",
            event,
        )
        return

    try:
        from ext import db
        from models import Booking
        from services.events.fanout import (
            fanout_booking_updated,
            fanout_booking_updated_to_company,
        )

        # Récupérer le booking pour la notification
        with suppress(Exception):
            db.session.rollback()

        booking = db.session.get(Booking, int(booking_id))
        if booking:
            # Préparer un payload riche (utile côté app entreprise)
            try:
                booking_data = booking.to_dict() if hasattr(booking, "to_dict") else {}
            except Exception:
                booking_data = {}
            booking_data.setdefault("id", int(getattr(booking, "id", booking_id)))
            booking_data["booking_id"] = int(getattr(booking, "id", booking_id))
            booking_data["driver_id"] = int(
                getattr(booking, "driver_id", driver_id) or driver_id
            )
            booking_data["company_id"] = int(
                getattr(booking, "company_id", event.get("company_id") or 0) or 0
            )
            status_raw = getattr(
                getattr(booking, "status", None),
                "value",
                getattr(booking, "status", None),
            )
            booking_data["status"] = (
                str(status_raw).lower() if status_raw is not None else None
            )
            booking_data["actor_role"] = actor_role
            booking_data["actor_id"] = actor_id
            booking_data["changes"] = event.get("changes")

            company_id = int(
                getattr(booking, "company_id", event.get("company_id") or 0) or 0
            )
            status_val = booking_data.get("status")  # pour P0: detecter driver progress

            # ✅ P0: Si le chauffeur est l'initiateur OU si le statut indique une action chauffeur
            # (en_route / in_progress / completed / return_completed), on notifie UNIQUEMENT
            # l'entreprise. Jamais de push au chauffeur pour ses propres actions.
            is_driver_actor = (
                actor_role == "driver"
                and actor_id is not None
                and int(actor_id) == int(driver_id)
            ) or (
                str(status_val or "").lower()
                in {"en_route", "in_progress", "completed", "return_completed"}
            )
            if is_driver_actor:
                if company_id:
                    fanout_booking_updated_to_company(
                        company_id=company_id,
                        booking_id=int(booking_id),
                        booking_data=booking_data,
                        send_push=True,
                    )
                logger.debug(
                    "[EventBus] BookingUpdatedEvent from driver / driver progress -> company only (no push driver), booking %s",
                    booking_id,
                )
                return

            # Sinon: on notifie le chauffeur (et on garde l'entreprise à jour via Socket.IO)
            changes = event.get("changes") or {}
            changes_keys = set(changes.keys()) if isinstance(changes, dict) else set()
            should_send_push = bool(
                changes_keys.intersection(
                    {"scheduled_time", "pickup_location", "dropoff_location", "notes"}
                )
            ) or booking_data.get("status") in {"cancelled", "canceled"}

            fanout_booking_updated(
                driver_id=int(driver_id),
                booking_id=int(booking_id),
                booking_data=booking_data,
                send_push=should_send_push,
            )
            if company_id:
                fanout_booking_updated_to_company(
                    company_id=company_id,
                    booking_id=int(booking_id),
                    booking_data=booking_data,
                    # ✅ si on envoie une notif au chauffeur pour un changement important,
                    # on envoie aussi une notif entreprise (multi-device).
                    send_push=should_send_push,
                )
            logger.debug(
                "[EventBus] Notified driver %s about booking update %s (actor_role=%s)",
                driver_id,
                booking_id,
                actor_role,
            )
    except (ValueError, TypeError) as e:
        # Erreurs de validation attendues : conversion de types
        logger.warning(
            "[EventBus] Failed to notify driver about booking update (validation error: %s): %s",
            type(e).__name__,
            e,
        )
    except (ConnectionError, OSError) as e:
        # Erreurs réseau attendues : Socket.IO indisponible
        logger.warning(
            "[EventBus] Failed to notify driver about booking update (network error: %s): %s",
            type(e).__name__,
            e,
        )
    except Exception:
        # Handler "safe" : ne pas faire échouer le système si notification échoue
        logger.exception("[EventBus] Failed to notify driver about booking update")


def handle_booking_cancelled(event: dict[str, Any]) -> None:
    """Handler pour BookingCancelledEvent.

    Actions:
    - Notifie le driver via SocketIO (annulation de booking)
    """
    booking_id = event.get("booking_id")
    driver_id = event.get("driver_id")

    if not booking_id or not driver_id:
        logger.warning(
            "[EventBus] BookingCancelledEvent missing booking_id or driver_id: %s",
            event,
        )
        return

    try:
        from ext import db
        from models import Booking
        from services.notifications.core import notify_booking_cancelled

        booking_data = None
        with suppress(Exception):
            db.session.rollback()
        booking = db.session.get(Booking, int(booking_id))
        if booking and hasattr(booking, "to_dict"):
            with suppress(Exception):
                booking_data = booking.to_dict()

        notify_booking_cancelled(
            int(driver_id),
            int(booking_id),
            booking_data=booking_data,
        )
        logger.debug(
            "[EventBus] Notified driver %s about booking cancellation %s",
            driver_id,
            booking_id,
        )
    except (ValueError, TypeError) as e:
        # Erreurs de validation attendues : conversion de types
        logger.warning(
            "[EventBus] Failed to notify driver about booking cancellation (validation error: %s): %s",
            type(e).__name__,
            e,
        )
    except (ConnectionError, OSError) as e:
        # Erreurs réseau attendues : Socket.IO indisponible
        logger.warning(
            "[EventBus] Failed to notify driver about booking cancellation (network error: %s): %s",
            type(e).__name__,
            e,
        )
    except Exception:
        # Handler "safe" : ne pas faire échouer le système si notification échoue
        logger.exception(
            "[EventBus] Failed to notify driver about booking cancellation"
        )
