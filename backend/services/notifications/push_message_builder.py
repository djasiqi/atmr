# backend/services/notifications/push_message_builder.py
"""Builder unique pour les messages push métier (nom client + contexte).

Produit title/body lisibles (nom client, date/heure, lieu court) et data
contenant booking_id, event, client_display_name (optionnel), deep_link.
Supporte le mode discret (pas de nom sur lockscreen).
"""

from __future__ import annotations

from typing import Any, Literal

# Événements canoniques pour le payload
EVENT_ASSIGNED = "ASSIGNED"
EVENT_STATUS_UPDATED = "STATUS_UPDATED"
EVENT_COMPLETED = "COMPLETED"
EVENT_CANCELLED = "CANCELLED"
EVENT_REASSIGNED = "REASSIGNED"

RecipientRole = Literal["driver", "company"]

# Longueur minimale d'une chaîne ISO datetime pour extraire HH:MM (positions 11:16)
_MIN_ISO_STR_LEN_FOR_TIME = 16


def _as_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _get_booking_id(booking_or_context: Any) -> int:
    """Retourne l'ID de la course depuis un modèle Booking ou un dict."""
    if hasattr(booking_or_context, "id"):
        return int(booking_or_context.id) if booking_or_context.id is not None else 0
    if isinstance(booking_or_context, dict):
        return int(booking_or_context.get("id") or booking_or_context.get("booking_id") or 0)
    return 0


def _get_client_display_name(booking_or_context: Any) -> str:
    """Nom affichable du client (booking.client + user ou customer_name). Sans N+1 si appelant a eager-loadé."""
    if hasattr(booking_or_context, "customer_full_name"):
        return _as_str(booking_or_context.customer_full_name, "Client")
    if hasattr(booking_or_context, "customer_name"):
        return _as_str(booking_or_context.customer_name, "Client")
    if isinstance(booking_or_context, dict):
        return _as_str(
            booking_or_context.get("client_display_name")
            or booking_or_context.get("client_name")
            or booking_or_context.get("customer_name"),
            "Client",
        )
    return "Client"


def _get_time_short(booking_or_context: Any) -> str:
    """Heure courte type 13:00 ou 'Aujourd'hui 13:00'."""
    if isinstance(booking_or_context, dict):
        tf = booking_or_context.get("time_formatted") or booking_or_context.get("time_formatted_local")
        if tf:
            return _as_str(tf)
        ts = booking_or_context.get("scheduled_time")
        if ts and isinstance(ts, str) and "T" in ts:
            return ts.replace("Z", "")[11:16] if len(ts) >= _MIN_ISO_STR_LEN_FOR_TIME else ""
    if hasattr(booking_or_context, "scheduled_time") and booking_or_context.scheduled_time:
        from shared.time_utils import split_date_time_local

        dt = booking_or_context.scheduled_time
        _, time_local = split_date_time_local(dt) if dt else (None, None)
        return _as_str(time_local)
    return ""


def _get_location_short(value: Any, max_len: int = 32) -> str:
    raw = _as_str(value).replace("\n", " ").strip()
    if not raw:
        return ""
    return f"{raw[:max_len]}…" if len(raw) > max_len else raw


def _get_pickup_short(booking_or_context: Any) -> str:
    if hasattr(booking_or_context, "pickup_location"):
        return _get_location_short(booking_or_context.pickup_location)
    if isinstance(booking_or_context, dict):
        return _get_location_short(
            booking_or_context.get("pickup_location") or booking_or_context.get("pickup_address")
        )
    return ""


def _get_dropoff_short(booking_or_context: Any) -> str:
    if hasattr(booking_or_context, "dropoff_location"):
        return _get_location_short(booking_or_context.dropoff_location)
    if isinstance(booking_or_context, dict):
        return _get_location_short(
            booking_or_context.get("dropoff_location") or booking_or_context.get("dropoff_address")
        )
    return ""


def _get_amount_chf(booking_or_context: Any) -> str:
    if hasattr(booking_or_context, "amount"):
        a = booking_or_context.amount
        return f"{float(a):.0f} CHF" if a is not None else ""
    if isinstance(booking_or_context, dict):
        a = booking_or_context.get("amount")
        return f"{float(a):.0f} CHF" if a is not None else ""
    return ""


def _build_deep_link(booking_id: int, recipient_role: RecipientRole) -> str:
    if recipient_role == "company":
        return f"atmr://enterprise/rides/{booking_id}"
    return f"atmr://booking/{booking_id}"


def build_push_message(
    event: str,
    booking_or_context: Any,
    recipient_role: RecipientRole,
    *,
    actor: dict[str, Any] | None = None,
    discrete_mode: bool = False,
    status: str | None = None,
    changes_preview: str | None = None,
) -> dict[str, Any]:
    """Construit un message push métier {title, body, data}.

    Args:
        event: ASSIGNED | STATUS_UPDATED | COMPLETED | CANCELLED | REASSIGNED
        booking_or_context: modèle Booking ou dict (client_name, pickup_location, etc.)
        recipient_role: "driver" | "company"
        actor: optionnel {"first_name": "Driss"} pour statut (sinon "Un chauffeur")
        discrete_mode: True => pas de nom client dans title/body (lockscreen)
        status: pour STATUS_UPDATED (ex. "en_route", "completed")
        changes_preview: pour STATUS_UPDATED, ligne courte type "Départ: Ernest-Pictet 9"

    Returns:
        {"title": str, "body": str, "data": dict} avec data = booking_id, event, client_display_name?, deep_link
    """
    bid = _get_booking_id(booking_or_context)
    client = "" if discrete_mode else _get_client_display_name(booking_or_context)
    time_short = _get_time_short(booking_or_context)
    pickup_short = _get_pickup_short(booking_or_context)
    dropoff_short = _get_dropoff_short(booking_or_context)
    amount_chf = _get_amount_chf(booking_or_context)
    deep_link = _build_deep_link(bid, recipient_role)

    # type legacy pour compat mobile (notification_type)
    type_map = {
        EVENT_ASSIGNED: "booking_assigned" if recipient_role == "company" else "booking",
        EVENT_STATUS_UPDATED: "booking_updated",
        EVENT_COMPLETED: "booking_updated",
        EVENT_CANCELLED: "booking_cancelled",
        EVENT_REASSIGNED: "booking_reassigned",
    }
    notification_type = type_map.get(event, "booking_updated")

    data: dict[str, Any] = {
        "booking_id": bid,
        "event": event,
        "deep_link": deep_link,
        "deepLink": deep_link,  # compat mobile (camelCase)
        "type": notification_type,
    }
    if not discrete_mode:
        data["client_display_name"] = _get_client_display_name(booking_or_context)
    if status:
        data["status"] = status

    title = ""
    body = ""

    if event == EVENT_ASSIGNED:
        title = "Assignation : Nouvelle course assignée"
        if discrete_mode:
            body = "Nouvelle course assignée. Ouvrez l'application pour les détails."
        elif recipient_role == "driver":
            parts = [f"Vous êtes assigné pour le transport de {client}"]
            if time_short:
                parts.append(time_short)
            if dropoff_short:
                parts.append(f"→ {dropoff_short}")
            body = " • ".join(parts) if parts else f"Vous êtes assigné pour le transport de {client}."
        else:
            parts = [f"Course assignée • {client}"]
            if time_short:
                parts.append(time_short)
            if dropoff_short:
                parts.append(dropoff_short)
            body = " • ".join(parts) if parts else f"Course assignée • {client}."

    elif event == EVENT_STATUS_UPDATED:
        title = "Statut : Mise à jour de course"
        actor_name = "Un chauffeur"
        if actor:
            fn = _as_str(actor.get("first_name"))
            ln = _as_str(actor.get("last_name"))
            actor_name = f"{fn} {ln}".strip() or actor.get("username") or "Un chauffeur"
        if discrete_mode:
            body = "Mise à jour de course. Ouvrez l'application pour les détails."
        elif status == "en_route":
            if client:
                loc = f"Départ: {pickup_short}" if pickup_short else dropoff_short
                body = f"{actor_name} est en route pour {client}"
                if loc:
                    body += f" • {loc}"
                body += "."
            else:
                body = changes_preview or "Chauffeur en route."
        elif status == "completed":
            title = "Terminé : Course terminée"
            if client:
                body = f"Course terminée • {client}"
                if amount_chf:
                    body += f" • {amount_chf}"
                body += "."
            else:
                body = amount_chf or "Course terminée."
        else:
            body = changes_preview or f"Mise à jour pour {client}." if client else "Mise à jour de course."

    elif event == EVENT_COMPLETED:
        title = "Terminé : Course terminée"
        if discrete_mode:
            body = "Course terminée. Ouvrez l'application pour les détails."
        else:
            body = f"Course terminée • {client}" + (f" • {amount_chf}" if amount_chf else "") + "."

    elif event == EVENT_CANCELLED:
        title = "Course annulée"
        if discrete_mode:
            body = "Une course a été annulée."
        else:
            body = f"La course {client} a été annulée." if client else "La course a été annulée."

    elif event == EVENT_REASSIGNED:
        title = "Course réassignée"
        if discrete_mode:
            body = "La course a été réassignée à un autre chauffeur."
        else:
            body = f"La course {client} a été réassignée à un autre chauffeur." if client else "La course a été réassignée."
        data["deep_link"] = data["deepLink"] = "atmr://bookings"

    else:
        title = "Mise à jour de course"
        body = changes_preview or (f"{client} • " if client else "") or "Ouvrez l'application pour les détails."

    return {"title": title, "body": body, "data": data}
