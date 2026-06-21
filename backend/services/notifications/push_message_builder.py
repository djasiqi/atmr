# backend/services/notifications/push_message_builder.py
"""Builder unique pour les messages push métier (nom client + contexte).

Produit title/body lisibles (nom client, date/heure, lieu court) et data
contenant booking_id, event, client_display_name (optionnel), deep_link.
Supporte le mode discret (pas de nom sur lockscreen).

Refonte 2026: templates Driver→Company et Company→Driver distincts,
anti-spam (collapse_key, dedupe_key, throttle), jamais d'ID visible.
"""

from __future__ import annotations

from typing import Any, Literal

# Événements canoniques pour le payload
EVENT_ASSIGNED = "ASSIGNED"
EVENT_STATUS_UPDATED = "STATUS_UPDATED"
EVENT_COMPLETED = "COMPLETED"
EVENT_CANCELLED = "CANCELLED"
EVENT_REASSIGNED = "REASSIGNED"
EVENT_TIME_CHANGE = "TIME_CHANGE"
EVENT_ADDRESS_CHANGE = "ADDRESS_CHANGE"

RecipientRole = Literal["driver", "company"]

# Types de changement (Company→Driver)
CHANGE_TYPE_ASSIGN = "assign"
CHANGE_TYPE_STATUS = "status"
CHANGE_TYPE_TIME_CHANGE = "time_change"
CHANGE_TYPE_ADDRESS_CHANGE = "address_change"
CHANGE_TYPE_DETAILS_CHANGE = "details_change"
CHANGE_TYPE_CANCEL = "cancel"

# Mapping unique statut → label FR (jamais d'ID dans title/body)
STATUS_LABEL_MAP_FR: dict[str, str] = {
    "pending": "En attente",
    "accepted": "Acceptée",
    "assigned": "Assignée",
    "en_route": "En route",
    "in_progress": "À bord",
    "completed": "Terminée",
    "return_completed": "Retour terminé",
    "canceled": "Annulée",
    "cancelled": "Annulée",
}

# Longueur minimale d'une chaîne ISO datetime pour extraire HH:MM (positions 11:16)
_MIN_ISO_STR_LEN_FOR_TIME = 16
_ADDRESS_MAX_LEN = 40
# Limite Android UX: body tronqué si trop long (60-70 chars recommandés pour pickup→dropoff)
_ROUTE_LINE_MAX_LEN = 65
# Chat: preview max 90 chars, throttle 3s
_CHAT_PREVIEW_MAX_LEN = 90
_CHAT_THROTTLE_SECONDS = 3

# Types de chat (messagerie)
CHAT_TYPE_TEAM = "team"
CHAT_TYPE_DIRECT = "direct"
CHAT_TYPE_BOOKING = "booking"


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
        return int(
            booking_or_context.get("id") or booking_or_context.get("booking_id") or 0
        )
    return 0


def _get_client_display_name(booking_or_context: Any) -> str:
    """Nom affichable du client. Fallback "Client", jamais "Non spécifié"."""
    raw = ""
    if hasattr(booking_or_context, "customer_full_name"):
        raw = _as_str(booking_or_context.customer_full_name)
    elif hasattr(booking_or_context, "customer_name"):
        raw = _as_str(booking_or_context.customer_name)
    elif isinstance(booking_or_context, dict):
        raw = _as_str(
            booking_or_context.get("client_display_name")
            or booking_or_context.get("client_name")
            or booking_or_context.get("customer_name")
        )
    if not raw or raw.lower() == "non spécifié":
        return "Client"
    return raw


def _get_driver_display_name(booking_or_context: Any) -> str:
    """Nom affichable du chauffeur. Fallback "Chauffeur"."""
    if isinstance(booking_or_context, dict):
        driver = booking_or_context.get("driver")
        if isinstance(driver, dict):
            fn = _as_str(driver.get("first_name"))
            ln = _as_str(driver.get("last_name"))
            full = f"{fn} {ln}".strip()
            if full:
                return full
            return (
                _as_str(driver.get("full_name"))
                or _as_str(driver.get("username"))
                or "Chauffeur"
            )
        return _as_str(booking_or_context.get("driver_name"), "Chauffeur")
    if hasattr(booking_or_context, "driver") and booking_or_context.driver:
        d = booking_or_context.driver
        u = getattr(d, "user", None)
        if u:
            fn = _as_str(getattr(u, "first_name", ""))
            ln = _as_str(getattr(u, "last_name", ""))
            full = f"{fn} {ln}".strip()
            if full:
                return full
            return _as_str(getattr(u, "username", ""), "Chauffeur")
    return "Chauffeur"


def _shorten_address_for_display(
    location: str | None,
    *,
    medical_facility: str | None = None,
    hospital_service: str | None = None,
    max_len: int = _ADDRESS_MAX_LEN,
) -> str:
    """Adresse courte: avant première virgule + (HUG Radiologie) si medical_facility/service."""
    raw = _as_str(location).replace("\n", " ").strip()
    if not raw:
        return ""
    base = raw.split(",")[0].strip() if "," in raw else raw
    base = base[:max_len] + ("…" if len(base) > max_len else "")
    suffix_parts = []
    if medical_facility:
        suffix_parts.append(_as_str(medical_facility))
    if hospital_service:
        suffix_parts.append(f"({_as_str(hospital_service)})")
    if suffix_parts:
        suffix = " ".join(suffix_parts).strip()
        if suffix:
            return f"{base} ({suffix})" if base else suffix
    return base


def _get_time_short(booking_or_context: Any) -> str:
    """Heure courte type 13:00 ou 'Aujourd'hui 13:00'."""
    if isinstance(booking_or_context, dict):
        tf = booking_or_context.get("time_formatted") or booking_or_context.get(
            "time_formatted_local"
        )
        if tf:
            return _as_str(tf)
        ts = booking_or_context.get("scheduled_time")
        if ts and isinstance(ts, str) and "T" in ts:
            return (
                ts.replace("Z", "")[11:16]
                if len(ts) >= _MIN_ISO_STR_LEN_FOR_TIME
                else ""
            )
    if (
        hasattr(booking_or_context, "scheduled_time")
        and booking_or_context.scheduled_time
    ):
        from shared.time_utils import split_date_time_local

        dt = booking_or_context.scheduled_time
        _, time_local = split_date_time_local(dt) if dt else (None, None)
        return _as_str(time_local)
    return ""


def _get_location_short(value: Any, max_len: int = 32) -> str:  # pyright: ignore[reportUnusedFunction]
    raw = _as_str(value).replace("\n", " ").strip()
    if not raw:
        return ""
    return f"{raw[:max_len]}…" if len(raw) > max_len else raw


def _get_medical_facility(booking_or_context: Any) -> str | None:
    raw = None
    if isinstance(booking_or_context, dict):
        raw = booking_or_context.get("medical_facility")
    elif hasattr(booking_or_context, "medical_facility"):
        raw = getattr(booking_or_context, "medical_facility", None)
    s = _as_str(raw)
    if not s or s.lower() == "non spécifié":
        return None
    return s


def _get_hospital_service(booking_or_context: Any) -> str | None:
    raw = None
    if isinstance(booking_or_context, dict):
        raw = booking_or_context.get("hospital_service")
    elif hasattr(booking_or_context, "hospital_service"):
        raw = getattr(booking_or_context, "hospital_service", None)
    s = _as_str(raw)
    if not s or s.lower() == "non spécifié":
        return None
    return s


def _format_route_line(
    pickup_short: str,
    dropoff_short: str,
    *,
    max_total: int = _ROUTE_LINE_MAX_LEN,
) -> str:
    """Format pickup → dropoff en limitant la longueur totale (Android UX).

    Sur certains Android, le body est tronqué si trop long.
    Ex: "Av. Ernest-Pictet 9 → HUG (Radiologie)" ✅
    Pas: "Avenue Ernest-Pictet 9, 1203 Genève → HUG, Rue Gabrielle-Perret-Gentil 4" ❌
    """
    combined = f"{pickup_short} → {dropoff_short}"
    if len(combined) <= max_total:
        return combined
    sep = " → "
    budget = max_total - len(sep)
    max_each = budget // 2
    p = (
        pickup_short[: max_each - 1] + "…"
        if len(pickup_short) > max_each
        else pickup_short
    )
    d = (
        dropoff_short[: max_each - 1] + "…"
        if len(dropoff_short) > max_each
        else dropoff_short
    )
    return f"{p}{sep}{d}"


def _get_pickup_short(booking_or_context: Any) -> str:
    loc = None
    if hasattr(booking_or_context, "pickup_location"):
        loc = booking_or_context.pickup_location
    elif isinstance(booking_or_context, dict):
        loc = booking_or_context.get("pickup_location") or booking_or_context.get(
            "pickup_address"
        )
    if not loc:
        return ""
    return _shorten_address_for_display(loc, max_len=_ADDRESS_MAX_LEN)


def _get_dropoff_short(booking_or_context: Any) -> str:
    loc = None
    if hasattr(booking_or_context, "dropoff_location"):
        loc = booking_or_context.dropoff_location
    elif isinstance(booking_or_context, dict):
        loc = booking_or_context.get("dropoff_location") or booking_or_context.get(
            "dropoff_address"
        )
    if not loc:
        return ""
    mf = _get_medical_facility(booking_or_context)
    hs = _get_hospital_service(booking_or_context)
    return _shorten_address_for_display(
        loc, medical_facility=mf, hospital_service=hs, max_len=_ADDRESS_MAX_LEN
    )


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
        return f"lirie://enterprise/rides/{booking_id}"
    return f"lirie://booking/{booking_id}"


def _get_status_label(status: str | None) -> str:
    """Retourne le label FR du statut. Jamais vide."""
    if not status:
        return "Mise à jour"
    s = str(status).lower().strip()
    return STATUS_LABEL_MAP_FR.get(s, "Mise à jour")


def _extract_time_hhmm(value: Any) -> str:  # pyright: ignore[reportUnusedFunction]
    """Extrait HH:MM d'un datetime ISO ou string."""
    if value is None:
        return ""
    s = str(value)
    if "T" in s and len(s) >= _MIN_ISO_STR_LEN_FOR_TIME:
        return s.replace("Z", "")[11:16]
    return ""


# ---------- Chat / Messagerie ----------


def normalize_preview(content: str | None, max_len: int = _CHAT_PREVIEW_MAX_LEN) -> str:
    """Normalise le preview d'un message: trim, \\n→espace, collapse spaces.

    Jamais vide: fallback "Nouveau message". Jamais "Non spécifié".
    """
    if content is None:
        return "Nouveau message"
    s = str(content).strip()
    s = s.replace("\n", " ").replace("\r", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    s = s.strip()
    if not s or s.lower() == "non spécifié":
        return "Nouveau message"
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def safe_sender_name(sender_name: str | None, *, fallback: str = "Message") -> str:
    """Nom affichable de l'expéditeur. Jamais vide."""
    s = _as_str(sender_name)
    if not s or s.lower() == "non spécifié":
        return fallback
    return s


def build_chat_push(
    *,
    chat_type: str,
    sender_name: str | None = None,
    message_preview: str | None = None,
    message_id: int | None = None,
    thread_id: str | int | None = None,
    company_id: int | None = None,
    booking_id: int | None = None,
    client_name: str | None = None,
    recipient_role: RecipientRole = "driver",
) -> dict[str, Any]:
    """Construit un message push pour la messagerie (chat équipe / direct / course).

    Templates:
    - team: "Équipe • {SENDER_NAME}"
    - direct: "{SENDER_NAME}"
    - booking: "Course • {CLIENT_NAME}" (jamais d'ID brut)

    Returns:
        {title, body, data, collapse_key, dedupe_key, throttle_seconds}
    """
    sender = safe_sender_name(sender_name)
    body = normalize_preview(message_preview)
    tid = str(thread_id or company_id or "team")
    msg_id = message_id or 0

    if chat_type == CHAT_TYPE_TEAM:
        title = f"Équipe • {sender}"
        collapse_key = f"chat:team:{company_id}" if company_id else "chat:team"
    elif chat_type == CHAT_TYPE_DIRECT:
        title = sender
        collapse_key = f"chat:thread:{tid}"
    elif chat_type == CHAT_TYPE_BOOKING:
        client = _as_str(client_name, "Course")
        title = f"Course • {client}"
        collapse_key = (
            f"chat:booking:{booking_id or 0}" if booking_id else f"chat:thread:{tid}"
        )
    else:
        title = sender
        collapse_key = f"chat:thread:{tid}"

    dedupe_key = (
        f"chat:{tid}:msg:{msg_id}"
        if msg_id
        else f"chat:{tid}:msg:hash:{hash(body) % 10**8}"
    )

    if recipient_role == "company":
        deep_link = "lirie://enterprise/chat"
    elif msg_id:
        deep_link = f"lirie://chat/message/{msg_id}"
    elif tid and tid != "team":
        deep_link = f"lirie://chat/thread/{tid}"
    else:
        deep_link = "lirie://chat"

    data: dict[str, Any] = {
        "type": "chat_message",
        "message_id": msg_id,
        "thread_id": tid,
        "company_id": company_id,
        "sender_name": sender,
        "deep_link": deep_link,
        "deepLink": deep_link,
        "recipient_role": recipient_role,
    }
    if booking_id:
        data["booking_id"] = booking_id
    if client_name:
        data["client_name"] = client_name

    return {
        "title": title,
        "body": body,
        "data": data,
        "collapse_key": collapse_key,
        "dedupe_key": dedupe_key,
        "throttle_seconds": _CHAT_THROTTLE_SECONDS,
    }


def build_push_for_driver_to_company(
    booking_or_context: Any,
    *,
    status: str | None = None,
    discrete_mode: bool = False,
) -> dict[str, Any]:
    """Template Driver→Company: chauffeur change statut (EN_ROUTE, IN_PROGRESS, COMPLETED).

    Titre: "Course • {STATUT_LABEL}"
    Body L1: "{DRIVER_NAME} → {CLIENT_NAME}"
    Body L2 (optionnel): "{PICKUP_SHORT} → {DROPOFF_SHORT}"

    Returns:
        {title, body, data, collapse_key, dedupe_key, throttle_seconds, is_significant}
    """
    bid = _get_booking_id(booking_or_context)
    status_label = _get_status_label(status)
    driver_name = (
        "Chauffeur" if discrete_mode else _get_driver_display_name(booking_or_context)
    )
    client_name = (
        "Client" if discrete_mode else _get_client_display_name(booking_or_context)
    )
    pickup_short = _get_pickup_short(booking_or_context)
    dropoff_short = _get_dropoff_short(booking_or_context)

    title = f"Course • {status_label}"

    body_lines: list[str] = []
    body_lines.append(f"{driver_name} → {client_name}")
    if pickup_short and dropoff_short:
        body_lines.append(_format_route_line(pickup_short, dropoff_short))
    elif pickup_short:
        body_lines.append(pickup_short)
    elif dropoff_short:
        body_lines.append(dropoff_short)
    body = "\n".join(body_lines)

    deep_link = _build_deep_link(bid, "company")
    data: dict[str, Any] = {
        "booking_id": bid,
        "event": "booking_updated",
        "event_type": "status",
        "type": "booking_updated",
        "status": (str(status).lower() if status else None),
        "deep_link": deep_link,
        "deepLink": deep_link,
        "client_name": client_name,
        "driver_name": driver_name,
    }
    if pickup_short:
        data["pickup_short"] = pickup_short
    if dropoff_short:
        data["dropoff_short"] = dropoff_short

    collapse_key = f"booking:{bid}"
    dedupe_key = f"booking:{bid}:event:status:status:{str(status or '').lower()}"
    is_significant = str(status or "").lower() in {
        "completed",
        "return_completed",
        "canceled",
        "cancelled",
    }
    throttle_seconds = 0 if is_significant else 8

    return {
        "title": title,
        "body": body,
        "data": data,
        "collapse_key": collapse_key,
        "dedupe_key": dedupe_key,
        "throttle_seconds": throttle_seconds,
        "is_significant": is_significant,
    }


def build_push_for_company_to_driver(
    booking_or_context: Any,
    *,
    change_type: str,
    status: str | None = None,
    old_time: str | None = None,
    new_time: str | None = None,
    address_change_type: str | None = None,
    new_address_short: str | None = None,
    details_change_labels: list[str] | None = None,
    discrete_mode: bool = False,
) -> dict[str, Any]:
    """Template Company→Driver: assignation, modif heure/lieu, annulation.

    Cas A assign: "Nouvelle course • Assignée"
    Cas B time_change: "Course • Horaire modifié"
    Cas C address_change: "Course • Adresse modifiée"
    Cas D cancel: "Course • Annulée"

    Returns:
        {title, body, data, collapse_key, dedupe_key, throttle_seconds, is_significant}
    """
    bid = _get_booking_id(booking_or_context)
    client_name = (
        "Client" if discrete_mode else _get_client_display_name(booking_or_context)
    )
    time_short = _get_time_short(booking_or_context)
    pickup_short = _get_pickup_short(booking_or_context)
    dropoff_short = _get_dropoff_short(booking_or_context)
    deep_link = _build_deep_link(bid, "driver")

    data: dict[str, Any] = {
        "booking_id": bid,
        "event_type": change_type,
        "type": "booking_updated",
        "deep_link": deep_link,
        "deepLink": deep_link,
        "client_name": client_name,
    }
    if pickup_short:
        data["pickup_short"] = pickup_short
    if dropoff_short:
        data["dropoff_short"] = dropoff_short

    collapse_key = f"booking:{bid}"
    # dedupe_key inclut event_type (pas seulement status) → cas "statut inchangé mais info modifiée"
    dedupe_key = f"booking:{bid}:event:{change_type}:status:{str(status or '').lower()}"
    is_significant = change_type in {
        CHANGE_TYPE_CANCEL,
        CHANGE_TYPE_ASSIGN,
        CHANGE_TYPE_TIME_CHANGE,
        CHANGE_TYPE_ADDRESS_CHANGE,
        CHANGE_TYPE_DETAILS_CHANGE,
    }
    throttle_seconds = 0 if is_significant else 8

    title = ""
    body_lines: list[str] = []

    if change_type == CHANGE_TYPE_ASSIGN:
        title = "Nouvelle course • Assignée"
        body_lines.append(f"{client_name} • {time_short or ''}".strip().rstrip(" •"))
        if pickup_short and dropoff_short:
            body_lines.append(_format_route_line(pickup_short, dropoff_short))
        body = "\n".join(s for s in body_lines if s)
        data["event"] = "booking_assigned"

    elif change_type == CHANGE_TYPE_TIME_CHANGE:
        title = f"Mission #{bid} • Horaire modifié"
        body_lines.append(client_name)
        if old_time and new_time:
            body_lines.append(f"{old_time} → {new_time}")
        elif new_time:
            body_lines.append(f"Nouvelle heure : {new_time}")
        body = "\n".join(s for s in body_lines if s)
        data["event"] = "time_change"
        data["old_time"] = old_time
        data["new_time"] = new_time

    elif change_type == CHANGE_TYPE_ADDRESS_CHANGE:
        title = f"Mission #{bid} • Adresse modifiée"
        body_lines.append(client_name)
        if address_change_type == "pickup" and new_address_short:
            body_lines.append(f"Départ modifié : {new_address_short}")
        elif address_change_type == "dropoff" and new_address_short:
            body_lines.append(f"Destination modifiée : {new_address_short}")
        elif address_change_type == "pickup":
            body_lines.append("Départ modifié")
        elif address_change_type == "dropoff":
            body_lines.append("Destination modifiée")
        body = "\n".join(s for s in body_lines if s)
        data["event"] = "address_change"

    elif change_type == CHANGE_TYPE_DETAILS_CHANGE:
        title = f"Mission #{bid} • Informations modifiées"
        body_lines.append(client_name)
        if details_change_labels:
            body_lines.append(", ".join(details_change_labels[:4]))
        else:
            body_lines.append("Informations mission mises à jour")
        body = "\n".join(s for s in body_lines if s)
        data["event"] = "details_change"
        if details_change_labels:
            data["details_change_labels"] = details_change_labels

    elif change_type == CHANGE_TYPE_CANCEL:
        title = "Course • Annulée"
        parts = [client_name]
        if time_short:
            parts.append(time_short)
        body = " • ".join(parts)
        data["event"] = "booking_cancelled"
        data["type"] = "booking_cancelled"

    else:
        title = "Course • Mise à jour"
        body = client_name
        data["event"] = "booking_updated"

    return {
        "title": title,
        "body": body,
        "data": data,
        "collapse_key": collapse_key,
        "dedupe_key": dedupe_key,
        "throttle_seconds": throttle_seconds,
        "is_significant": is_significant,
    }


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
        EVENT_ASSIGNED: "booking_assigned",
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
        title = "Nouvelle course • Assignée"
        if discrete_mode:
            body = "Nouvelle course assignée. Ouvrez l'application pour les détails."
        elif recipient_role == "driver":
            body_lines = [
                f"{client} • {time_short}".strip().rstrip(" •")
                if time_short
                else client
            ]
            if pickup_short and dropoff_short:
                body_lines.append(_format_route_line(pickup_short, dropoff_short))
            body = "\n".join(s for s in body_lines if s)
        else:
            driver_name = (
                "Chauffeur"
                if discrete_mode
                else _get_driver_display_name(booking_or_context)
            )
            body_lines = [f"{driver_name} → {client}"]
            if pickup_short and dropoff_short:
                body_lines.append(_format_route_line(pickup_short, dropoff_short))
            body = "\n".join(body_lines)

    elif event == EVENT_STATUS_UPDATED:
        title = f"Course • {_get_status_label(status)}"
        actor_name = "Chauffeur"
        if actor:
            fn = _as_str(actor.get("first_name"))
            ln = _as_str(actor.get("last_name"))
            actor_name = (
                f"{fn} {ln}".strip() or _as_str(actor.get("username")) or "Chauffeur"
            )
        if discrete_mode:
            body = "Mise à jour de course. Ouvrez l'application pour les détails."
        else:
            body_lines = [f"{actor_name} → {client}"]
            if pickup_short and dropoff_short:
                body_lines.append(_format_route_line(pickup_short, dropoff_short))
            body = "\n".join(body_lines)

    elif event == EVENT_COMPLETED:
        title = "Course • Terminée"
        if discrete_mode:
            body = "Course terminée. Ouvrez l'application pour les détails."
        else:
            body_lines = [client]
            if amount_chf:
                body_lines.append(amount_chf)
            body = " • ".join(body_lines)

    elif event == EVENT_CANCELLED:
        title = "Course • Annulée"
        if discrete_mode:
            body = "Une course a été annulée."
        else:
            parts = [client]
            if time_short:
                parts.append(time_short)
            body = " • ".join(parts)

    elif event == EVENT_REASSIGNED:
        title = "Course réassignée"
        if discrete_mode:
            body = "La course a été réassignée à un autre chauffeur."
        else:
            body = (
                f"La course {client} a été réassignée à un autre chauffeur."
                if client
                else "La course a été réassignée."
            )
        data["deep_link"] = data["deepLink"] = "lirie://bookings"

    else:
        title = "Mise à jour de course"
        body = (
            changes_preview
            or (f"{client} • " if client else "")
            or "Ouvrez l'application pour les détails."
        )

    if event == EVENT_ASSIGNED and bid:
        assigned_dedupe_key = f"booking:{bid}:event:assigned"
        data["mission_id"] = bid
        data["dedupe_key"] = assigned_dedupe_key

    return {"title": title, "body": body, "data": data}
