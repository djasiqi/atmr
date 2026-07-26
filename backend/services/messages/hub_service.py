"""Driver message hub: threads, unread, system messages."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, or_
from sqlalchemy.exc import IntegrityError

from ext import db
from models import Booking, Driver, Message, SenderRole
from services.messaging.message_idempotence import (
    find_idempotent_message,
    note_duplicate_hit,
)
from services.monitoring.chat_metrics import inc_chat_message_sent

logger = logging.getLogger(__name__)

THREAD_DISPATCH = "dispatch"
THREAD_TEAM = "team"
THREAD_SUPPORT = "support"
THREAD_PREFIX_MISSION = "mission:"
DIRECT_PREFIX = "direct:"


def mission_thread_id(booking_id: int) -> str:
    return f"{THREAD_PREFIX_MISSION}{booking_id}"


def parse_mission_thread(thread_id: str) -> int | None:
    if not thread_id.startswith(THREAD_PREFIX_MISSION):
        return None
    try:
        return int(thread_id.split(":", 1)[1])
    except (IndexError, ValueError):
        return None


def direct_thread_id(peer_user_id: int) -> str:
    return f"{DIRECT_PREFIX}{peer_user_id}"


def parse_direct_thread(thread_id: str) -> int | None:
    """Peer user id depuis ``direct:{peer_user_id}`` (pas ``direct:a:b``)."""
    if not thread_id.startswith(DIRECT_PREFIX):
        return None
    parts = thread_id.split(":")
    if len(parts) != 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _driver_display_name(driver: Driver) -> str:
    user = getattr(driver, "user", None)
    if user is None and getattr(driver, "user_id", None):
        from models import User

        user = User.query.get(int(driver.user_id))
    if user is not None:
        first = (getattr(user, "first_name", None) or "").strip()
        last = (getattr(user, "last_name", None) or "").strip()
        full = f"{first} {last}".strip()
        if full:
            return full
        username = getattr(user, "username", None)
        if username:
            return str(username)
    return f"Chauffeur #{driver.id}"


def _is_direct_message(message: Message) -> bool:
    if getattr(message, "receiver_id", None) is not None:
        return True
    tid = getattr(message, "thread_id", None)
    return bool(tid and str(tid).startswith(DIRECT_PREFIX))


def _message_in_thread(
    message: Message,
    thread_id: str,
    *,
    my_user_id: int | None,
    peer_user_id: int | None = None,
) -> bool:
    if thread_id.startswith(DIRECT_PREFIX):
        if peer_user_id is None:
            peer_user_id = parse_direct_thread(thread_id)
        if peer_user_id is None or my_user_id is None:
            return False
        sender = getattr(message, "sender_id", None)
        receiver = getattr(message, "receiver_id", None)
        return (sender == my_user_id and receiver == peer_user_id) or (
            sender == peer_user_id and receiver == my_user_id
        )
    if thread_id.startswith(THREAD_PREFIX_MISSION):
        return getattr(message, "thread_id", None) == thread_id
    if thread_id == THREAD_TEAM:
        if _is_direct_message(message):
            return False
        return getattr(message, "thread_id", None) == THREAD_TEAM
    if thread_id == THREAD_DISPATCH:
        if _is_direct_message(message):
            return False
        tid = getattr(message, "thread_id", None)
        if tid == THREAD_DISPATCH:
            return True
        return bool(tid in (None, "") and getattr(message, "receiver_id", None) is None)
    if thread_id == THREAD_SUPPORT:
        return getattr(message, "thread_id", None) == THREAD_SUPPORT
    return getattr(message, "thread_id", None) == thread_id


def _terminal_statuses() -> set[str]:
    return {"COMPLETED", "CANCELLED", "REASSIGNED", "NO_SHOW", "FAILED"}


def _booking_label(booking: Booking) -> str:
    client = getattr(booking, "client", None)
    name = None
    if client is not None:
        name = getattr(client, "full_name", None) or getattr(client, "first_name", None)
    if name:
        return str(name)
    return f"Mission #{booking.id}"


def _last_message_preview(messages: list[Message]) -> str | None:
    if not messages:
        return None
    last = messages[-1]
    msg_type = getattr(last, "message_type", None) or "text"
    if msg_type == "system":
        return last.content or "Événement mission"
    if getattr(last, "image_url", None):
        return "Photo"
    if getattr(last, "pdf_url", None):
        return "Document"
    if getattr(last, "audio_url", None) if hasattr(last, "audio_url") else None:
        return "Message vocal"
    return (last.content or "").strip() or None


def _message_priority_rank(priority: str | None) -> int:
    if priority == "urgent":
        return 3
    if priority == "important":
        return 2
    return 1


def count_company_team_members(company_id: int) -> int:
    """Chauffeurs actifs avec compte utilisateur — taille réelle du canal équipe."""
    return (
        Driver.query.filter_by(company_id=company_id, is_active=True)
        .filter(Driver.user_id.isnot(None))
        .count()
    )


def list_driver_colleagues(
    company_id: int,
    driver: Driver,
    *,
    limit_messages: int = 200,
) -> list[dict[str, Any]]:
    """Liste des collègues (roster entreprise + fils DM existants)."""
    my_user_id = getattr(driver, "user_id", None)
    messages_asc = list(
        reversed(
            Message.query.filter(Message.company_id == company_id)
            .order_by(Message.timestamp.desc())
            .limit(limit_messages)
            .all()
        )
    )

    colleague_meta: dict[int, dict[str, Any]] = {}
    if my_user_id is not None:
        for m in messages_asc:
            if not _is_direct_message(m):
                continue
            sender = getattr(m, "sender_id", None)
            receiver = getattr(m, "receiver_id", None)
            if sender == my_user_id and receiver and receiver != my_user_id:
                peer = int(receiver)
            elif receiver == my_user_id and sender and sender != my_user_id:
                peer = int(sender)
            else:
                continue
            colleague_meta.setdefault(peer, {"peer_user_id": peer})

    from repositories.driver_repository import DriverRepository

    company_drivers = DriverRepository().find_models_by_company_id(company_id)
    for peer_driver in company_drivers:
        if not getattr(peer_driver, "is_active", True):
            continue
        peer_uid = getattr(peer_driver, "user_id", None)
        if peer_uid is None or peer_uid == my_user_id:
            continue
        peer_uid = int(peer_uid)
        colleague_meta.setdefault(
            peer_uid,
            {"peer_user_id": peer_uid, "driver_id": int(peer_driver.id)},
        )

    rows: list[dict[str, Any]] = []
    for peer_uid, meta in colleague_meta.items():
        peer_driver = next(
            (d for d in company_drivers if getattr(d, "user_id", None) == peer_uid),
            None,
        )
        title = (
            _driver_display_name(peer_driver)
            if peer_driver is not None
            else f"Collègue #{peer_uid}"
        )
        rows.append(
            {
                "peer_user_id": peer_uid,
                "driver_id": meta.get("driver_id"),
                "title": title,
                "thread_id": direct_thread_id(peer_uid),
            }
        )
    rows.sort(key=lambda r: r.get("title") or "")
    return rows


def build_driver_threads(
    company_id: int,
    driver: Driver,
    *,
    limit_messages: int = 200,
) -> list[dict[str, Any]]:
    """Build inbox: mission, équipe, dispatch, collègues, support, archives."""
    my_user_id = getattr(driver, "user_id", None)
    messages = (
        Message.query.filter(Message.company_id == company_id)
        .order_by(Message.timestamp.desc())
        .limit(limit_messages)
        .all()
    )
    messages_asc = list(reversed(messages))

    bookings = (
        Booking.query.filter(
            Booking.company_id == company_id,
            Booking.driver_id == driver.id,
        )
        .order_by(Booking.scheduled_time.desc().nullslast())
        .limit(40)
        .all()
    )

    active_booking: Booking | None = None
    archived: list[Booking] = []
    for booking in bookings:
        status = str(getattr(booking, "status", "") or "").upper()
        if active_booking is None and status not in _terminal_statuses():
            active_booking = booking
        elif status in _terminal_statuses():
            archived.append(booking)

    def thread_messages(
        thread_id: str, peer_user_id: int | None = None
    ) -> list[Message]:
        return [
            m
            for m in messages_asc
            if _message_in_thread(
                m, thread_id, my_user_id=my_user_id, peer_user_id=peer_user_id
            )
        ]

    def unread_for_thread(thread_id: str, peer_user_id: int | None = None) -> int:
        if thread_id.startswith(DIRECT_PREFIX) and my_user_id is not None:
            return sum(
                1
                for m in thread_messages(thread_id, peer_user_id)
                if getattr(m, "receiver_id", None) == my_user_id
                and not bool(getattr(m, "is_read", False))
            )
        return sum(
            1
            for m in thread_messages(thread_id, peer_user_id)
            if not bool(getattr(m, "is_read", False))
            and getattr(m, "sender_id", None) != my_user_id
        )

    def last_for_thread(
        thread_id: str, peer_user_id: int | None = None
    ) -> Message | None:
        msgs = thread_messages(thread_id, peer_user_id)
        return msgs[-1] if msgs else None

    def priority_for_thread(thread_id: str, peer_user_id: int | None = None) -> str:
        msgs = thread_messages(thread_id, peer_user_id)
        if not msgs:
            return "normal"
        return max(
            (getattr(m, "priority", None) or "normal" for m in msgs),
            key=_message_priority_rank,
        )

    def thread_payload(
        *,
        thread_id: str,
        section: str,
        title: str,
        subtitle: str | None,
        peer_user_id: int | None = None,
        booking_id: int | None = None,
        status: str | None = None,
        scheduled_time: str | None = None,
        pickup: str | None = None,
        dropoff: str | None = None,
    ) -> dict[str, Any]:
        last = last_for_thread(thread_id, peer_user_id)
        return {
            "thread_id": thread_id,
            "section": section,
            "title": title,
            "subtitle": subtitle,
            "peer_user_id": peer_user_id,
            "booking_id": booking_id,
            "status": status,
            "scheduled_time": scheduled_time,
            "pickup_location": pickup,
            "dropoff_location": dropoff,
            "unread_count": unread_for_thread(thread_id, peer_user_id),
            "priority": priority_for_thread(thread_id, peer_user_id),
            "last_message_preview": _last_message_preview(
                thread_messages(thread_id, peer_user_id)
            ),
            "last_message_at": last.timestamp.isoformat()
            if last and last.timestamp
            else None,
        }

    threads: list[dict[str, Any]] = []

    if active_booking is not None:
        tid = mission_thread_id(active_booking.id)
        if last_for_thread(tid) is not None:
            threads.append(
                thread_payload(
                    thread_id=tid,
                    section="mission_active",
                    title=_booking_label(active_booking),
                    subtitle=f"Mission #{active_booking.id}",
                    booking_id=active_booking.id,
                    status=str(getattr(active_booking, "status", "") or ""),
                    scheduled_time=active_booking.scheduled_time.isoformat()
                    if getattr(active_booking, "scheduled_time", None)
                    else None,
                    pickup=getattr(active_booking, "pickup_location", None),
                    dropoff=getattr(active_booking, "dropoff_location", None),
                )
            )

    threads.append(
        thread_payload(
            thread_id=THREAD_TEAM,
            section="team",
            title="Équipe chauffeurs",
            subtitle="Canal groupe · tous les collègues",
        )
    )

    threads.append(
        thread_payload(
            thread_id=THREAD_DISPATCH,
            section="dispatch",
            title="Dispatch",
            subtitle="Exploitation & régulation",
        )
    )

    colleague_roster = list_driver_colleagues(
        company_id, driver, limit_messages=limit_messages
    )
    colleague_rows: list[dict[str, Any]] = []
    for row in colleague_roster:
        peer_uid = int(row["peer_user_id"])
        colleague_rows.append(
            thread_payload(
                thread_id=str(row["thread_id"]),
                section="colleagues",
                title=str(row["title"]),
                subtitle="Message direct",
                peer_user_id=peer_uid,
            )
        )

    colleague_rows.sort(key=lambda row: row.get("last_message_at") or "", reverse=True)
    colleague_rows.sort(key=lambda row: row.get("unread_count") or 0, reverse=True)
    threads.extend(colleague_rows)

    threads.append(
        thread_payload(
            thread_id=THREAD_SUPPORT,
            section="support",
            title="Support",
            subtitle="Assistance LIRIE",
        )
    )

    for booking in archived[:15]:
        tid = mission_thread_id(booking.id)
        if last_for_thread(tid) is None:
            continue
        threads.append(
            thread_payload(
                thread_id=tid,
                section="archives",
                title=_booking_label(booking),
                subtitle=f"Mission #{booking.id}",
                booking_id=booking.id,
                status=str(getattr(booking, "status", "") or ""),
            )
        )

    return threads


def _messages_for_conversation_legacy(
    conversation_id: int,
    company_id: int,
    *,
    legacy_thread_id: str | None = None,
    before: datetime | None = None,
    limit: int = 40,
) -> list[Message]:
    """Lecture de secours par conversation_id (évite un fil vide après refus permissions)."""
    conv_filters = [Message.conversation_id == conversation_id]
    if legacy_thread_id:
        conv_filters.append(
            and_(
                Message.conversation_id.is_(None),
                Message.company_id == company_id,
                Message.thread_id == legacy_thread_id,
            )
        )
    query = Message.query.filter(or_(*conv_filters))
    if before:
        query = query.filter(Message.timestamp < before)
    rows = query.order_by(Message.timestamp.desc()).limit(limit).all()
    rows.reverse()
    dirty = False
    for msg in rows:
        if msg.conversation_id is None:
            msg.conversation_id = conversation_id
            dirty = True
    if dirty:
        db.session.commit()
    return rows


def get_thread_messages(
    company_id: int,
    thread_id: str,
    *,
    before: datetime | None = None,
    limit: int = 40,
    reader_user_id: int | None = None,
    driver: Driver | None = None,
    reader_user=None,
) -> list[Message]:
    """Messages d'un fil — isolation stricte par conversation_id ou thread_id exact."""
    from models import User
    from services.messaging.conversation_service import ConversationService
    from services.messaging.permission_service import MessagingPermissionService

    hub_reader = reader_user
    if hub_reader is None and reader_user_id:
        hub_reader = User.query.get(int(reader_user_id))

    conv = None
    if hub_reader is not None:
        role = str(getattr(hub_reader.role, "value", hub_reader.role)).upper()
        try:
            if role == "COMPANY":
                conv = ConversationService.resolve_by_legacy_thread_for_company(
                    company_id, thread_id
                )
                if conv is not None:
                    MessagingPermissionService.assert_can_read(hub_reader, conv)
                    return ConversationService.get_messages(
                        conv, hub_reader, before=before, limit=limit
                    )
            elif driver and reader_user_id:
                conv = ConversationService.resolve_by_legacy_thread(
                    company_id, thread_id, driver
                )
                if conv is not None:
                    if thread_id == THREAD_DISPATCH:
                        ConversationService._sync_dispatch_driver_participants(
                            conv, company_id
                        )
                    MessagingPermissionService.assert_can_read(hub_reader, conv)
                    return ConversationService.get_messages(
                        conv, hub_reader, before=before, limit=limit
                    )
        except PermissionError as perm_err:
            logger.warning(
                "get_thread_messages permission denied company_id=%s thread_id=%s: %s",
                company_id,
                thread_id,
                perm_err,
            )
            if (
                conv is not None
                and thread_id == THREAD_DISPATCH
                and driver is not None
                and ConversationService._sync_dispatch_driver_participants(
                    conv, company_id
                )
            ):
                try:
                    MessagingPermissionService.assert_can_read(hub_reader, conv)
                    return ConversationService.get_messages(
                        conv, hub_reader, before=before, limit=limit
                    )
                except PermissionError:
                    pass
            if conv is not None:
                rows = _messages_for_conversation_legacy(
                    conv.id,
                    company_id,
                    legacy_thread_id=getattr(conv, "legacy_thread_id", None)
                    or thread_id,
                    before=before,
                    limit=limit,
                )
                if rows:
                    return rows

    if thread_id == THREAD_DISPATCH and conv is None:
        try:
            if driver is not None:
                conv = ConversationService.resolve_by_legacy_thread(
                    company_id, thread_id, driver
                )
            elif hub_reader is not None:
                conv = ConversationService.resolve_by_legacy_thread_for_company(
                    company_id, thread_id
                )
        except Exception:
            conv = None

    query = Message.query.filter(Message.company_id == company_id)
    if thread_id == THREAD_DISPATCH and conv is not None:
        query = query.filter(
            or_(
                Message.thread_id == THREAD_DISPATCH,
                Message.conversation_id == conv.id,
            )
        )
    else:
        query = query.filter(Message.thread_id == thread_id)
    if thread_id.startswith(DIRECT_PREFIX) and reader_user_id:
        peer_uid = parse_direct_thread(thread_id)
        if peer_uid is not None:
            query = query.filter(
                or_(
                    and_(
                        Message.sender_id == reader_user_id,
                        Message.receiver_id == peer_uid,
                    ),
                    and_(
                        Message.sender_id == peer_uid,
                        Message.receiver_id == reader_user_id,
                    ),
                )
            )
    if before:
        query = query.filter(Message.timestamp < before)
    rows = query.order_by(Message.timestamp.desc()).limit(limit).all()
    rows.reverse()
    return rows


def mark_thread_read(
    company_id: int,
    thread_id: str,
    _reader_role: SenderRole,
    *,
    reader_user_id: int | None = None,
    _driver: Driver | None = None,
) -> int:
    """Mark inbound messages as read for a thread."""
    rows = Message.query.filter(
        Message.company_id == company_id,
        Message.thread_id == thread_id,
        Message.is_read.is_(False),
    ).all()
    updated = 0
    for msg in rows:
        if getattr(msg, "sender_id", None) == reader_user_id:
            continue
        msg.is_read = True
        updated += 1
    if updated:
        db.session.commit()
    return updated


def ack_message(message_id: int, company_id: int) -> bool:
    msg = Message.query.filter_by(id=message_id, company_id=company_id).first()
    if not msg:
        return False
    msg.acked_at = datetime.now(UTC)
    msg.is_read = True
    db.session.commit()
    return True


def count_unread(company_id: int, driver_role: SenderRole = SenderRole.DRIVER) -> int:
    return int(
        Message.query.filter(
            Message.company_id == company_id,
            Message.is_read.is_(False),
            Message.sender_role != driver_role,
        ).count()
    )


EMERGENCY_LABELS = {
    "patient_absent": "Patient absent",
    "retard_important": "Retard important",
    "panne_vehicule": "Panne véhicule",
    "incident": "Incident",
    "besoin_assistance": "Besoin d'assistance",
}


def report_hub_emergency(
    driver: Driver,
    *,
    issue_type: str,
    booking_id: int | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Fanout dispatch + journal système dans le fil mission/dispatch."""
    from services.events.fanout import fanout_urgent_alert

    company_id = int(driver.company_id)
    label = EMERGENCY_LABELS.get(issue_type, issue_type.replace("_", " ").title())
    thread_id = mission_thread_id(booking_id) if booking_id else THREAD_DISPATCH

    content = f"⚠ {label}"
    if note and note.strip():
        content = f"{content} — {note.strip()}"

    system_msg = create_system_message(
        company_id,
        thread_id=thread_id,
        booking_id=booking_id,
        content=content,
        priority="urgent",
        driver=driver,
        reporter=driver,
    )

    fanout_urgent_alert(
        company_id=company_id,
        alert_id=f"hub-{driver.id}-{int(time.time())}",
        alert_type=f"driver_hub_{issue_type}",
        message=content,
        severity="critical",
        booking_id=booking_id,
        driver_id=driver.id,
    )

    return {
        "ok": True,
        "issue_type": issue_type,
        "thread_id": thread_id,
        "company_id": company_id,
        "system_message_id": system_msg.id,
        "message": {
            **system_msg.serialize,
            "sender_name": _driver_display_name(driver),
        },
        "latitude": latitude,
        "longitude": longitude,
    }


MAX_HUB_MESSAGE_LENGTH = 1000


def send_driver_hub_message(
    user,
    driver: Driver,
    company_id: int,
    *,
    thread_id: str,
    body: dict[str, Any],
) -> tuple[dict[str, Any] | None, tuple[dict[str, str], int] | None]:
    """Crée un message chauffeur (fallback HTTP si Socket.IO indisponible)."""
    from services.messaging.conversation_service import ConversationService
    from services.messaging.permission_service import MessagingPermissionService
    from services.security.spam import can_send_message

    content_raw = body.get("content")
    content = (content_raw or "").strip() if content_raw else ""
    image_url = body.get("image_url") or body.get("image")
    pdf_url = body.get("pdf_url") or body.get("pdf")
    pdf_filename = body.get("pdf_filename")
    pdf_size = body.get("pdf_size")
    has_image = bool(image_url)
    has_pdf = bool(pdf_url)
    if has_image and has_pdf:
        return None, ({"error": "Limite : une image ou un PDF par message."}, 400)
    if not (content or has_image or has_pdf):
        return None, (
            {"error": "Le message doit contenir du texte, une image ou un PDF."},
            400,
        )
    if content and len(content) > MAX_HUB_MESSAGE_LENGTH:
        return None, (
            {"error": f"Message trop long (max {MAX_HUB_MESSAGE_LENGTH} caractères)."},
            400,
        )

    allowed_spam, spam_error = can_send_message(int(user.id))
    if not allowed_spam:
        return None, (
            {"error": spam_error or "Trop de messages. Attendez 1 seconde."},
            429,
        )

    receiver_id = body.get("receiver_id")
    if receiver_id is not None:
        try:
            receiver_id = int(receiver_id)
            if receiver_id <= 0:
                raise ValueError
        except (TypeError, ValueError):
            return None, ({"error": "receiver_id invalide."}, 400)

    booking_id_raw = body.get("booking_id")
    booking_id = None
    if booking_id_raw is not None:
        try:
            booking_id = int(booking_id_raw)
        except (TypeError, ValueError):
            booking_id = None

    tid = str(thread_id or THREAD_TEAM).strip() or THREAD_TEAM
    if not tid and booking_id:
        tid = mission_thread_id(booking_id)
    if not tid:
        tid = direct_thread_id(receiver_id) if receiver_id else THREAD_TEAM

    conversation_id_val = body.get("conversation_id")
    conv_obj = None
    try:
        if conversation_id_val:
            from models import Conversation

            conv_obj = Conversation.query.get(int(conversation_id_val))
        else:
            conv_obj = ConversationService.resolve_by_legacy_thread(
                company_id, tid, driver
            )
        if tid.startswith("direct:") and conv_obj is None:
            return None, (
                {"error": "Collègue introuvable ou message direct refusé."},
                403,
            )
        if conv_obj is not None:
            MessagingPermissionService.assert_can_write(user, conv_obj)
            conversation_id_val = conv_obj.id
    except PermissionError as perm_err:
        return None, ({"error": str(perm_err)}, 403)
    except Exception:
        logger.exception("send_driver_hub_message conversation resolve failed")

    message_type = (body.get("message_type") or "text").strip() or "text"
    priority = (body.get("priority") or "normal").strip() or "normal"
    client_message_id = body.get("client_message_id") or body.get("_localId")
    timestamp = datetime.now(UTC)
    content_final = content if content else None

    message: Message | None = None
    if client_message_id:
        message = find_idempotent_message(int(user.id), str(client_message_id))
        if message is not None:
            note_duplicate_hit(channel="rest")

    if message is None:
        message = Message(
            sender_id=int(user.id),
            receiver_id=receiver_id,
            company_id=company_id,
            sender_role=SenderRole.DRIVER,
            content=content_final,
            timestamp=timestamp,
            image_url=image_url if has_image else None,
            pdf_url=pdf_url if has_pdf else None,
            pdf_filename=pdf_filename if has_pdf else None,
            pdf_size=int(pdf_size) if has_pdf and pdf_size else None,
            thread_id=tid,
            booking_id=booking_id,
            message_type=message_type,
            priority=priority,
            client_message_id=str(client_message_id) if client_message_id else None,
            conversation_id=int(conversation_id_val) if conversation_id_val else None,
            visibility_tags=["operational"],
        )
        try:
            db.session.add(message)
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            if client_message_id:
                message = find_idempotent_message(int(user.id), str(client_message_id))
                if message is not None:
                    note_duplicate_hit(channel="rest")
            if message is None:
                raise

    thread_type = tid.split(":", 1)[0] if ":" in tid else tid
    inc_chat_message_sent(channel="rest", thread_type=thread_type)

    sender_name = getattr(user, "first_name", None)
    payload: dict[str, Any] = {
        **message.serialize,
        "sender_name": sender_name,
        "type": "chat",
        "_localId": client_message_id,
    }
    _fanout_hub_message_socket(
        company_id,
        payload,
        conversation_id=conversation_id_val,
        receiver_id=receiver_id,
        thread_id=tid,
    )
    return payload, None


def send_company_hub_message(
    user,
    company_id: int,
    *,
    thread_id: str,
    body: dict[str, Any],
) -> tuple[dict[str, Any] | None, tuple[dict[str, str], int] | None]:
    """Crée un message exploitation (canal dispatch, chauffeur, mission)."""
    from services.messaging.conversation_service import ConversationService
    from services.messaging.permission_service import MessagingPermissionService
    from services.security.spam import can_send_message

    content_raw = body.get("content")
    content = (content_raw or "").strip() if content_raw else ""
    image_url = body.get("image_url") or body.get("image")
    pdf_url = body.get("pdf_url") or body.get("pdf")
    pdf_filename = body.get("pdf_filename")
    pdf_size = body.get("pdf_size")
    has_image = bool(image_url)
    has_pdf = bool(pdf_url)
    if has_image and has_pdf:
        return None, ({"error": "Limite : une image ou un PDF par message."}, 400)
    if not (content or has_image or has_pdf):
        return None, (
            {"error": "Le message doit contenir du texte, une image ou un PDF."},
            400,
        )
    if content and len(content) > MAX_HUB_MESSAGE_LENGTH:
        return None, (
            {"error": f"Message trop long (max {MAX_HUB_MESSAGE_LENGTH} caractères)."},
            400,
        )

    allowed_spam, spam_error = can_send_message(int(user.id))
    if not allowed_spam:
        return None, (
            {"error": spam_error or "Trop de messages. Attendez 1 seconde."},
            429,
        )

    booking_id_raw = body.get("booking_id")
    booking_id = None
    if booking_id_raw is not None:
        try:
            booking_id = int(booking_id_raw)
        except (TypeError, ValueError):
            booking_id = None

    tid = str(thread_id or THREAD_DISPATCH).strip() or THREAD_DISPATCH
    if booking_id and tid == THREAD_DISPATCH:
        tid = mission_thread_id(booking_id)

    conversation_id_val = body.get("conversation_id")
    conv_obj = None
    try:
        if conversation_id_val:
            from models import Conversation

            conv_obj = Conversation.query.get(int(conversation_id_val))
        else:
            conv_obj = ConversationService.resolve_by_legacy_thread_for_company(
                company_id, tid
            )
        if conv_obj is not None:
            MessagingPermissionService.assert_can_write(user, conv_obj)
            conversation_id_val = conv_obj.id
    except PermissionError as perm_err:
        return None, ({"error": str(perm_err)}, 403)
    except Exception:
        logger.exception("send_company_hub_message conversation resolve failed")

    message_type = (body.get("message_type") or "text").strip() or "text"
    priority = (body.get("priority") or "normal").strip() or "normal"
    client_message_id = body.get("client_message_id") or body.get("_localId")
    timestamp = datetime.now(UTC)
    content_final = content if content else None

    message: Message | None = None
    if client_message_id:
        message = find_idempotent_message(int(user.id), str(client_message_id))
        if message is not None:
            note_duplicate_hit(channel="rest")

    if message is None:
        message = Message(
            sender_id=int(user.id),
            receiver_id=None,
            company_id=company_id,
            sender_role=SenderRole.COMPANY,
            content=content_final,
            timestamp=timestamp,
            image_url=image_url if has_image else None,
            pdf_url=pdf_url if has_pdf else None,
            pdf_filename=pdf_filename if has_pdf else None,
            pdf_size=int(pdf_size) if has_pdf and pdf_size else None,
            thread_id=tid,
            booking_id=booking_id,
            message_type=message_type,
            priority=priority,
            client_message_id=str(client_message_id) if client_message_id else None,
            conversation_id=int(conversation_id_val) if conversation_id_val else None,
            visibility_tags=["operational"],
        )
        try:
            db.session.add(message)
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            if client_message_id:
                message = find_idempotent_message(int(user.id), str(client_message_id))
                if message is not None:
                    note_duplicate_hit(channel="rest")
            if message is None:
                raise

    thread_type = tid.split(":", 1)[0] if ":" in tid else tid
    inc_chat_message_sent(channel="rest", thread_type=thread_type)

    sender_name = getattr(user, "first_name", None)
    payload: dict[str, Any] = {
        **message.serialize,
        "sender_name": sender_name,
        "type": "chat",
        "_localId": client_message_id,
    }
    _fanout_hub_message_socket(
        company_id,
        payload,
        conversation_id=conversation_id_val,
        receiver_id=None,
        thread_id=tid,
    )
    return payload, None


def _fanout_hub_message_socket(
    company_id: int,
    payload: dict[str, Any],
    *,
    conversation_id: int | None,
    receiver_id: int | None,
    thread_id: str | None = None,
) -> None:
    try:
        from ext import socketio
        from services.messaging.channel_routing import emit_chat_message

        emit_chat_message(
            socketio.emit,
            "team_chat_message",
            payload,
            company_id=company_id,
            thread_id=thread_id or payload.get("thread_id"),
            conversation_id=conversation_id,
            receiver_id=receiver_id,
        )
    except Exception:
        logger.exception("hub message socket fanout failed")


def create_system_message(
    company_id: int,
    *,
    thread_id: str,
    booking_id: int | None,
    content: str,
    priority: str = "normal",
    driver: Driver | None = None,
    reporter: Driver | None = None,
    fanout: bool = True,
) -> Message:
    conversation_id_val: int | None = None
    try:
        from services.messaging.conversation_service import ConversationService

        if booking_id is not None:
            conv = ConversationService.ensure_mission_conversation(
                company_id, int(booking_id)
            )
            conversation_id_val = conv.id
        elif thread_id == THREAD_DISPATCH:
            conv = ConversationService.ensure_company_dispatch_conversation(company_id)
            conversation_id_val = conv.id
            if reporter is not None:
                ConversationService._sync_dispatch_driver_participants(conv, company_id)
        elif driver is not None:
            conv = ConversationService.resolve_by_legacy_thread(
                company_id, thread_id, driver
            )
            if conv is not None:
                conversation_id_val = conv.id
    except Exception:
        logger.exception(
            "create_system_message conversation resolve failed company_id=%s thread_id=%s",
            company_id,
            thread_id,
        )

    if conversation_id_val is None and thread_id == THREAD_DISPATCH:
        try:
            from services.messaging.conversation_service import ConversationService

            conv = ConversationService.ensure_company_dispatch_conversation(company_id)
            conversation_id_val = conv.id
            if reporter is not None:
                ConversationService._sync_dispatch_driver_participants(conv, company_id)
        except Exception:
            logger.exception(
                "create_system_message dispatch ensure failed company_id=%s",
                company_id,
            )

    sender_id: int | None = None
    sender_role = SenderRole.COMPANY
    visibility_tags = ["system", "operational"]
    reporter_name = "Système"
    if reporter is not None:
        user_id = getattr(reporter, "user_id", None)
        if user_id:
            sender_id = int(user_id)
        sender_role = SenderRole.DRIVER
        reporter_name = _driver_display_name(reporter)
        visibility_tags = ["system", "operational", "driver_report"]

    msg = Message(
        company_id=company_id,
        sender_id=sender_id,
        receiver_id=None,
        sender_role=sender_role,
        content=content,
        thread_id=thread_id,
        booking_id=booking_id,
        conversation_id=conversation_id_val,
        message_type="system",
        priority=priority,
        is_read=False,
        timestamp=datetime.now(UTC),
        visibility_tags=visibility_tags,
    )
    db.session.add(msg)
    db.session.commit()

    if fanout:
        payload: dict[str, Any] = {
            **msg.serialize,
            "sender_name": reporter_name,
            "type": "chat",
        }
        _fanout_hub_message_socket(
            company_id,
            payload,
            conversation_id=conversation_id_val,
            receiver_id=None,
            thread_id=thread_id,
        )

    return msg
