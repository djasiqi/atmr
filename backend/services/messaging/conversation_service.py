"""Conversation lifecycle, inbox, messages, read state."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

from ext import db
from sqlalchemy import and_, or_
from models import Booking, Company, Conversation, ConversationParticipant, Driver, Message, MessageRead, User
from models.messaging_enums import (
    ConversationContext,
    ConversationType,
    DEFAULT_MESSAGE_VISIBILITY_TAGS,
    ParticipantRole,
)
from services.messaging.legacy_thread import (
    DIRECT_PREFIX,
    THREAD_SUPPORT,
    company_dispatch_legacy_thread_id,
    company_driver_channel_legacy_thread_id,
    company_group_legacy_thread_id,
    conversation_id_to_legacy_thread,
    direct_pair_legacy_thread_id,
    mission_thread_id,
    parse_direct_pair_legacy_thread,
    parse_direct_thread,
    parse_mission_thread,
)
from services.messaging.permission_service import MessagingPermissionService

MESSAGING_BACKFILL_ON_INBOX = os.environ.get("MESSAGING_BACKFILL_ON_INBOX", "1") == "1"
MESSAGING_BACKFILL_ON_CONNECT = os.environ.get("MESSAGING_BACKFILL_ON_CONNECT", "0") == "1"

logger = logging.getLogger(__name__)

_TERMINAL = {"COMPLETED", "CANCELLED", "CANCELED", "REASSIGNED", "NO_SHOW", "FAILED", "RETURN_COMPLETED"}


class ConversationService:
    # --- Provisioning ---

    @staticmethod
    def _fanout_join_conversation_room(conv: Conversation) -> None:
        try:
            from services.messaging.conversation_room_fanout import (
                join_conversation_participants,
            )

            join_conversation_participants(int(conv.id))
        except Exception:
            logger.exception(
                "[conversation] dynamic room join failed conv_id=%s", conv.id
            )

    @staticmethod
    def ensure_mission_conversation(company_id: int, booking_id: int) -> Conversation:
        existing = Conversation.query.filter_by(
            company_id=company_id,
            conversation_type=ConversationType.MISSION.value,
            context_type=ConversationContext.MISSION.value,
            context_id=booking_id,
        ).first()
        if existing:
            return existing

        booking = Booking.query.filter_by(id=booking_id, company_id=company_id).first()
        title = _booking_label(booking) if booking else f"Mission #{booking_id}"
        conv = Conversation(
            company_id=company_id,
            conversation_type=ConversationType.MISSION.value,
            context_type=ConversationContext.MISSION.value,
            context_id=booking_id,
            title=title,
            legacy_thread_id=mission_thread_id(booking_id),
            conversation_metadata=Conversation.default_metadata(),
        )
        db.session.add(conv)
        db.session.flush()
        if booking and booking.driver_id:
            driver = Driver.query.get(booking.driver_id)
            if driver and driver.user_id:
                ConversationService._add_participant(
                    conv,
                    int(driver.user_id),
                    ParticipantRole.DRIVER,
                    can_write=True,
                )
        ConversationService._ensure_dispatch_participants(conv, company_id)
        db.session.commit()
        ConversationService._fanout_join_conversation_room(conv)
        return conv

    @staticmethod
    def ensure_company_driver_conversation(company_id: int, driver: Driver) -> Conversation:
        driver_user_id = int(driver.user_id) if driver.user_id else None
        existing = (
            Conversation.query.filter_by(
                company_id=company_id,
                conversation_type=ConversationType.COMPANY.value,
                context_type=ConversationContext.COMPANY.value,
                context_id=driver.id,
            ).first()
        )
        if existing:
            return existing

        user = User.query.get(driver_user_id) if driver_user_id else None
        name = _user_display(user) if user else f"Chauffeur #{driver.id}"
        company = Company.query.get(company_id)
        conv = Conversation(
            company_id=company_id,
            conversation_type=ConversationType.COMPANY.value,
            context_type=ConversationContext.COMPANY.value,
            context_id=driver.id,
            title=name,
            legacy_thread_id=company_driver_channel_legacy_thread_id(int(driver.id)),
            conversation_metadata=Conversation.default_metadata(),
        )
        db.session.add(conv)
        db.session.flush()
        if driver_user_id:
            ConversationService._add_participant(
                conv, driver_user_id, ParticipantRole.DRIVER, can_write=True
            )
        ConversationService._ensure_dispatch_participants(conv, company_id)
        if company and company.user_id:
            ConversationService._add_participant(
                conv,
                int(company.user_id),
                ParticipantRole.COMPANY,
                can_write=True,
                can_manage=True,
            )
        db.session.commit()
        ConversationService._fanout_join_conversation_room(conv)
        return conv

    @staticmethod
    def ensure_company_group_conversation(
        company_id: int,
        *,
        title: str = "Équipe chauffeurs",
        created_by: int | None = None,
    ) -> Conversation:
        existing = Conversation.query.filter_by(
            company_id=company_id,
            conversation_type=ConversationType.GROUP.value,
            context_type=ConversationContext.SUPERVISION.value,
            legacy_thread_id=company_group_legacy_thread_id(),
        ).first()
        if existing:
            return existing

        conv = Conversation(
            company_id=company_id,
            conversation_type=ConversationType.GROUP.value,
            context_type=ConversationContext.SUPERVISION.value,
            context_id=company_id,
            title=title,
            created_by=created_by,
            legacy_thread_id=company_group_legacy_thread_id(),
            conversation_metadata=Conversation.default_metadata(),
        )
        db.session.add(conv)
        db.session.flush()
        drivers = Driver.query.filter_by(company_id=company_id, is_active=True).all()
        for d in drivers:
            if d.user_id:
                ConversationService._add_participant(
                    conv, int(d.user_id), ParticipantRole.DRIVER, can_write=True
                )
        db.session.commit()
        ConversationService._fanout_join_conversation_room(conv)
        return conv

    @staticmethod
    def ensure_company_dispatch_conversation(company_id: int) -> Conversation:
        """Canal dispatch partagé : entreprise + tous les chauffeurs actifs."""
        existing = Conversation.query.filter_by(
            company_id=company_id,
            conversation_type=ConversationType.COMPANY.value,
            context_type=ConversationContext.COMPANY.value,
            context_id=company_id,
            legacy_thread_id=company_dispatch_legacy_thread_id(),
        ).first()
        if existing:
            ConversationService._sync_dispatch_driver_participants(existing, company_id)
            return existing

        company = Company.query.get(company_id)
        conv = Conversation(
            company_id=company_id,
            conversation_type=ConversationType.COMPANY.value,
            context_type=ConversationContext.COMPANY.value,
            context_id=company_id,
            title="Dispatch",
            legacy_thread_id=company_dispatch_legacy_thread_id(),
            conversation_metadata=Conversation.default_metadata(),
        )
        db.session.add(conv)
        db.session.flush()
        ConversationService._ensure_dispatch_participants(conv, company_id)
        drivers = Driver.query.filter_by(company_id=company_id, is_active=True).all()
        for d in drivers:
            if d.user_id:
                ConversationService._add_participant(
                    conv, int(d.user_id), ParticipantRole.DRIVER, can_write=True
                )
        db.session.commit()
        ConversationService._fanout_join_conversation_room(conv)
        return conv

    @staticmethod
    def ensure_direct_driver_conversation(
        company_id: int, driver: Driver, peer_user_id: int
    ) -> Conversation:
        """Conversation 1-1 entre deux chauffeurs actifs de la même entreprise."""
        my_uid = int(driver.user_id) if driver.user_id else None
        if my_uid is None:
            raise PermissionError("Chauffeur sans compte utilisateur")
        peer_uid = int(peer_user_id)
        user = User.query.get(my_uid)
        if user:
            MessagingPermissionService.assert_can_direct_message_peer(user, peer_uid)

        pair_legacy = direct_pair_legacy_thread_id(my_uid, peer_uid)
        existing = Conversation.query.filter_by(
            company_id=company_id, legacy_thread_id=pair_legacy
        ).first()
        if existing:
            return existing

        peer_driver = Driver.query.filter_by(
            user_id=peer_uid, company_id=company_id, is_active=True
        ).first()
        if not peer_driver:
            raise PermissionError("Collègue introuvable")

        peer_user = User.query.get(peer_uid)
        title = _user_display(peer_user)
        conv = Conversation(
            company_id=company_id,
            conversation_type=ConversationType.DIRECT.value,
            context_type=ConversationContext.COMPANY.value,
            context_id=int(peer_driver.id),
            title=title,
            legacy_thread_id=pair_legacy,
            conversation_metadata=Conversation.default_metadata(),
        )
        db.session.add(conv)
        db.session.flush()
        ConversationService._add_participant(
            conv, my_uid, ParticipantRole.DRIVER, can_write=True
        )
        ConversationService._add_participant(
            conv, peer_uid, ParticipantRole.DRIVER, can_write=True
        )
        db.session.commit()
        ConversationService._fanout_join_conversation_room(conv)
        return conv

    @staticmethod
    def _add_participant(
        conv: Conversation,
        user_id: int,
        role: ParticipantRole | str,
        *,
        can_write: bool = True,
        can_manage: bool = False,
        conversation: Conversation | None = None,
    ) -> ConversationParticipant:
        target = conversation or conv
        existing = ConversationParticipant.query.filter_by(
            conversation_id=target.id, user_id=user_id
        ).first()
        if existing:
            return existing
        part = ConversationParticipant(
            conversation_id=target.id,
            user_id=user_id,
            participant_role=(
                role.value if isinstance(role, ParticipantRole) else str(role).upper()
            ),
            can_read=True,
            can_write=can_write,
            can_manage=can_manage,
        )
        db.session.add(part)
        return part

    @staticmethod
    def _ensure_dispatch_participants(conv: Conversation, company_id: int) -> None:
        company = Company.query.get(company_id)
        if company and company.user_id:
            ConversationService._add_participant(
                conv,
                int(company.user_id),
                ParticipantRole.DISPATCH,
                can_write=True,
                can_manage=True,
            )

    @staticmethod
    def _sync_dispatch_driver_participants(conv: Conversation, company_id: int) -> bool:
        """Ajoute les chauffeurs actifs absents du canal dispatch partagé."""
        existing_user_ids = {
            int(p.user_id)
            for p in ConversationParticipant.query.filter_by(
                conversation_id=conv.id, left_at=None
            ).all()
        }
        removed_user_ids = {
            int(p.user_id)
            for p in ConversationParticipant.query.filter(
                ConversationParticipant.conversation_id == conv.id,
                ConversationParticipant.left_at.isnot(None),
            ).all()
        }
        changed = False
        drivers = Driver.query.filter_by(company_id=company_id, is_active=True).all()
        for d in drivers:
            uid = int(d.user_id) if d.user_id else None
            if uid is None or uid in existing_user_ids or uid in removed_user_ids:
                continue
            ConversationService._add_participant(
                conv, uid, ParticipantRole.DRIVER, can_write=True
            )
            existing_user_ids.add(uid)
            changed = True
        if changed:
            db.session.commit()
        return changed

    @staticmethod
    def conversation_ids_for_user(user_id: int, *, limit: int = 50) -> list[int]:
        rows = (
            ConversationParticipant.query.filter_by(user_id=int(user_id), left_at=None)
            .order_by(ConversationParticipant.conversation_id.desc())
            .limit(limit)
            .all()
        )
        return [int(r.conversation_id) for r in rows if r.conversation_id is not None]

    # --- Inbox ---

    @staticmethod
    def build_driver_inbox(driver: Driver) -> dict[str, Any]:
        company_id = int(driver.company_id)
        user_id = int(driver.user_id) if driver.user_id else None
        if not user_id:
            return _empty_inbox()

        if MESSAGING_BACKFILL_ON_INBOX:
            ConversationService.backfill_company_conversations(company_id, driver)
        ConversationService.ensure_company_group_conversation(company_id)

        conv_ids = [
            p.conversation_id
            for p in ConversationParticipant.query.filter_by(
                user_id=user_id, left_at=None
            ).all()
        ]
        conversations = (
            Conversation.query.filter(Conversation.id.in_(conv_ids))
            .order_by(Conversation.id.desc())
            .all()
            if conv_ids
            else []
        )

        mission_active: list[dict] = []
        urgent: list[dict] = []
        company_rows: list[dict] = []
        groups: list[dict] = []
        colleagues: list[dict] = []
        archives: list[dict] = []

        read_ids = ConversationService._load_read_message_ids(user_id)
        for conv in conversations:
            if conv.archived_at:
                continue
            row = ConversationService._thread_row(conv, user_id, read_ids)
            ctype = str(conv.conversation_type)
            if ctype == ConversationType.MISSION.value:
                if not row.get("last_message_at"):
                    continue
                booking = (
                    Booking.query.get(conv.context_id) if conv.context_id else None
                )
                status = str(getattr(booking, "status", "") or "").upper()
                if status in _TERMINAL:
                    row["section"] = "archives"
                    archives.append(row)
                else:
                    row["section"] = "mission_active"
                    mission_active.append(row)
            elif ctype == ConversationType.COMPANY.value:
                legacy_tid = str(getattr(conv, "legacy_thread_id", "") or "")
                if legacy_tid == company_dispatch_legacy_thread_id():
                    row["section"] = "dispatch"
                    company_rows.append(row)
                else:
                    row["section"] = "company"
                    company_rows.append(row)
            elif ctype == ConversationType.GROUP.value:
                row["section"] = "groups"
                groups.append(row)
            elif ctype == ConversationType.DIRECT.value:
                peer_uid = _peer_user_id_in_conversation(conv, user_id)
                if peer_uid is None:
                    continue
                row["thread_id"] = f"{DIRECT_PREFIX}{peer_uid}"
                row["peer_user_id"] = peer_uid
                row["section"] = "colleagues"
                row["subtitle"] = "Message direct"
                colleagues.append(row)
            else:
                continue

        company_rows = _dedupe_thread_rows_by_id(company_rows, company_id=company_id)

        urgent_ids: set[int] = set()
        for row in mission_active + company_rows + groups + colleagues + archives:
            if row.get("priority") == "urgent" or _row_is_urgent_unread(row):
                cid = row.get("conversation_id")
                if cid and cid not in urgent_ids:
                    urgent_ids.add(cid)
                    urgent.append({**row, "section": "urgent"})

        _sort_threads(mission_active)
        _sort_threads(urgent)
        _sort_threads(company_rows)
        _sort_threads(groups)
        _sort_threads(colleagues)
        _sort_threads(archives)

        unread_total = sum(
            int(r.get("unread_count") or 0)
            for r in mission_active + company_rows + groups + colleagues + archives
        )

        return {
            "sections": {
                "mission_active": mission_active,
                "urgent": urgent,
                "company": company_rows,
                "groups": groups,
                "colleagues": colleagues,
                "archives": archives,
            },
            "threads": mission_active + urgent + company_rows + groups + colleagues + archives,
            "unread_total": unread_total,
        }

    @staticmethod
    def build_company_inbox(user: User) -> dict[str, Any]:
        """Inbox exploitation mobile : dispatch, chauffeurs (1-1), missions."""
        company = getattr(user, "company", None)
        if company is None:
            return _empty_company_inbox()
        company_id = int(company.id)
        user_id = int(user.id)

        ConversationService.ensure_company_dispatch_conversation(company_id)

        conversations = (
            Conversation.query.filter_by(company_id=company_id)
            .filter(Conversation.archived_at.is_(None))
            .order_by(Conversation.id.desc())
            .all()
        )

        mission_active: list[dict] = []
        urgent: list[dict] = []
        dispatch_rows: list[dict] = []
        driver_rows: list[dict] = []
        archives: list[dict] = []

        read_ids = ConversationService._load_read_message_ids(user_id)
        for conv in conversations:
            row = ConversationService._thread_row(conv, user_id, read_ids)
            ctype = str(conv.conversation_type)
            if ctype == ConversationType.MISSION.value:
                if not row.get("last_message_at"):
                    continue
                booking = (
                    Booking.query.get(conv.context_id) if conv.context_id else None
                )
                status = str(getattr(booking, "status", "") or "").upper()
                if status in _TERMINAL:
                    row["section"] = "archives"
                    archives.append(row)
                else:
                    row["section"] = "mission_active"
                    mission_active.append(row)
            elif ctype == ConversationType.COMPANY.value:
                legacy_tid = str(getattr(conv, "legacy_thread_id", "") or "")
                if legacy_tid == company_dispatch_legacy_thread_id():
                    row["section"] = "dispatch"
                    dispatch_rows.append(row)
                else:
                    row["section"] = "drivers"
                    driver_rows.append(row)
            else:
                continue

        dispatch_rows = _dedupe_thread_rows_by_id(dispatch_rows, company_id=company_id)
        driver_rows = _dedupe_thread_rows_by_id(driver_rows, company_id=company_id)

        if not dispatch_rows:
            conv = ConversationService.ensure_company_dispatch_conversation(company_id)
            row = ConversationService._thread_row(conv, user_id, read_ids)
            row["section"] = "dispatch"
            dispatch_rows.append(row)

        urgent_ids: set[int] = set()
        for row in mission_active + dispatch_rows + driver_rows + archives:
            if row.get("priority") == "urgent" or _row_is_urgent_unread(row):
                cid = row.get("conversation_id")
                if cid and cid not in urgent_ids:
                    urgent_ids.add(cid)
                    urgent.append({**row, "section": "urgent"})

        _sort_threads(mission_active)
        _sort_threads(urgent)
        _sort_threads(dispatch_rows)
        _sort_threads(driver_rows)
        _sort_threads(archives)

        unread_total = sum(
            int(r.get("unread_count") or 0)
            for r in mission_active + dispatch_rows + driver_rows + archives
        )

        return {
            "sections": {
                "mission_active": mission_active,
                "urgent": urgent,
                "dispatch": dispatch_rows,
                "drivers": driver_rows,
                "archives": archives,
            },
            "threads": mission_active + urgent + dispatch_rows + driver_rows + archives,
            "unread_total": unread_total,
        }

    @staticmethod
    def resolve_by_legacy_thread_for_company(
        company_id: int, thread_id: str
    ) -> Conversation | None:
        if thread_id == company_dispatch_legacy_thread_id():
            return ConversationService.ensure_company_dispatch_conversation(company_id)
        conv = Conversation.query.filter_by(
            company_id=company_id, legacy_thread_id=thread_id
        ).first()
        if conv:
            return conv
        booking_id = parse_mission_thread(thread_id)
        if booking_id:
            return ConversationService.ensure_mission_conversation(company_id, booking_id)
        if thread_id.startswith("company_driver:"):
            try:
                driver_id = int(thread_id.split(":", 1)[1])
            except (IndexError, ValueError):
                return None
            driver = Driver.query.filter_by(id=driver_id, company_id=company_id).first()
            if driver:
                return ConversationService.ensure_company_driver_conversation(
                    company_id, driver
                )
        return None

    @staticmethod
    def hub_threads_for_company(
        user: User, inbox: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if inbox is None:
            inbox = ConversationService.build_company_inbox(user)
        threads: list[dict] = []
        section_map = {
            "mission_active": "mission_active",
            "urgent": "urgent",
            "dispatch": "dispatch",
            "drivers": "drivers",
            "archives": "archives",
        }
        seen_conv: set[int] = set()
        seen_thread: set[str] = set()
        for key, section_key in section_map.items():
            for row in inbox["sections"].get(key, []):
                cid = row.get("conversation_id")
                tid = str(row.get("thread_id") or "")
                if cid is not None and int(cid) in seen_conv:
                    continue
                if cid is None and tid and tid in seen_thread:
                    continue
                if cid is not None:
                    seen_conv.add(int(cid))
                if tid:
                    seen_thread.add(tid)
                threads.append({**row, "section": str(row.get("section") or section_key)})
        return threads

    @staticmethod
    def _thread_row(
        conv: Conversation,
        user_id: int,
        read_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        last = (
            Message.query.filter_by(conversation_id=conv.id)
            .order_by(Message.timestamp.desc())
            .first()
        )
        unread = ConversationService.unread_count_for_user(
            conv.id, user_id, read_ids=read_ids
        )
        priority = "normal"
        if last and getattr(last, "priority", None) in ("urgent", "important"):
            priority = str(last.priority)
        legacy = conversation_id_to_legacy_thread(conv)
        booking_id = conv.context_id if conv.conversation_type == ConversationType.MISSION.value else None
        return {
            "conversation_id": conv.id,
            "thread_id": legacy or str(conv.id),
            "section": "company",
            "title": conv.title,
            "subtitle": _subtitle_for(conv),
            "booking_id": booking_id,
            "status": _booking_status(booking_id),
            "unread_count": unread,
            "priority": priority,
            "last_message_preview": _preview(last),
            "last_message_at": last.timestamp.isoformat() if last and last.timestamp else None,
            "last_message_from_self": bool(last and last.sender_id == user_id),
            "conversation_type": conv.conversation_type,
            "context_type": conv.context_type,
        }

    @staticmethod
    def _load_read_message_ids(user_id: int) -> set[int]:
        rows = (
            MessageRead.query.filter_by(user_id=user_id)
            .with_entities(MessageRead.message_id)
            .all()
        )
        return {int(r[0]) for r in rows if r[0] is not None}

    @staticmethod
    def unread_count_for_user(
        conversation_id: int,
        user_id: int,
        *,
        read_ids: set[int] | None = None,
    ) -> int:
        if read_ids is None:
            read_ids = ConversationService._load_read_message_ids(user_id)
        query = Message.query.filter(
            Message.conversation_id == conversation_id,
            Message.sender_id != user_id,
        )
        if read_ids:
            query = query.filter(~Message.id.in_(read_ids))
        return int(query.count())

    @staticmethod
    def get_messages(
        conversation: Conversation,
        user: User,
        *,
        before: datetime | None = None,
        limit: int = 40,
    ) -> list[Message]:
        MessagingPermissionService.assert_can_read(user, conversation)
        legacy_tid = getattr(conversation, "legacy_thread_id", None)
        conv_filters = [Message.conversation_id == conversation.id]
        if legacy_tid:
            conv_filters.append(
                and_(
                    Message.conversation_id.is_(None),
                    Message.company_id == conversation.company_id,
                    Message.thread_id == legacy_tid,
                )
            )
        q = Message.query.filter(or_(*conv_filters))
        if before:
            q = q.filter(Message.timestamp < before)
        rows = q.order_by(Message.timestamp.desc()).limit(limit).all()
        rows.reverse()
        dirty = False
        for msg in rows:
            if msg.conversation_id is None:
                msg.conversation_id = conversation.id
                dirty = True
        if dirty:
            db.session.commit()
        part = MessagingPermissionService.participant_for(conversation.id, user.id)
        return [
            m
            for m in rows
            if MessagingPermissionService.can_read_message(user, m, conversation, part)
        ]

    @staticmethod
    def mark_read(conversation: Conversation, user: User) -> int:
        MessagingPermissionService.assert_can_read(user, conversation)
        messages = Message.query.filter_by(conversation_id=conversation.id).all()
        updated = 0
        for msg in messages:
            if msg.sender_id == user.id:
                continue
            existing = MessageRead.query.filter_by(
                user_id=user.id, message_id=msg.id
            ).first()
            if not existing:
                db.session.add(
                    MessageRead(user_id=user.id, message_id=msg.id, read_at=datetime.now(UTC))
                )
                updated += 1
            if not msg.is_read:
                msg.is_read = True
                updated += 1
        if updated:
            db.session.commit()
        return updated

    @staticmethod
    def is_company_managed_dispatch(conversation: Conversation) -> bool:
        legacy_tid = str(getattr(conversation, "legacy_thread_id", "") or "")
        return (
            str(conversation.conversation_type) == ConversationType.COMPANY.value
            and legacy_tid == company_dispatch_legacy_thread_id()
        )

    @staticmethod
    def _participant_payload(part: ConversationParticipant) -> dict[str, Any]:
        row = dict(part.serialize)
        user = getattr(part, "user", None)
        display = _user_display(user) if user else f"Utilisateur #{part.user_id}"
        driver = (
            Driver.query.filter_by(user_id=int(part.user_id)).first()
            if part.user_id
            else None
        )
        row["display_name"] = display
        row["driver_id"] = int(driver.id) if driver else None
        role = str(part.participant_role).upper()
        row["role_label"] = (
            "Exploitation" if role == ParticipantRole.DISPATCH.value else "Chauffeur"
        )
        row["is_admin"] = role == ParticipantRole.DISPATCH.value
        last_msg = (
            Message.query.filter_by(
                conversation_id=int(part.conversation_id), sender_id=int(part.user_id)
            )
            .order_by(Message.timestamp.desc())
            .first()
        )
        row["last_activity_at"] = (
            last_msg.timestamp.isoformat()
            if last_msg and last_msg.timestamp
            else None
        )
        row["can_remove"] = (
            str(part.participant_role).upper() != ParticipantRole.DISPATCH.value
            and part.left_at is None
        )
        return row

    @staticmethod
    def list_dispatch_participants(
        conversation: Conversation, user: User
    ) -> dict[str, Any]:
        MessagingPermissionService.assert_can_read(user, conversation)
        if not ConversationService.is_company_managed_dispatch(conversation):
            raise PermissionError("Canal non gérable")
        parts = (
            ConversationParticipant.query.filter_by(
                conversation_id=conversation.id, left_at=None
            )
            .order_by(ConversationParticipant.participant_role.asc())
            .all()
        )
        active_user_ids = {int(p.user_id) for p in parts if p.user_id}
        available: list[dict[str, Any]] = []
        if MessagingPermissionService.can_manage_conversation(user, conversation):
            drivers = Driver.query.filter_by(
                company_id=int(conversation.company_id), is_active=True
            ).all()
            for d in drivers:
                uid = int(d.user_id) if d.user_id else None
                if uid is None or uid in active_user_ids:
                    continue
                available.append(
                    {
                        "driver_id": int(d.id),
                        "user_id": uid,
                        "display_name": _user_display(getattr(d, "user", None) or User.query.get(uid)),
                    }
                )
        return {
            "participants": [
                ConversationService._participant_payload(p) for p in parts
            ],
            "available_drivers": available,
            "can_manage": MessagingPermissionService.can_manage_conversation(
                user, conversation
            ),
        }

    @staticmethod
    def add_dispatch_participant(
        conversation: Conversation, user: User, *, driver_id: int
    ) -> dict[str, Any]:
        MessagingPermissionService.assert_can_manage(user, conversation)
        if not ConversationService.is_company_managed_dispatch(conversation):
            raise PermissionError("Canal non gérable")
        driver = Driver.query.filter_by(
            id=int(driver_id), company_id=int(conversation.company_id), is_active=True
        ).first()
        if not driver or not driver.user_id:
            raise PermissionError("Chauffeur introuvable")
        uid = int(driver.user_id)
        existing = ConversationParticipant.query.filter_by(
            conversation_id=conversation.id, user_id=uid
        ).first()
        if existing:
            if existing.left_at is not None:
                existing.left_at = None
                existing.can_read = True
                existing.can_write = True
                db.session.commit()
                ConversationService._append_channel_audit(
                    conversation,
                    event_type="participant_added",
                    actor_user=user,
                    detail=_user_display(getattr(driver, "user", None) or User.query.get(uid)),
                )
            return ConversationService._participant_payload(existing)
        part = ConversationService._add_participant(
            conversation, uid, ParticipantRole.DRIVER, can_write=True
        )
        db.session.commit()
        ConversationService._append_channel_audit(
            conversation,
            event_type="participant_added",
            actor_user=user,
            detail=_user_display(getattr(driver, "user", None) or User.query.get(uid)),
        )
        ConversationService._fanout_join_conversation_room(conversation)
        return ConversationService._participant_payload(part)

    @staticmethod
    def remove_dispatch_participant(
        conversation: Conversation, user: User, *, target_user_id: int
    ) -> dict[str, Any]:
        MessagingPermissionService.assert_can_manage(user, conversation)
        if not ConversationService.is_company_managed_dispatch(conversation):
            raise PermissionError("Canal non gérable")
        part = ConversationParticipant.query.filter_by(
            conversation_id=conversation.id,
            user_id=int(target_user_id),
            left_at=None,
        ).first()
        if not part:
            raise PermissionError("Participant introuvable")
        if str(part.participant_role).upper() == ParticipantRole.DISPATCH.value:
            raise PermissionError("Impossible de retirer l'exploitation du canal")
        part.left_at = datetime.now(UTC)
        removed_name = _user_display(getattr(part, "user", None) or User.query.get(int(target_user_id)))
        db.session.commit()
        ConversationService._append_channel_audit(
            conversation,
            event_type="participant_removed",
            actor_user=user,
            detail=removed_name,
        )
        return {"removed_user_id": int(target_user_id), "ok": True}

    @staticmethod
    def list_conversation_attachments(
        conversation: Conversation,
        user: User,
        *,
        limit: int = 80,
    ) -> list[dict[str, Any]]:
        MessagingPermissionService.assert_can_read(user, conversation)
        limit = max(1, min(200, int(limit)))
        rows = (
            Message.query.filter_by(conversation_id=conversation.id)
            .filter(
                or_(
                    Message.image_url.isnot(None),
                    Message.pdf_url.isnot(None),
                )
            )
            .order_by(Message.timestamp.desc())
            .limit(limit)
            .all()
        )
        items: list[dict[str, Any]] = []
        for msg in rows:
            if msg.image_url:
                items.append(
                    {
                        "id": f"{msg.id}-img",
                        "message_id": msg.id,
                        "kind": "photo",
                        "url": msg.image_url,
                        "label": "Photo",
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    }
                )
            if msg.pdf_url:
                items.append(
                    {
                        "id": f"{msg.id}-pdf",
                        "message_id": msg.id,
                        "kind": "document",
                        "url": msg.pdf_url,
                        "label": getattr(msg, "pdf_filename", None) or "Document PDF",
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    }
                )
            audio_url = getattr(msg, "audio_url", None)
            if audio_url:
                items.append(
                    {
                        "id": f"{msg.id}-audio",
                        "message_id": msg.id,
                        "kind": "audio",
                        "url": audio_url,
                        "label": "Message vocal",
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    }
                )
        return items

    @staticmethod
    def _channel_description(conversation: Conversation) -> str:
        meta = conversation.conversation_metadata or {}
        desc = meta.get("description") if isinstance(meta, dict) else None
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
        return "Gestion réservée à l'exploitation"

    @staticmethod
    def _dispatch_permissions(can_manage: bool) -> dict[str, bool]:
        return {
            "add_participants": can_manage,
            "send_files": True,
            "reply": True,
            "edit_channel": can_manage,
            "delete_messages": False,
        }

    @staticmethod
    def _append_channel_audit(
        conversation: Conversation,
        *,
        event_type: str,
        actor_user: User,
        detail: str,
    ) -> None:
        meta = dict(conversation.conversation_metadata or Conversation.default_metadata())
        log = list(meta.get("audit_log") or [])
        log.insert(
            0,
            {
                "at": datetime.now(UTC).isoformat(),
                "type": event_type,
                "actor": _user_display(actor_user),
                "detail": detail,
            },
        )
        meta["audit_log"] = log[:50]
        conversation.conversation_metadata = meta
        db.session.commit()

    @staticmethod
    def _channel_history(conversation: Conversation) -> list[dict[str, Any]]:
        meta = conversation.conversation_metadata or {}
        log = list(meta.get("audit_log") or []) if isinstance(meta, dict) else []
        history: list[dict[str, Any]] = []
        creator = (
            User.query.get(int(conversation.created_by))
            if conversation.created_by
            else None
        )
        creator_name = _user_display(creator) if creator else "Système"
        created_at = conversation.created_at.isoformat() if conversation.created_at else None
        history.append(
            {
                "at": created_at,
                "label": f"{creator_name} a créé le canal",
                "type": "channel_created",
            }
        )
        type_labels = {
            "participant_added": lambda d: f"{d} ajouté",
            "participant_removed": lambda d: f"{d} retiré",
            "channel_renamed": lambda d: f"Nom modifié · {d}",
            "description_updated": lambda d: "Description modifiée",
            "history_cleared": lambda d: f"Historique vidé · {d} message(s)",
        }
        for entry in log:
            if not isinstance(entry, dict):
                continue
            et = str(entry.get("type") or "")
            detail = str(entry.get("detail") or "")
            fn = type_labels.get(et)
            label = fn(detail) if fn else detail or et
            history.append(
                {
                    "at": entry.get("at"),
                    "label": label,
                    "type": et,
                }
            )
        return history

    @staticmethod
    def get_dispatch_channel_manage(
        conversation: Conversation, user: User
    ) -> dict[str, Any]:
        base = ConversationService.list_dispatch_participants(conversation, user)
        attachments = ConversationService.list_conversation_attachments(
            conversation, user, limit=120
        )
        counts = {"all": len(attachments), "photo": 0, "document": 0, "audio": 0}
        for item in attachments:
            kind = str(item.get("kind") or "")
            if kind in counts:
                counts[kind] += 1
        creator = (
            User.query.get(int(conversation.created_by))
            if conversation.created_by
            else None
        )
        can_manage = bool(base.get("can_manage"))
        return {
            **base,
            "channel": {
                "id": conversation.id,
                "title": conversation.title or "Dispatch",
                "description": ConversationService._channel_description(conversation),
                "channel_type_label": "Canal privé",
                "legacy_thread_id": conversation.legacy_thread_id,
                "created_at": conversation.created_at.isoformat()
                if conversation.created_at
                else None,
                "created_by_name": _user_display(creator) if creator else "—",
                "participant_count": len(base.get("participants") or []),
                "attachment_count": counts["all"],
            },
            "attachments_preview": attachments[:6],
            "attachments_all": attachments,
            "attachment_counts": counts,
            "permissions": ConversationService._dispatch_permissions(can_manage),
            "history": ConversationService._channel_history(conversation),
        }

    @staticmethod
    def update_dispatch_channel(
        conversation: Conversation,
        user: User,
        *,
        title: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        MessagingPermissionService.assert_can_manage(user, conversation)
        if not ConversationService.is_company_managed_dispatch(conversation):
            raise PermissionError("Canal non gérable")
        changed = False
        if title is not None:
            clean = title.strip()
            if clean and clean != (conversation.title or ""):
                conversation.title = clean[:255]
                ConversationService._append_channel_audit(
                    conversation,
                    event_type="channel_renamed",
                    actor_user=user,
                    detail=clean,
                )
                changed = True
        if description is not None:
            meta = dict(conversation.conversation_metadata or Conversation.default_metadata())
            clean_desc = description.strip()[:500]
            if clean_desc != meta.get("description"):
                meta["description"] = clean_desc
                conversation.conversation_metadata = meta
                ConversationService._append_channel_audit(
                    conversation,
                    event_type="description_updated",
                    actor_user=user,
                    detail=clean_desc,
                )
                changed = True
        if changed:
            db.session.commit()
        return ConversationService.get_dispatch_channel_manage(conversation, user)

    @staticmethod
    def clear_dispatch_channel_history(
        conversation: Conversation, user: User
    ) -> dict[str, Any]:
        MessagingPermissionService.assert_can_manage(user, conversation)
        if not ConversationService.is_company_managed_dispatch(conversation):
            raise PermissionError("Canal non gérable")
        legacy_tid = conversation_id_to_legacy_thread(conversation) or getattr(
            conversation, "legacy_thread_id", None
        )
        filters = [Message.conversation_id == conversation.id]
        if legacy_tid:
            filters.append(
                and_(
                    Message.conversation_id.is_(None),
                    Message.company_id == conversation.company_id,
                    Message.thread_id == legacy_tid,
                )
            )
        message_ids = [
            int(row[0])
            for row in Message.query.filter(or_(*filters))
            .with_entities(Message.id)
            .all()
        ]
        deleted = len(message_ids)
        if message_ids:
            MessageRead.query.filter(MessageRead.message_id.in_(message_ids)).delete(
                synchronize_session=False
            )
            Message.query.filter(Message.id.in_(message_ids)).delete(
                synchronize_session=False
            )
            db.session.commit()
        ConversationService._append_channel_audit(
            conversation,
            event_type="history_cleared",
            actor_user=user,
            detail=str(deleted),
        )
        return ConversationService.get_dispatch_channel_manage(conversation, user)

    @staticmethod
    def resolve_by_legacy_thread(
        company_id: int, thread_id: str, driver: Driver | None = None
    ) -> Conversation | None:
        peer_uid = parse_direct_thread(thread_id)
        if peer_uid is not None and driver and driver.user_id:
            pair_legacy = direct_pair_legacy_thread_id(int(driver.user_id), peer_uid)
            conv = Conversation.query.filter_by(
                company_id=company_id, legacy_thread_id=pair_legacy
            ).first()
            if conv:
                return conv
            return ConversationService.ensure_direct_driver_conversation(
                company_id, driver, peer_uid
            )

        if thread_id == company_dispatch_legacy_thread_id():
            return ConversationService.ensure_company_dispatch_conversation(company_id)

        conv = Conversation.query.filter_by(
            company_id=company_id, legacy_thread_id=thread_id
        ).first()
        if conv:
            return conv
        booking_id = parse_mission_thread(thread_id)
        if booking_id:
            return ConversationService.ensure_mission_conversation(company_id, booking_id)
        if thread_id.startswith("company_driver:") and driver:
            try:
                driver_ctx_id = int(thread_id.split(":", 1)[1])
            except (IndexError, ValueError):
                driver_ctx_id = None
            if driver_ctx_id is not None and int(driver.id) == driver_ctx_id:
                return ConversationService.ensure_company_driver_conversation(
                    company_id, driver
                )
        if thread_id == company_group_legacy_thread_id():
            return ConversationService.ensure_company_group_conversation(company_id)
        if thread_id == THREAD_SUPPORT:
            return None
        return None

    @staticmethod
    def backfill_company_conversations(company_id: int, driver: Driver) -> None:
        """Attach legacy messages to conversations and link driver."""
        ConversationService.ensure_company_dispatch_conversation(company_id)
        ConversationService.ensure_company_group_conversation(company_id)
        messages = (
            Message.query.filter_by(company_id=company_id)
            .filter(Message.conversation_id.is_(None))
            .limit(500)
            .all()
        )
        for msg in messages:
            tid = getattr(msg, "thread_id", None) or ""
            if tid.startswith("mission:") and msg.booking_id:
                conv = ConversationService.ensure_mission_conversation(
                    company_id, int(msg.booking_id)
                )
            elif tid == company_group_legacy_thread_id():
                conv = ConversationService.ensure_company_group_conversation(company_id)
            elif tid == company_dispatch_legacy_thread_id() or (
                not tid
                and str(getattr(msg.sender_role, "value", msg.sender_role)) == "COMPANY"
                and getattr(msg, "receiver_id", None) is None
            ):
                conv = ConversationService.ensure_company_dispatch_conversation(company_id)
                if not getattr(msg, "thread_id", None):
                    msg.thread_id = company_dispatch_legacy_thread_id()
            elif tid.startswith("company_driver:"):
                if driver:
                    conv = ConversationService.ensure_company_driver_conversation(
                        company_id, driver
                    )
                else:
                    continue
            elif (
                not tid
                and str(getattr(msg.sender_role, "value", msg.sender_role)) == "DRIVER"
                and getattr(msg, "receiver_id", None) is None
            ):
                conv = ConversationService.ensure_company_group_conversation(company_id)
                msg.thread_id = company_group_legacy_thread_id()
            elif tid == THREAD_SUPPORT:
                if not getattr(msg, "thread_id", None):
                    msg.thread_id = THREAD_SUPPORT
                continue
            elif tid.startswith("mission:"):
                bid = parse_mission_thread(tid)
                if bid:
                    conv = ConversationService.ensure_mission_conversation(company_id, bid)
                else:
                    continue
            elif tid.startswith(DIRECT_PREFIX):
                peer_uid = parse_direct_thread(tid)
                if peer_uid is None:
                    pair = parse_direct_pair_legacy_thread(tid)
                    if pair and driver and driver.user_id:
                        peer_uid = pair[1] if int(driver.user_id) == pair[0] else pair[0]
                if peer_uid and driver:
                    conv = ConversationService.ensure_direct_driver_conversation(
                        company_id, driver, int(peer_uid)
                    )
                else:
                    continue
            else:
                continue
            msg.conversation_id = conv.id
            if not msg.visibility_tags:
                msg.visibility_tags = list(DEFAULT_MESSAGE_VISIBILITY_TAGS)
        db.session.commit()

    # Hub-compatible thread list
    @staticmethod
    def hub_threads_for_driver(
        driver: Driver, inbox: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if inbox is None:
            inbox = ConversationService.build_driver_inbox(driver)
        threads: list[dict] = []
        section_map = {
            "mission_active": "mission_active",
            "urgent": "urgent",
            "company": "company",
            "groups": "team",
            "colleagues": "colleagues",
            "archives": "archives",
        }
        # company_rows peut contenir dispatch (partagé) et company (1-1 exploitation-chauffeur)
        seen: set[int] = set()
        for key, section_key in section_map.items():
            for row in inbox["sections"].get(key, []):
                cid = row.get("conversation_id")
                if cid in seen and key != "urgent":
                    continue
                if key != "urgent":
                    seen.add(cid)
                resolved = (
                    str(row.get("section") or section_key)
                    if key == "company"
                    else section_key
                )
                threads.append({**row, "section": resolved})

        from services.messages.hub_service import list_driver_colleagues

        company_id = int(driver.company_id)
        roster_peer_ids = {
            int(r.get("peer_user_id"))
            for r in threads
            if r.get("section") == "colleagues" and r.get("peer_user_id") is not None
        }
        for col in list_driver_colleagues(company_id, driver):
            peer = int(col["peer_user_id"])
            if peer in roster_peer_ids:
                continue
            threads.append(
                {
                    "thread_id": str(col["thread_id"]),
                    "section": "colleagues",
                    "title": str(col.get("title") or "Collègue"),
                    "subtitle": "Message direct",
                    "peer_user_id": peer,
                    "booking_id": None,
                    "status": None,
                    "unread_count": 0,
                    "priority": "normal",
                    "last_message_preview": "Démarrer une conversation",
                    "last_message_at": None,
                    "conversation_id": None,
                }
            )
        return threads


def _peer_user_id_in_conversation(conv: Conversation, viewer_user_id: int) -> int | None:
    for part in conv.participants or []:
        uid = int(part.user_id)
        if uid != int(viewer_user_id):
            return uid
    return None


def _empty_inbox() -> dict:
    return {
        "sections": {
            "mission_active": [],
            "urgent": [],
            "company": [],
            "groups": [],
            "colleagues": [],
            "archives": [],
        },
        "threads": [],
        "unread_total": 0,
    }


def _empty_company_inbox() -> dict:
    return {
        "sections": {
            "mission_active": [],
            "urgent": [],
            "dispatch": [],
            "drivers": [],
            "archives": [],
        },
        "threads": [],
        "unread_total": 0,
    }


def _booking_label(booking: Booking | None) -> str:
    if not booking:
        return "Mission"
    client = getattr(booking, "client", None)
    name = None
    if client:
        name = getattr(client, "full_name", None) or getattr(client, "first_name", None)
    return str(name) if name else f"Mission #{booking.id}"


def _user_display(user: User | None) -> str:
    if not user:
        return "Chauffeur"
    first = getattr(user, "first_name", "") or ""
    last = getattr(user, "last_name", "") or ""
    full = f"{first} {last}".strip()
    return full or "Chauffeur"


def _subtitle_for(conv: Conversation) -> str:
    if conv.conversation_type == ConversationType.MISSION.value:
        return f"Mission #{conv.context_id}"
    if conv.conversation_type == ConversationType.GROUP.value:
        return "Canal groupe"
    legacy_tid = str(getattr(conv, "legacy_thread_id", "") or "")
    if legacy_tid == company_dispatch_legacy_thread_id():
        return ConversationService._channel_description(conv)
    return "Exploitation & régulation"


def _booking_status(booking_id: int | None) -> str | None:
    if not booking_id:
        return None
    b = Booking.query.get(booking_id)
    return str(getattr(b, "status", "")) if b else None


def _preview(msg: Message | None) -> str | None:
    if not msg:
        return None
    if getattr(msg, "message_type", None) == "system":
        return msg.content or "Événement mission"
    if getattr(msg, "image_url", None):
        return "Photo"
    if getattr(msg, "pdf_url", None):
        return "Document"
    return (msg.content or "").strip() or None


def _row_is_urgent_unread(row: dict) -> bool:
    return row.get("priority") == "urgent" and (row.get("unread_count") or 0) > 0


def _dedupe_thread_rows_by_id(
    rows: list[dict], *, company_id: int | None = None
) -> list[dict]:
    """Une seule ligne par thread_id (évite dispatch dupliqué si plusieurs conversations legacy)."""
    dispatch_canonical_id: int | None = None
    if company_id is not None:
        try:
            dispatch_canonical_id = ConversationService.ensure_company_dispatch_conversation(
                company_id
            ).id
        except Exception:
            dispatch_canonical_id = None

    by_tid: dict[str, dict] = {}
    for row in rows:
        tid = str(row.get("thread_id") or "")
        if not tid:
            continue
        prev = by_tid.get(tid)
        if prev is None:
            by_tid[tid] = row
            continue
        if (
            tid == company_dispatch_legacy_thread_id()
            and dispatch_canonical_id is not None
        ):
            row_cid = int(row.get("conversation_id") or 0)
            prev_cid = int(prev.get("conversation_id") or 0)
            if row_cid == dispatch_canonical_id and prev_cid != dispatch_canonical_id:
                by_tid[tid] = row
                continue
            if prev_cid == dispatch_canonical_id and row_cid != dispatch_canonical_id:
                continue
        prev_ts = prev.get("last_message_at") or ""
        row_ts = row.get("last_message_at") or ""
        if row_ts > prev_ts:
            by_tid[tid] = row
        elif row_ts == prev_ts and int(row.get("unread_count") or 0) > int(
            prev.get("unread_count") or 0
        ):
            by_tid[tid] = row
    return list(by_tid.values())


def _sort_threads(rows: list[dict]) -> None:
    rows.sort(
        key=lambda r: (
            -(r.get("unread_count") or 0),
            r.get("last_message_at") or "",
        ),
        reverse=True,
    )
