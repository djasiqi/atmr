# models/message.py
"""Model Message - Gestion des messages entre chauffeurs et entreprises.
Extrait depuis models.py (lignes ~1926-2014).
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, ForeignKey, Index, Integer, Text, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from typing_extensions import override

from ext import db

from .base import _as_bool, _iso
from .enums import SenderRole


class Message(db.Model):
    __tablename__ = "message"
    __table_args__ = (
        Index(
            "ix_msg_company_receiver_unread_ts",
            "company_id",
            "receiver_id",
            "is_read",
            "timestamp"),
        CheckConstraint(
            "sender_role IN ('DRIVER','COMPANY')",
            name="check_sender_role_valid"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    company_id = Column(
        Integer,
        ForeignKey(
            "company.id",
            ondelete="CASCADE"),
        nullable=False,
        index=True)
    sender_id = Column(
        Integer,
        ForeignKey(
            "user.id",
            ondelete="SET NULL"),
        nullable=True,
        index=True)
    receiver_id = Column(
        Integer,
        ForeignKey(
            "user.id",
            ondelete="SET NULL"),
        nullable=True,
        index=True)

    sender_role = Column(
        SAEnum(
            SenderRole,
            name="sender_role"),
        nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    timestamp = Column(
        DateTime(
            timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True)
    is_read = Column(Boolean, nullable=False, default=False)

    # Relations
    sender = relationship("User", foreign_keys=[sender_id], lazy="joined")
    receiver = relationship("User", foreign_keys=[receiver_id], lazy="joined")
    company = relationship("Company", lazy="joined")

    @override
    def __repr__(self):
        return f"<Message {self.id} from {getattr(self.sender_role, 'value', self.sender_role)} ({self.sender_id})>"

    @property
    def serialize(self):
        sender_role_val = getattr(self.sender_role, "value", self.sender_role)
        sender_user = getattr(self, "sender", None)
        receiver_user = getattr(self, "receiver", None)
        company_obj = getattr(self, "company", None)
        sender_role_str = str(sender_role_val)
        if sender_role_str == "DRIVER":
            sender_name = getattr(sender_user, "first_name", None)
        else:
            sender_name = getattr(company_obj, "name", None)
        receiver_name = getattr(
            receiver_user,
            "first_name",
            None) if receiver_user else None
        return {
            "id": self.id,
            "company_id": self.company_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "sender_role": sender_role_val,
            "sender_name": sender_name,
            "receiver_name": receiver_name,
            "content": self.content,
            "timestamp": _iso(self.timestamp),
            "is_read": _as_bool(self.is_read),
        }

    # Validateurs
    @validates("company_id")
    def _v_company_id(self, _k, v):
        if not isinstance(v, int) or v <= 0:
            msg = "company_id invalide."
            raise ValueError(msg)
        return v

    @validates("sender_role")
    def _v_sender_role(self, _k, v):
        if isinstance(v, str):
            try:
                v = SenderRole(v.upper())
            except ValueError:
                msg = "Le rôle de l'expéditeur doit être 'DRIVER' ou 'COMPANY'"
                raise ValueError(msg) from None
        if not isinstance(v, SenderRole):
            msg = "sender_role invalide."
            raise ValueError(msg)
        return v

    @validates("content")
    def _v_content(self, _k, text):
        if not text or not str(text).strip():
            msg = "Le contenu du message ne peut pas être vide."
            raise ValueError(msg)
        return text.strip()
