# models/booking_message.py
"""Model BookingMessage — Mini-canal de communication entre entreprise et institution.

Messages texte lies a un booking, affiches dans le panel de detail.
Enum separe BookingMessageSender pour ne pas toucher SenderRole (driver/company).
"""

from __future__ import annotations

from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, validates

from ext import db

from .base import _iso

MAX_CONTENT_LENGTH = 500


class BookingMessageSender(str, PyEnum):
    """Enum separe — NE PAS ajouter a SenderRole (models/enums.py)."""

    COMPANY = "COMPANY"
    INSTITUTION = "INSTITUTION"
    CLIENT = "CLIENT"


class BookingMessage(db.Model):
    """Message dans le mini-chat d'un booking institution."""

    __tablename__ = "booking_messages"
    __table_args__ = (Index("ix_bmsg_booking_created", "booking_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id = Column(
        Integer,
        ForeignKey("booking.id", ondelete="CASCADE"),
        nullable=False,
    )
    sender_user_id = Column(
        Integer,
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )
    sender_type = Column(
        SAEnum(BookingMessageSender, name="bookingmessagesender"),
        nullable=False,
    )
    sender_label = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @validates("content")
    def _validate_content(self, _key, value):
        if value is None:
            msg = "Le contenu du message ne peut pas etre vide."
            raise ValueError(msg)
        text = value.strip() if isinstance(value, str) else str(value).strip()
        if not text:
            msg = "Le contenu du message ne peut pas etre vide."
            raise ValueError(msg)
        if len(text) > MAX_CONTENT_LENGTH:
            msg = "Le message ne peut pas depasser 500 caracteres."
            raise ValueError(msg)
        return text

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    @property
    def serialize(self):
        st = getattr(self, "sender_type", None)
        return {
            "id": self.id,
            "booking_id": self.booking_id,
            "sender_type": st.value if st else None,
            "sender_label": self.sender_label,
            "content": self.content,
            "created_at": _iso(self.created_at),
        }

    def __repr__(self):  # type: ignore[override]
        return f"<BookingMessage {self.id} booking={self.booking_id} type={self.sender_type}>"
