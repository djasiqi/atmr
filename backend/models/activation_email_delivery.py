"""Historique des livraisons email d'activation (Lot 1)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db

# Statuts livraison (matrice Lot 1)
EMAIL_DELIVERY_QUEUED = "queued"
EMAIL_DELIVERY_SENDING = "sending"
EMAIL_DELIVERY_SENT = "sent"
EMAIL_DELIVERY_FAILED = "failed"
EMAIL_DELIVERY_DELIVERED = "delivered"
EMAIL_DELIVERY_SOFT_BOUNCED = "soft_bounced"
EMAIL_DELIVERY_HARD_BOUNCED = "hard_bounced"
EMAIL_DELIVERY_SPAM = "spam"
EMAIL_DELIVERY_BLOCKED = "blocked"
EMAIL_DELIVERY_INVALID = "invalid"

EMAIL_DELIVERY_KIND_INITIAL = "initial"
EMAIL_DELIVERY_KIND_RESEND = "resend"

# Transitions autorisées (Lot 1 v5.1)
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    EMAIL_DELIVERY_QUEUED: frozenset({EMAIL_DELIVERY_SENDING, EMAIL_DELIVERY_FAILED}),
    EMAIL_DELIVERY_SENDING: frozenset(
        {
            EMAIL_DELIVERY_SENT,
            EMAIL_DELIVERY_FAILED,
            EMAIL_DELIVERY_DELIVERED,
            EMAIL_DELIVERY_SOFT_BOUNCED,
            EMAIL_DELIVERY_HARD_BOUNCED,
            EMAIL_DELIVERY_SPAM,
            EMAIL_DELIVERY_BLOCKED,
            EMAIL_DELIVERY_INVALID,
        }
    ),
    EMAIL_DELIVERY_SENT: frozenset(
        {
            EMAIL_DELIVERY_DELIVERED,
            EMAIL_DELIVERY_SOFT_BOUNCED,
            EMAIL_DELIVERY_HARD_BOUNCED,
            EMAIL_DELIVERY_SPAM,
            EMAIL_DELIVERY_BLOCKED,
            EMAIL_DELIVERY_INVALID,
        }
    ),
    EMAIL_DELIVERY_FAILED: frozenset(),
    EMAIL_DELIVERY_DELIVERED: frozenset(
        {
            EMAIL_DELIVERY_HARD_BOUNCED,
            EMAIL_DELIVERY_SPAM,
            EMAIL_DELIVERY_BLOCKED,
        }
    ),
    EMAIL_DELIVERY_SOFT_BOUNCED: frozenset(
        {
            EMAIL_DELIVERY_DELIVERED,
            EMAIL_DELIVERY_HARD_BOUNCED,
            EMAIL_DELIVERY_SPAM,
            EMAIL_DELIVERY_BLOCKED,
            EMAIL_DELIVERY_INVALID,
        }
    ),
    EMAIL_DELIVERY_HARD_BOUNCED: frozenset(),
    EMAIL_DELIVERY_SPAM: frozenset(),
    EMAIL_DELIVERY_BLOCKED: frozenset(),
    EMAIL_DELIVERY_INVALID: frozenset(),
}

# Statuts webhook déjà avancés — ne pas rétrograder vers sent
WEBHOOK_ADVANCED_STATUSES = frozenset(
    {
        EMAIL_DELIVERY_DELIVERED,
        EMAIL_DELIVERY_SOFT_BOUNCED,
        EMAIL_DELIVERY_HARD_BOUNCED,
        EMAIL_DELIVERY_SPAM,
        EMAIL_DELIVERY_BLOCKED,
        EMAIL_DELIVERY_INVALID,
    }
)

SENDING_LEASE_MINUTES = 5


class ActivationEmailDelivery(db.Model):
    __tablename__ = "activation_email_deliveries"
    __table_args__ = (
        Index("ix_act_email_del_session_id", "activation_session_pk"),
        Index("ix_act_email_del_delivery_id", "email_delivery_id", unique=True),
        Index("ix_act_email_del_provider_msg", "provider_message_id"),
        Index("ix_act_email_del_token_hash", "email_token_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    activation_session_pk: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("activation_session.id", ondelete="CASCADE"),
        nullable=False,
    )
    email_delivery_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        default=lambda: str(uuid.uuid4()),
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=EMAIL_DELIVERY_QUEUED
    )
    token_key_version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="1"
    )
    email_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    provider_message_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    provider_accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sending_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, onupdate=func.now()
    )

    session = relationship(
        "ActivationSession",
        back_populates="email_deliveries",
        foreign_keys=[activation_session_pk],
    )


class BrevoWebhookEvent(db.Model):
    """Idempotence webhooks Brevo (clé SHA-256 calculée, pas le id Brevo)."""

    __tablename__ = "brevo_webhook_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )
    event_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provider_message_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    email_delivery_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
