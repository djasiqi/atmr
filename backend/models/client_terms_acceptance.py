"""Registre d'acceptation des conditions du client privé.

Deux documents distincts : CGU (``terms_of_service``) et CGV de transport
(``transport_terms``). Une acceptation est une insertion. Le service
applicatif n'expose ni mise à jour ni suppression.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from ext import db

DOCUMENT_TERMS_OF_SERVICE = "terms_of_service"
DOCUMENT_TRANSPORT_TERMS = "transport_terms"
VERIFICATION_OTP_SMS = "otp_sms"
VERIFICATION_NOT_VERIFIED = "not_verified"
TERMS_LOCALE_FR_CH = "fr-CH"


class LegalDocumentVersion(db.Model):
    """Version publiée et figée d'un document contractuel."""

    __tablename__ = "legal_document_version"
    __table_args__ = (
        UniqueConstraint(
            "document_type",
            "terms_version",
            name="uq_legal_document_version_identity",
        ),
        CheckConstraint(
            "document_type IN ('terms_of_service', 'transport_terms')",
            name="ck_legal_document_version_type",
        ),
        CheckConstraint(
            "char_length(terms_hash) = 64",
            name="ck_legal_document_version_hash_len",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_type: Mapped[str] = mapped_column(String(32), nullable=False)
    terms_version: Mapped[str] = mapped_column(String(32), nullable=False)
    terms_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    locale: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=TERMS_LOCALE_FR_CH
    )
    canonical_body: Mapped[str] = mapped_column(Text, nullable=False)
    requires_reacceptance: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ClientTermsAcceptance(db.Model):
    """Preuve d'acceptation. Une nouvelle acceptation insère une nouvelle ligne."""

    __tablename__ = "client_terms_acceptance"
    __table_args__ = (
        Index("ix_client_terms_acceptance_user_id", "user_id"),
        Index(
            "ix_client_terms_acceptance_user_document_accepted",
            "user_id",
            "document_type",
            "accepted_at",
        ),
        CheckConstraint(
            "document_type IN ('terms_of_service', 'transport_terms')",
            name="ck_client_terms_acceptance_type",
        ),
        CheckConstraint(
            "verification_method IN ('otp_sms', 'not_verified')",
            name="ck_client_terms_acceptance_verification",
        ),
        CheckConstraint(
            "char_length(terms_hash) = 64",
            name="ck_client_terms_acceptance_hash_len",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="RESTRICT"),
        nullable=False,
    )
    document_version_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("legal_document_version.id", ondelete="RESTRICT"),
        nullable=False,
    )
    document_type: Mapped[str] = mapped_column(String(32), nullable=False)
    terms_version: Mapped[str] = mapped_column(String(32), nullable=False)
    terms_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    accepted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    email_snapshot: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email_verified_at_snapshot: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    phone_snapshot: Mapped[str | None] = mapped_column(String(32), nullable=True)
    phone_verified_at_snapshot: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    verification_method: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
