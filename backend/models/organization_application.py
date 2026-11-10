"""Demande de préinscription Entreprise ou Institution.

Une ligne n'est pas une société, ni une institution, ni une habilitation.
Aucun portail n'est ouvert depuis ce dossier.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from ext import db

from .enums import OrganizationApplicationKind, OrganizationApplicationStatus


class OrganizationApplication(db.Model):
    """Dossier déposé par un représentant, avant toute validation LIRIE."""

    __tablename__ = "organization_application"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('company', 'institution')",
            name="ck_org_application_kind",
        ),
        CheckConstraint(
            "status IN ("
            "'email_unverified', 'received', 'in_review', 'needs_info', "
            "'approved', 'refused', 'abandoned')",
            name="ck_org_application_status",
        ),
        CheckConstraint(
            "("
            "kind = 'company' AND legal_name IS NOT NULL "
            "AND establishment_name IS NULL AND establishment_category IS NULL"
            ") OR ("
            "kind = 'institution' AND establishment_name IS NOT NULL "
            "AND establishment_category IS NOT NULL "
            "AND legal_name IS NULL AND ide_number IS NULL"
            ")",
            name="ck_org_application_kind_fields",
        ),
        Index(
            "uq_org_application_open_email_kind",
            "applicant_email",
            "kind",
            unique=True,
            postgresql_where=text(
                "status IN ('email_unverified', 'received', 'in_review', 'needs_info')"
            ),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=OrganizationApplicationStatus.EMAIL_UNVERIFIED.value,
        server_default=OrganizationApplicationStatus.EMAIL_UNVERIFIED.value,
    )
    applicant_first_name: Mapped[str] = mapped_column(String(80), nullable=False)
    applicant_last_name: Mapped[str] = mapped_column(String(80), nullable=False)
    applicant_function: Mapped[str] = mapped_column(String(80), nullable=False)
    applicant_email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    applicant_phone: Mapped[str] = mapped_column(String(30), nullable=False)
    legal_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    ide_number: Mapped[str | None] = mapped_column(String(20), nullable=True)
    establishment_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    establishment_category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    street: Mapped[str] = mapped_column(String(255), nullable=False)
    postal_code: Mapped[str] = mapped_column(String(12), nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=False)
    country: Mapped[str] = mapped_column(String(2), nullable=False, default="CH")
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    matched_company_id: Mapped[int | None] = mapped_column(
        ForeignKey("company.id", ondelete="SET NULL"),
        nullable=True,
    )
    matched_institution_id: Mapped[int | None] = mapped_column(
        ForeignKey("institutions.id", ondelete="SET NULL"),
        nullable=True,
    )
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )
    email_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    submitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        onupdate=func.now(),
    )

    events = relationship(
        "OrganizationApplicationEvent",
        back_populates="application",
        cascade="all, delete-orphan",
    )
    tokens = relationship(
        "OrganizationApplicationToken",
        back_populates="application",
        cascade="all, delete-orphan",
    )

    def organization_label(self) -> str:
        if self.kind == OrganizationApplicationKind.COMPANY.value:
            return self.legal_name or ""
        return self.establishment_name or ""


class OrganizationApplicationEvent(db.Model):
    """Historique d'un dossier. Les notes internes ne sortent pas du suivi public."""

    __tablename__ = "organization_application_event"
    __table_args__ = (
        CheckConstraint(
            "visibility IN ('public', 'internal')",
            name="ck_org_application_event_visibility",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("organization_application.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    visibility: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="public",
        server_default="public",
    )
    actor_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    application = relationship("OrganizationApplication", back_populates="events")


class OrganizationApplicationToken(db.Model):
    """Empreinte d'un lien. Le secret en clair n'est pas conservé."""

    __tablename__ = "organization_application_token"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("organization_application.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    application = relationship("OrganizationApplication", back_populates="tokens")
