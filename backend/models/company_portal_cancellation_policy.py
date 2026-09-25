"""Politique d'annulation / no-show / attente présentée au client PORTAL.

Propriété de l'entreprise de transport. LIRIE stocke, versionne, présente et
fige le snapshot — sans inventer de montants.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from ext import db


class CompanyPortalCancellationPolicy(db.Model):
    """Version immuable des conditions d'annulation d'une entreprise (PORTAL)."""

    __tablename__ = "company_portal_cancellation_policy"
    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "version",
            name="uq_company_portal_cancellation_policy_version",
        ),
        Index(
            "uq_company_portal_cancellation_policy_current",
            "company_id",
            unique=True,
            postgresql_where=text("is_current IS TRUE"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="RESTRICT"),
        nullable=False,
    )
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    body_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    effective_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
