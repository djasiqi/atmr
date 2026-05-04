# models/curator_team.py
"""Models CuratorTeam & CuratorTeamMember — Équipes de curateurs (curatelle).

Les curateurs sont organisés en équipes qui gèrent un sous-ensemble de protégés.
Le rôle institution_curator ne voit que les patients assignés à son/ses équipe(s).
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db

from .base import _iso


class CuratorTeam(db.Model):
    """Équipe de curateurs au sein d'une institution de type curatelle."""

    __tablename__ = "curator_teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id = Column(
        String(36),
        default=lambda: str(uuid.uuid4()),
        unique=True,
        nullable=False,
        index=True,
    )
    institution_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("institutions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relations
    members = relationship(
        "CuratorTeamMember",
        back_populates="team",
        cascade="all, delete-orphan",
    )
    patients = relationship(
        "InstitutionPatient",
        backref="curator_team",
        foreign_keys="InstitutionPatient.curator_team_id",
    )

    @property
    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "public_id": self.public_id,
            "institution_id": self.institution_id,
            "name": self.name,
            "members_count": len(self.members) if self.members else 0,
            "patients_count": len(self.patients) if self.patients else 0,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


class CuratorTeamMember(db.Model):
    """Appartenance d'un utilisateur à une équipe de curateurs."""

    __tablename__ = "curator_team_members"
    __table_args__ = (
        UniqueConstraint("team_id", "user_id", name="uq_curator_team_member"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("curator_teams.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relations
    team = relationship("CuratorTeam", back_populates="members")
    user = relationship("User")

    @property
    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "team_id": self.team_id,
            "user_id": self.user_id,
            "user_public_id": self.user.public_id if self.user else None,
            "user_name": (
                f"{self.user.first_name} {self.user.last_name}" if self.user else None
            ),
            "user_email": self.user.email if self.user else None,
            "created_at": _iso(self.created_at),
        }
