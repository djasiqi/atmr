# models/patient_identity.py
"""Patient Master Index — Synchronisation cross-plateforme via AVS.

Tables:
- PatientIdentity: identité unique d'un patient (clé = hash AVS)
- PatientIdentityLink: liens vers les enregistrements entity-specific
- PatientSyncEvent: outbox pour la propagation asynchrone des modifications
- PatientAuditLog: traçabilité des actions sensibles
- PatientMatchRejection: matchs rejetés pour ne pas les reproposer
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Date,
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
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db

from .base import _iso


class PatientIdentity(db.Model):
    """Identité unique d'un patient, indexée par le hash HMAC-SHA256 du numéro AVS.

    Le numéro AVS n'est JAMAIS stocké en clair ici. Seul le hash (avec pepper serveur)
    et les 4 derniers chiffres sont conservés pour l'indexation et le debug.
    """

    __tablename__ = "patient_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Index AVS sécurisé
    avs_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True,
        comment="HMAC-SHA256(pepper, avs_normalise) — jamais l'AVS en clair",
    )
    avs_last4: Mapped[str | None] = mapped_column(
        String(4), nullable=True,
        comment="4 derniers chiffres pour debug/UX",
    )
    avs_status: Mapped[str] = mapped_column(
        String(10), nullable=False, default="unknown",
        comment="valid, invalid, unknown",
    )
    avs_verified_at = Column(DateTime(timezone=True), nullable=True)
    avs_verified_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True,
    )

    # Données canoniques (meilleure info disponible, priorité curatelle)
    canonical_first_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    canonical_last_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    canonical_dob: Mapped[Any] = mapped_column(Date, nullable=True)
    canonical_source: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True,
        comment='Source par champ: {"first_name": "curatelle", "dob": "clinic"}',
    )
    canonical_updated_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True,
    )

    # Optimistic locking
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Confiance du lien
    confidence_level: Mapped[str] = mapped_column(
        String(10), nullable=False, default="high",
        comment="high (AVS validé), medium (nom+DOB confirmé), low (manuel)",
    )

    # Source de vérité (curatelle)
    source_institution_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("institutions.id", ondelete="SET NULL"), nullable=True,
    )
    source_patient_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("institution_patients.id", ondelete="SET NULL"), nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relations
    links = relationship(
        "PatientIdentityLink",
        back_populates="identity",
        cascade="all, delete-orphan",
    )
    sync_events = relationship(
        "PatientSyncEvent",
        back_populates="patient_identity",
        cascade="all, delete-orphan",
    )

    @property
    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "avs_last4": self.avs_last4,
            "avs_status": self.avs_status,
            "canonical_first_name": self.canonical_first_name,
            "canonical_last_name": self.canonical_last_name,
            "canonical_dob": self.canonical_dob.isoformat() if self.canonical_dob else None,
            "confidence_level": self.confidence_level,
            "version": self.version,
            "active_links_count": sum(1 for lnk in self.links if lnk.is_active),
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


class PatientIdentityLink(db.Model):
    """Lien entre une PatientIdentity et un enregistrement entity-specific.

    entity_type: 'institution_patient' ou 'client'
    entity_id: PK de l'entité liée
    """

    __tablename__ = "patient_identity_links"
    __table_args__ = (
        UniqueConstraint(
            "patient_identity_id", "entity_type", "entity_id",
            name="uq_identity_link",
        ),
        # Un seul lien actif par entité (empêche double liaison)
        Index(
            "ix_identity_link_active_entity",
            "entity_type", "entity_id",
            unique=True,
            postgresql_where=text("is_active = true"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_identity_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("patient_identities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity_type: Mapped[str] = mapped_column(
        String(30), nullable=False,
        comment="institution_patient ou client",
    )
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)

    link_method: Mapped[str] = mapped_column(
        String(20), nullable=False,
        comment="avs_exact, name_dob_confirmed, manual",
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    linked_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    linked_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True,
    )

    # Soft detach
    detached_at = Column(DateTime(timezone=True), nullable=True)
    detached_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True,
    )
    detach_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # Relations
    identity = relationship("PatientIdentity", back_populates="links")

    @property
    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "patient_identity_id": self.patient_identity_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "link_method": self.link_method,
            "is_active": self.is_active,
            "linked_at": _iso(self.linked_at),
            "detached_at": _iso(self.detached_at),
            "detach_reason": self.detach_reason,
        }


class PatientSyncEvent(db.Model):
    """Outbox event pour la propagation asynchrone des modifications patient.

    Traité par un worker Celery avec FOR UPDATE SKIP LOCKED.
    """

    __tablename__ = "patient_sync_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_identity_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("patient_identities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    source_entity_type: Mapped[str] = mapped_column(String(30), nullable=False)
    source_entity_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Delta avec before/after pour debug + rollback
    changed_fields: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False,
        comment='{"field": {"before": "old", "after": "new"}}',
    )

    # Clé déterministe : même mutation = même key
    idempotency_key: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False,
    )
    event_version: Mapped[int] = mapped_column(Integer, nullable=False)

    # Statut du traitement
    status: Mapped[str] = mapped_column(
        String(15), nullable=False, default="pending",
        comment="pending, processing, success, partial_failure, failed",
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3)

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    processed_at = Column(DateTime(timezone=True), nullable=True)

    # Relations
    patient_identity = relationship("PatientIdentity", back_populates="sync_events")

    @property
    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "patient_identity_id": self.patient_identity_id,
            "source_entity_type": self.source_entity_type,
            "source_entity_id": self.source_entity_id,
            "changed_fields_keys": list(self.changed_fields.keys()) if self.changed_fields else [],
            "status": self.status,
            "retry_count": self.retry_count,
            "created_at": _iso(self.created_at),
            "processed_at": _iso(self.processed_at),
        }


class PatientAuditLog(db.Model):
    """Traçabilité des actions sensibles sur les identités patient."""

    __tablename__ = "patient_audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    actor_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True,
    )
    action: Mapped[str] = mapped_column(
        String(50), nullable=False,
        comment="READ_IDENTITY_LINKS, MERGE, DETACH, LINK_CONFIRMED, SYNC_TRIGGERED, SYNC_APPLIED, MATCH_REJECTED, SYNC_MANUAL_RETRY",
    )
    entity_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    entity_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True,
        comment="Détails complémentaires",
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PatientMatchRejection(db.Model):
    """Enregistrement des matchs rejetés pour ne pas les reproposer."""

    __tablename__ = "patient_match_rejections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("institution_patients.id", ondelete="CASCADE"),
        nullable=False,
    )
    identity_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("patient_identities.id", ondelete="CASCADE"),
        nullable=False,
    )
    rejected_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True,
    )
    rejected_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PatientLinkSuggestion(db.Model):
    """Suggestion de lien entre un patient source et une entité cible.

    Créée automatiquement lors de la création d'un patient sans AVS
    quand un match par nom+prénom+DOB est trouvé. Requiert une
    confirmation humaine avant de créer le lien effectif.
    """

    __tablename__ = "patient_link_suggestions"
    __table_args__ = (
        Index(
            "ix_link_suggestion_pending_unique",
            "source_patient_id", "target_entity_type", "target_entity_id",
            unique=True,
            postgresql_where=text("status = 'pending'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_patient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("institution_patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_identity_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("patient_identities.id", ondelete="SET NULL"),
        nullable=True,
    )
    target_entity_type: Mapped[str] = mapped_column(
        String(30), nullable=False,
        comment="institution_patient ou client",
    )
    target_entity_id: Mapped[int] = mapped_column(Integer, nullable=False)

    match_score: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    match_signals: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True,
        comment='{"name_exact": true, "dob_exact": true, ...}',
    )

    status: Mapped[str] = mapped_column(
        String(15), nullable=False, default="pending",
        comment="pending, confirmed, rejected, expired",
    )
    resolved_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True,
    )
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at = Column(
        DateTime(timezone=True),
        server_default=text("now() + interval '30 days'"),
        nullable=False,
    )

    @property
    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_patient_id": self.source_patient_id,
            "target_identity_id": self.target_identity_id,
            "target_entity_type": self.target_entity_type,
            "target_entity_id": self.target_entity_id,
            "match_score": self.match_score,
            "match_signals": self.match_signals,
            "status": self.status,
            "created_at": _iso(self.created_at),
            "expires_at": _iso(self.expires_at),
            "resolved_at": _iso(self.resolved_at),
        }
