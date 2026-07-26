"""Synchronise les commentaires SQL des colonnes avec les modèles SQLAlchemy.

Revision ID: 20260726_col_comments
Revises: 20260726_sync_schema
Create Date: 2026-07-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260726_col_comments"
down_revision = "20260726_sync_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("company", schema=None) as batch_op:
        batch_op.alter_column(
            "security_policy",
            existing_type=sa.TEXT(),
            comment="JSON: require_2fa_roles, password_expiry_days, max_session_days, enforcement_mode",
            existing_nullable=True,
        )

    with op.batch_alter_table("company_billing_settings", schema=None) as batch_op:
        batch_op.alter_column(
            "cancellation_policy",
            existing_type=postgresql.JSONB(astext_type=sa.Text()),
            comment="Policy d'annulation parametrable: tiers, min/max, overrides",
            existing_nullable=True,
        )

    with op.batch_alter_table("company_notifications", schema=None) as batch_op:
        batch_op.alter_column(
            "event_type",
            existing_type=sa.VARCHAR(length=50),
            comment="Type: booking_message, new_request, etc.",
            existing_nullable=False,
        )
        batch_op.alter_column(
            "dedupe_key",
            existing_type=sa.VARCHAR(length=200),
            comment="Cle de deduplication: {event_type}:{booking_id}:{status_or_actor}",
            existing_nullable=True,
        )

    with op.batch_alter_table("institution_notifications", schema=None) as batch_op:
        batch_op.alter_column(
            "event_type",
            existing_type=sa.VARCHAR(length=50),
            comment="Type d'événement: request_sent, offer_accepted, etc.",
            existing_comment="Type: request_sent, offer_accepted, request_converted, booking_status_updated, request_cancelled, booking_cancelled",
            existing_nullable=False,
        )
        batch_op.alter_column(
            "title",
            existing_type=sa.VARCHAR(length=200),
            comment="Titre court de la notification",
            existing_nullable=False,
        )
        batch_op.alter_column(
            "message",
            existing_type=sa.TEXT(),
            comment="Message descriptif de la notification",
            existing_nullable=False,
        )
        batch_op.alter_column(
            "dedupe_key",
            existing_type=sa.VARCHAR(length=200),
            comment="Cle de deduplication: {event_type}:{booking_id}:{status_or_actor}",
            existing_nullable=True,
        )
        batch_op.drop_table_comment(
            existing_comment="Notifications in-app pour les institutions"
        )

    with op.batch_alter_table("institution_patients", schema=None) as batch_op:
        batch_op.alter_column(
            "access_notes",
            existing_type=sa.TEXT(),
            comment="Notes d'accès (ascenseur, rampe, concierge...)",
            existing_comment="Notes accès chauffeur",
            existing_nullable=True,
        )
        batch_op.alter_column(
            "residence_name",
            existing_type=sa.VARCHAR(length=200),
            comment="Établissement de résidence (EMS, foyer, etc.)",
            existing_comment="Établissement de résidence",
            existing_nullable=True,
            existing_server_default=sa.text("NULL::character varying"),
        )
        batch_op.alter_column(
            "insurance_name",
            existing_type=sa.VARCHAR(length=200),
            comment="Nom de la caisse maladie",
            existing_comment="Nom caisse maladie",
            existing_nullable=True,
            existing_server_default=sa.text("NULL::character varying"),
        )
        batch_op.alter_column(
            "insurance_number",
            existing_type=sa.VARCHAR(length=50),
            comment="Numéro d'assuré",
            existing_comment="Numéro assuré",
            existing_nullable=True,
            existing_server_default=sa.text("NULL::character varying"),
        )
        batch_op.alter_column(
            "guardian_name",
            existing_type=sa.VARCHAR(length=200),
            comment="Nom du curateur / représentant légal",
            existing_comment="Nom du curateur",
            existing_nullable=True,
            existing_server_default=sa.text("NULL::character varying"),
        )
        batch_op.alter_column(
            "guardian_address",
            existing_type=sa.VARCHAR(length=500),
            comment="Adresse complète du curateur (utilisée pour facturation)",
            existing_comment="Adresse complète du curateur (facturation)",
            existing_nullable=True,
            existing_server_default=sa.text("NULL::character varying"),
        )
        batch_op.alter_column(
            "curator_team_id",
            existing_type=sa.INTEGER(),
            comment="Équipe de curateurs en charge de ce patient",
            existing_nullable=True,
        )
        batch_op.alter_column(
            "data_source_flags",
            existing_type=postgresql.JSONB(astext_type=sa.Text()),
            comment='Ex: {"address": "sync_curatelle", "phone": "local"}',
            existing_nullable=True,
        )

    with op.batch_alter_table("institution_settings", schema=None) as batch_op:
        batch_op.alter_column(
            "default_pickup_mode",
            existing_type=sa.VARCHAR(length=20),
            comment="Mode par défaut du lieu de départ: institution | domicile",
            existing_nullable=False,
            existing_server_default=sa.text("'institution'::character varying"),
        )
        batch_op.alter_column(
            "entry_points",
            existing_type=postgresql.JSONB(astext_type=sa.Text()),
            comment="Points d'accueil suggérés (ex: Réception, Urgences)",
            existing_nullable=False,
            existing_server_default=sa.text("'[]'::jsonb"),
        )
        batch_op.alter_column(
            "default_contact_phone",
            existing_type=sa.VARCHAR(length=50),
            comment="Téléphone standard institution (pré-rempli contact sur place)",
            existing_nullable=True,
        )

    with op.batch_alter_table("patient_audit_logs", schema=None) as batch_op:
        batch_op.alter_column(
            "action",
            existing_type=sa.VARCHAR(length=50),
            comment="READ_IDENTITY_LINKS, MERGE, DETACH, LINK_CONFIRMED, SYNC_TRIGGERED, SYNC_APPLIED, MATCH_REJECTED, SYNC_MANUAL_RETRY",
            existing_nullable=False,
        )
        batch_op.alter_column(
            "metadata_json",
            existing_type=postgresql.JSONB(astext_type=sa.Text()),
            comment="Détails complémentaires",
            existing_nullable=True,
        )

    with op.batch_alter_table("patient_identities", schema=None) as batch_op:
        batch_op.alter_column(
            "avs_hash",
            existing_type=sa.VARCHAR(length=64),
            comment="HMAC-SHA256(pepper, avs_normalise) — jamais l'AVS en clair",
            existing_nullable=False,
        )
        batch_op.alter_column(
            "avs_last4",
            existing_type=sa.VARCHAR(length=4),
            comment="4 derniers chiffres pour debug/UX",
            existing_nullable=True,
        )
        batch_op.alter_column(
            "avs_status",
            existing_type=sa.VARCHAR(length=10),
            comment="valid, invalid, unknown",
            existing_nullable=False,
            existing_server_default=sa.text("'unknown'::character varying"),
        )
        batch_op.alter_column(
            "canonical_source",
            existing_type=postgresql.JSONB(astext_type=sa.Text()),
            comment='Source par champ: {"first_name": "curatelle", "dob": "clinic"}',
            existing_nullable=True,
        )
        batch_op.alter_column(
            "confidence_level",
            existing_type=sa.VARCHAR(length=10),
            comment="high (AVS validé), medium (nom+DOB confirmé), low (manuel)",
            existing_nullable=False,
            existing_server_default=sa.text("'high'::character varying"),
        )

    with op.batch_alter_table("patient_identity_links", schema=None) as batch_op:
        batch_op.alter_column(
            "entity_type",
            existing_type=sa.VARCHAR(length=30),
            comment="institution_patient ou client",
            existing_nullable=False,
        )
        batch_op.alter_column(
            "link_method",
            existing_type=sa.VARCHAR(length=20),
            comment="avs_exact, name_dob_confirmed, manual",
            existing_nullable=False,
        )

    with op.batch_alter_table("patient_link_suggestions", schema=None) as batch_op:
        batch_op.alter_column(
            "target_entity_type",
            existing_type=sa.VARCHAR(length=30),
            comment="institution_patient ou client",
            existing_nullable=False,
        )
        batch_op.alter_column(
            "match_signals",
            existing_type=postgresql.JSONB(astext_type=sa.Text()),
            comment='{"name_exact": true, "dob_exact": true, ...}',
            existing_nullable=True,
        )
        batch_op.alter_column(
            "status",
            existing_type=sa.VARCHAR(length=15),
            comment="pending, confirmed, rejected, expired",
            existing_nullable=False,
            existing_server_default=sa.text("'pending'::character varying"),
        )

    with op.batch_alter_table("patient_sync_events", schema=None) as batch_op:
        batch_op.alter_column(
            "changed_fields",
            existing_type=postgresql.JSONB(astext_type=sa.Text()),
            comment='{"field": {"before": "old", "after": "new"}}',
            existing_nullable=False,
        )
        batch_op.alter_column(
            "status",
            existing_type=sa.VARCHAR(length=15),
            comment="pending, processing, success, partial_failure, failed",
            existing_nullable=False,
            existing_server_default=sa.text("'pending'::character varying"),
        )

    with op.batch_alter_table("transport_requests", schema=None) as batch_op:
        batch_op.alter_column(
            "pickup_time_confirmed",
            existing_type=sa.BOOLEAN(),
            comment="Heure de départ confirmée (scheduled_time = départ uniquement)",
            existing_nullable=False,
            existing_server_default=sa.text("false"),
        )
        batch_op.alter_column(
            "pickup_type",
            existing_type=sa.VARCHAR(length=20),
            comment="institution | domicile | other",
            existing_nullable=True,
        )
        batch_op.alter_column(
            "dropoff_type",
            existing_type=sa.VARCHAR(length=20),
            comment="institution | domicile | other",
            existing_nullable=True,
        )
        batch_op.alter_column(
            "pickup_entry_point",
            existing_type=sa.VARCHAR(length=100),
            comment="Point d'accueil départ",
            existing_nullable=True,
        )
        batch_op.alter_column(
            "dropoff_entry_point",
            existing_type=sa.VARCHAR(length=100),
            comment="Point d'accueil arrivée",
            existing_nullable=True,
        )


def downgrade() -> None:
    # Commentaires SQL : pas de rollback automatique (cosmétique).
    pass
