"""Alignement schéma DB / modèles SQLAlchemy (guest checkout, rollback client, user).

Revision ID: 20260726_sync_schema
Revises: b79c3a9a4958
Create Date: 2026-07-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260726_sync_schema"
down_revision = "b79c3a9a4958"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Guest checkout : réservations sans client lié (voir guest_saferpay.py)
    op.alter_column(
        "booking",
        "client_id",
        existing_type=sa.Integer(),
        nullable=True,
    )

    # Colonne rollback conservée par 20260410_client_type_v2 — plus nécessaire
    op.drop_column("client", "_old_client_type")

    # Institution / guest : username optionnel (modèle User)
    op.alter_column(
        "user",
        "username",
        existing_type=sa.String(length=100),
        nullable=True,
    )
    op.alter_column(
        "user",
        "totp_enabled",
        existing_type=sa.Boolean(),
        nullable=False,
        server_default=sa.text("false"),
    )
    op.alter_column(
        "user",
        "recovery_codes_remaining",
        existing_type=sa.Integer(),
        nullable=False,
        server_default=sa.text("0"),
    )

    # patient_link_suggestions créée en JSON, modèles en JSONB
    op.execute(
        "ALTER TABLE patient_link_suggestions "
        "ALTER COLUMN match_signals TYPE JSONB USING match_signals::jsonb"
    )

    # Nettoyer les références orphelines avant d'ajouter la FK
    op.execute(
        """
        UPDATE message
        SET booking_id = NULL
        WHERE booking_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM booking b WHERE b.id = message.booking_id
          )
        """
    )

    # FK manquante sur message.booking_id (colonne ajoutée sans contrainte)
    op.create_foreign_key(
        op.f("fk_message_booking_id_booking"),
        "message",
        "booking",
        ["booking_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Aligner ON DELETE avec le modèle PlatformRunbookExecution
    op.drop_constraint(
        op.f("fk_prunbook_exec_rollback_of"),
        "platform_runbook_execution",
        type_="foreignkey",
    )
    op.create_foreign_key(
        op.f("fk_prunbook_exec_rollback_of"),
        "platform_runbook_execution",
        "platform_runbook_execution",
        ["rollback_of_execution_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("fk_prunbook_exec_rollback_of"),
        "platform_runbook_execution",
        type_="foreignkey",
    )
    op.create_foreign_key(
        op.f("fk_prunbook_exec_rollback_of"),
        "platform_runbook_execution",
        "platform_runbook_execution",
        ["rollback_of_execution_id"],
        ["id"],
    )

    op.drop_constraint(
        op.f("fk_message_booking_id_booking"),
        "message",
        type_="foreignkey",
    )

    op.execute(
        "ALTER TABLE patient_link_suggestions "
        "ALTER COLUMN match_signals TYPE JSON USING match_signals::json"
    )

    op.alter_column(
        "user",
        "recovery_codes_remaining",
        existing_type=sa.Integer(),
        nullable=True,
        server_default=sa.text("0"),
    )
    op.alter_column(
        "user",
        "totp_enabled",
        existing_type=sa.Boolean(),
        nullable=True,
        server_default=sa.text("false"),
    )
    op.alter_column(
        "user",
        "username",
        existing_type=sa.String(length=100),
        nullable=False,
    )

    op.add_column(
        "client",
        sa.Column("_old_client_type", sa.String(length=20), nullable=True),
    )

    op.alter_column(
        "booking",
        "client_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
