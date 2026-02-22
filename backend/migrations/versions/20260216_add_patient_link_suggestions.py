"""Add patient_link_suggestions table.

Table pour les suggestions de lien patient en attente de confirmation humaine.
Utilisee quand un patient est cree sans AVS et qu'un match par nom+prenom+DOB
est trouve cross-plateforme.

Revision ID: 20260216_link_sugg
Revises: 20260213_sched_type
Create Date: 2026-02-16
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


revision = "20260216_link_sugg"
down_revision = "20260213_sched_type"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "patient_link_suggestions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "source_patient_id",
            sa.Integer(),
            sa.ForeignKey("institution_patients.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "target_identity_id",
            sa.Integer(),
            sa.ForeignKey("patient_identities.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("target_entity_type", sa.String(30), nullable=False),
        sa.Column("target_entity_id", sa.Integer(), nullable=False),
        sa.Column("match_score", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("match_signals", JSON, nullable=True),
        sa.Column(
            "status",
            sa.String(15),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "resolved_by_user_id",
            sa.Integer(),
            sa.ForeignKey("user.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now() + interval '30 days'"),
            nullable=False,
        ),
    )

    op.execute("""
        CREATE UNIQUE INDEX ix_link_suggestion_pending_unique
        ON patient_link_suggestions (source_patient_id, target_entity_type, target_entity_id)
        WHERE status = 'pending'
    """)


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_link_suggestion_pending_unique")
    op.drop_table("patient_link_suggestions")
