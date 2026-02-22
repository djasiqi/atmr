"""add institution_settings table and billing columns to institutions

Revision ID: 055c847af0bf
Revises: 20260204_audit_immut
Create Date: 2026-02-09 13:17:21.156746

P1: Table institution_settings (1:1 via institution_id unique FK)
    + 3 colonnes billing sur institutions (billing_email, billing_address, vat_number)
    + Backfill: une row settings pour chaque institution existante
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "055c847af0bf"
down_revision = "20260204_audit_immut"
branch_labels = None
depends_on = None


def upgrade():
    # 1. Creer table institution_settings
    op.create_table(
        "institution_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("institution_id", sa.Integer(), nullable=False),
        sa.Column(
            "timeout_same_day_minutes",
            sa.Integer(),
            nullable=False,
            server_default="5",
        ),
        sa.Column(
            "timeout_default_minutes",
            sa.Integer(),
            nullable=False,
            server_default="60",
        ),
        sa.Column(
            "default_billing_intent",
            sa.String(length=50),
            nullable=False,
            server_default="patient",
        ),
        sa.Column(
            "default_vat_rate",
            sa.Numeric(precision=5, scale=2),
            nullable=True,
        ),
        sa.Column(
            "default_payment_terms_days",
            sa.Integer(),
            nullable=False,
            server_default="30",
        ),
        sa.Column(
            "notification_emails",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "notify_request_sent",
            sa.Boolean(),
            nullable=False,
            server_default="true",
        ),
        sa.Column(
            "notify_offer_accepted",
            sa.Boolean(),
            nullable=False,
            server_default="true",
        ),
        sa.Column(
            "notify_request_expired",
            sa.Boolean(),
            nullable=False,
            server_default="true",
        ),
        sa.Column(
            "timezone",
            sa.String(length=50),
            nullable=False,
            server_default="Europe/Zurich",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["institution_id"], ["institutions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_institution_settings_institution_id",
        "institution_settings",
        ["institution_id"],
        unique=True,
    )

    # 2. Ajouter colonnes billing a institutions (IF NOT EXISTS car peuvent exister deja)
    op.execute(
        "ALTER TABLE institutions ADD COLUMN IF NOT EXISTS billing_email VARCHAR(255)"
    )
    op.execute(
        "ALTER TABLE institutions ADD COLUMN IF NOT EXISTS billing_address TEXT"
    )
    op.execute(
        "ALTER TABLE institutions ADD COLUMN IF NOT EXISTS vat_number VARCHAR(50)"
    )

    # 3. Backfill: creer une row settings pour chaque institution existante
    op.execute(
        """
        INSERT INTO institution_settings (institution_id)
        SELECT id FROM institutions
        WHERE id NOT IN (SELECT institution_id FROM institution_settings)
        """
    )


def downgrade():
    op.drop_column("institutions", "vat_number")
    op.drop_column("institutions", "billing_address")
    op.drop_column("institutions", "billing_email")
    op.drop_index(
        "ix_institution_settings_institution_id",
        table_name="institution_settings",
    )
    op.drop_table("institution_settings")
