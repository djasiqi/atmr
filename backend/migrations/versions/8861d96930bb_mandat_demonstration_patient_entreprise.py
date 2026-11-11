"""mandat demonstration patient entreprise

Revision ID: 8861d96930bb
Revises: 151674730380
Create Date: 2026-11-01 12:17:57.211840

Généré par Alembic, puis limité à la table du mandat de démonstration.
Le reste de l'autogénération reprenait un écart de modèles déjà présent
et ne fait pas partie de ce lot.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "8861d96930bb"
down_revision = "151674730380"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "patient_company_convention_demo",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("client_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("debtor_label", sa.String(length=160), nullable=False),
        sa.Column("representative_name", sa.String(length=160), nullable=True),
        sa.Column("tariff_version", sa.String(length=64), nullable=False),
        sa.Column("service_scope", sa.Text(), nullable=False),
        sa.Column("duration_label", sa.String(length=160), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("document_version", sa.String(length=32), nullable=False),
        sa.Column(
            "legally_effective",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
        sa.Column(
            "specimen_trace", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('demo_draft', 'demo_specimen_signed', 'demo_valid_simulated')",
            name="ck_convention_demo_status",
        ),
        sa.CheckConstraint(
            "legally_effective = false", name="ck_convention_demo_not_legal"
        ),
        sa.ForeignKeyConstraint(["client_id"], ["client.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "client_id", "company_id", name="uq_convention_demo_client_company"
        ),
    )
    op.create_index(
        op.f("ix_patient_company_convention_demo_client_id"),
        "patient_company_convention_demo",
        ["client_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_patient_company_convention_demo_company_id"),
        "patient_company_convention_demo",
        ["company_id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        op.f("ix_patient_company_convention_demo_company_id"),
        table_name="patient_company_convention_demo",
    )
    op.drop_index(
        op.f("ix_patient_company_convention_demo_client_id"),
        table_name="patient_company_convention_demo",
    )
    op.drop_table("patient_company_convention_demo")
