"""versions de mandat

Revision ID: 9b59a05626a7
Revises: 360789deedb3
Create Date: 2026-11-01 21:37:57.556693

Généré par Alembic, puis limité à la table du mandat de démonstration.
Le reste de l'autogénération reprenait des index déjà présents hors de ce lot.
La clé d'idempotence est remplie avant d'être exigée, pour conserver les lignes
historiques. ``ck_convention_demo_not_legal`` n'est pas modifié.
"""

from alembic import op
import sqlalchemy as sa


revision = "9b59a05626a7"
down_revision = "360789deedb3"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "patient_company_convention_demo",
        sa.Column("version_number", sa.Integer(), server_default="1", nullable=False),
    )
    op.add_column(
        "patient_company_convention_demo",
        sa.Column("active", sa.Boolean(), server_default="true", nullable=False),
    )
    op.add_column(
        "patient_company_convention_demo",
        sa.Column("idempotency_key", sa.String(length=80), nullable=True),
    )
    op.execute(
        "UPDATE patient_company_convention_demo "
        "SET idempotency_key = 'legacy-' || id::text "
        "WHERE idempotency_key IS NULL"
    )
    op.alter_column(
        "patient_company_convention_demo",
        "idempotency_key",
        existing_type=sa.String(length=80),
        nullable=False,
    )
    op.drop_constraint(
        "uq_convention_demo_client_company",
        "patient_company_convention_demo",
        type_="unique",
    )
    op.create_index(
        "uq_convention_demo_active",
        "patient_company_convention_demo",
        ["client_id", "company_id"],
        unique=True,
        postgresql_where=sa.text("active"),
    )
    op.create_unique_constraint(
        "uq_convention_demo_operation",
        "patient_company_convention_demo",
        ["client_id", "company_id", "idempotency_key"],
    )
    op.create_unique_constraint(
        "uq_convention_demo_version",
        "patient_company_convention_demo",
        ["client_id", "company_id", "version_number"],
    )
    op.create_check_constraint(
        "ck_convention_demo_version",
        "patient_company_convention_demo",
        "version_number > 0",
    )
    op.create_check_constraint(
        "ck_convention_demo_key",
        "patient_company_convention_demo",
        "char_length(idempotency_key) > 0",
    )


def downgrade():
    op.drop_constraint(
        "ck_convention_demo_key",
        "patient_company_convention_demo",
        type_="check",
    )
    op.drop_constraint(
        "ck_convention_demo_version",
        "patient_company_convention_demo",
        type_="check",
    )
    op.drop_constraint(
        "uq_convention_demo_version",
        "patient_company_convention_demo",
        type_="unique",
    )
    op.drop_constraint(
        "uq_convention_demo_operation",
        "patient_company_convention_demo",
        type_="unique",
    )
    op.drop_index(
        "uq_convention_demo_active",
        table_name="patient_company_convention_demo",
        postgresql_where=sa.text("active"),
    )
    op.create_unique_constraint(
        "uq_convention_demo_client_company",
        "patient_company_convention_demo",
        ["client_id", "company_id"],
    )
    op.drop_column("patient_company_convention_demo", "idempotency_key")
    op.drop_column("patient_company_convention_demo", "active")
    op.drop_column("patient_company_convention_demo", "version_number")
