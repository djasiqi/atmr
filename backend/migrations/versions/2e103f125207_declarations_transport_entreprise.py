"""declarations transport entreprise

Revision ID: 2e103f125207
Revises: ed461584cdae
Create Date: 2026-10-21 09:13:27.117857

Deux déclarations nullable sur ``company``. NULL reste non déclaré.
Le fauteuil fourni n'est enregistrable à vrai que si un stock est indiqué.

La génération automatique a aussi vu d'autres écarts de modèle déjà couverts
par des révisions non appliquées sur cette base : ils sont exclus.
"""

from alembic import op
import sqlalchemy as sa


revision = "2e103f125207"
down_revision = "ed461584cdae"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "company",
        sa.Column("provides_standard_assistance", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "company",
        sa.Column("provides_manual_wheelchair", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "company",
        sa.Column("manual_wheelchair_count", sa.Integer(), nullable=True),
    )
    op.create_check_constraint(
        "ck_company_manual_wheelchair_stock",
        "company",
        "(provides_manual_wheelchair IS NOT TRUE OR "
        "(manual_wheelchair_count IS NOT NULL AND manual_wheelchair_count >= 1)) "
        "AND (provides_manual_wheelchair IS NOT FALSE OR manual_wheelchair_count IS NULL) "
        "AND (provides_manual_wheelchair IS NOT NULL OR manual_wheelchair_count IS NULL)",
    )


def downgrade():
    op.drop_constraint("ck_company_manual_wheelchair_stock", "company", type_="check")
    op.drop_column("company", "manual_wheelchair_count")
    op.drop_column("company", "provides_manual_wheelchair")
    op.drop_column("company", "provides_standard_assistance")
