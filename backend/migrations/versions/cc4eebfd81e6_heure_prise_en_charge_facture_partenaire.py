"""heure prise en charge facture partenaire

Revision ID: cc4eebfd81e6
Revises: aaaf36d71923
Create Date: 2026-10-02 18:46:46.186327

"""

from alembic import op
import sqlalchemy as sa


revision = "cc4eebfd81e6"
down_revision = "aaaf36d71923"
branch_labels = None
depends_on = None


def upgrade():
    # Colonnes détectées par autogenerate. Le reste du diff local
    # (index et colonnes hors facture partenaire) n'appartient pas à ce chantier.
    op.add_column(
        "partner_invoice_lines",
        sa.Column("scheduled_pickup_at", sa.String(length=5), nullable=True),
    )
    op.add_column(
        "partner_invoice_lines",
        sa.Column("boarded_at", sa.String(length=5), nullable=True),
    )
    op.add_column(
        "partner_invoice_lines",
        sa.Column("pickup_time_source", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "partner_invoices",
        sa.Column(
            "line_time_mode",
            sa.String(length=32),
            server_default="none",
            nullable=False,
        ),
    )


def downgrade():
    op.drop_column("partner_invoices", "line_time_mode")
    op.drop_column("partner_invoice_lines", "pickup_time_source")
    op.drop_column("partner_invoice_lines", "boarded_at")
    op.drop_column("partner_invoice_lines", "scheduled_pickup_at")
