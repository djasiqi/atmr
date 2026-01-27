"""add_sent_at_to_partner_invoices

Ajoute la colonne sent_at à partner_invoices pour le workflow Brouillon → Envoyer → Envoyée.

Revision ID: add_sent_at_partner_inv
Revises: 5ad79a6e8d27
Create Date: 2026-01-27 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = "add_sent_at_partner_inv"
down_revision = "5ad79a6e8d27"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [c["name"] for c in inspector.get_columns("partner_invoices")]
    if "sent_at" not in columns:
        op.add_column(
            "partner_invoices",
            sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade():
    op.drop_column("partner_invoices", "sent_at")
