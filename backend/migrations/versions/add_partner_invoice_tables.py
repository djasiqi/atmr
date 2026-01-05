"""add_partner_invoice_tables

Revision ID: p1a2r3t4n5i7
Revises: m1e2r3g4e5h6
Create Date: 2025-12-17 16:00:00.000000

Ajoute les tables pour la facturation mensuelle consolidée des partenaires.
- Table partner_invoices : factures mensuelles consolidées
- Table partner_invoice_transfers : liaison entre factures et transferts
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "p1a2r3t4n5i7"
down_revision = "m1e2r3g4e5h6"
branch_labels = None
depends_on = None


def upgrade():
    """
    Ajoute les tables pour la facturation mensuelle consolidée des partenaires.
    
    ⚠️ NOTE: Cette migration est idempotente - elle vérifie l'existence
    des tables avant de les créer. En production, les tables peuvent
    avoir été créées manuellement avant l'application de cette migration.
    """
    # Vérifier l'existence des tables avant de les créer
    from sqlalchemy import inspect
    from sqlalchemy.engine import reflection
    
    bind = op.get_bind()
    inspector = reflection.Inspector.from_engine(bind)
    existing_tables = inspector.get_table_names()
    existing_indexes = {}
    existing_foreign_keys = {}
    for table_name in existing_tables:
        existing_indexes[table_name] = [idx["name"] for idx in inspector.get_indexes(table_name)]
        existing_foreign_keys[table_name] = [fk["name"] for fk in inspector.get_foreign_keys(table_name)]
    
    # Créer la table partner_invoices (si elle n'existe pas déjà)
    if "partner_invoices" not in existing_tables:
        op.create_table(
            "partner_invoices",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("partnership_id", sa.Integer(), nullable=False),
            sa.Column("period_year", sa.Integer(), nullable=False),
            sa.Column("period_month", sa.Integer(), nullable=False),
            sa.Column("invoice_number", sa.String(length=100), nullable=False),
            sa.Column("subtotal_amount", sa.Numeric(10, 2), nullable=False),
            sa.Column("vat_amount", sa.Numeric(10, 2), nullable=False),
            sa.Column("total_amount", sa.Numeric(10, 2), nullable=False),
            sa.Column(
                "currency", sa.String(length=3), nullable=False, server_default="CHF"
            ),
            sa.Column(
                "status", sa.String(length=20), nullable=False, server_default="draft"
            ),
            sa.Column(
                "issued_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("due_date", sa.DateTime(timezone=True), nullable=False),
            sa.Column("paid_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("pdf_url", sa.String(length=500), nullable=True),
            sa.Column("notes", sa.String(length=1000), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.ForeignKeyConstraint(
                ["partnership_id"],
                ["partnerships.id"],
                ondelete="CASCADE",
            ),
            sa.UniqueConstraint(
                "partnership_id",
                "period_year",
                "period_month",
                name="unique_partner_invoice_period",
            ),
        )
    
    # Mettre à jour les index et foreign keys si la table existe maintenant
    if "partner_invoices" in inspector.get_table_names():
        existing_indexes["partner_invoices"] = [idx["name"] for idx in inspector.get_indexes("partner_invoices")]
        existing_foreign_keys["partner_invoices"] = [fk["name"] for fk in inspector.get_foreign_keys("partner_invoices")]

    # Créer les index (si ils n'existent pas déjà)
    if "ix_partner_invoices_partnership_id" not in existing_indexes.get("partner_invoices", []):
        op.create_index(
            "ix_partner_invoices_partnership_id",
            "partner_invoices",
            ["partnership_id"],
        )
    if "ix_partner_invoices_period_year" not in existing_indexes.get("partner_invoices", []):
        op.create_index(
            "ix_partner_invoices_period_year",
            "partner_invoices",
            ["period_year"],
        )
    if "ix_partner_invoices_period_month" not in existing_indexes.get("partner_invoices", []):
        op.create_index(
            "ix_partner_invoices_period_month",
            "partner_invoices",
            ["period_month"],
        )
    if "ix_partner_invoices_invoice_number" not in existing_indexes.get("partner_invoices", []):
        op.create_index(
            "ix_partner_invoices_invoice_number",
            "partner_invoices",
            ["invoice_number"],
            unique=True,
        )

    # Créer la table de liaison partner_invoice_transfers (si elle n'existe pas déjà)
    if "partner_invoice_transfers" not in existing_tables:
        op.create_table(
            "partner_invoice_transfers",
            sa.Column(
                "partner_invoice_id",
                sa.Integer(),
                nullable=False,
            ),
            sa.Column(
                "booking_transfer_id",
                sa.Integer(),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("partner_invoice_id", "booking_transfer_id"),
            sa.ForeignKeyConstraint(
                ["partner_invoice_id"],
                ["partner_invoices.id"],
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["booking_transfer_id"],
                ["booking_transfers.id"],
                ondelete="CASCADE",
            ),
        )


def downgrade():
    # Supprimer la table de liaison
    op.drop_table("partner_invoice_transfers")

    # Supprimer les index
    op.drop_index("ix_partner_invoices_invoice_number", table_name="partner_invoices")
    op.drop_index("ix_partner_invoices_period_month", table_name="partner_invoices")
    op.drop_index("ix_partner_invoices_period_year", table_name="partner_invoices")
    op.drop_index("ix_partner_invoices_partnership_id", table_name="partner_invoices")

    # Supprimer la table partner_invoices
    op.drop_table("partner_invoices")
