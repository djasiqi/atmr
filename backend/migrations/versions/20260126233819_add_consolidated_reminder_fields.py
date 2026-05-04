"""add_consolidated_reminder_fields

Revision ID: 20260126233819
Revises: 292d4fc6604a
Create Date: 2026-01-26 23:38:19

Ajoute les champs nécessaires pour les rappels consolidés :
- principal_amount : montant de la facture initiale
- reminder_fee_amount : frais de rappel
- total_due : total à payer (principal + frais)
- qr_reference : référence QR-bill pour le rappel
- status : statut du rappel (OPEN/PAID)
- paid_at : date de paiement
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260126233819"
down_revision = "292d4fc6604a"
branch_labels = None
depends_on = None


def upgrade():
    # Ajouter les nouveaux champs à invoice_reminders
    op.add_column(
        "invoice_reminders",
        sa.Column(
            "principal_amount", sa.Numeric(10, 2), nullable=False, server_default="0.00"
        ),
    )
    op.add_column(
        "invoice_reminders",
        sa.Column(
            "reminder_fee_amount",
            sa.Numeric(10, 2),
            nullable=False,
            server_default="0.00",
        ),
    )
    op.add_column(
        "invoice_reminders",
        sa.Column(
            "total_due", sa.Numeric(10, 2), nullable=False, server_default="0.00"
        ),
    )
    op.add_column(
        "invoice_reminders",
        sa.Column("qr_reference", sa.String(50), nullable=True),
    )
    op.add_column(
        "invoice_reminders",
        sa.Column("status", sa.String(20), nullable=False, server_default="OPEN"),
    )
    op.add_column(
        "invoice_reminders",
        sa.Column("paid_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Index pour les requêtes fréquentes
    op.create_index(
        "ix_invoice_reminders_status",
        "invoice_reminders",
        ["status"],
    )
    op.create_index(
        "ix_invoice_reminders_invoice_status",
        "invoice_reminders",
        ["invoice_id", "status"],
    )

    # Ajouter reminder_id à invoice_payments pour la ventilation automatique
    op.add_column(
        "invoice_payments",
        sa.Column("reminder_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_invoice_payment_reminder",
        "invoice_payments",
        "invoice_reminders",
        ["reminder_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_invoice_payments_reminder",
        "invoice_payments",
        ["reminder_id"],
    )


def downgrade():
    # Supprimer les index et FK pour invoice_payments
    op.drop_index("ix_invoice_payments_reminder", table_name="invoice_payments")
    op.drop_constraint(
        "fk_invoice_payment_reminder", "invoice_payments", type_="foreignkey"
    )
    op.drop_column("invoice_payments", "reminder_id")

    # Supprimer les index
    op.drop_index("ix_invoice_reminders_invoice_status", table_name="invoice_reminders")
    op.drop_index("ix_invoice_reminders_status", table_name="invoice_reminders")

    # Supprimer les colonnes
    op.drop_column("invoice_reminders", "paid_at")
    op.drop_column("invoice_reminders", "status")
    op.drop_column("invoice_reminders", "qr_reference")
    op.drop_column("invoice_reminders", "total_due")
    op.drop_column("invoice_reminders", "reminder_fee_amount")
    op.drop_column("invoice_reminders", "principal_amount")
