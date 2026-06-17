"""add_due_date_to_invoice_reminders

Revision ID: c5847c06ae2d
Revises: a9a8b1c76e15
Create Date: 2026-06-17 11:24:18.429495

"""

from alembic import op
import sqlalchemy as sa


revision = "c5847c06ae2d"
down_revision = "a9a8b1c76e15"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "invoice_reminders",
        sa.Column("due_date", sa.DateTime(timezone=True), nullable=True),
    )
    # Rétro-remplissage : date du rappel + délai de paiement entreprise
    op.execute(
        """
        UPDATE invoice_reminders AS r
        SET due_date = r.generated_at + (
            COALESCE(cbs.payment_terms_days, 10) || ' days'
        )::interval
        FROM invoices AS i
        LEFT JOIN company_billing_settings AS cbs ON cbs.company_id = i.company_id
        WHERE r.invoice_id = i.id
          AND r.generated_at IS NOT NULL
          AND r.due_date IS NULL
        """
    )


def downgrade():
    op.drop_column("invoice_reminders", "due_date")
