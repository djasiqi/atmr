"""Add critical indexes for performance

Revision ID: f3a9c7b8d1e2
Revises: e252718b5271
Create Date: 2025-10-15 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "f3a9c7b8d1e2"
down_revision = "d8a4d7185c60"
branch_labels = None
depends_on = None


def upgrade():
    """
    Ajoute les index critiques manquants pour améliorer les performances
    des requêtes fréquentes.
    """

    # ========== BOOKING ==========

    # 1. invoice_line_id (FK sans index)
    op.create_index(
        "ix_booking_invoice_line",
        "booking",
        ["invoice_line_id"],
        unique=False,
        postgresql_where=sa.text("invoice_line_id IS NOT NULL"),
    )

    # 2. Composite pour filtres company + status + date
    op.create_index(
        "ix_booking_company_status_scheduled", "booking", ["company_id", "status", "scheduled_time"], unique=False
    )

    # ========== INVOICE ==========

    # 3. Composite company + status + due_date (rapports factures en retard)
    op.create_index("ix_invoice_company_status_due", "invoices", ["company_id", "status", "due_date"], unique=False)

    # ========== ASSIGNMENT ==========

    # 4. dispatch_run_id (FK sans index explicite)
    op.create_index(
        "ix_assignment_dispatch_run",
        "assignment",
        ["dispatch_run_id"],
        unique=False,
        postgresql_where=sa.text("dispatch_run_id IS NOT NULL"),
    )

    # ========== DRIVER_STATUS ==========

    # 5. current_assignment_id
    op.create_index(
        "ix_driver_status_assignment",
        "driver_status",
        ["current_assignment_id"],
        unique=False,
        postgresql_where=sa.text("current_assignment_id IS NOT NULL"),
    )

    # ========== REALTIME_EVENT ==========

    # 6. Timestamp pour requêtes historiques
    op.create_index("ix_realtime_event_timestamp", "realtime_event", ["timestamp"], unique=False)


def downgrade():
    """Rollback: supprime tous les index créés"""
    op.drop_index("ix_booking_invoice_line", table_name="booking")
    op.drop_index("ix_booking_company_status_scheduled", table_name="booking")
    op.drop_index("ix_invoice_company_status_due", table_name="invoices")
    op.drop_index("ix_assignment_dispatch_run", table_name="assignment")
    op.drop_index("ix_driver_status_assignment", table_name="driver_status")
    op.drop_index("ix_realtime_event_timestamp", table_name="realtime_event")
