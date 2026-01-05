"""✅ P1: Add assignment indexes for performance optimization

Ajout d'index composites pour optimiser les requêtes fréquentes sur Assignment:
- ix_assignment_driver_booking: Optimise _get_driver_previous_booking() et requêtes driver+booking
- ix_assignment_booking_driver: Optimise les joins Assignment-Booking dans realtime_optimizer

Revision ID: p1_assignment_indexes
Revises: add_decision_explanation
Create Date: 2025-01-28 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "p1_assignment_indexes"
down_revision = "add_decision_explanation"
branch_labels = None
depends_on = None


def upgrade():
    """Ajoute les index composites pour optimiser les requêtes Assignment."""
    # ✅ P1: Index composite pour requêtes driver + booking
    # Optimise _get_driver_previous_booking() et autres requêtes qui filtrent par driver_id
    # et joignent avec booking
    op.create_index(
        "ix_assignment_driver_booking",
        "assignment",
        ["driver_id", "booking_id"],
        unique=False,
    )

    # ✅ P1: Index composite pour requêtes booking + driver
    # Optimise les joins Assignment-Booking dans realtime_optimizer et autres
    # requêtes qui filtrent par booking_id et joignent avec driver
    op.create_index(
        "ix_assignment_booking_driver",
        "assignment",
        ["booking_id", "driver_id"],
        unique=False,
    )

    # ✅ P1: Index composite pour requêtes booking + status
    # Optimise les requêtes qui filtrent par booking_id et status
    op.create_index(
        "ix_assignment_booking_status",
        "assignment",
        ["booking_id", "status"],
        unique=False,
    )

    # ✅ P1: Index sur created_at pour requêtes temporelles
    # Optimise les requêtes qui trient par date de création (DESC)
    op.create_index(
        "ix_assignment_created_at",
        "assignment",
        [sa.text("created_at DESC")],
        unique=False,
        postgresql_ops={"created_at": "DESC"},
    )

    # ✅ P1: Index composite pour recherches par client avec tri temporel
    # Optimise les requêtes qui filtrent par client_id et trient par scheduled_time DESC
    op.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS ix_booking_client_time "
            "ON booking(client_id, scheduled_time DESC)"
        )
    )


def downgrade():
    """Supprime les index ajoutés."""
    op.execute(sa.text("DROP INDEX IF EXISTS ix_booking_client_time"))
    op.execute(sa.text("DROP INDEX IF EXISTS ix_assignment_created_at"))
    op.drop_index("ix_assignment_booking_status", table_name="assignment")
    op.drop_index("ix_assignment_booking_driver", table_name="assignment")
    op.drop_index("ix_assignment_driver_booking", table_name="assignment")
