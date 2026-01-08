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
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = "p1_assignment_indexes"
down_revision = "add_decision_explanation"
branch_labels = None
depends_on = None


def upgrade():
    """Ajoute les index composites pour optimiser les requêtes Assignment.

    Utilise CREATE INDEX IF NOT EXISTS pour rendre l'opération idempotente (Postgres).
    Tous les index sont créés via SQL natif pour supporter les fonctionnalités PostgreSQL
    (IF NOT EXISTS, DESC) et éviter les limitations d'Alembic.
    """
    # ✅ P1: Index composite pour requêtes driver + booking
    # Optimise _get_driver_previous_booking() et autres requêtes qui filtrent par driver_id
    # et joignent avec booking
    op.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_assignment_driver_booking "
            "ON assignment (driver_id, booking_id)"
        )
    )

    # ✅ P1: Index composite pour requêtes booking + driver
    # Optimise les joins Assignment-Booking dans realtime_optimizer et autres
    # requêtes qui filtrent par booking_id et joignent avec driver
    op.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_assignment_booking_driver "
            "ON assignment (booking_id, driver_id)"
        )
    )

    # ✅ P1: Index composite pour requêtes booking + status
    # Optimise les requêtes qui filtrent par booking_id et status
    op.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_assignment_booking_status "
            "ON assignment (booking_id, status)"
        )
    )

    # ✅ P1: Index sur created_at pour requêtes temporelles
    # Optimise les requêtes qui trient par date de création (DESC)
    # SQL natif requis pour l'ordre DESC (non supporté par op.create_index)
    op.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_assignment_created_at "
            "ON assignment (created_at DESC)"
        )
    )

    # ✅ P1: Index composite pour recherches par client avec tri temporel
    # Optimise les requêtes qui filtrent par client_id et trient par scheduled_time DESC
    op.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_booking_client_time "
            "ON booking (client_id, scheduled_time DESC)"
        )
    )


def downgrade():
    """Supprime les index ajoutés.

    Utilise DROP INDEX IF EXISTS pour permettre un rollback sûr même si
    certains index n'ont pas été créés (idempotence).
    """
    # Retirer tous les indexes créés (ordre inverse de création)
    op.execute(text("DROP INDEX IF EXISTS ix_booking_client_time"))
    op.execute(text("DROP INDEX IF EXISTS ix_assignment_created_at"))
    op.execute(text("DROP INDEX IF EXISTS ix_assignment_booking_status"))
    op.execute(text("DROP INDEX IF EXISTS ix_assignment_booking_driver"))
    op.execute(text("DROP INDEX IF EXISTS ix_assignment_driver_booking"))
