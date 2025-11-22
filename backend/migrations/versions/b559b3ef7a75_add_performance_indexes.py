"""add_performance_indexes

Ajout d'index de performance pour optimiser les requêtes fréquentes:
- Assignment: index sur (booking_id, created_at) et (dispatch_run_id, status)
- Booking: index composite sur (status, scheduled_time, company_id)
- Améliore les performances des requêtes de dispatch et de recherche

Revision ID: b559b3ef7a75
Revises: fix_circular_fk_20251018
Create Date: 2025-10-20 16:0.1:27.151468

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "b559b3ef7a75"
down_revision = "fix_circular_fk_20251018"
branch_labels = None
depends_on = None


def upgrade():
    # ✅ PERF: Index pour rechercher les assignments par booking avec tri chronologique
    # Utilisé par: requêtes de tracking, historique des assignments
    op.create_index("ix_assignment_booking_created", "assignment", ["booking_id", "created_at"], unique=False)

    # ✅ PERF: Index composite pour filtrer assignments par dispatch_run et status
    # Utilisé par: affichage des résultats de dispatch, filtrage par statut
    op.create_index("ix_assignment_dispatch_run_status", "assignment", ["dispatch_run_id", "status"], unique=False)

    # ✅ PERF: Index composite optimisé pour requêtes booking
    # Remplace/complète les index existants pour queries multi-colonnes
    # Utilisé par: filtrage bookings par company, status et période
    op.create_index(
        "ix_booking_status_scheduled_company", "booking", ["status", "scheduled_time", "company_id"], unique=False
    )


def downgrade():
    # Suppression des index dans l'ordre inverse
    op.drop_index("ix_booking_status_scheduled_company", table_name="booking")
    op.drop_index("ix_assignment_dispatch_run_status", table_name="assignment")
    op.drop_index("ix_assignment_booking_created", table_name="assignment")
