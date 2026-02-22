"""Add scheduled_time_type to transport_requests.

Permet de distinguer si l'horaire indiqué est une heure de départ
ou une heure de rendez-vous (arrivée).

Revision ID: 20260213_sched_type
Revises: (see previous head)
Create Date: 2026-02-13
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260213_sched_type"
down_revision = "20260213_notif"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE transport_requests
        ADD COLUMN IF NOT EXISTS scheduled_time_type VARCHAR(20) NOT NULL DEFAULT 'departure'
    """)
    op.execute("""
        COMMENT ON COLUMN transport_requests.scheduled_time_type
        IS 'departure = heure de départ, arrival = heure du rendez-vous'
    """)


def downgrade():
    op.drop_column("transport_requests", "scheduled_time_type")
