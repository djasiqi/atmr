"""delay_events.assignment_id ON DELETE CASCADE

Revision ID: a7b8c9d0e1f2
Revises: 20260126233819
Create Date: 2026-01-27

Évite ForeignKeyViolation lorsqu'on supprime un assignment encore référencé par
delay_events (ex: update_driver_booking_status RELEASE). Les delay_events sont
un log technique rattaché à l'assignation ; CASCADE les supprime avec elle.
"""
from alembic import op

revision = "a7b8c9d0e1f2"
down_revision = "20260126233819"
branch_labels = None
depends_on = None

# Nom par défaut PostgreSQL pour la FK delay_events(assignment_id) -> assignment(id)
FK_NAME = "delay_events_assignment_id_fkey"


def upgrade():
    op.drop_constraint(FK_NAME, "delay_events", type_="foreignkey")
    op.create_foreign_key(
        FK_NAME,
        "delay_events",
        "assignment",
        ["assignment_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade():
    op.drop_constraint(FK_NAME, "delay_events", type_="foreignkey")
    op.create_foreign_key(
        FK_NAME,
        "delay_events",
        "assignment",
        ["assignment_id"],
        ["id"],
        ondelete=None,
    )
