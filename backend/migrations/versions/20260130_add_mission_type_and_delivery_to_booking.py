"""add mission_type and delivery_description to booking

Revision ID: 20260130_mission_delivery
Revises: f2b0c6600828
Create Date: 2026-01-30

"""

from alembic import op
import sqlalchemy as sa


revision = "20260130_mission_delivery"
down_revision = "f2b0c6600828"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "mission_type",
                sa.String(50),
                nullable=False,
                server_default="patient_transport",
            )
        )
        batch_op.add_column(sa.Column("delivery_description", sa.Text(), nullable=True))

    # server_default s'applique automatiquement aux lignes existantes


def downgrade():
    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.drop_column("delivery_description")
        batch_op.drop_column("mission_type")
