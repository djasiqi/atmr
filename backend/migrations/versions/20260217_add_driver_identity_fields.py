"""Add driver identity and emergency contact fields

Revision ID: 20260217_drv_identity
Revises: 20260217_comp_notif
Create Date: 2026-02-17
"""

from alembic import op
import sqlalchemy as sa

revision = "20260217_drv_identity"
down_revision = "20260217_comp_notif"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("driver", sa.Column("avs_number", sa.String(16), nullable=True))
    op.add_column("driver", sa.Column("nationality", sa.String(60), nullable=True))
    op.add_column(
        "driver", sa.Column("emergency_contact_name", sa.String(120), nullable=True)
    )
    op.add_column(
        "driver", sa.Column("emergency_contact_phone", sa.String(30), nullable=True)
    )


def downgrade():
    op.drop_column("driver", "emergency_contact_phone")
    op.drop_column("driver", "emergency_contact_name")
    op.drop_column("driver", "nationality")
    op.drop_column("driver", "avs_number")
