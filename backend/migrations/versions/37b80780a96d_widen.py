"""widen booking.hospital_service to 255

Revision ID: 37b80780a96d
Revises: 395bd3663e8d
Create Date: 2026-09-23 17:19:23.016888

Le service de destination (jusqu'à 255 caractères sur le trajet) était recopié
dans booking.hospital_service, limité à 100 caractères.
"""

import sqlalchemy as sa
from alembic import op

revision = "37b80780a96d"
down_revision = "395bd3663e8d"
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column(
        "booking",
        "hospital_service",
        existing_type=sa.VARCHAR(length=100),
        type_=sa.String(length=255),
        existing_nullable=True,
    )


def downgrade():
    op.alter_column(
        "booking",
        "hospital_service",
        existing_type=sa.String(length=255),
        type_=sa.VARCHAR(length=100),
        existing_nullable=True,
    )
