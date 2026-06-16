"""p3 mission timestamp without tz

Convertit les horaires mission institution de timestamptz vers timestamp
(heure murale Genève) en préservant la face horloge via AT TIME ZONE.

Revision ID: eb21b7cf8467
Revises: ed0ca76e0f2f
Create Date: 2026-06-16 16:45:24.248199

"""

from alembic import op
import sqlalchemy as sa


revision = "eb21b7cf8467"
down_revision = "ed0ca76e0f2f"
branch_labels = None
depends_on = None

_GENEVA_TZ = "Europe/Zurich"


def upgrade():
    op.alter_column(
        "transport_requests",
        "scheduled_time",
        existing_type=sa.DateTime(timezone=True),
        type_=sa.DateTime(timezone=False),
        existing_nullable=True,
        postgresql_using=f"scheduled_time AT TIME ZONE '{_GENEVA_TZ}'",
    )
    op.alter_column(
        "transport_requests",
        "return_time",
        existing_type=sa.DateTime(timezone=True),
        type_=sa.DateTime(timezone=False),
        existing_nullable=True,
        postgresql_using=f"return_time AT TIME ZONE '{_GENEVA_TZ}'",
    )
    op.alter_column(
        "transport_request_legs",
        "scheduled_time",
        existing_type=sa.DateTime(timezone=True),
        type_=sa.DateTime(timezone=False),
        existing_nullable=True,
        postgresql_using=f"scheduled_time AT TIME ZONE '{_GENEVA_TZ}'",
    )


def downgrade():
    op.alter_column(
        "transport_request_legs",
        "scheduled_time",
        existing_type=sa.DateTime(timezone=False),
        type_=sa.DateTime(timezone=True),
        existing_nullable=True,
        postgresql_using=f"scheduled_time AT TIME ZONE '{_GENEVA_TZ}'",
    )
    op.alter_column(
        "transport_requests",
        "return_time",
        existing_type=sa.DateTime(timezone=False),
        type_=sa.DateTime(timezone=True),
        existing_nullable=True,
        postgresql_using=f"return_time AT TIME ZONE '{_GENEVA_TZ}'",
    )
    op.alter_column(
        "transport_requests",
        "scheduled_time",
        existing_type=sa.DateTime(timezone=False),
        type_=sa.DateTime(timezone=True),
        existing_nullable=True,
        postgresql_using=f"scheduled_time AT TIME ZONE '{_GENEVA_TZ}'",
    )
