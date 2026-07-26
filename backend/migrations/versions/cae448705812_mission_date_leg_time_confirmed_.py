"""mission_date_leg_time_confirmed_nullable_departure

Revision ID: cae448705812
Revises: 41910172af52
Create Date: 2026-06-12 02:15:02.172763

"""

from alembic import op
import sqlalchemy as sa


revision = "cae448705812"
down_revision = "41910172af52"
branch_labels = None
depends_on = None


def upgrade():
    # mission_date (backfill depuis scheduled_time)
    op.add_column(
        "transport_requests",
        sa.Column("mission_date", sa.Date(), nullable=True),
    )
    op.execute(
        """
        UPDATE transport_requests
        SET mission_date = (scheduled_time AT TIME ZONE 'UTC')::date
        WHERE scheduled_time IS NOT NULL
        """
    )
    op.execute(
        """
        UPDATE transport_requests
        SET mission_date = CURRENT_DATE
        WHERE mission_date IS NULL
        """
    )
    op.alter_column("transport_requests", "mission_date", nullable=False)

    # départ confirmé (legacy : départ si scheduled_time présent et type departure)
    op.add_column(
        "transport_requests",
        sa.Column(
            "pickup_time_confirmed",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
    )
    op.execute(
        """
        UPDATE transport_requests
        SET pickup_time_confirmed = true
        WHERE scheduled_time IS NOT NULL
          AND COALESCE(scheduled_time_type, 'departure') = 'departure'
        """
    )

    # scheduled_time mission nullable (départ uniquement, peut être absent)
    op.alter_column(
        "transport_requests",
        "scheduled_time",
        existing_type=sa.DateTime(timezone=True),
        nullable=True,
    )
    op.execute(
        """
        UPDATE transport_requests
        SET scheduled_time = NULL
        WHERE scheduled_time IS NOT NULL
          AND COALESCE(scheduled_time_type, 'departure') = 'arrival'
        """
    )

    op.create_index(
        "ix_transport_requests_mission_date",
        "transport_requests",
        ["institution_id", "mission_date"],
        unique=False,
    )

    # time_confirmed persisté sur les legs
    op.add_column(
        "transport_request_legs",
        sa.Column(
            "time_confirmed",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
    )
    op.execute(
        """
        UPDATE transport_request_legs
        SET time_confirmed = (scheduled_time IS NOT NULL)
        """
    )


def downgrade():
    op.drop_column("transport_request_legs", "time_confirmed")
    op.drop_index("ix_transport_requests_mission_date", table_name="transport_requests")
    op.alter_column(
        "transport_requests",
        "scheduled_time",
        existing_type=sa.DateTime(timezone=True),
        nullable=False,
    )
    op.drop_column("transport_requests", "pickup_time_confirmed")
    op.drop_column("transport_requests", "mission_date")
