"""client_booking_contract_event

Revision ID: 9105ba261622
Revises: 3220451c51ae
Create Date: 2026-09-23 17:57:36.080147

Table générée par ``flask db revision --autogenerate``.
Les écarts d'index sans rapport avec cet événement ont été retirés.
Le trigger d'immutabilité suit le registre d'acceptation.
"""

from alembic import op
import sqlalchemy as sa


revision = "9105ba261622"
down_revision = "3220451c51ae"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "client_booking_contract_event",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=False),
        sa.Column("actor_type", sa.String(length=16), nullable=False),
        sa.Column("customer_name_snapshot", sa.String(length=200), nullable=False),
        sa.Column("email_snapshot", sa.String(length=255), nullable=True),
        sa.Column("phone_snapshot", sa.String(length=32), nullable=True),
        sa.Column("passenger_name_snapshot", sa.String(length=200), nullable=True),
        sa.Column("billed_to_type_snapshot", sa.String(length=50), nullable=False),
        sa.Column("debtor_resolution", sa.String(length=16), nullable=False),
        sa.Column("carrier_status", sa.String(length=16), nullable=False),
        sa.Column("company_id_snapshot", sa.Integer(), nullable=True),
        sa.Column("pickup_snapshot", sa.String(length=500), nullable=False),
        sa.Column("dropoff_snapshot", sa.String(length=500), nullable=False),
        sa.Column("scheduled_time_snapshot", sa.DateTime(), nullable=True),
        sa.Column("is_round_trip_snapshot", sa.Boolean(), nullable=False),
        sa.Column("return_scheduled_time_snapshot", sa.DateTime(), nullable=True),
        sa.Column("wheelchair_need_snapshot", sa.Boolean(), nullable=False),
        sa.Column("estimated_amount_snapshot", sa.Float(), nullable=False),
        sa.Column("pricing_status", sa.String(length=16), nullable=False),
        sa.Column("amount_is_contractual", sa.Boolean(), nullable=False),
        sa.Column("terms_of_service_acceptance_id", sa.Integer(), nullable=True),
        sa.Column("transport_terms_acceptance_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "actor_type IN ('client')",
            name="ck_client_booking_contract_event_actor",
        ),
        sa.CheckConstraint(
            "carrier_status IN ('not_assigned', 'assigned')",
            name="ck_client_booking_contract_event_carrier",
        ),
        sa.CheckConstraint(
            "debtor_resolution IN ('partial', 'resolved')",
            name="ck_client_booking_contract_event_debtor",
        ),
        sa.CheckConstraint(
            "event_type IN ('BOOKING_CREATED', 'BOOKING_MODIFIED', 'BOOKING_CANCELLED')",
            name="ck_client_booking_contract_event_type",
        ),
        sa.CheckConstraint(
            "pricing_status = 'estimated'",
            name="ck_client_booking_contract_event_pricing",
        ),
        sa.CheckConstraint(
            "amount_is_contractual IS FALSE",
            name="ck_client_booking_contract_event_not_contractual",
        ),
        sa.ForeignKeyConstraint(["actor_user_id"], ["user.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["terms_of_service_acceptance_id"],
            ["client_terms_acceptance.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["transport_terms_acceptance_id"],
            ["client_terms_acceptance.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_client_booking_contract_event_initial",
        "client_booking_contract_event",
        ["booking_id"],
        unique=True,
        postgresql_where=sa.text("event_type = 'BOOKING_CREATED'"),
    )
    op.create_index(
        "uq_client_booking_contract_event_sequence",
        "client_booking_contract_event",
        ["booking_id", "sequence_number"],
        unique=True,
    )
    op.execute(
        """
        CREATE TRIGGER client_booking_contract_event_no_update
        BEFORE UPDATE ON client_booking_contract_event
        FOR EACH ROW
        EXECUTE FUNCTION legal_acceptance_prevent_modification();
        """
    )
    op.execute(
        """
        CREATE TRIGGER client_booking_contract_event_no_delete
        BEFORE DELETE ON client_booking_contract_event
        FOR EACH ROW
        EXECUTE FUNCTION legal_acceptance_prevent_modification();
        """
    )


def downgrade():
    op.execute(
        "DROP TRIGGER IF EXISTS client_booking_contract_event_no_delete "
        "ON client_booking_contract_event;"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS client_booking_contract_event_no_update "
        "ON client_booking_contract_event;"
    )
    op.drop_index(
        "uq_client_booking_contract_event_sequence",
        table_name="client_booking_contract_event",
    )
    op.drop_index(
        "uq_client_booking_contract_event_initial",
        table_name="client_booking_contract_event",
    )
    op.drop_table("client_booking_contract_event")
