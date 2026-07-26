"""external_carrier_snapshot

Revision ID: 5f8a87e796bb
Revises: cae448705812
Create Date: 2026-06-13 13:01:42.236450

"""

from alembic import op
import sqlalchemy as sa


revision = "5f8a87e796bb"
down_revision = "cae448705812"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "transport_requests",
        sa.Column(
            "carrier_source",
            sa.String(length=20),
            server_default="lirie",
            nullable=False,
        ),
    )
    op.add_column(
        "transport_requests",
        sa.Column("external_carrier_name", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "transport_requests",
        sa.Column("external_carrier_phone", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "transport_requests",
        sa.Column("external_carrier_reference", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "transport_requests",
        sa.Column("external_carrier_reason", sa.String(length=120), nullable=True),
    )
    op.add_column(
        "transport_requests",
        sa.Column("assigned_externally_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "transport_requests",
        sa.Column("externalized_by_user_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "transport_requests",
        sa.Column("executed_externally_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "transport_requests",
        sa.Column("executed_externally_by_user_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "transport_requests",
        sa.Column("external_execution_notes", sa.Text(), nullable=True),
    )
    op.alter_column(
        "transport_requests",
        "status",
        existing_type=sa.VARCHAR(length=20),
        type_=sa.String(length=32),
        existing_nullable=False,
        existing_server_default=sa.text("'DRAFT'::character varying"),
    )
    op.create_index(
        "ix_transport_requests_carrier_source",
        "transport_requests",
        ["institution_id", "carrier_source"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_transport_requests_externalized_by_user_id",
        "transport_requests",
        "user",
        ["externalized_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_transport_requests_executed_externally_by_user_id",
        "transport_requests",
        "user",
        ["executed_externally_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade():
    op.drop_constraint(
        "fk_transport_requests_executed_externally_by_user_id",
        "transport_requests",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_transport_requests_externalized_by_user_id",
        "transport_requests",
        type_="foreignkey",
    )
    op.drop_index(
        "ix_transport_requests_carrier_source", table_name="transport_requests"
    )
    op.alter_column(
        "transport_requests",
        "status",
        existing_type=sa.String(length=32),
        type_=sa.VARCHAR(length=20),
        existing_nullable=False,
        existing_server_default=sa.text("'DRAFT'::character varying"),
    )
    op.drop_column("transport_requests", "external_execution_notes")
    op.drop_column("transport_requests", "executed_externally_by_user_id")
    op.drop_column("transport_requests", "executed_externally_at")
    op.drop_column("transport_requests", "externalized_by_user_id")
    op.drop_column("transport_requests", "assigned_externally_at")
    op.drop_column("transport_requests", "external_carrier_reason")
    op.drop_column("transport_requests", "external_carrier_reference")
    op.drop_column("transport_requests", "external_carrier_phone")
    op.drop_column("transport_requests", "external_carrier_name")
    op.drop_column("transport_requests", "carrier_source")
