"""parcours canonique reservation entreprise

Revision ID: b41d4242fe1a
Revises: 8b1974a79318
Create Date: 2026-09-27 11:12:51.733050

"""

import sqlalchemy as sa
from alembic import op

revision = "b41d4242fe1a"
down_revision = "8b1974a79318"
branch_labels = None
depends_on = None


def upgrade():
    # Généré par Alembic, puis limité aux objets de ce contrat.
    # L'autogenerate avait aussi proposé des index et colonnes sans lien
    # avec la mission entreprise (dérive du schéma local).
    op.create_table(
        "company_booking_route_steps",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("anchor_booking_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("location", sa.String(length=500), nullable=False),
        sa.Column("latitude", sa.Numeric(precision=10, scale=7), nullable=True),
        sa.Column("longitude", sa.Numeric(precision=10, scale=7), nullable=True),
        sa.Column("arrival_at", sa.DateTime(), nullable=True),
        sa.Column("departure_at", sa.DateTime(), nullable=True),
        sa.Column(
            "time_confirmed",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
        sa.Column("destination_kind", sa.String(length=20), nullable=True),
        sa.Column("establishment", sa.String(length=200), nullable=True),
        sa.Column("service", sa.String(length=255), nullable=True),
        sa.Column("doctor", sa.String(length=200), nullable=True),
        sa.Column("access_notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["anchor_booking_id"], ["booking.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "anchor_booking_id",
            "position",
            name="uq_company_route_step_anchor_position",
        ),
    )
    op.create_index(
        op.f("ix_company_booking_route_steps_anchor_booking_id"),
        "company_booking_route_steps",
        ["anchor_booking_id"],
        unique=False,
    )
    op.create_table(
        "company_manual_booking_requests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "company_id",
            "idempotency_key",
            name="uq_company_manual_booking_request_key",
        ),
    )
    op.create_index(
        op.f("ix_company_manual_booking_requests_company_id"),
        "company_manual_booking_requests",
        ["company_id"],
        unique=False,
    )
    op.create_table(
        "company_manual_booking_request_occurrences",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.Integer(), nullable=False),
        sa.Column("occurrence_index", sa.Integer(), nullable=False),
        sa.Column("anchor_booking_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["anchor_booking_id"], ["booking.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["company_manual_booking_requests.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "request_id",
            "anchor_booking_id",
            name="uq_company_manual_request_occurrence_anchor",
        ),
        sa.UniqueConstraint(
            "request_id",
            "occurrence_index",
            name="uq_company_manual_request_occurrence_index",
        ),
    )
    op.create_index(
        op.f("ix_company_manual_booking_request_occurrences_request_id"),
        "company_manual_booking_request_occurrences",
        ["request_id"],
        unique=False,
    )
    op.create_index(
        "ix_company_manual_request_occurrence_anchor",
        "company_manual_booking_request_occurrences",
        ["anchor_booking_id"],
        unique=False,
    )
    op.add_column(
        "booking", sa.Column("passenger_name", sa.String(length=200), nullable=True)
    )
    op.add_column(
        "booking",
        sa.Column("external_reference", sa.String(length=120), nullable=True),
    )
    op.add_column(
        "booking",
        sa.Column(
            "needs_assistance",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.add_column(
        "booking", sa.Column("requester_name", sa.String(length=200), nullable=True)
    )
    op.add_column(
        "booking", sa.Column("requester_phone", sa.String(length=30), nullable=True)
    )
    op.add_column(
        "booking",
        sa.Column("requester_service", sa.String(length=120), nullable=True),
    )
    op.add_column(
        "booking", sa.Column("pricing_mode", sa.String(length=20), nullable=True)
    )
    op.add_column(
        "booking",
        sa.Column(
            "preferential_amount",
            sa.Numeric(precision=10, scale=2),
            nullable=True,
        ),
    )


def downgrade():
    op.drop_column("booking", "preferential_amount")
    op.drop_column("booking", "pricing_mode")
    op.drop_column("booking", "requester_service")
    op.drop_column("booking", "requester_phone")
    op.drop_column("booking", "requester_name")
    op.drop_column("booking", "needs_assistance")
    op.drop_column("booking", "external_reference")
    op.drop_column("booking", "passenger_name")
    op.drop_index(
        "ix_company_manual_request_occurrence_anchor",
        table_name="company_manual_booking_request_occurrences",
    )
    op.drop_index(
        op.f("ix_company_manual_booking_request_occurrences_request_id"),
        table_name="company_manual_booking_request_occurrences",
    )
    op.drop_table("company_manual_booking_request_occurrences")
    op.drop_index(
        op.f("ix_company_manual_booking_requests_company_id"),
        table_name="company_manual_booking_requests",
    )
    op.drop_table("company_manual_booking_requests")
    op.drop_index(
        op.f("ix_company_booking_route_steps_anchor_booking_id"),
        table_name="company_booking_route_steps",
    )
    op.drop_table("company_booking_route_steps")
