"""add institution_patients and transport_requests tables

Revision ID: 20260204_patients_requests
Revises: 20260204_api_keys
Create Date: 2026-02-04

Ajoute les tables pour la gestion des patients et demandes de transport
institutionnelles (portail DPI).
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260204_patients_requests"
down_revision = "20260204_api_keys"
branch_labels = None
depends_on = None


def upgrade():
    # ========== Table institution_patients ==========
    op.create_table(
        "institution_patients",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("institution_id", sa.Integer(), nullable=False),
        sa.Column("external_reference", sa.String(length=100), nullable=True),
        sa.Column("first_name", sa.String(length=100), nullable=False),
        sa.Column("last_name", sa.String(length=100), nullable=False),
        sa.Column("dob", sa.Date(), nullable=True),
        sa.Column("gender", sa.String(length=20), nullable=True),
        sa.Column("address", sa.String(length=255), nullable=True),
        sa.Column("city", sa.String(length=100), nullable=True),
        sa.Column("postal_code", sa.String(length=20), nullable=True),
        sa.Column("phone", sa.String(length=50), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["institution_id"],
            ["institutions.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("public_id"),
    )

    # Index institution_patients
    op.create_index(
        "ix_institution_patients_institution_id",
        "institution_patients",
        ["institution_id"],
        unique=False,
    )
    op.create_index(
        "ix_institution_patients_public_id",
        "institution_patients",
        ["public_id"],
        unique=True,
    )
    op.create_index(
        "ix_institution_patients_name",
        "institution_patients",
        ["institution_id", "last_name", "first_name"],
        unique=False,
    )
    # Unique external_reference par institution (si présent)
    op.create_index(
        "uq_institution_patient_external_ref",
        "institution_patients",
        ["institution_id", "external_reference"],
        unique=True,
        postgresql_where=sa.text("external_reference IS NOT NULL"),
    )

    # ========== Table transport_requests ==========
    op.create_table(
        "transport_requests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("institution_id", sa.Integer(), nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("external_reference", sa.String(length=100), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=True),
        sa.Column(
            "mission_type",
            sa.String(length=50),
            nullable=False,
            server_default="patient_transport",
        ),
        sa.Column("delivery_description", sa.Text(), nullable=True),
        sa.Column("scheduled_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("pickup_location", sa.String(length=255), nullable=False),
        sa.Column("pickup_lat", sa.Numeric(precision=10, scale=7), nullable=True),
        sa.Column("pickup_lng", sa.Numeric(precision=10, scale=7), nullable=True),
        sa.Column("pickup_floor", sa.String(length=50), nullable=True),
        sa.Column("pickup_door_code", sa.String(length=50), nullable=True),
        sa.Column("dropoff_location", sa.String(length=255), nullable=False),
        sa.Column("dropoff_lat", sa.Numeric(precision=10, scale=7), nullable=True),
        sa.Column("dropoff_lng", sa.Numeric(precision=10, scale=7), nullable=True),
        sa.Column("dropoff_floor", sa.String(length=50), nullable=True),
        sa.Column("dropoff_door_code", sa.String(length=50), nullable=True),
        sa.Column(
            "is_round_trip",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("return_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("mobility", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("floor_elevator_info", sa.Text(), nullable=True),
        sa.Column(
            "contact_on_site", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "billing_intent",
            sa.String(length=50),
            nullable=False,
            server_default="patient",
        ),
        sa.Column(
            "billing_details", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "status", sa.String(length=20), nullable=False, server_default="DRAFT"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("converted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("booking_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["institution_id"],
            ["institutions.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["user.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["patient_id"],
            ["institution_patients.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["booking_id"],
            ["booking.id"],
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint("public_id"),
        sa.CheckConstraint(
            "mission_type = 'patient_transport' OR delivery_description IS NOT NULL",
            name="chk_delivery_description_required",
        ),
    )

    # Index transport_requests
    op.create_index(
        "ix_transport_requests_public_id",
        "transport_requests",
        ["public_id"],
        unique=True,
    )
    op.create_index(
        "ix_transport_requests_institution_id",
        "transport_requests",
        ["institution_id"],
        unique=False,
    )
    op.create_index(
        "ix_transport_requests_status",
        "transport_requests",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_transport_requests_institution_status",
        "transport_requests",
        ["institution_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_transport_requests_scheduled",
        "transport_requests",
        ["institution_id", "scheduled_time"],
        unique=False,
    )
    # Unique external_reference par institution
    op.create_index(
        "uq_transport_request_ext_ref",
        "transport_requests",
        ["institution_id", "external_reference"],
        unique=True,
    )


def downgrade():
    # Drop transport_requests indexes
    op.drop_index("uq_transport_request_ext_ref", table_name="transport_requests")
    op.drop_index("ix_transport_requests_scheduled", table_name="transport_requests")
    op.drop_index(
        "ix_transport_requests_institution_status", table_name="transport_requests"
    )
    op.drop_index("ix_transport_requests_status", table_name="transport_requests")
    op.drop_index(
        "ix_transport_requests_institution_id", table_name="transport_requests"
    )
    op.drop_index("ix_transport_requests_public_id", table_name="transport_requests")

    # Drop transport_requests table
    op.drop_table("transport_requests")

    # Drop institution_patients indexes
    op.drop_index(
        "uq_institution_patient_external_ref", table_name="institution_patients"
    )
    op.drop_index("ix_institution_patients_name", table_name="institution_patients")
    op.drop_index(
        "ix_institution_patients_public_id", table_name="institution_patients"
    )
    op.drop_index(
        "ix_institution_patients_institution_id", table_name="institution_patients"
    )

    # Drop institution_patients table
    op.drop_table("institution_patients")
