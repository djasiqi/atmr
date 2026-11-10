"""registre incident operationnel

Revision ID: a7e3c91b04d2
Revises: f44ce1cd5095
Create Date: 2026-10-25 14:20:00.000000

Migration limitée aux deux tables du registre opérationnel.
L'autogénération sur l'arbre courant proposerait aussi des tables
d'autres branches absentes de f44ce1cd5095, déjà appliquée sur atmr_test.
Ces opérations ont été écartées. Les clés étrangères de paiement et
d'attestation ne sont pas modifiées.
"""

from alembic import op
import sqlalchemy as sa


revision = "a7e3c91b04d2"
down_revision = "f44ce1cd5095"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "operational_incident",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("idempotency_key", sa.String(length=64), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("owner_company_id", sa.Integer(), nullable=True),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column("actor_role", sa.String(length=32), nullable=False),
        sa.Column("motif", sa.Text(), nullable=False),
        sa.Column(
            "group_ids_text", sa.Text(), server_default=sa.text("'[]'"), nullable=False
        ),
        sa.Column(
            "completed_ids_text",
            sa.Text(),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.Column(
            "remaining_ids_text",
            sa.Text(),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.Column(
            "state",
            sa.String(length=16),
            server_default=sa.text("'signale'"),
            nullable=False,
        ),
        sa.Column(
            "formally_released",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "state = 'signale'", name="ck_operational_incident_state"
        ),
        sa.CheckConstraint(
            "formally_released = false",
            name="ck_operational_incident_not_released",
        ),
        sa.ForeignKeyConstraint(
            ["booking_id"], ["booking.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["actor_user_id"], ["user.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
        sa.UniqueConstraint("idempotency_key"),
    )
    op.create_index(
        "ix_operational_incident_state",
        "operational_incident",
        ["state"],
    )
    op.create_index(
        "ix_operational_incident_booking",
        "operational_incident",
        ["booking_id"],
    )
    op.create_table(
        "operational_incident_event",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("incident_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("actor_role", sa.String(length=32), nullable=True),
        sa.Column("detail", sa.Text(), server_default=sa.text("''"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["incident_id"],
            ["operational_incident.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_operational_incident_event_incident",
        "operational_incident_event",
        ["incident_id"],
    )


def downgrade():
    op.drop_index(
        "ix_operational_incident_event_incident",
        table_name="operational_incident_event",
    )
    op.drop_table("operational_incident_event")
    op.drop_index(
        "ix_operational_incident_booking", table_name="operational_incident"
    )
    op.drop_index(
        "ix_operational_incident_state", table_name="operational_incident"
    )
    op.drop_table("operational_incident")
