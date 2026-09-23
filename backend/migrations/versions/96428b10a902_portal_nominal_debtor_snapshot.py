"""portal_nominal_debtor_snapshot

Revision ID: 96428b10a902
Revises: 9105ba261622
Create Date: 2026-09-23 18:22:53.326881

Colonnes générées par ``flask db revision --autogenerate``.
Les écarts d'index sans rapport ont été retirés.
Aucun UPDATE : les événements déjà ``partial`` restent sans identité inventée.
La contrainte d'identité est celle du modèle ; Alembic ne l'a pas émise.
Les ALTER sont directs pour conserver les triggers d'immutabilité.
"""

from alembic import op
import sqlalchemy as sa


revision = "96428b10a902"
down_revision = "9105ba261622"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "client_booking_contract_event",
        sa.Column("debtor_type_snapshot", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "client_booking_contract_event",
        sa.Column("debtor_user_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "client_booking_contract_event",
        sa.Column("debtor_name_snapshot", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "client_booking_contract_event",
        sa.Column("debtor_email_snapshot", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "client_booking_contract_event",
        sa.Column("debtor_phone_snapshot", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "client_booking_contract_event",
        sa.Column(
            "debtor_billing_address_snapshot",
            sa.String(length=255),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "fk_client_booking_contract_event_debtor_user_id",
        "client_booking_contract_event",
        "user",
        ["debtor_user_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_check_constraint(
        "ck_client_booking_contract_event_debtor_identity",
        "client_booking_contract_event",
        "("
        "debtor_resolution = 'partial' "
        "AND debtor_type_snapshot IS NULL "
        "AND debtor_user_id IS NULL"
        ") OR ("
        "debtor_resolution = 'resolved' "
        "AND debtor_type_snapshot = 'account_holder' "
        "AND debtor_user_id IS NOT NULL "
        "AND debtor_name_snapshot IS NOT NULL "
        "AND btrim(debtor_name_snapshot) <> ''"
        ")",
    )


def downgrade():
    op.drop_constraint(
        "ck_client_booking_contract_event_debtor_identity",
        "client_booking_contract_event",
        type_="check",
    )
    op.drop_constraint(
        "fk_client_booking_contract_event_debtor_user_id",
        "client_booking_contract_event",
        type_="foreignkey",
    )
    op.drop_column("client_booking_contract_event", "debtor_billing_address_snapshot")
    op.drop_column("client_booking_contract_event", "debtor_phone_snapshot")
    op.drop_column("client_booking_contract_event", "debtor_email_snapshot")
    op.drop_column("client_booking_contract_event", "debtor_name_snapshot")
    op.drop_column("client_booking_contract_event", "debtor_user_id")
    op.drop_column("client_booking_contract_event", "debtor_type_snapshot")
