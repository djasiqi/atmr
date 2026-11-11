"""cycle de traitement des incidents operationnels

Revision ID: b8d2f04c91aa
Revises: a7e3c91b04d2
Create Date: 2026-10-25 20:10:00.000000

Élargit uniquement le registre opérationnel déjà présent sur atmr_test.
Les clés étrangères de paiement et d'attestation ne sont pas modifiées.
"""

from alembic import op
import sqlalchemy as sa


revision = "b8d2f04c91aa"
down_revision = "a7e3c91b04d2"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "operational_incident",
        sa.Column("close_justification", sa.Text(), nullable=True),
    )
    op.drop_constraint(
        "ck_operational_incident_state",
        "operational_incident",
        type_="check",
    )
    op.create_check_constraint(
        "ck_operational_incident_state",
        "operational_incident",
        "state IN ('signale', 'analyse', 'resolu', 'cloture')",
    )
    op.create_check_constraint(
        "ck_operational_incident_close_reason",
        "operational_incident",
        "state <> 'cloture' OR ("
        "close_justification IS NOT NULL "
        "AND length(btrim(close_justification)) > 0)",
    )


def downgrade():
    op.drop_constraint(
        "ck_operational_incident_close_reason",
        "operational_incident",
        type_="check",
    )
    op.drop_constraint(
        "ck_operational_incident_state",
        "operational_incident",
        type_="check",
    )
    op.create_check_constraint(
        "ck_operational_incident_state",
        "operational_incident",
        "state = 'signale'",
    )
    op.drop_column("operational_incident", "close_justification")
