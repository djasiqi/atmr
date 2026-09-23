"""portal booking mutation evidence

Revision ID: 02c0836f2261
Revises: 22e5139ffb5c
Create Date: 2026-09-23 21:41:23.196257

Colonnes générées par ``flask db revision --autogenerate``.
Les écarts d'index sans rapport ont été retirés.
``ADD COLUMN`` direct : ``batch_alter_table`` recréerait la table et perdrait
les triggers d'immutabilité. Aucun événement historique n'est inséré.
"""

import sqlalchemy as sa
from alembic import op

revision = "02c0836f2261"
down_revision = "22e5139ffb5c"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "client_booking_contract_event",
        sa.Column("status_before", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "client_booking_contract_event",
        sa.Column("status_after", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "client_booking_contract_event",
        sa.Column("cancellation_reason", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "client_booking_contract_event",
        sa.Column("changed_fields", sa.Text(), nullable=True),
    )


def downgrade():
    op.drop_column("client_booking_contract_event", "changed_fields")
    op.drop_column("client_booking_contract_event", "cancellation_reason")
    op.drop_column("client_booking_contract_event", "status_after")
    op.drop_column("client_booking_contract_event", "status_before")
