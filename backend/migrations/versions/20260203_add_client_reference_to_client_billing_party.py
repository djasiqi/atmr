"""add client_reference to client_billing_parties (numéro SPC, etc.)

Revision ID: 20260203_spc
Revises:
Create Date: 2026-02-03

"""
import sqlalchemy as sa
from alembic import op

revision = "20260203_spc"
down_revision = "20260202_cancellation_fields"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("client_billing_parties", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("client_reference", sa.String(length=80), nullable=True)
        )


def downgrade():
    with op.batch_alter_table("client_billing_parties", schema=None) as batch_op:
        batch_op.drop_column("client_reference")
