"""contract_special_conditions_partner_v120

Revision ID: a6a422986202
Revises: b5b935af8e86
Create Date: 2026-08-04 20:59:34.470966

"""

from alembic import op
import sqlalchemy as sa


revision = "a6a422986202"
down_revision = "b5b935af8e86"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table(
        "company_platform_billing_config", schema=None
    ) as batch_op:
        batch_op.add_column(
            sa.Column("contract_special_conditions", sa.Text(), nullable=True)
        )


def downgrade():
    with op.batch_alter_table(
        "company_platform_billing_config", schema=None
    ) as batch_op:
        batch_op.drop_column("contract_special_conditions")
