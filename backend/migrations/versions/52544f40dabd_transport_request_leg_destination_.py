"""transport_request_leg_destination_billing_override

Revision ID: 52544f40dabd
Revises: d7e4a1b92f03
Create Date: 2026-06-18 00:15:28.556395

Facturation multi-payeurs : override par destination sur les legs.
"""

from alembic import op
import sqlalchemy as sa


revision = "52544f40dabd"
down_revision = "d7e4a1b92f03"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("transport_request_legs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "destination_billing_override", sa.String(length=50), nullable=True
            )
        )
        batch_op.add_column(
            sa.Column(
                "is_return_stop",
                sa.Boolean(),
                server_default="false",
                nullable=False,
            )
        )


def downgrade():
    with op.batch_alter_table("transport_request_legs", schema=None) as batch_op:
        batch_op.drop_column("is_return_stop")
        batch_op.drop_column("destination_billing_override")
