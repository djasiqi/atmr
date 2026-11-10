"""numero professionnel visible par les clients

Revision ID: e3a22ba34db5
Revises: 2e103f125207
Create Date: 2026-10-23 15:17:42.052763

"""
from alembic import op
import sqlalchemy as sa


revision = "e3a22ba34db5"
down_revision = "2e103f125207"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "company",
        sa.Column("client_contact_phone", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "company",
        sa.Column(
            "client_contact_published",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
    )


def downgrade():
    op.drop_column("company", "client_contact_published")
    op.drop_column("company", "client_contact_phone")
