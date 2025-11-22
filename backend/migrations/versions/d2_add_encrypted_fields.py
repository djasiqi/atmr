"""✅ D2: Ajoute colonnes pour stockage chiffré (migration progressive).

Revision ID: d2_encrypted_fields
Revises: d2_audit_logs
Create Date: 2025-01-28 11:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "d2_encrypted_fields"
down_revision = "d7e8f9a1b2c3"  # rl_suggestions_001 (révision actuelle de la base)
branch_labels = None
depends_on = None


def upgrade():
    # User table : colonnes chiffrées (doublons temporaires)
    op.add_column("user", sa.Column("phone_encrypted", sa.Text(), nullable=True))
    op.add_column("user", sa.Column("email_encrypted", sa.Text(), nullable=True))
    op.add_column("user", sa.Column("first_name_encrypted", sa.Text(), nullable=True))
    op.add_column("user", sa.Column("last_name_encrypted", sa.Text(), nullable=True))
    op.add_column("user", sa.Column("address_encrypted", sa.Text(), nullable=True))

    # Client table : colonnes chiffrées
    op.add_column(
        "client", sa.Column("contact_phone_encrypted", sa.Text(), nullable=True)
    )
    op.add_column("client", sa.Column("gp_name_encrypted", sa.Text(), nullable=True))
    op.add_column("client", sa.Column("gp_phone_encrypted", sa.Text(), nullable=True))
    op.add_column(
        "client", sa.Column("billing_address_encrypted", sa.Text(), nullable=True)
    )

    # Colonne pour indiquer si les données sont migrées
    op.add_column(
        "user",
        sa.Column(
            "encryption_migrated", sa.Boolean(), server_default="false", nullable=False
        ),
    )
    op.add_column(
        "client",
        sa.Column(
            "encryption_migrated", sa.Boolean(), server_default="false", nullable=False
        ),
    )


def downgrade():
    op.drop_column("client", "encryption_migrated")
    op.drop_column("client", "billing_address_encrypted")
    op.drop_column("client", "gp_phone_encrypted")
    op.drop_column("client", "gp_name_encrypted")
    op.drop_column("client", "contact_phone_encrypted")
    op.drop_column("user", "encryption_migrated")
    op.drop_column("user", "address_encrypted")
    op.drop_column("user", "last_name_encrypted")
    op.drop_column("user", "first_name_encrypted")
    op.drop_column("user", "email_encrypted")
    op.drop_column("user", "phone_encrypted")
