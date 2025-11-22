"""Add secret_rotation table for monitoring secret rotations via Vault.

Revision ID: b2c3d4e5f6a7
Revises: d7e8f9a1b2c3
Create Date: 2025-01-28 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "b2c3d4e5f6a7"
down_revision = "d7e8f9a1b2c3"  # Point vers dernière migration
branch_labels = None
depends_on = None


def upgrade():
    # Table pour stocker l'historique des rotations de secrets
    op.create_table(
        "secret_rotation",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("secret_type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column(
            "rotated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("environment", sa.String(length=20), nullable=False),
        sa.Column(
            "rotation_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("task_id", sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Indexes pour performance et requêtes fréquentes
    op.create_index(
        "ix_secret_rotation_secret_type", "secret_rotation", ["secret_type"]
    )
    op.create_index("ix_secret_rotation_status", "secret_rotation", ["status"])
    op.create_index("ix_secret_rotation_rotated_at", "secret_rotation", ["rotated_at"])
    op.create_index(
        "ix_secret_rotation_environment", "secret_rotation", ["environment"]
    )
    op.create_index("ix_secret_rotation_task_id", "secret_rotation", ["task_id"])


def downgrade():
    # Supprimer les indexes
    op.drop_index("ix_secret_rotation_task_id", table_name="secret_rotation")
    op.drop_index("ix_secret_rotation_environment", table_name="secret_rotation")
    op.drop_index("ix_secret_rotation_rotated_at", table_name="secret_rotation")
    op.drop_index("ix_secret_rotation_status", table_name="secret_rotation")
    op.drop_index("ix_secret_rotation_secret_type", table_name="secret_rotation")

    # Supprimer la table
    op.drop_table("secret_rotation")
