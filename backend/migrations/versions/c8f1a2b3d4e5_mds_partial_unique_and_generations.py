"""mds_partial_unique_and_generations

Revision ID: c8f1a2b3d4e5
Revises: b7428dc318e7
Create Date: 2026-07-30 17:40:00.000000

F1b : unicité partielle (sessions actives uniquement) + générations séparées
(session_epoch, credential_generation, refresh_generation) + operation_type
sur AuthRotationResult.

Downgrade non sûr si plusieurs lignes historiques existent pour une installation.
"""

from alembic import op
import sqlalchemy as sa


revision = "c8f1a2b3d4e5"
down_revision = "b7428dc318e7"
branch_labels = None
depends_on = None


def upgrade():
    # Générations séparées (backfill depuis generation)
    op.add_column(
        "mobile_device_session",
        sa.Column("session_epoch", sa.Integer(), server_default="1", nullable=False),
    )
    op.add_column(
        "mobile_device_session",
        sa.Column(
            "credential_generation", sa.Integer(), server_default="1", nullable=False
        ),
    )
    op.add_column(
        "mobile_device_session",
        sa.Column(
            "refresh_generation", sa.Integer(), server_default="1", nullable=False
        ),
    )
    op.execute(
        """
        UPDATE mobile_device_session
        SET session_epoch = 1,
            credential_generation = COALESCE(generation, 1),
            refresh_generation = 1
        """
    )

    op.add_column(
        "auth_rotation_result",
        sa.Column(
            "operation_type",
            sa.String(length=32),
            server_default="refresh",
            nullable=False,
        ),
    )

    # Unicité partielle : une seule session active par installation
    op.drop_constraint(
        "uq_mobile_device_session_user_installation",
        "mobile_device_session",
        type_="unique",
    )
    op.create_index(
        "uq_mobile_device_session_active_installation",
        "mobile_device_session",
        ["user_id", "device_installation_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade():
    # Non sûr : plusieurs lignes historiques peuvent bloquer l'unicité absolue.
    op.drop_index(
        "uq_mobile_device_session_active_installation",
        table_name="mobile_device_session",
    )
    op.create_unique_constraint(
        "uq_mobile_device_session_user_installation",
        "mobile_device_session",
        ["user_id", "device_installation_id"],
    )
    op.drop_column("auth_rotation_result", "operation_type")
    op.drop_column("mobile_device_session", "refresh_generation")
    op.drop_column("mobile_device_session", "credential_generation")
    op.drop_column("mobile_device_session", "session_epoch")
