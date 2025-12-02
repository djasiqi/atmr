"""add_app_version_config_table

Revision ID: 019e8e5179d9
Revises: f6e0dfb9f5da
Create Date: 2025-12-02 23:57:01.568886

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "019e8e5179d9"
down_revision = "f6e0dfb9f5da"
branch_labels = None
depends_on = None


def upgrade():
    # Créer table app_version_config
    op.create_table(
        "app_version_config",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("platform", sa.String(length=20), nullable=False),
        sa.Column("min_required_version", sa.String(length=20), nullable=False),
        sa.Column("latest_version", sa.String(length=20), nullable=False),
        sa.Column("store_url", sa.String(length=500), nullable=True),
        sa.Column("update_message", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("platform", name="uq_app_version_config_platform"),
    )

    # Index pour performance
    op.create_index(
        "ix_app_version_config_platform", "app_version_config", ["platform"]
    )

    # Insérer les configurations par défaut (valeurs de départ)
    # Note: Ces valeurs devront être mises à jour manuellement selon les versions réelles
    op.execute(
        """
        INSERT INTO app_version_config (platform, min_required_version, latest_version, store_url, update_message)
        VALUES 
        ('android', '1.0.0', '1.0.3', 'https://play.google.com/store/apps/details?id=com.drinjasiqi.atmr', NULL),
        ('ios', '1.0.0', '1.0.3', NULL, NULL)
        ON CONFLICT (platform) DO NOTHING;
        """
    )


def downgrade():
    op.drop_index("ix_app_version_config_platform", table_name="app_version_config")
    op.drop_table("app_version_config")
