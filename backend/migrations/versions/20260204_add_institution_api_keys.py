"""add institution_api_keys table for DPI authentication

Revision ID: 20260204_api_keys
Revises: 20260204_institution
Create Date: 2026-02-04

Ajoute la table institution_api_keys pour permettre aux logiciels DPI
de s'authentifier via X-API-Key header sans interface web.
"""

import sqlalchemy as sa
from alembic import op

revision = "20260204_api_keys"
down_revision = "20260204_institution"
branch_labels = None
depends_on = None


def upgrade():
    # Créer la table institution_api_keys
    op.create_table(
        "institution_api_keys",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("institution_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("key_prefix", sa.String(length=20), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("scopes", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["institution_id"],
            ["institutions.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["user.id"],
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint("key_hash"),
    )

    # Créer les index
    op.create_index(
        "ix_institution_api_keys_institution_id",
        "institution_api_keys",
        ["institution_id"],
        unique=False,
    )
    op.create_index(
        "ix_institution_api_keys_key_prefix",
        "institution_api_keys",
        ["key_prefix"],
        unique=False,
    )
    op.create_index(
        "ix_institution_api_keys_key_hash",
        "institution_api_keys",
        ["key_hash"],
        unique=True,
    )


def downgrade():
    # Supprimer les index
    op.drop_index(
        "ix_institution_api_keys_key_hash", table_name="institution_api_keys"
    )
    op.drop_index(
        "ix_institution_api_keys_key_prefix", table_name="institution_api_keys"
    )
    op.drop_index(
        "ix_institution_api_keys_institution_id", table_name="institution_api_keys"
    )

    # Supprimer la table
    op.drop_table("institution_api_keys")
