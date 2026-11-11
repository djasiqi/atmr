"""identites externes google et apple

Revision ID: 99a1e3b30d27
Revises: 255990200183
Create Date: 2026-10-12 23:36:08.905533

"""

from alembic import op
import sqlalchemy as sa


revision = "99a1e3b30d27"
down_revision = "255990200183"
branch_labels = None
depends_on = None


def upgrade():
    # Conservé depuis l'autogénération : uniquement les identités externes.
    # Le reste du diff décrivait un écart historique de schéma, sans lien avec
    # Google ou Apple, et n'est pas appliqué.
    op.create_table(
        "external_auth_transaction",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("state_hash", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=16), nullable=False),
        sa.Column("nonce_hash", sa.String(length=64), nullable=False),
        sa.Column("code_verifier", sa.String(length=128), nullable=True),
        sa.Column("intent", sa.String(length=16), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column(
            "status", sa.String(length=16), server_default="pending", nullable=False
        ),
        sa.Column("ticket_hash", sa.String(length=64), nullable=True),
        sa.Column("provider_sub", sa.String(length=255), nullable=True),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column(
            "email_verified", sa.Boolean(), server_default="false", nullable=False
        ),
        sa.Column("first_name", sa.String(length=100), nullable=True),
        sa.Column("last_name", sa.String(length=100), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "intent IN ('login', 'link')", name="ck_external_auth_transaction_intent"
        ),
        sa.CheckConstraint(
            "provider IN ('google', 'apple')",
            name="ck_external_auth_transaction_provider",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'ready', 'used', 'failed')",
            name="ck_external_auth_transaction_status",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
            name="fk_external_auth_transaction_user",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("state_hash"),
        sa.UniqueConstraint("ticket_hash"),
    )
    op.create_index(
        "ix_external_auth_transaction_user_id",
        "external_auth_transaction",
        ["user_id"],
        unique=False,
    )
    op.create_table(
        "user_external_identity",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=16), nullable=False),
        sa.Column("provider_sub", sa.String(length=255), nullable=False),
        sa.Column("email_at_link", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "provider IN ('google', 'apple')", name="ck_user_external_identity_provider"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
            name="fk_user_external_identity_user",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider", "provider_sub", name="uq_user_external_identity_provider_sub"
        ),
        sa.UniqueConstraint(
            "user_id", "provider", name="uq_user_external_identity_user_provider"
        ),
    )
    op.create_index(
        op.f("ix_user_external_identity_user_id"),
        "user_external_identity",
        ["user_id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        op.f("ix_user_external_identity_user_id"), table_name="user_external_identity"
    )
    op.drop_table("user_external_identity")
    op.drop_index(
        "ix_external_auth_transaction_user_id", table_name="external_auth_transaction"
    )
    op.drop_table("external_auth_transaction")
