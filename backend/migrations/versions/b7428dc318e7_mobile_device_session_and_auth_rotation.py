"""mobile_device_session_and_auth_rotation

Revision ID: b7428dc318e7
Revises: 20260729_msg_audio
Create Date: 2026-07-30 12:29:55.874060

Migration chirurgicale : uniquement MobileDeviceSession, AuthRotationResult
et colonnes session sur refresh_token (pas les diffs parasites autogenerate).
"""

from alembic import op
import sqlalchemy as sa


revision = "b7428dc318e7"
down_revision = "20260729_msg_audio"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "mobile_device_session",
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=True),
        sa.Column("device_installation_id", sa.String(length=255), nullable=False),
        sa.Column("device_name", sa.String(length=255), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "active",
                "revoked",
                "security_revoked",
                "account_disabled",
                name="mobile_device_session_status",
            ),
            server_default="active",
            nullable=False,
        ),
        sa.Column("credential_hash", sa.String(length=64), nullable=False),
        sa.Column("previous_credential_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "previous_credential_valid_until",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column("previous_generation", sa.Integer(), nullable=True),
        sa.Column("generation", sa.Integer(), server_default="1", nullable=False),
        sa.Column("revocation_secret_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_refresh_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_reason", sa.String(length=255), nullable=True),
        sa.Column("revoked_by_user_id", sa.Integer(), nullable=True),
        sa.Column("last_context_id", sa.String(length=128), nullable=True),
        sa.Column("last_app_version", sa.String(length=64), nullable=True),
        sa.Column("last_platform", sa.String(length=32), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("session_id"),
        sa.UniqueConstraint(
            "user_id",
            "device_installation_id",
            name="uq_mobile_device_session_user_installation",
        ),
    )
    op.create_index(
        "ix_mobile_device_session_device_installation_id",
        "mobile_device_session",
        ["device_installation_id"],
        unique=False,
    )
    op.create_index(
        "ix_mobile_device_session_status",
        "mobile_device_session",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_mobile_device_session_user_id",
        "mobile_device_session",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "auth_rotation_result",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("idempotency_key_hash", sa.String(length=64), nullable=False),
        sa.Column("request_generation", sa.Integer(), nullable=False),
        sa.Column("successor_generation", sa.Integer(), nullable=False),
        sa.Column("response_ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("encryption_key_id", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["mobile_device_session.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "session_id",
            "idempotency_key_hash",
            name="uq_auth_rotation_result_session_idempotency",
        ),
    )
    op.create_index(
        "ix_auth_rotation_result_expires_at",
        "auth_rotation_result",
        ["expires_at"],
        unique=False,
    )

    op.add_column(
        "refresh_token",
        sa.Column("session_id", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "refresh_token",
        sa.Column("session_generation", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_refresh_token_session_id",
        "refresh_token",
        ["session_id"],
        unique=False,
    )


def downgrade():
    op.drop_index("ix_refresh_token_session_id", table_name="refresh_token")
    op.drop_column("refresh_token", "session_generation")
    op.drop_column("refresh_token", "session_id")
    op.drop_index(
        "ix_auth_rotation_result_expires_at", table_name="auth_rotation_result"
    )
    op.drop_table("auth_rotation_result")
    op.drop_index(
        "ix_mobile_device_session_user_id", table_name="mobile_device_session"
    )
    op.drop_index(
        "ix_mobile_device_session_status", table_name="mobile_device_session"
    )
    op.drop_index(
        "ix_mobile_device_session_device_installation_id",
        table_name="mobile_device_session",
    )
    op.drop_table("mobile_device_session")
    op.execute("DROP TYPE IF EXISTS mobile_device_session_status")
