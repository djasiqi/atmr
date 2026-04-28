"""Ajoute la table activation_session pour onboarding client 2 preuves.

Revision ID: 20260409_activation_session
Revises: 20260328_co_plat_susp
Create Date: 2026-04-09
"""

from alembic import op

revision = "20260409_activation_session"
down_revision = "20260328_co_plat_susp"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS activation_session (
            id SERIAL PRIMARY KEY,
            activation_session_id VARCHAR(36) NOT NULL UNIQUE,
            user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
            email_token_hash VARCHAR(64),
            email_token_expires_at TIMESTAMP WITH TIME ZONE,
            email_verified_at TIMESTAMP WITH TIME ZONE,
            sms_code_hash VARCHAR(64),
            sms_expires_at TIMESTAMP WITH TIME ZONE,
            sms_attempts INTEGER NOT NULL DEFAULT 0,
            sms_locked_until TIMESTAMP WITH TIME ZONE,
            phone_verified_at TIMESTAMP WITH TIME ZONE,
            resend_count_email INTEGER NOT NULL DEFAULT 0,
            resend_count_sms INTEGER NOT NULL DEFAULT 0,
            last_email_sent_at TIMESTAMP WITH TIME ZONE,
            last_sms_sent_at TIMESTAMP WITH TIME ZONE,
            consumed_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_activation_session_user_id ON activation_session (user_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_activation_session_session_id ON activation_session (activation_session_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_activation_session_email_token_hash ON activation_session (email_token_hash)"
    )


def downgrade():
    op.execute("DROP TABLE IF EXISTS activation_session")
