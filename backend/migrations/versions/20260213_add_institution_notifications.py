"""Add institution_notifications table for in-app notifications.

Revision ID: 20260213_notif
Revises: None
Create Date: 2026-02-13
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "20260213_notif"
down_revision = "patient_identity_001"
branch_labels = None
depends_on = None


def upgrade():
    """Crée la table institution_notifications."""
    op.execute("""
        CREATE TABLE IF NOT EXISTS institution_notifications (
            id SERIAL PRIMARY KEY,
            institution_id INTEGER NOT NULL
                REFERENCES institutions(id) ON DELETE CASCADE,
            event_type VARCHAR(50) NOT NULL,
            title VARCHAR(200) NOT NULL,
            message TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}',
            is_read BOOLEAN NOT NULL DEFAULT FALSE,
            dedupe_key VARCHAR(200),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)

    # Index composite pour les requêtes fréquentes (non-lues par institution)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_inst_notif_institution_read_created
        ON institution_notifications (institution_id, is_read, created_at DESC);
    """)

    # Index pour lister toutes les notifications d'une institution
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_inst_notif_institution_created
        ON institution_notifications (institution_id, created_at DESC);
    """)

    # Dedupe constraint
    op.execute("""
        ALTER TABLE institution_notifications
        ADD CONSTRAINT uq_inst_notif_dedupe
        UNIQUE (institution_id, dedupe_key);
    """)

    # Commentaires
    op.execute(
        "COMMENT ON TABLE institution_notifications "
        "IS 'Notifications in-app pour les institutions';"
    )
    op.execute(
        "COMMENT ON COLUMN institution_notifications.event_type "
        "IS 'Type: request_sent, offer_accepted, request_converted, "
        "booking_status_updated, request_cancelled, booking_cancelled';"
    )
    op.execute(
        "COMMENT ON COLUMN institution_notifications.metadata "
        "IS 'Données supplémentaires: request_id, booking_id, company_name, etc.';"
    )


def downgrade():
    op.execute("DROP TABLE IF EXISTS institution_notifications;")
