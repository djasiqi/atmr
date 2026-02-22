"""Ajouter tables Patient Master Index : patient_identities, links, sync events, audit, rejections.

Revision ID: patient_identity_001
Revises: curator_teams_001
Create Date: 2026-02-11
"""

from alembic import op


revision = "patient_identity_001"
down_revision = "curator_teams_001"
branch_labels = None
depends_on = None


def upgrade():
    # ── Table patient_identities ──
    op.execute("""
        CREATE TABLE IF NOT EXISTS patient_identities (
            id SERIAL PRIMARY KEY,
            avs_hash VARCHAR(64) NOT NULL UNIQUE,
            avs_last4 VARCHAR(4),
            avs_status VARCHAR(10) NOT NULL DEFAULT 'unknown',
            avs_verified_at TIMESTAMPTZ,
            avs_verified_by_user_id INTEGER REFERENCES "user"(id) ON DELETE SET NULL,

            canonical_first_name VARCHAR(100),
            canonical_last_name VARCHAR(100),
            canonical_dob DATE,
            canonical_source JSONB,
            canonical_updated_by_user_id INTEGER REFERENCES "user"(id) ON DELETE SET NULL,

            version INTEGER NOT NULL DEFAULT 1,
            confidence_level VARCHAR(10) NOT NULL DEFAULT 'high',

            source_institution_id INTEGER REFERENCES institutions(id) ON DELETE SET NULL,
            source_patient_id INTEGER REFERENCES institution_patients(id) ON DELETE SET NULL,

            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_patient_identities_avs_hash
        ON patient_identities (avs_hash)
    """)

    # ── Table patient_identity_links ──
    op.execute("""
        CREATE TABLE IF NOT EXISTS patient_identity_links (
            id SERIAL PRIMARY KEY,
            patient_identity_id INTEGER NOT NULL
                REFERENCES patient_identities(id) ON DELETE CASCADE,
            entity_type VARCHAR(30) NOT NULL,
            entity_id INTEGER NOT NULL,
            link_method VARCHAR(20) NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,

            linked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            linked_by_user_id INTEGER REFERENCES "user"(id) ON DELETE SET NULL,

            detached_at TIMESTAMPTZ,
            detached_by_user_id INTEGER REFERENCES "user"(id) ON DELETE SET NULL,
            detach_reason VARCHAR(200),

            CONSTRAINT uq_identity_link
                UNIQUE (patient_identity_id, entity_type, entity_id)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_patient_identity_links_identity_id
        ON patient_identity_links (patient_identity_id)
    """)
    # Unique partial index : une seule identité active par entité
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_identity_link_active_entity
        ON patient_identity_links (entity_type, entity_id)
        WHERE is_active = true
    """)

    # ── Table patient_sync_events (outbox) ──
    op.execute("""
        CREATE TABLE IF NOT EXISTS patient_sync_events (
            id SERIAL PRIMARY KEY,
            patient_identity_id INTEGER NOT NULL
                REFERENCES patient_identities(id) ON DELETE CASCADE,
            source_entity_type VARCHAR(30) NOT NULL,
            source_entity_id INTEGER NOT NULL,
            changed_fields JSONB NOT NULL,
            idempotency_key VARCHAR(64) NOT NULL UNIQUE,
            event_version INTEGER NOT NULL,
            status VARCHAR(15) NOT NULL DEFAULT 'pending',
            error TEXT,
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 3,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            processed_at TIMESTAMPTZ
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_patient_sync_events_identity_id
        ON patient_sync_events (patient_identity_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_patient_sync_events_status
        ON patient_sync_events (status) WHERE status = 'pending'
    """)

    # ── Table patient_audit_logs ──
    op.execute("""
        CREATE TABLE IF NOT EXISTS patient_audit_logs (
            id SERIAL PRIMARY KEY,
            actor_user_id INTEGER REFERENCES "user"(id) ON DELETE SET NULL,
            action VARCHAR(50) NOT NULL,
            entity_type VARCHAR(30),
            entity_id INTEGER,
            metadata_json JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # ── Table patient_match_rejections ──
    op.execute("""
        CREATE TABLE IF NOT EXISTS patient_match_rejections (
            id SERIAL PRIMARY KEY,
            patient_id INTEGER NOT NULL
                REFERENCES institution_patients(id) ON DELETE CASCADE,
            identity_id INTEGER NOT NULL
                REFERENCES patient_identities(id) ON DELETE CASCADE,
            rejected_by_user_id INTEGER REFERENCES "user"(id) ON DELETE SET NULL,
            rejected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)


def downgrade():
    op.execute("DROP TABLE IF EXISTS patient_match_rejections")
    op.execute("DROP TABLE IF EXISTS patient_audit_logs")
    op.execute("DROP TABLE IF EXISTS patient_sync_events")
    op.execute("DROP TABLE IF EXISTS patient_identity_links")
    op.execute("DROP TABLE IF EXISTS patient_identities")
