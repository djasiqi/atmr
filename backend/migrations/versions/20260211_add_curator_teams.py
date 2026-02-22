"""Ajouter tables curator_teams et curator_team_members + FK sur institution_patients.

Revision ID: curator_teams_001
Revises: (depends on last migration)
Create Date: 2026-02-11
"""

from alembic import op


revision = "curator_teams_001"
down_revision = "linked_inst_001"
branch_labels = None
depends_on = None


def upgrade():
    # ── Table curator_teams ──
    op.execute("""
        CREATE TABLE IF NOT EXISTS curator_teams (
            id SERIAL PRIMARY KEY,
            public_id VARCHAR(36) NOT NULL UNIQUE,
            institution_id INTEGER NOT NULL
                REFERENCES institutions(id) ON DELETE CASCADE,
            name VARCHAR(200) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_curator_teams_institution_id
        ON curator_teams (institution_id)
    """)

    # ── Table curator_team_members ──
    op.execute("""
        CREATE TABLE IF NOT EXISTS curator_team_members (
            id SERIAL PRIMARY KEY,
            team_id INTEGER NOT NULL
                REFERENCES curator_teams(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL
                REFERENCES "user"(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_curator_team_member UNIQUE (team_id, user_id)
        )
    """)

    # ── FK curator_team_id sur institution_patients ──
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'institution_patients'
                  AND column_name = 'curator_team_id'
            ) THEN
                ALTER TABLE institution_patients
                    ADD COLUMN curator_team_id INTEGER
                    REFERENCES curator_teams(id) ON DELETE SET NULL;
            END IF;
        END $$
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_institution_patients_curator_team_id
        ON institution_patients (curator_team_id)
    """)

    # ── data_source_flags (JSON) sur institution_patients ──
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'institution_patients'
                  AND column_name = 'data_source_flags'
            ) THEN
                ALTER TABLE institution_patients
                    ADD COLUMN data_source_flags JSONB;
            END IF;
        END $$
    """)


def downgrade():
    op.execute("ALTER TABLE institution_patients DROP COLUMN IF EXISTS data_source_flags")
    op.execute("ALTER TABLE institution_patients DROP COLUMN IF EXISTS curator_team_id")
    op.execute("DROP TABLE IF EXISTS curator_team_members")
    op.execute("DROP TABLE IF EXISTS curator_teams")
