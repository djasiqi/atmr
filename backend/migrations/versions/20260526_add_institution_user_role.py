"""Ajoute INSTITUTION à l'enum PostgreSQL user_role.

Revision ID: 20260526_institution_role
Revises: 20260526_merge_heads
Create Date: 2026-05-26
"""

from alembic import op

revision = "20260526_institution_role"
down_revision = "20260526_merge_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_enum e
                JOIN pg_type t ON e.enumtypid = t.oid
                WHERE t.typname = 'user_role'
                AND e.enumlabel = 'INSTITUTION'
            ) THEN
                ALTER TYPE user_role ADD VALUE 'INSTITUTION';
            END IF;
        END $$;
    """)


def downgrade() -> None:
    # Suppression d'une valeur d'enum non supportée sans recréer le type.
    pass
