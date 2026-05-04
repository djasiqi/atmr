"""Add guardianship detail fields to institution_patients.

Revision ID: 20260211_guard
Revises: 20260209_catchup_sync_all_models
Create Date: 2026-02-11
"""

from alembic import op

# revision identifiers
revision = "20260211_guard"
down_revision = "058be909ad9c"
branch_labels = None
depends_on = None


def upgrade():
    """Ajoute les champs détaillés de curatelle aux patients institution."""

    # Type de curatelle (curatorship, opad, lawyer, family, other)
    op.execute("""
        ALTER TABLE institution_patients
        ADD COLUMN IF NOT EXISTS guardianship_type VARCHAR(30) DEFAULT NULL;
    """)

    # Organisation du curateur
    op.execute("""
        ALTER TABLE institution_patients
        ADD COLUMN IF NOT EXISTS guardian_organization VARCHAR(200) DEFAULT NULL;
    """)

    # Email du curateur
    op.execute("""
        ALTER TABLE institution_patients
        ADD COLUMN IF NOT EXISTS guardian_email VARCHAR(200) DEFAULT NULL;
    """)

    # Adresse complète du curateur (pour facturation)
    op.execute("""
        ALTER TABLE institution_patients
        ADD COLUMN IF NOT EXISTS guardian_address VARCHAR(500) DEFAULT NULL;
    """)

    # Commentaires documentation
    op.execute(
        "COMMENT ON COLUMN institution_patients.guardianship_type "
        "IS 'Type de curatelle: curatorship, opad, lawyer, family, other';"
    )
    op.execute(
        "COMMENT ON COLUMN institution_patients.guardian_organization "
        "IS 'Organisation du curateur (OPAD Genève, Étude Me. Dupont, etc.)';"
    )
    op.execute(
        "COMMENT ON COLUMN institution_patients.guardian_email IS 'Email du curateur';"
    )
    op.execute(
        "COMMENT ON COLUMN institution_patients.guardian_address "
        "IS 'Adresse complète du curateur (facturation)';"
    )


def downgrade():
    op.execute(
        "ALTER TABLE institution_patients DROP COLUMN IF EXISTS guardianship_type;"
    )
    op.execute(
        "ALTER TABLE institution_patients DROP COLUMN IF EXISTS guardian_organization;"
    )
    op.execute("ALTER TABLE institution_patients DROP COLUMN IF EXISTS guardian_email;")
    op.execute(
        "ALTER TABLE institution_patients DROP COLUMN IF EXISTS guardian_address;"
    )
