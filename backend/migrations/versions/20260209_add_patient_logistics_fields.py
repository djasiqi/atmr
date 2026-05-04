"""Ajouter champs logistiques + administratifs patient institution.

- Logistique (IMAD/domicile): code porte, étage, notes accès, résidence
- Administratif (clinique/EMS): AVS, assurance, curatelle

Revision ID: pat_logistics_001
Revises: (dépend de la dernière migration)
Create Date: 2026-02-09
"""

from alembic import op


# revision identifiers
revision = "pat_logistics_001"
down_revision = "20260204_patients_requests"
branch_labels = None
depends_on = None


def upgrade():
    """Ajouter colonnes logistiques + administratives (idempotent)."""
    # ── Logistique (IMAD / domicile) ──
    op.execute("""
        ALTER TABLE institution_patients
        ADD COLUMN IF NOT EXISTS door_code VARCHAR(50) DEFAULT NULL;
    """)
    op.execute("""
        ALTER TABLE institution_patients
        ADD COLUMN IF NOT EXISTS floor VARCHAR(20) DEFAULT NULL;
    """)
    op.execute("""
        ALTER TABLE institution_patients
        ADD COLUMN IF NOT EXISTS access_notes TEXT DEFAULT NULL;
    """)
    op.execute("""
        ALTER TABLE institution_patients
        ADD COLUMN IF NOT EXISTS residence_name VARCHAR(200) DEFAULT NULL;
    """)

    # ── Administratif (clinique / EMS) ──
    op.execute("""
        ALTER TABLE institution_patients
        ADD COLUMN IF NOT EXISTS avs_number VARCHAR(16) DEFAULT NULL;
    """)
    op.execute("""
        ALTER TABLE institution_patients
        ADD COLUMN IF NOT EXISTS insurance_name VARCHAR(200) DEFAULT NULL;
    """)
    op.execute("""
        ALTER TABLE institution_patients
        ADD COLUMN IF NOT EXISTS insurance_number VARCHAR(50) DEFAULT NULL;
    """)
    op.execute("""
        ALTER TABLE institution_patients
        ADD COLUMN IF NOT EXISTS has_guardianship BOOLEAN NOT NULL DEFAULT FALSE;
    """)
    op.execute("""
        ALTER TABLE institution_patients
        ADD COLUMN IF NOT EXISTS guardian_name VARCHAR(200) DEFAULT NULL;
    """)
    op.execute("""
        ALTER TABLE institution_patients
        ADD COLUMN IF NOT EXISTS guardian_phone VARCHAR(50) DEFAULT NULL;
    """)

    # ── Commentaires documentation ──
    op.execute(
        "COMMENT ON COLUMN institution_patients.door_code IS 'Code porte / digicode';"
    )
    op.execute(
        "COMMENT ON COLUMN institution_patients.floor IS 'Étage (ex: 3, RDC, 2B)';"
    )
    op.execute(
        "COMMENT ON COLUMN institution_patients.access_notes IS 'Notes accès chauffeur';"
    )
    op.execute(
        "COMMENT ON COLUMN institution_patients.residence_name IS 'Établissement de résidence';"
    )
    op.execute(
        "COMMENT ON COLUMN institution_patients.avs_number IS 'Numéro AVS (756.XXXX.XXXX.XX)';"
    )
    op.execute(
        "COMMENT ON COLUMN institution_patients.insurance_name IS 'Nom caisse maladie';"
    )
    op.execute(
        "COMMENT ON COLUMN institution_patients.insurance_number IS 'Numéro assuré';"
    )
    op.execute(
        "COMMENT ON COLUMN institution_patients.has_guardianship IS 'Patient sous curatelle';"
    )
    op.execute(
        "COMMENT ON COLUMN institution_patients.guardian_name IS 'Nom du curateur';"
    )
    op.execute(
        "COMMENT ON COLUMN institution_patients.guardian_phone IS 'Téléphone du curateur';"
    )


def downgrade():
    """Retirer les colonnes ajoutées (réversible)."""
    op.execute("ALTER TABLE institution_patients DROP COLUMN IF EXISTS guardian_phone;")
    op.execute("ALTER TABLE institution_patients DROP COLUMN IF EXISTS guardian_name;")
    op.execute(
        "ALTER TABLE institution_patients DROP COLUMN IF EXISTS has_guardianship;"
    )
    op.execute(
        "ALTER TABLE institution_patients DROP COLUMN IF EXISTS insurance_number;"
    )
    op.execute("ALTER TABLE institution_patients DROP COLUMN IF EXISTS insurance_name;")
    op.execute("ALTER TABLE institution_patients DROP COLUMN IF EXISTS avs_number;")
    op.execute("ALTER TABLE institution_patients DROP COLUMN IF EXISTS residence_name;")
    op.execute("ALTER TABLE institution_patients DROP COLUMN IF EXISTS access_notes;")
    op.execute("ALTER TABLE institution_patients DROP COLUMN IF EXISTS floor;")
    op.execute("ALTER TABLE institution_patients DROP COLUMN IF EXISTS door_code;")
