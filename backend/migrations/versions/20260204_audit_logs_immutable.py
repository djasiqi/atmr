"""make_audit_logs_immutable

Revision ID: 20260204_audit_immut
Revises: 20260204_offers
Create Date: 2026-02-04

ÉTAPE 5: Rendre la table audit_logs immuable via triggers PostgreSQL.
- BEFORE UPDATE: RAISE EXCEPTION
- BEFORE DELETE: RAISE EXCEPTION

Ceci garantit l'intégrité des logs d'audit pour la traçabilité.
"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "20260204_audit_immut"
down_revision = "20260204_offers"
branch_labels = None
depends_on = None


def upgrade():
    # 1. Créer la fonction qui bloque les modifications
    op.execute("""
        CREATE OR REPLACE FUNCTION audit_logs_prevent_modification()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION 'Modification of audit_logs is not allowed. Audit logs are immutable for compliance.';
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # 2. Créer le trigger BEFORE UPDATE
    op.execute("""
        CREATE TRIGGER audit_logs_no_update
        BEFORE UPDATE ON audit_logs
        FOR EACH ROW
        EXECUTE FUNCTION audit_logs_prevent_modification();
    """)

    # 3. Créer le trigger BEFORE DELETE
    op.execute("""
        CREATE TRIGGER audit_logs_no_delete
        BEFORE DELETE ON audit_logs
        FOR EACH ROW
        EXECUTE FUNCTION audit_logs_prevent_modification();
    """)


def downgrade():
    # 1. Supprimer les triggers
    op.execute("DROP TRIGGER IF EXISTS audit_logs_no_delete ON audit_logs;")
    op.execute("DROP TRIGGER IF EXISTS audit_logs_no_update ON audit_logs;")

    # 2. Supprimer la fonction
    op.execute("DROP FUNCTION IF EXISTS audit_logs_prevent_modification();")
