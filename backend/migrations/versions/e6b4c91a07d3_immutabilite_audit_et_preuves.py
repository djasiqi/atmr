"""immutabilite des journaux et preuves directes

Revision ID: e6b4c91a07d3
Revises: c4e8a17b02f6
Create Date: 2026-10-25 21:20:00.000000

Bloque les UPDATE et DELETE ordinaires sur audit_logs.
Bloque la suppression d'un paiement confirme et de toute attestation.
L'insertion de nouveaux evenements reste autorisee.
Aucune ligne existante n'est reecrite.
"""

from alembic import op


revision = "e6b4c91a07d3"
down_revision = "c4e8a17b02f6"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        CREATE OR REPLACE FUNCTION audit_logs_prevent_modification()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION 'Modification of audit_logs is not allowed. Audit logs are immutable for compliance.';
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute("DROP TRIGGER IF EXISTS audit_logs_no_update ON audit_logs;")
    op.execute(
        """
        CREATE TRIGGER audit_logs_no_update
        BEFORE UPDATE ON audit_logs
        FOR EACH ROW
        EXECUTE FUNCTION audit_logs_prevent_modification();
        """
    )
    op.execute("DROP TRIGGER IF EXISTS audit_logs_no_delete ON audit_logs;")
    op.execute(
        """
        CREATE TRIGGER audit_logs_no_delete
        BEFORE DELETE ON audit_logs
        FOR EACH ROW
        EXECUTE FUNCTION audit_logs_prevent_modification();
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION payment_completed_prevent_delete()
        RETURNS TRIGGER AS $$
        BEGIN
            IF upper(OLD.status::text) = 'COMPLETED' THEN
                RAISE EXCEPTION 'Suppression d''un paiement confirme interdite.';
            END IF;
            RETURN OLD;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute("DROP TRIGGER IF EXISTS payment_completed_no_delete ON payment;")
    op.execute(
        """
        CREATE TRIGGER payment_completed_no_delete
        BEFORE DELETE ON payment
        FOR EACH ROW
        EXECUTE FUNCTION payment_completed_prevent_delete();
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION transport_attestation_prevent_delete()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION 'Suppression d''une attestation interdite.';
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        "DROP TRIGGER IF EXISTS transport_attestation_no_delete ON transport_attestation;"
    )
    op.execute(
        """
        CREATE TRIGGER transport_attestation_no_delete
        BEFORE DELETE ON transport_attestation
        FOR EACH ROW
        EXECUTE FUNCTION transport_attestation_prevent_delete();
        """
    )


def downgrade():
    op.execute(
        "DROP TRIGGER IF EXISTS transport_attestation_no_delete ON transport_attestation;"
    )
    op.execute("DROP FUNCTION IF EXISTS transport_attestation_prevent_delete();")
    op.execute("DROP TRIGGER IF EXISTS payment_completed_no_delete ON payment;")
    op.execute("DROP FUNCTION IF EXISTS payment_completed_prevent_delete();")
    op.execute("DROP TRIGGER IF EXISTS audit_logs_no_delete ON audit_logs;")
    op.execute("DROP TRIGGER IF EXISTS audit_logs_no_update ON audit_logs;")
    op.execute("DROP FUNCTION IF EXISTS audit_logs_prevent_modification();")
