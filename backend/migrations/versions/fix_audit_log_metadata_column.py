"""Fix: Renommer colonne metadata vers additional_metadata dans audit_logs.

Le nom 'metadata' est réservé dans SQLAlchemy Declarative API.

Revision ID: fix_audit_metadata
Revises: d2_audit_logs
Create Date: 2025-10-29 13:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "fix_audit_metadata"
down_revision = "d2_audit_logs"  # Après création table audit_logs
branch_labels = None
depends_on = None


def upgrade():
    # Renommer colonne metadata vers additional_metadata
    op.alter_column("audit_logs", "metadata", new_column_name="additional_metadata")


def downgrade():
    # Renommer back vers metadata
    op.alter_column("audit_logs", "additional_metadata", new_column_name="metadata")
