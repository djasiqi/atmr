"""add_created_by_display_name_to_transport_requests

Revision ID: 9c313d7f5a0a
Revises: ad0020bf5f62
Create Date: 2026-08-31 08:31:37.683431

DB-SCHEMA-CONSOLIDATION-17
--------------------------
Autogenerate Alembic contre head ``ad0020bf5f62`` (DB vierge ``atmr_schema17``)
a détecté l'ajout intentionnel :

    transport_requests.created_by_display_name  VARCHAR(255) NULL

ainsi qu'un grand volume de faux positifs (renommages d'index / contraintes
uniques déjà présents sous d'autres noms dans le schéma historique).

Après inspection manuelle :
- DROP TABLE / DROP COLUMN hors cible / ALTER TYPE = AUCUN
- changements non liés = REJETÉS (non appliqués)
- seul le changement CNY acteur est conservé

"""

from alembic import op
import sqlalchemy as sa


revision = "9c313d7f5a0a"
down_revision = "ad0020bf5f62"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("transport_requests", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("created_by_display_name", sa.String(length=255), nullable=True)
        )


def downgrade():
    with op.batch_alter_table("transport_requests", schema=None) as batch_op:
        batch_op.drop_column("created_by_display_name")
