"""borne de lecture des conversations apres reaffectation

Revision ID: 9ddf4ec5763f
Revises: 99a1e3b30d27
Create Date: 2026-10-14 11:58:27.247165

"""
from alembic import op
import sqlalchemy as sa


revision = "9ddf4ec5763f"
down_revision = "99a1e3b30d27"
branch_labels = None
depends_on = None


def upgrade():
    # Colonne détectée par autogenerate. Le reste du diff (index et
    # contraintes déjà présents) a été retiré pour ne pas réécrire le schéma.
    op.add_column(
        "conversation_participant",
        sa.Column("read_cutoff_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade():
    op.drop_column("conversation_participant", "read_cutoff_at")
