"""assignment revision monotone mission state

Révision monotone du lifecycle mission (P1 MISSION-STATE) : incrémentée à
chaque transition d'Assignment, exposée au mobile pour ignorer les snapshots
périmés (anti-régression UI).

Généré par autogenerate puis réduit à la seule colonne `assignment.revision`
(le diff brut contenait du drift dev sans rapport avec ce changement).

Revision ID: ad0020bf5f62
Revises: 14d1b170291f
Create Date: 2026-08-27 11:11:27.692631

"""

import sqlalchemy as sa
from alembic import op

revision = "ad0020bf5f62"
down_revision = "14d1b170291f"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "assignment",
        sa.Column("revision", sa.Integer(), server_default="0", nullable=False),
    )


def downgrade():
    op.drop_column("assignment", "revision")
