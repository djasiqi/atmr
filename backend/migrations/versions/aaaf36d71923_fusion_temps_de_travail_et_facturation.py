"""fusion_temps_de_travail_et_facturation

Revision ID: aaaf36d71923
Revises: ('001d2f093160', '1aa803e70656')
Create Date: 2026-10-02 01:15:55.776704

Fusionne les deux têtes issues de ``b2ca98b6cb2d`` : le temps de travail
(``001d2f093160``) et la facturation (``1aa803e70656``). Aucun changement
de schéma.
"""

from alembic import op
import sqlalchemy as sa


revision = "aaaf36d71923"
down_revision = ("001d2f093160", "1aa803e70656")
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
