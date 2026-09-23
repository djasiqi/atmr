"""portal terms requires reacceptance

Revision ID: 22e5139ffb5c
Revises: 98dac28420c7
Create Date: 2026-09-23 21:23:50.223936

Colonne générée par ``flask db revision --autogenerate``.
Les écarts d'index sans rapport avec cette décision de publication ont été retirés.
``ADD COLUMN`` direct : ``batch_alter_table`` recréerait la table et perdrait
les triggers d'immutabilité.
"""

import sqlalchemy as sa
from alembic import op

revision = "22e5139ffb5c"
down_revision = "98dac28420c7"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "legal_document_version",
        sa.Column(
            "requires_reacceptance",
            sa.Boolean(),
            server_default="true",
            nullable=False,
        ),
    )


def downgrade():
    op.drop_column("legal_document_version", "requires_reacceptance")
