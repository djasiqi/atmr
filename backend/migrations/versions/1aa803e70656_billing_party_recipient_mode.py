"""billing_party_recipient_mode

Revision ID: 1aa803e70656
Revises: b2ca98b6cb2d
Create Date: 2026-10-01 19:10:11.537453

Colonne explicite du rôle de destinataire (auto / care_of / debtor).
Autogénérée depuis une base au head Git ``b2ca98b6cb2d``, puis restreinte
aux deux tables concernées : aucun drift, aucun UPDATE de données
existantes (défaut serveur ``auto``).
"""

import sqlalchemy as sa
from alembic import op

revision = "1aa803e70656"
down_revision = "b2ca98b6cb2d"
branch_labels = None
depends_on = None

_RECIPIENT_MODE = sa.Enum(
    "auto",
    "care_of",
    "debtor",
    name="billing_party_recipient_mode",
)


def upgrade():
    bind = op.get_bind()
    _RECIPIENT_MODE.create(bind, checkfirst=True)
    op.add_column(
        "billing_parties",
        sa.Column(
            "recipient_mode",
            _RECIPIENT_MODE,
            server_default="auto",
            nullable=False,
        ),
    )
    op.add_column(
        "client_billing_parties",
        sa.Column("recipient_mode", _RECIPIENT_MODE, nullable=True),
    )


def downgrade():
    op.drop_column("client_billing_parties", "recipient_mode")
    op.drop_column("billing_parties", "recipient_mode")
    _RECIPIENT_MODE.drop(op.get_bind(), checkfirst=True)
