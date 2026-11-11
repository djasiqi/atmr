"""identifiant tentative message reservation

Revision ID: ed461584cdae
Revises: 9d7a508da66b
Create Date: 2026-10-21 08:21:06.856444

Colonne nullable ``client_message_id`` et index unique partiel sur
``booking_messages``. Une tentative rejouée retrouve le message déjà
enregistré. Les lignes sans identifiant (clients Web actuels) restent
autorisées.

La génération automatique a aussi vu d'autres écarts de modèle déjà couverts
par des révisions non appliquées sur cette base : ils sont exclus.
"""

from alembic import op
import sqlalchemy as sa


revision = "ed461584cdae"
down_revision = "9d7a508da66b"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "booking_messages",
        sa.Column("client_message_id", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "uq_booking_messages_client_attempt",
        "booking_messages",
        ["booking_id", "client_message_id"],
        unique=True,
        postgresql_where=sa.text("client_message_id IS NOT NULL"),
    )


def downgrade():
    op.drop_index(
        "uq_booking_messages_client_attempt",
        table_name="booking_messages",
        postgresql_where=sa.text("client_message_id IS NOT NULL"),
    )
    op.drop_column("booking_messages", "client_message_id")
