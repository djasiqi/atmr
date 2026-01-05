"""add_message_file_fields

Revision ID: add_message_file_fields
Revises:
Create Date: 2025-01-18 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "add_message_file_fields"
down_revision = "24bbcb82c891"  # Dernière migration (add_vat_columns_to_invoices)
branch_labels = None
depends_on = None


def upgrade():
    # Ajouter les colonnes pour les fichiers dans la table message
    # Vérifier si les colonnes existent déjà avant de les ajouter (pour éviter les erreurs lors des merges)
    from contextlib import suppress

    from sqlalchemy import inspect

    bind = op.get_bind()
    inspector = inspect(bind)

    # Obtenir la liste des colonnes existantes
    existing_columns = [col["name"] for col in inspector.get_columns("message")]

    # Ajouter seulement les colonnes qui n'existent pas déjà
    if "image_url" not in existing_columns:
        op.add_column(
            "message", sa.Column("image_url", sa.String(length=500), nullable=True)
        )
    if "pdf_url" not in existing_columns:
        op.add_column(
            "message", sa.Column("pdf_url", sa.String(length=500), nullable=True)
        )
    if "pdf_filename" not in existing_columns:
        op.add_column(
            "message", sa.Column("pdf_filename", sa.String(length=255), nullable=True)
        )
    if "pdf_size" not in existing_columns:
        op.add_column("message", sa.Column("pdf_size", sa.Integer(), nullable=True))


def downgrade():
    # Supprimer les colonnes en cas de rollback
    op.drop_column("message", "pdf_size")
    op.drop_column("message", "pdf_filename")
    op.drop_column("message", "pdf_url")
    op.drop_column("message", "image_url")
