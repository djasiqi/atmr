"""Add dispatch modes to company

Revision ID: a8f4c9e2b1d3
Revises: f3a9c7b8d1e2
Create Date: 2025-0.1-17 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "a8f4c9e2b1d3"
down_revision = "f3a9c7b8d1e2"
branch_labels = None
depends_on = None


def upgrade():
    """
    Ajoute les champs dispatch_mode et autonomous_config à la table company.
    - dispatch_mode: Mode de fonctionnement (manual, semi_auto, fully_auto)
    - autonomous_config: Configuration JSON pour le dispatch autonome
    """
    # Créer l'enum pour dispatch_mode
    dispatch_mode_enum = sa.Enum(
        "MANUAL", "SEMI_AUTO", "FULLY_AUTO", name="dispatchmode"
    )
    dispatch_mode_enum.create(op.get_bind(), checkfirst=True)

    # Ajouter la colonne dispatch_mode avec valeur par défaut
    op.add_column(
        "company",
        sa.Column(
            "dispatch_mode",
            dispatch_mode_enum,
            nullable=False,
            server_default="SEMI_AUTO",
            comment="Mode de fonctionnement du dispatch: MANUAL, SEMI_AUTO, FULLY_AUTO",
        ),
    )

    # Ajouter la colonne autonomous_config (JSON stocké en TEXT)
    op.add_column(
        "company",
        sa.Column(
            "autonomous_config",
            sa.Text(),
            nullable=True,
            comment="Configuration JSON pour le dispatch autonome",
        ),
    )

    # Créer un index sur dispatch_mode pour les requêtes de filtrage
    op.create_index(
        "ix_company_dispatch_mode", "company", ["dispatch_mode"], unique=False
    )


def downgrade():
    """
    Supprime les champs ajoutés pour le dispatch autonome.
    """
    # Supprimer l'index
    op.drop_index("ix_company_dispatch_mode", table_name="company")

    # Supprimer les colonnes
    op.drop_column("company", "autonomous_config")
    op.drop_column("company", "dispatch_mode")

    # Supprimer l'enum (si plus utilisé ailleurs)
    dispatch_mode_enum = sa.Enum(
        "MANUAL", "SEMI_AUTO", "FULLY_AUTO", name="dispatchmode"
    )
    dispatch_mode_enum.drop(op.get_bind(), checkfirst=True)
