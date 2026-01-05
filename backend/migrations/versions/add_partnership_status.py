"""add_partnership_status

Revision ID: p1a2r3t4n5s7
Revises: p1a2r3t4n5i7
Create Date: 2025-12-17 17:00:00.000000

Ajoute le champ status au modèle Partnership pour gérer les demandes (pending, accepted, rejected).
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "p1a2r3t4n5s7"
down_revision = "p1a2r3t4n5i7"
branch_labels = None
depends_on = None


def upgrade():
    """
    Ajoute le champ status au modèle Partnership pour gérer les demandes (pending, accepted, rejected).
    
    ⚠️ NOTE: Cette migration est idempotente - elle vérifie l'existence
    de la colonne avant de l'ajouter. En production, la colonne peut
    avoir été créée manuellement avant l'application de cette migration.
    """
    # Vérifier l'existence de la colonne avant de l'ajouter
    from sqlalchemy import inspect
    from sqlalchemy.engine import reflection
    
    bind = op.get_bind()
    inspector = reflection.Inspector.from_engine(bind)
    existing_columns = [col["name"] for col in inspector.get_columns("partnerships")]
    existing_indexes = [idx["name"] for idx in inspector.get_indexes("partnerships")]
    
    # Créer le type enum partnership_status (si il n'existe pas déjà)
    op.execute(
        """
        DO $$ BEGIN
            CREATE TYPE partnership_status AS ENUM ('PENDING', 'ACCEPTED', 'REJECTED');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
        """
    )

    # Ajouter la colonne status (si elle n'existe pas déjà)
    if "status" not in existing_columns:
        op.add_column(
            "partnerships",
            sa.Column(
                "status",
                sa.Enum("PENDING", "ACCEPTED", "REJECTED", name="partnership_status"),
                nullable=False,
                server_default="PENDING",
            ),
        )
        
        # Mettre à jour les partenariats existants en "ACCEPTED" (seulement si la colonne vient d'être créée)
        op.execute("UPDATE partnerships SET status = 'ACCEPTED' WHERE is_active = true")
        op.execute("UPDATE partnerships SET status = 'REJECTED' WHERE is_active = false")

    # Créer l'index (si il n'existe pas déjà)
    if "ix_partnerships_status" not in existing_indexes:
        op.create_index("ix_partnerships_status", "partnerships", ["status"])


def downgrade():
    # Supprimer l'index
    op.drop_index("ix_partnerships_status", table_name="partnerships")

    # Supprimer la colonne
    op.drop_column("partnerships", "status")

    # Supprimer le type enum (optionnel, peut être utilisé ailleurs)
    op.execute("DROP TYPE IF EXISTS partnership_status")
