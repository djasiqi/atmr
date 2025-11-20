"""ensure_admin_value_in_user_role_enum

S'assure que l'enum user_role contient la valeur "admin" (en minuscule)
pour éviter les erreurs "invalid input value for enum user_role: 'admin'".

Revision ID: a1b2c3d4e5f6
Revises: 8186648ac54e
Create Date: 2025-01-28 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "8186648ac54e"
branch_labels = None
depends_on = None


def upgrade():
    """
    Ajoute la valeur 'admin' (minuscule) à l'enum user_role si elle n'existe pas déjà.
    PostgreSQL permet d'avoir à la fois 'ADMIN' et 'admin' dans le même enum.
    """
    # Vérifier si 'admin' existe déjà dans l'enum user_role
    # Si non, l'ajouter
    op.execute("""
        DO $$ 
        BEGIN 
            IF NOT EXISTS (
                SELECT 1 
                FROM pg_enum e 
                JOIN pg_type t ON e.enumtypid = t.oid 
                WHERE t.typname = 'user_role' 
                AND e.enumlabel = 'admin'
            ) THEN 
                ALTER TYPE user_role ADD VALUE 'admin';
            END IF;
        END $$;
    """)


def downgrade():
    """
    Supprimer une valeur d'enum n'est pas trivial en PostgreSQL.
    On laisse vide car la suppression d'une valeur d'enum nécessite
    de recréer le type ou de migrer les données.
    """
    pass

