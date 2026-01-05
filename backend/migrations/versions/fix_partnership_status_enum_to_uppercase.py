"""fix_partnership_status_enum_to_uppercase

Revision ID: fix_partnership_status_uppercase
Revises: p1a2r3t4n5s7
Create Date: 2025-12-17 18:00:00.000000

Modifie l'enum partnership_status pour utiliser des majuscules (PENDING, ACCEPTED, REJECTED)
au lieu de minuscules (pending, accepted, rejected).
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "fix_partnership_status_uppercase"
down_revision = "p1a2r3t4n5s7"
branch_labels = None
depends_on = None


def upgrade():
    # 1. Supprimer la valeur par défaut qui dépend de l'enum
    op.execute("ALTER TABLE partnerships ALTER COLUMN status DROP DEFAULT")

    # 2. Convertir la colonne en text temporairement pour permettre la mise à jour
    op.execute("ALTER TABLE partnerships ALTER COLUMN status TYPE text")

    # 3. Mettre à jour les données existantes : convertir minuscules en majuscules
    op.execute("UPDATE partnerships SET status = 'PENDING' WHERE status = 'pending'")
    op.execute("UPDATE partnerships SET status = 'ACCEPTED' WHERE status = 'accepted'")
    op.execute("UPDATE partnerships SET status = 'REJECTED' WHERE status = 'rejected'")

    # 4. Supprimer l'ancien enum
    op.execute("DROP TYPE IF EXISTS partnership_status CASCADE")

    # 4. Créer le nouveau enum avec des majuscules
    op.execute(
        """
        DO $$ BEGIN
            CREATE TYPE partnership_status AS ENUM ('PENDING', 'ACCEPTED', 'REJECTED');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
        """
    )

    # 5. Remettre le type enum sur la colonne
    op.execute(
        "ALTER TABLE partnerships ALTER COLUMN status TYPE partnership_status USING status::partnership_status"
    )

    # 6. Remettre la valeur par défaut
    op.execute(
        "ALTER TABLE partnerships ALTER COLUMN status SET DEFAULT 'PENDING'::partnership_status"
    )


def downgrade():
    # 1. Mettre à jour les données existantes : convertir majuscules en minuscules
    op.execute(
        "UPDATE partnerships SET status = 'pending' WHERE status::text = 'PENDING'"
    )
    op.execute(
        "UPDATE partnerships SET status = 'accepted' WHERE status::text = 'ACCEPTED'"
    )
    op.execute(
        "UPDATE partnerships SET status = 'rejected' WHERE status::text = 'REJECTED'"
    )

    # 2. Supprimer l'ancien enum
    op.execute("ALTER TABLE partnerships ALTER COLUMN status TYPE text")
    op.execute("DROP TYPE IF EXISTS partnership_status")

    # 3. Créer le nouveau enum avec des minuscules
    op.execute(
        """
        DO $$ BEGIN
            CREATE TYPE partnership_status AS ENUM ('pending', 'accepted', 'rejected');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
        """
    )

    # 4. Remettre le type enum sur la colonne
    op.execute(
        "ALTER TABLE partnerships ALTER COLUMN status TYPE partnership_status USING status::partnership_status"
    )
