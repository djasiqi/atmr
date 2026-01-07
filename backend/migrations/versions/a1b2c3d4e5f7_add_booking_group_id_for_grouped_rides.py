"""add_booking_group_id_for_grouped_rides

Revision ID: a1b2c3d4e5f7
Revises: f7e8d9c0b1a2
Create Date: 2025-12-17 14:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f7"
down_revision = "f7e8d9c0b1a2"
branch_labels = None
depends_on = None


def upgrade():
    """
    Ajoute le champ booking_group_id à la table booking pour permettre
    de grouper des courses (même heure, même départ, même destination).

    ⚠️ NOTE: Cette migration est idempotente - elle vérifie l'existence
    de la colonne avant de l'ajouter. En production, la colonne peut
    avoir été créée manuellement avant l'application de cette migration.
    """
    # Vérifier l'existence de la colonne avant de l'ajouter
    from sqlalchemy import inspect
    from sqlalchemy.engine import reflection

    bind = op.get_bind()
    inspector = reflection.Inspector.from_engine(bind)
    existing_columns = [col["name"] for col in inspector.get_columns("booking")]
    existing_indexes = [idx["name"] for idx in inspector.get_indexes("booking")]
    existing_foreign_keys = [fk["name"] for fk in inspector.get_foreign_keys("booking")]

    # Ajouter le champ booking_group_id à la table booking (si elle n'existe pas déjà)
    if "booking_group_id" not in existing_columns:
        with op.batch_alter_table("booking", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("booking_group_id", sa.Integer(), nullable=True)
            )
    else:
        # La colonne existe déjà - on skip l'ajout mais on vérifie les contraintes
        pass

    # Créer la foreign key (si elle n'existe pas déjà)
    if "fk_booking_booking_group" not in existing_foreign_keys:
        with op.batch_alter_table("booking", schema=None) as batch_op:
            batch_op.create_foreign_key(
                "fk_booking_booking_group",
                "booking",
                ["booking_group_id"],
                ["id"],
                ondelete="SET NULL",
            )

    # Créer l'index (si il n'existe pas déjà)
    if "ix_booking_booking_group_id" not in existing_indexes:
        with op.batch_alter_table("booking", schema=None) as batch_op:
            batch_op.create_index(
                "ix_booking_booking_group_id",
                ["booking_group_id"],
                unique=False,
            )


def downgrade():
    """
    Rollback: supprime le champ booking_group_id et ses contraintes.
    """
    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.drop_index("ix_booking_booking_group_id")
        batch_op.drop_constraint("fk_booking_booking_group", type_="foreignkey")
        batch_op.drop_column("booking_group_id")
