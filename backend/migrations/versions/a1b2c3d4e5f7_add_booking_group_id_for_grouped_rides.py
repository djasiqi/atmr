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
    """
    # Ajouter le champ booking_group_id à la table booking
    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.add_column(sa.Column("booking_group_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_booking_booking_group",
            "booking",
            ["booking_group_id"],
            ["id"],
            ondelete="SET NULL",
        )
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
