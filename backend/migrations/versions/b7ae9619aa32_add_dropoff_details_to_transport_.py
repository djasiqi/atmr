"""add dropoff details to transport_request_legs

Ajoute les colonnes établissement / service / médecin au point d'arrivée
de chaque étape (leg) afin que les destinations supplémentaires (2, 3, 4...)
puissent porter les mêmes détails que la destination principale.

Revision ID: b7ae9619aa32
Revises: 20260611_institution_timeline
Create Date: 2026-06-11 21:31:07.834463

"""
from alembic import op
import sqlalchemy as sa


revision = "b7ae9619aa32"
down_revision = "20260611_institution_timeline"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("transport_request_legs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("dropoff_establishment", sa.String(length=255), nullable=True)
        )
        batch_op.add_column(
            sa.Column("dropoff_service", sa.String(length=255), nullable=True)
        )
        batch_op.add_column(
            sa.Column("dropoff_doctor", sa.String(length=255), nullable=True)
        )


def downgrade():
    with op.batch_alter_table("transport_request_legs", schema=None) as batch_op:
        batch_op.drop_column("dropoff_doctor")
        batch_op.drop_column("dropoff_service")
        batch_op.drop_column("dropoff_establishment")
