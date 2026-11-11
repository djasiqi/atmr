"""carnet privé de passagers

Revision ID: c3a91e70b4f2
Revises: b7e4c91a02d8
Create Date: 2026-10-15 00:40:00.000000

Migration manuelle : table du carnet et copie figée sur la réservation.
Aucune suppression d'autres objets.
"""

from alembic import op
import sqlalchemy as sa


revision = "c3a91e70b4f2"
down_revision = "b7e4c91a02d8"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "private_passenger",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("owner_user_id", sa.Integer(), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=False),
        sa.Column("phone", sa.String(length=30), nullable=True),
        sa.Column("access_notes", sa.String(length=500), nullable=True),
        sa.Column(
            "wheelchair_need",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "assistance_required",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("assistance_note", sa.String(length=200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["owner_user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
    )
    op.create_index(
        "ix_private_passenger_owner",
        "private_passenger",
        ["owner_user_id"],
    )
    op.add_column(
        "booking",
        sa.Column("passenger_phone", sa.String(length=30), nullable=True),
    )
    op.add_column(
        "booking",
        sa.Column("passenger_access_notes", sa.Text(), nullable=True),
    )


def downgrade():
    op.drop_column("booking", "passenger_access_notes")
    op.drop_column("booking", "passenger_phone")
    op.drop_index("ix_private_passenger_owner", table_name="private_passenger")
    op.drop_table("private_passenger")
