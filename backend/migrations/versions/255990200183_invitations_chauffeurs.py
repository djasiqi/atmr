"""invitations chauffeurs

Revision ID: 255990200183
Revises: 8e76ed1ac6c6
Create Date: 2026-10-12 20:34:03.147211

"""

from alembic import op
import sqlalchemy as sa


revision = "255990200183"
down_revision = "8e76ed1ac6c6"
branch_labels = None
depends_on = None


def upgrade():
    # Conservé depuis l'autogénération : uniquement la table d'invitation.
    # Le reste du diff décrivait un écart historique de schéma, sans lien avec
    # les invitations chauffeur, et n'est pas appliqué.
    op.create_table(
        "driver_invitation",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("invited_by_user_id", sa.Integer(), nullable=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("first_name", sa.String(length=80), nullable=False),
        sa.Column("last_name", sa.String(length=80), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accepted_user_id", sa.Integer(), nullable=True),
        sa.Column("created_driver_id", sa.Integer(), nullable=True),
        sa.Column("resend_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
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
        sa.CheckConstraint(
            "status IN ('pending', 'accepted', 'cancelled', 'expired', 'revoked')",
            name="ck_driver_invitation_status",
        ),
        sa.ForeignKeyConstraint(
            ["accepted_user_id"],
            ["user.id"],
            name="fk_driver_invitation_accepted_user",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["company.id"],
            name="fk_driver_invitation_company",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_driver_id"],
            ["driver.id"],
            name="fk_driver_invitation_created_driver",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["invited_by_user_id"],
            ["user.id"],
            name="fk_driver_invitation_invited_by",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index(
        "ix_driver_invitation_company_status",
        "driver_invitation",
        ["company_id", "status"],
        unique=False,
    )
    op.create_index(
        "uq_driver_invitation_pending_email",
        "driver_invitation",
        ["company_id", "email"],
        unique=True,
        postgresql_where=sa.text("status = 'pending'"),
    )


def downgrade():
    op.drop_index(
        "uq_driver_invitation_pending_email",
        table_name="driver_invitation",
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.drop_index("ix_driver_invitation_company_status", table_name="driver_invitation")
    op.drop_table("driver_invitation")
