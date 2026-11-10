"""habilitations portail grant

Revision ID: a643270a0e07
Revises: cc4eebfd81e6
Create Date: 2026-10-12 00:03:35.702008

"""

from alembic import op
import sqlalchemy as sa


revision = "a643270a0e07"
down_revision = "cc4eebfd81e6"
branch_labels = None
depends_on = None


def upgrade():
    # Conservé depuis l'autogénération : uniquement portal_grant.
    # Le reste du diff autogénéré décrivait un écart historique de schéma,
    # sans lien avec les habilitations, et n'est pas appliqué.
    op.create_table(
        "portal_grant",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("portal", sa.String(length=32), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=True),
        sa.Column("institution_id", sa.Integer(), nullable=True),
        sa.Column("internal_role", sa.String(length=50), nullable=True),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="active",
            nullable=False,
        ),
        sa.Column(
            "granted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(portal = 'client' AND company_id IS NULL "
            "AND institution_id IS NULL) OR "
            "(portal IN ('company', 'driver') AND company_id IS NOT NULL "
            "AND institution_id IS NULL) OR "
            "(portal = 'institution' AND institution_id IS NOT NULL "
            "AND company_id IS NULL)",
            name="ck_portal_grant_organization",
        ),
        sa.CheckConstraint(
            "(status <> 'revoked') OR (revoked_at IS NOT NULL)",
            name="ck_portal_grant_revoked_at",
        ),
        sa.CheckConstraint(
            "portal IN ('client', 'company', 'driver', 'institution')",
            name="ck_portal_grant_portal",
        ),
        sa.CheckConstraint(
            "status IN ('pending_activation', 'active', 'revoked')",
            name="ck_portal_grant_status",
        ),
        sa.ForeignKeyConstraint(
            ["company_id"], ["company.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["institution_id"], ["institutions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_portal_grant_user_status",
        "portal_grant",
        ["user_id", "status"],
        unique=False,
    )
    op.create_index(
        "uq_portal_grant_user_client",
        "portal_grant",
        ["user_id"],
        unique=True,
        postgresql_where=sa.text("portal = 'client'"),
    )
    op.create_index(
        "uq_portal_grant_user_company_portal",
        "portal_grant",
        ["user_id", "portal", "company_id"],
        unique=True,
        postgresql_where=sa.text("company_id IS NOT NULL"),
    )
    op.create_index(
        "uq_portal_grant_user_institution_portal",
        "portal_grant",
        ["user_id", "portal", "institution_id"],
        unique=True,
        postgresql_where=sa.text("institution_id IS NOT NULL"),
    )


def downgrade():
    op.drop_index(
        "uq_portal_grant_user_institution_portal",
        table_name="portal_grant",
        postgresql_where=sa.text("institution_id IS NOT NULL"),
    )
    op.drop_index(
        "uq_portal_grant_user_company_portal",
        table_name="portal_grant",
        postgresql_where=sa.text("company_id IS NOT NULL"),
    )
    op.drop_index(
        "uq_portal_grant_user_client",
        table_name="portal_grant",
        postgresql_where=sa.text("portal = 'client'"),
    )
    op.drop_index("ix_portal_grant_user_status", table_name="portal_grant")
    op.drop_table("portal_grant")
