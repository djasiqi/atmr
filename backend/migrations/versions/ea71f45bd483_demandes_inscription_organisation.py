"""demandes inscription organisation

Revision ID: ea71f45bd483
Revises: a643270a0e07
Create Date: 2026-10-12 10:43:47.022456

"""

from alembic import op
import sqlalchemy as sa


revision = "ea71f45bd483"
down_revision = "a643270a0e07"
branch_labels = None
depends_on = None


def upgrade():
    # Conservé depuis l'autogénération : uniquement les dossiers de préinscription.
    # Le reste du diff décrivait un écart historique de schéma, sans lien avec
    # ces demandes, et n'est pas appliqué.
    op.create_table(
        "organization_application",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="email_unverified",
            nullable=False,
        ),
        sa.Column("applicant_first_name", sa.String(length=80), nullable=False),
        sa.Column("applicant_last_name", sa.String(length=80), nullable=False),
        sa.Column("applicant_function", sa.String(length=80), nullable=False),
        sa.Column("applicant_email", sa.String(length=255), nullable=False),
        sa.Column("applicant_phone", sa.String(length=30), nullable=False),
        sa.Column("legal_name", sa.String(length=200), nullable=True),
        sa.Column("ide_number", sa.String(length=20), nullable=True),
        sa.Column("establishment_name", sa.String(length=200), nullable=True),
        sa.Column("establishment_category", sa.String(length=50), nullable=True),
        sa.Column("street", sa.String(length=255), nullable=False),
        sa.Column("postal_code", sa.String(length=12), nullable=False),
        sa.Column("city", sa.String(length=100), nullable=False),
        sa.Column("country", sa.String(length=2), nullable=False),
        sa.Column("review_note", sa.Text(), nullable=True),
        sa.Column("matched_company_id", sa.Integer(), nullable=True),
        sa.Column("matched_institution_id", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "kind IN ('company', 'institution')",
            name="ck_org_application_kind",
        ),
        sa.CheckConstraint(
            "status IN ("
            "'email_unverified', 'received', 'in_review', 'needs_info', "
            "'approved', 'refused', 'abandoned')",
            name="ck_org_application_status",
        ),
        sa.CheckConstraint(
            "("
            "kind = 'company' AND legal_name IS NOT NULL "
            "AND establishment_name IS NULL AND establishment_category IS NULL"
            ") OR ("
            "kind = 'institution' AND establishment_name IS NOT NULL "
            "AND establishment_category IS NOT NULL "
            "AND legal_name IS NULL AND ide_number IS NULL"
            ")",
            name="ck_org_application_kind_fields",
        ),
        sa.ForeignKeyConstraint(
            ["matched_company_id"], ["company.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["matched_institution_id"], ["institutions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
    )
    op.create_index(
        "ix_organization_application_applicant_email",
        "organization_application",
        ["applicant_email"],
        unique=False,
    )
    op.create_index(
        "uq_org_application_open_email_kind",
        "organization_application",
        ["applicant_email", "kind"],
        unique=True,
        postgresql_where=sa.text(
            "status IN ('email_unverified', 'received', 'in_review', 'needs_info')"
        ),
    )
    op.create_table(
        "organization_application_event",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("application_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["application_id"],
            ["organization_application.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_organization_application_event_application_id",
        "organization_application_event",
        ["application_id"],
        unique=False,
    )
    op.create_table(
        "organization_application_token",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("application_id", sa.Integer(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["application_id"],
            ["organization_application.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index(
        "ix_organization_application_token_application_id",
        "organization_application_token",
        ["application_id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_organization_application_token_application_id",
        table_name="organization_application_token",
    )
    op.drop_table("organization_application_token")
    op.drop_index(
        "ix_organization_application_event_application_id",
        table_name="organization_application_event",
    )
    op.drop_table("organization_application_event")
    op.drop_index(
        "uq_org_application_open_email_kind",
        table_name="organization_application",
    )
    op.drop_index(
        "ix_organization_application_applicant_email",
        table_name="organization_application",
    )
    op.drop_table("organization_application")
