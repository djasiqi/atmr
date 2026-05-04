"""add_request_offers_and_preferences

Revision ID: 20260204_offers
Revises: 20260204_inst_patients
Create Date: 2026-02-04

ÉTAPE 4: Ajoute les tables pour l'orchestration des offres de transport:
- request_offers: Offres envoyées aux entreprises
- institution_transport_preferences: Préférences de transporteurs par institution
- transport_requests.accepted_by_company_id: Entreprise acceptante
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260204_offers"
down_revision = "20260204_patients_requests"
branch_labels = None
depends_on = None


def upgrade():
    # 1. Créer table request_offers
    op.create_table(
        "request_offers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("transport_request_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column(
            "mode", sa.String(length=20), nullable=False, server_default="broadcast"
        ),
        sa.Column("order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "status", sa.String(length=20), nullable=False, server_default="PENDING"
        ),
        sa.Column(
            "sent_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("responded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["transport_request_id"],
            ["transport_requests.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["company.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "transport_request_id",
            "company_id",
            name="uq_request_offer_request_company",
        ),
    )
    op.create_index(
        "ix_request_offers_company_status",
        "request_offers",
        ["company_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_request_offers_request_id",
        "request_offers",
        ["transport_request_id"],
        unique=False,
    )
    op.create_index(
        "ix_request_offers_expires_at",
        "request_offers",
        ["expires_at"],
        unique=False,
    )

    # 2. Créer table institution_transport_preferences
    op.create_table(
        "institution_transport_preferences",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("institution_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["institution_id"],
            ["institutions.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["company.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "institution_id", "company_id", name="uq_institution_transport_preference"
        ),
    )
    op.create_index(
        "ix_institution_transport_pref_institution",
        "institution_transport_preferences",
        ["institution_id"],
        unique=False,
    )

    # 3. Ajouter colonne accepted_by_company_id à transport_requests
    op.add_column(
        "transport_requests",
        sa.Column("accepted_by_company_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_transport_requests_accepted_company",
        "transport_requests",
        "company",
        ["accepted_by_company_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_transport_requests_accepted_company",
        "transport_requests",
        ["accepted_by_company_id"],
        unique=False,
    )


def downgrade():
    # 1. Supprimer colonne accepted_by_company_id de transport_requests
    op.drop_index(
        "ix_transport_requests_accepted_company", table_name="transport_requests"
    )
    op.drop_constraint(
        "fk_transport_requests_accepted_company",
        "transport_requests",
        type_="foreignkey",
    )
    op.drop_column("transport_requests", "accepted_by_company_id")

    # 2. Supprimer table institution_transport_preferences
    op.drop_index(
        "ix_institution_transport_pref_institution",
        table_name="institution_transport_preferences",
    )
    op.drop_table("institution_transport_preferences")

    # 3. Supprimer table request_offers
    op.drop_index("ix_request_offers_expires_at", table_name="request_offers")
    op.drop_index("ix_request_offers_request_id", table_name="request_offers")
    op.drop_index("ix_request_offers_company_status", table_name="request_offers")
    op.drop_table("request_offers")
