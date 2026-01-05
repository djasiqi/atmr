"""add_partnership_system

Revision ID: p1a2r3t4n5s6
Revises: 9fbfec587490
Create Date: 2025-01-XX XX:XX:XX.XXXXXX

Ajoute le système de partenariat et sous-traitance entre entreprises.
- Table partnerships : partenariats entre entreprises
- Table booking_transfers : transferts de courses à des partenaires
- Colonne executing_company_id dans bookings : entreprise qui exécute réellement
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "p1a2r3t4n5s6"  # Hash unique pour la migration partnership
down_revision = (
    "9fbfec587490"  # Dernière migration: s3_add_password_history_and_expiration
)
branch_labels = None
depends_on = None


def upgrade():
    # Créer les types d'enum pour les transferts (si ils n'existent pas déjà)
    # Note: Utilisation de DO $$ BEGIN ... EXCEPTION pour éviter les erreurs si les types existent déjà
    op.execute(
        """
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'transfermodel') THEN
                CREATE TYPE transfermodel AS ENUM (
                    'SUBCONTRACT', 'ASSIGN_TO_PARTNER', 'MARKETPLACE'
                );
            END IF;
        END $$;
    """
    )

    op.execute(
        """
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'transferstatus') THEN
                CREATE TYPE transferstatus AS ENUM (
                    'PENDING', 'ACCEPTED', 'REJECTED', 'COMPLETED', 'CANCELLED'
                );
            END IF;
        END $$;
    """
    )

    # Créer la table partnerships
    op.create_table(
        "partnerships",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("owner_company_id", sa.Integer(), nullable=False),
        sa.Column("partner_company_id", sa.Integer(), nullable=False),
        sa.Column(
            "default_transfer_model",
            postgresql.ENUM(
                "SUBCONTRACT",
                "ASSIGN_TO_PARTNER",
                "MARKETPLACE",
                name="transfermodel",
                create_type=False,  # Ne pas créer le type, il existe déjà
            ),
            nullable=False,
            server_default="SUBCONTRACT",
        ),
        sa.Column("default_margin_percent", sa.Numeric(5, 2), nullable=True),
        sa.Column("default_partner_tariff_percent", sa.Numeric(5, 2), nullable=True),
        sa.Column(
            "auto_accept_rules", sa.Boolean(), nullable=False, server_default="false"
        ),
        sa.Column("auto_invoice", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "payment_terms_days", sa.Integer(), nullable=False, server_default="30"
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_company_id"],
            ["company.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["partner_company_id"],
            ["company.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "owner_company_id", "partner_company_id", name="unique_partnership"
        ),
    )

    # Créer les index pour partnerships
    op.create_index(
        "ix_partnerships_owner_company",
        "partnerships",
        ["owner_company_id"],
    )
    op.create_index(
        "ix_partnerships_partner_company",
        "partnerships",
        ["partner_company_id"],
    )

    # Créer la table booking_transfers
    op.create_table(
        "booking_transfers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("partnership_id", sa.Integer(), nullable=False),
        sa.Column(
            "transfer_model",
            postgresql.ENUM(
                "SUBCONTRACT",
                "ASSIGN_TO_PARTNER",
                "MARKETPLACE",
                name="transfermodel",
                create_type=False,  # Ne pas créer le type, il existe déjà
            ),
            nullable=False,
        ),
        sa.Column("owner_company_id", sa.Integer(), nullable=False),
        sa.Column("executing_company_id", sa.Integer(), nullable=False),
        sa.Column("client_price", sa.Numeric(10, 2), nullable=False),
        sa.Column("partner_cost", sa.Numeric(10, 2), nullable=True),
        sa.Column(
            "platform_fee", sa.Numeric(10, 2), nullable=False, server_default="0"
        ),
        sa.Column("currency", sa.String(3), nullable=False, server_default="CHF"),
        sa.Column("vat_rate", sa.Numeric(5, 2), nullable=False, server_default="0"),
        sa.Column("vat_included", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "status",
            postgresql.ENUM(
                "PENDING",
                "ACCEPTED",
                "REJECTED",
                "COMPLETED",
                "CANCELLED",
                name="transferstatus",
                create_type=False,  # Ne pas créer le type, il existe déjà
            ),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column(
            "requested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rejected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_validated", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("validated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("validated_by", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["partnership_id"], ["partnerships.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["owner_company_id"], ["company.id"]),
        sa.ForeignKeyConstraint(["executing_company_id"], ["company.id"]),
        sa.ForeignKeyConstraint(["validated_by"], ["user.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    # Créer les index pour booking_transfers
    op.create_index("ix_booking_transfers_booking", "booking_transfers", ["booking_id"])
    op.create_index(
        "ix_booking_transfers_partnership", "booking_transfers", ["partnership_id"]
    )
    op.create_index(
        "ix_booking_transfers_owner_company", "booking_transfers", ["owner_company_id"]
    )
    op.create_index(
        "ix_booking_transfers_executing_company",
        "booking_transfers",
        ["executing_company_id"],
    )
    op.create_index("ix_booking_transfers_status", "booking_transfers", ["status"])

    # Ajouter la colonne executing_company_id à bookings
    op.add_column(
        "booking",
        sa.Column(
            "executing_company_id",
            sa.Integer(),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "fk_bookings_executing_company",
        "booking",
        "company",
        ["executing_company_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_booking_executing_company",
        "booking",
        ["executing_company_id"],
    )


def downgrade():
    # Supprimer la colonne executing_company_id de bookings
    op.drop_index("ix_booking_executing_company", table_name="booking")
    op.drop_constraint("fk_bookings_executing_company", "booking", type_="foreignkey")
    op.drop_column("booking", "executing_company_id")

    # Supprimer la table booking_transfers
    op.drop_index("ix_booking_transfers_status", table_name="booking_transfers")
    op.drop_index(
        "ix_booking_transfers_executing_company", table_name="booking_transfers"
    )
    op.drop_index("ix_booking_transfers_owner_company", table_name="booking_transfers")
    op.drop_index("ix_booking_transfers_partnership", table_name="booking_transfers")
    op.drop_index("ix_booking_transfers_booking", table_name="booking_transfers")
    op.drop_table("booking_transfers")

    # Supprimer la table partnerships
    op.drop_index("ix_partnerships_partner_company", table_name="partnerships")
    op.drop_index("ix_partnerships_owner_company", table_name="partnerships")
    op.drop_table("partnerships")

    # Supprimer les types d'enum
    op.execute("DROP TYPE IF EXISTS transferstatus")
    op.execute("DROP TYPE IF EXISTS transfermodel")
