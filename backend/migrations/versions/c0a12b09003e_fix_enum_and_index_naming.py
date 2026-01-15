"""fix_enum_and_index_naming

Revision ID: c0a12b09003e
Revises: 4a4c71c80d0c
Create Date: 2026-01-15 03:39:06.948552

"""

from contextlib import suppress

from alembic import op
import sqlalchemy as sa


revision = "c0a12b09003e"
down_revision = "cd360327d324"  # Après la migration device_tokens
branch_labels = None
depends_on = None


def upgrade():
    """Corrige les écarts entre modèles SQLAlchemy et schéma DB.

    - Migre les types ENUM : transfermodel -> transfer_model, transferstatus -> transfer_status
    - Renomme les index pour suivre la convention _id
    - Corrige les contraintes uniques
    """
    from sqlalchemy.dialects import postgresql

    # ========== 1. Créer les nouveaux types ENUM ==========
    # Vérifier si les types existent déjà avant de les créer
    op.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'transfer_model') THEN
                CREATE TYPE transfer_model AS ENUM (
                    'SUBCONTRACT', 'ASSIGN_TO_PARTNER', 'MARKETPLACE'
                );
            END IF;
        END $$;
    """)

    op.execute("""
        DO $$ BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'transfer_status') THEN
                CREATE TYPE transfer_status AS ENUM (
                    'PENDING', 'ACCEPTED', 'REJECTED', 'COMPLETED', 'CANCELLED'
                );
            END IF;
        END $$;
    """)

    # ========== 2. Migrer booking_transfers.transfer_model ==========
    # Convertir la colonne vers le nouveau type ENUM
    op.execute("""
        ALTER TABLE booking_transfers
        ALTER COLUMN transfer_model TYPE transfer_model
        USING transfer_model::text::transfer_model;
    """)

    # ========== 3. Migrer booking_transfers.status ==========
    # Supprimer le default temporairement, puis changer le type, puis remettre le default
    op.execute("""
        ALTER TABLE booking_transfers
        ALTER COLUMN status DROP DEFAULT;
    """)
    op.execute("""
        ALTER TABLE booking_transfers
        ALTER COLUMN status TYPE transfer_status
        USING status::text::transfer_status;
    """)
    op.execute("""
        ALTER TABLE booking_transfers
        ALTER COLUMN status SET DEFAULT 'PENDING'::transfer_status;
    """)

    # ========== 4. Migrer partnerships.default_transfer_model ==========
    # Supprimer le default temporairement, puis changer le type, puis remettre le default
    op.execute("""
        ALTER TABLE partnerships
        ALTER COLUMN default_transfer_model DROP DEFAULT;
    """)
    op.execute("""
        ALTER TABLE partnerships
        ALTER COLUMN default_transfer_model TYPE transfer_model
        USING default_transfer_model::text::transfer_model;
    """)
    op.execute("""
        ALTER TABLE partnerships
        ALTER COLUMN default_transfer_model SET DEFAULT 'SUBCONTRACT'::transfer_model;
    """)

    # ========== 5. Renommer les index de booking_transfers ==========
    with op.batch_alter_table("booking_transfers", schema=None) as batch_op:
        # Supprimer les anciens index
        with suppress(Exception):  # Index peut ne pas exister
            batch_op.drop_index("ix_booking_transfers_booking")
        with suppress(Exception):
            batch_op.drop_index("ix_booking_transfers_executing_company")
        with suppress(Exception):
            batch_op.drop_index("ix_booking_transfers_owner_company")
        with suppress(Exception):
            batch_op.drop_index("ix_booking_transfers_partnership")
        with suppress(Exception):
            batch_op.drop_index("ix_booking_transfers_status")

        # Créer les nouveaux index avec la convention _id
        batch_op.create_index(
            "ix_booking_transfers_booking_id", ["booking_id"], unique=False
        )
        batch_op.create_index(
            "ix_booking_transfers_executing_company_id",
            ["executing_company_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_booking_transfers_owner_company_id",
            ["owner_company_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_booking_transfers_partnership_id", ["partnership_id"], unique=False
        )

    # ========== 6. Renommer l'index de booking ==========
    # Vérifier si l'index existe avant de le supprimer
    from sqlalchemy import inspect

    bind = op.get_bind()
    inspector = inspect(bind)
    existing_indexes = [idx["name"] for idx in inspector.get_indexes("booking")]

    with op.batch_alter_table("booking", schema=None) as batch_op:
        if "ix_booking_executing_company" in existing_indexes:
            batch_op.drop_index("ix_booking_executing_company")
        # Créer le nouvel index seulement s'il n'existe pas déjà
        if "ix_booking_executing_company_id" not in existing_indexes:
            batch_op.create_index(
                "ix_booking_executing_company_id",
                ["executing_company_id"],
                unique=False,
            )

    # ========== 7. Renommer les index de partnerships ==========
    with op.batch_alter_table("partnerships", schema=None) as batch_op:
        with suppress(Exception):
            batch_op.drop_index("ix_partnerships_owner_company")
        with suppress(Exception):
            batch_op.drop_index("ix_partnerships_partner_company")
        batch_op.create_index(
            "ix_partnerships_owner_company_id", ["owner_company_id"], unique=False
        )
        batch_op.create_index(
            "ix_partnerships_partner_company_id", ["partner_company_id"], unique=False
        )

    # ========== 8. Corriger company_billing_profile (contrainte unique) ==========
    with op.batch_alter_table("company_billing_profile", schema=None) as batch_op:
        with suppress(Exception):
            batch_op.drop_constraint(
                "company_billing_profile_company_id_key", type_="unique"
            )
        with suppress(Exception):
            batch_op.drop_index("ix_company_billing_profile_company_id")
        batch_op.create_index(
            "ix_company_billing_profile_company_id", ["company_id"], unique=True
        )

    # ========== 9. Corriger partner_invoices (index unique) ==========
    with op.batch_alter_table("partner_invoices", schema=None) as batch_op:
        with suppress(Exception):
            batch_op.drop_index("ix_partner_invoices_invoice_number")
        batch_op.create_unique_constraint(None, ["invoice_number"])

    # Note: Les anciens types ENUM (transfermodel, transferstatus) ne sont pas supprimés
    # car ils pourraient être utilisés ailleurs. Ils peuvent être supprimés manuellement
    # après vérification qu'ils ne sont plus utilisés.


def downgrade():
    """Rollback des changements."""
    from sqlalchemy.dialects import postgresql

    # ========== Rollback partner_invoices ==========
    with op.batch_alter_table("partner_invoices", schema=None) as batch_op:
        with suppress(Exception):
            batch_op.drop_constraint(None, type_="unique")
        batch_op.create_index(
            "ix_partner_invoices_invoice_number", ["invoice_number"], unique=True
        )

    # ========== Rollback company_billing_profile ==========
    with op.batch_alter_table("company_billing_profile", schema=None) as batch_op:
        with suppress(Exception):
            batch_op.drop_index("ix_company_billing_profile_company_id")
        batch_op.create_index(
            "ix_company_billing_profile_company_id", ["company_id"], unique=False
        )
        batch_op.create_unique_constraint(
            "company_billing_profile_company_id_key",
            ["company_id"],
            postgresql_nulls_not_distinct=False,
        )

    # ========== Rollback partnerships ==========
    with op.batch_alter_table("partnerships", schema=None) as batch_op:
        with suppress(Exception):
            batch_op.drop_index("ix_partnerships_partner_company_id")
            batch_op.drop_index("ix_partnerships_owner_company_id")
        batch_op.create_index(
            "ix_partnerships_partner_company", ["partner_company_id"], unique=False
        )
        batch_op.create_index(
            "ix_partnerships_owner_company", ["owner_company_id"], unique=False
        )
        op.execute("""
            ALTER TABLE partnerships
            ALTER COLUMN default_transfer_model TYPE transfermodel
            USING default_transfer_model::text::transfermodel;
        """)

    # ========== Rollback booking ==========
    with op.batch_alter_table("booking", schema=None) as batch_op:
        with suppress(Exception):
            batch_op.drop_index("ix_booking_executing_company_id")
        batch_op.create_index(
            "ix_booking_executing_company", ["executing_company_id"], unique=False
        )

    # ========== Rollback booking_transfers ==========
    with op.batch_alter_table("booking_transfers", schema=None) as batch_op:
        with suppress(Exception):
            batch_op.drop_index("ix_booking_transfers_partnership_id")
            batch_op.drop_index("ix_booking_transfers_owner_company_id")
            batch_op.drop_index("ix_booking_transfers_executing_company_id")
            batch_op.drop_index("ix_booking_transfers_booking_id")
        batch_op.create_index("ix_booking_transfers_status", ["status"], unique=False)
        batch_op.create_index(
            "ix_booking_transfers_partnership", ["partnership_id"], unique=False
        )
        batch_op.create_index(
            "ix_booking_transfers_owner_company", ["owner_company_id"], unique=False
        )
        batch_op.create_index(
            "ix_booking_transfers_executing_company",
            ["executing_company_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_booking_transfers_booking", ["booking_id"], unique=False
        )
        op.execute("""
            ALTER TABLE booking_transfers
            ALTER COLUMN status TYPE transferstatus
            USING status::text::transferstatus;
        """)
        op.execute("""
            ALTER TABLE booking_transfers
            ALTER COLUMN transfer_model TYPE transfermodel
            USING transfer_model::text::transfermodel;
        """)

    # Note: Les nouveaux types ENUM (transfer_model, transfer_status) ne sont pas supprimés
    # car ils pourraient être utilisés ailleurs. Ils peuvent être supprimés manuellement
    # après vérification qu'ils ne sont plus utilisés.
