"""client_terms_acceptance_registry

Revision ID: 3220451c51ae
Revises: 37b80780a96d
Create Date: 2026-09-23 17:39:53.403715

Tables générées par ``flask db revision --autogenerate``.
Les écarts d'index sans rapport avec ce registre ont été retirés.
Les triggers d'immutabilité suivent le modèle déjà utilisé pour audit_logs.
Aucune acceptation historique n'est insérée.
"""

from alembic import op
import sqlalchemy as sa


revision = "3220451c51ae"
down_revision = "37b80780a96d"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "legal_document_version",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_type", sa.String(length=32), nullable=False),
        sa.Column("terms_version", sa.String(length=32), nullable=False),
        sa.Column("terms_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "locale",
            sa.String(length=16),
            server_default="fr-CH",
            nullable=False,
        ),
        sa.Column("canonical_body", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "document_type IN ('terms_of_service', 'transport_terms')",
            name="ck_legal_document_version_type",
        ),
        sa.CheckConstraint(
            "char_length(terms_hash) = 64",
            name="ck_legal_document_version_hash_len",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_type",
            "terms_version",
            name="uq_legal_document_version_identity",
        ),
    )
    op.create_table(
        "client_terms_acceptance",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("document_version_id", sa.Integer(), nullable=False),
        sa.Column("document_type", sa.String(length=32), nullable=False),
        sa.Column("terms_version", sa.String(length=32), nullable=False),
        sa.Column("terms_hash", sa.String(length=64), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("email_snapshot", sa.String(length=255), nullable=True),
        sa.Column("email_verified_at_snapshot", sa.DateTime(timezone=True), nullable=True),
        sa.Column("phone_snapshot", sa.String(length=32), nullable=True),
        sa.Column(
            "phone_verified_at_snapshot",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column("verification_method", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "document_type IN ('terms_of_service', 'transport_terms')",
            name="ck_client_terms_acceptance_type",
        ),
        sa.CheckConstraint(
            "verification_method IN ('otp_sms', 'not_verified')",
            name="ck_client_terms_acceptance_verification",
        ),
        sa.CheckConstraint(
            "char_length(terms_hash) = 64",
            name="ck_client_terms_acceptance_hash_len",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["legal_document_version.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_client_terms_acceptance_user_document_accepted",
        "client_terms_acceptance",
        ["user_id", "document_type", "accepted_at"],
        unique=False,
    )
    op.create_index(
        "ix_client_terms_acceptance_user_id",
        "client_terms_acceptance",
        ["user_id"],
        unique=False,
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION legal_acceptance_prevent_modification()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION
                'Modification of % is not allowed. Terms records are append-only.',
                TG_TABLE_NAME;
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER legal_document_version_no_update
        BEFORE UPDATE ON legal_document_version
        FOR EACH ROW
        EXECUTE FUNCTION legal_acceptance_prevent_modification();
        """
    )
    op.execute(
        """
        CREATE TRIGGER legal_document_version_no_delete
        BEFORE DELETE ON legal_document_version
        FOR EACH ROW
        EXECUTE FUNCTION legal_acceptance_prevent_modification();
        """
    )
    op.execute(
        """
        CREATE TRIGGER client_terms_acceptance_no_update
        BEFORE UPDATE ON client_terms_acceptance
        FOR EACH ROW
        EXECUTE FUNCTION legal_acceptance_prevent_modification();
        """
    )
    op.execute(
        """
        CREATE TRIGGER client_terms_acceptance_no_delete
        BEFORE DELETE ON client_terms_acceptance
        FOR EACH ROW
        EXECUTE FUNCTION legal_acceptance_prevent_modification();
        """
    )


def downgrade():
    op.execute(
        "DROP TRIGGER IF EXISTS client_terms_acceptance_no_delete ON client_terms_acceptance;"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS client_terms_acceptance_no_update ON client_terms_acceptance;"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS legal_document_version_no_delete ON legal_document_version;"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS legal_document_version_no_update ON legal_document_version;"
    )
    op.execute("DROP FUNCTION IF EXISTS legal_acceptance_prevent_modification();")
    op.drop_index(
        "ix_client_terms_acceptance_user_id",
        table_name="client_terms_acceptance",
    )
    op.drop_index(
        "ix_client_terms_acceptance_user_document_accepted",
        table_name="client_terms_acceptance",
    )
    op.drop_table("client_terms_acceptance")
    op.drop_table("legal_document_version")
