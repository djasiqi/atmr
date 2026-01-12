"""Add CompanyBillingProfile table for centralized billing configuration

Revision ID: 7c5a9ec81135
Revises: 94d80b1d14f0
Create Date: 2026-01-09 12:07:04.008426

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "7c5a9ec81135"
down_revision = "94d80b1d14f0"
branch_labels = None
depends_on = None


def upgrade():
    # Créer la nouvelle table company_billing_profile
    op.create_table(
        "company_billing_profile",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column(
            "legal_name",
            sa.String(length=200),
            nullable=False,
            comment="Nom légal de l'entreprise pour factures",
        ),
        sa.Column(
            "brand_name",
            sa.String(length=200),
            nullable=True,
            comment="Nom commercial (si différent du nom légal)",
        ),
        sa.Column(
            "uid_ide",
            sa.String(length=20),
            nullable=False,
            comment="Numéro IDE/UID suisse (format: CHE-XXX.XXX.XXX)",
        ),
        sa.Column(
            "street_name",
            sa.String(length=70),
            nullable=False,
            comment="Nom de rue (sans numéro)",
        ),
        sa.Column(
            "building_number",
            sa.String(length=16),
            nullable=False,
            comment="Numéro de bâtiment (peut contenir lettres: 12A)",
        ),
        sa.Column(
            "postal_code",
            sa.String(length=16),
            nullable=False,
            comment="Code postal (4 chiffres pour Suisse)",
        ),
        sa.Column(
            "city",
            sa.String(length=35),
            nullable=False,
            comment="Ville",
        ),
        sa.Column(
            "country_code",
            sa.String(length=2),
            nullable=False,
            comment="Code pays ISO 3166-1 alpha-2",
        ),
        sa.Column(
            "billing_email",
            sa.String(length=100),
            nullable=False,
            comment="Email pour envoi factures",
        ),
        sa.Column(
            "billing_phone",
            sa.String(length=20),
            nullable=False,
            comment="Téléphone facturation (format international recommandé)",
        ),
        sa.Column(
            "vat_registered",
            sa.Boolean(),
            nullable=False,
            comment="Entreprise assujettie à la TVA",
        ),
        sa.Column(
            "vat_number",
            sa.String(length=50),
            nullable=True,
            comment="Numéro TVA (si assujetti)",
        ),
        sa.Column(
            "vat_rate",
            sa.Numeric(precision=5, scale=2),
            nullable=True,
            comment="Taux TVA par défaut (ex: 7.7 pour 7.7%)",
        ),
        sa.Column(
            "iban",
            sa.String(length=200),
            nullable=False,
            comment="IBAN chiffré (format CHxx xxxx xxxx xxxx xxxx x)",
        ),
        sa.Column(
            "qr_iban",
            sa.String(length=200),
            nullable=True,
            comment="QR-IBAN chiffré (uniquement si références QRR)",
        ),
        sa.Column(
            "payment_reference_mode",
            sa.String(length=10),
            nullable=False,
            comment="Mode référence paiement: NONE, SCOR (ISO 11649), QRR (ESR)",
        ),
        sa.Column(
            "creditor_reference_base",
            sa.String(length=20),
            nullable=True,
            comment="Base pour générer références QRR (si mode=QRR)",
        ),
        sa.Column(
            "payment_terms_days",
            sa.Integer(),
            nullable=False,
            comment="Délai de paiement en jours",
        ),
        sa.Column(
            "overdue_fee",
            sa.Numeric(precision=10, scale=2),
            nullable=False,
            comment="Frais de retard (CHF)",
        ),
        sa.Column(
            "legal_footer",
            sa.Text(),
            nullable=True,
            comment="Texte légal pied de page facture",
        ),
        sa.Column(
            "is_address_validated",
            sa.Boolean(),
            nullable=False,
            comment="Adresse validée (structure + existence)",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["company.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("company_id"),
    )

    op.create_index(
        op.f("ix_company_billing_profile_company_id"),
        "company_billing_profile",
        ["company_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_company_billing_profile_uid_ide"),
        "company_billing_profile",
        ["uid_ide"],
        unique=False,
    )


def downgrade():
    # Supprimer la table company_billing_profile
    op.drop_index(
        op.f("ix_company_billing_profile_uid_ide"),
        table_name="company_billing_profile",
    )
    op.drop_index(
        op.f("ix_company_billing_profile_company_id"),
        table_name="company_billing_profile",
    )
    op.drop_table("company_billing_profile")
