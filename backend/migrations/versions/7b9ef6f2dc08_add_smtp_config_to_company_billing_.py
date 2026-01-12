"""add_smtp_config_to_company_billing_settings

Revision ID: 7b9ef6f2dc08
Revises: 7c5a9ec81135
Create Date: 2026-01-09 14:59:41.011846

"""

from alembic import op
import sqlalchemy as sa


revision = "7b9ef6f2dc08"
down_revision = "7c5a9ec81135"
branch_labels = None
depends_on = None


def upgrade():
    # Ajouter les colonnes de configuration SMTP par entreprise
    op.add_column(
        "company_billing_settings",
        sa.Column("smtp_server", sa.String(length=200), nullable=True),
    )
    op.add_column(
        "company_billing_settings",
        sa.Column("smtp_port", sa.Integer(), nullable=True, server_default="587"),
    )
    op.add_column(
        "company_billing_settings",
        sa.Column("smtp_use_tls", sa.Boolean(), nullable=False, server_default="true"),
    )
    op.add_column(
        "company_billing_settings",
        sa.Column("smtp_use_ssl", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "company_billing_settings",
        sa.Column("smtp_username", sa.String(length=200), nullable=True),
    )
    op.add_column(
        "company_billing_settings",
        sa.Column("smtp_password", sa.String(length=200), nullable=True),
    )  # Stocke la valeur chiffrée
    op.add_column(
        "company_billing_settings",
        sa.Column("smtp_enabled", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade():
    # Supprimer les colonnes SMTP
    op.drop_column("company_billing_settings", "smtp_enabled")
    op.drop_column("company_billing_settings", "smtp_password")
    op.drop_column("company_billing_settings", "smtp_username")
    op.drop_column("company_billing_settings", "smtp_use_ssl")
    op.drop_column("company_billing_settings", "smtp_use_tls")
    op.drop_column("company_billing_settings", "smtp_port")
    op.drop_column("company_billing_settings", "smtp_server")
