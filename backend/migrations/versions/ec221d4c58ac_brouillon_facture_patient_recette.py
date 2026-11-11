"""brouillon facture patient recette

Revision ID: ec221d4c58ac
Revises: 8ff19373999b
Create Date: 2026-10-27 21:16:01.324463

La dérive d'index non liée, détectée par l'autogénération, a été retirée.
Les contraintes de revue du représentant, absentes de la détection, sont
conservées car elles font partie du modèle.
"""

from alembic import op
import sqlalchemy as sa


revision = "ec221d4c58ac"
down_revision = "8ff19373999b"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("patient_billing_dossier", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "recipe_contract_fixture",
                sa.Boolean(),
                server_default=sa.text("false"),
                nullable=False,
            )
        )
        batch_op.add_column(sa.Column("draft_invoice_id", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("representative_name", sa.String(length=160), nullable=True)
        )
        batch_op.add_column(
            sa.Column("representative_kind", sa.String(length=40), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "representative_proof_expected", sa.String(length=160), nullable=True
            )
        )
        batch_op.add_column(
            sa.Column(
                "representative_powers_verified",
                sa.Boolean(),
                server_default=sa.text("false"),
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column(
                "representative_decision",
                sa.String(length=32),
                server_default=sa.text("'absent'"),
                nullable=False,
            )
        )
        batch_op.create_foreign_key(
            "fk_patient_billing_dossier_draft_invoice_id",
            "invoices",
            ["draft_invoice_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_check_constraint(
            "ck_patient_billing_powers",
            "representative_powers_verified = false",
        )
        batch_op.create_check_constraint(
            "ck_patient_billing_representative",
            "representative_decision IN ('absent', 'pending_review')",
        )


def downgrade():
    with op.batch_alter_table("patient_billing_dossier", schema=None) as batch_op:
        batch_op.drop_constraint("ck_patient_billing_representative", type_="check")
        batch_op.drop_constraint("ck_patient_billing_powers", type_="check")
        batch_op.drop_constraint(
            "fk_patient_billing_dossier_draft_invoice_id", type_="foreignkey"
        )
        batch_op.drop_column("representative_decision")
        batch_op.drop_column("representative_powers_verified")
        batch_op.drop_column("representative_proof_expected")
        batch_op.drop_column("representative_kind")
        batch_op.drop_column("representative_name")
        batch_op.drop_column("draft_invoice_id")
        batch_op.drop_column("recipe_contract_fixture")
