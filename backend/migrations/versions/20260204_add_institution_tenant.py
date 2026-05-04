"""add institution tenant support

Revision ID: 20260204_institution
Revises: 20260203_spc
Create Date: 2026-02-04

Ajoute le support multi-tenant institutionnel:
- Table institutions (cliniques, EMS, IMAD, hôpitaux)
- Colonnes user.institution_id et user.institution_role
- Index pour optimiser les recherches
"""

import sqlalchemy as sa
from alembic import op

revision = "20260204_institution"
down_revision = "20260203_spc"
branch_labels = None
depends_on = None


def upgrade():
    # 1. Créer la table institutions
    op.create_table(
        "institutions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("institution_type", sa.String(length=50), nullable=True),
        sa.Column("address", sa.String(length=255), nullable=True),
        sa.Column("contact_email", sa.String(length=255), nullable=True),
        sa.Column("contact_phone", sa.String(length=50), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
    )
    op.create_index(
        op.f("ix_institutions_public_id"), "institutions", ["public_id"], unique=True
    )

    # 2. Ajouter les colonnes institution_id et institution_role à la table user
    with op.batch_alter_table("user", schema=None) as batch_op:
        batch_op.add_column(sa.Column("institution_id", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("institution_role", sa.String(length=50), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_user_institution_id",
            "institutions",
            ["institution_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "idx_user_institution_id", ["institution_id"], unique=False
        )

    # 3. Ajouter institution_id à la table audit_logs (si elle existe)
    # Vérifier si la table existe avant de la modifier
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "audit_logs" in inspector.get_table_names():
        with op.batch_alter_table("audit_logs", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("institution_id", sa.Integer(), nullable=True)
            )
            batch_op.create_index(
                "ix_audit_logs_institution_id", ["institution_id"], unique=False
            )


def downgrade():
    # 1. Supprimer la colonne institution_id de audit_logs (si elle existe)
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "audit_logs" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("audit_logs")]
        if "institution_id" in columns:
            with op.batch_alter_table("audit_logs", schema=None) as batch_op:
                batch_op.drop_index("ix_audit_logs_institution_id")
                batch_op.drop_column("institution_id")

    # 2. Supprimer les colonnes de la table user
    with op.batch_alter_table("user", schema=None) as batch_op:
        batch_op.drop_index("idx_user_institution_id")
        batch_op.drop_constraint("fk_user_institution_id", type_="foreignkey")
        batch_op.drop_column("institution_role")
        batch_op.drop_column("institution_id")

    # 3. Supprimer la table institutions
    op.drop_index(op.f("ix_institutions_public_id"), table_name="institutions")
    op.drop_table("institutions")
