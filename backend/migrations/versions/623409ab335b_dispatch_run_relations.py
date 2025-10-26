"""dispatch_run relations

Revision ID: 623409ab335b
Revises: 19fc92c2ba30
Create Date: 2025-10-07 10:34:08.865530
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "623409ab335b"
down_revision = "19fc92c2ba30"
branch_labels = None
depends_on = None


def upgrade():
    # --- assignment.dispatch_run_id : nullable + FK SET NULL ---
    with op.batch_alter_table("assignment", schema=None) as batch_op:
        batch_op.alter_column("dispatch_run_id",
                              existing_type=sa.INTEGER(),
                              nullable=True)
        batch_op.drop_constraint("assignment_dispatch_run_id_fkey", type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_assignment_dispatch_run_id",
            "dispatch_run",
            ["dispatch_run_id"],
            ["id"],
            ondelete="SET NULL"
        )

    # --- dispatch_run.status : drop ancien CHECK (lowercase), uppercasing des données, passage à Enum UPPER ---
    bind = op.get_bind()
    insp = sa.inspect(bind)
    ck_names = {c["name"] for c in insp.get_check_constraints("dispatch_run")}
    if "ck_dispatch_run_status" in ck_names:
        op.drop_constraint("ck_dispatch_run_status", "dispatch_run", type_="check")

    # Normalise les données en UPPER avant de poser la nouvelle contrainte Enum
    op.execute("UPDATE dispatch_run SET status = UPPER(status) WHERE status IS NOT NULL;")

    with op.batch_alter_table("dispatch_run", schema=None) as batch_op:
        batch_op.alter_column(
            "status",
            existing_type=sa.VARCHAR(length=20),
            type_=sa.Enum("PENDING", "RUNNING", "COMPLETED", "FAILED",
                          name="dispatch_status", native_enum=False),
            existing_nullable=False
        )
        batch_op.create_index(
            "ix_dispatch_run_company_status_day",
            ["company_id", "status", "day"],
            unique=False
        )


def downgrade():
    # --- dispatch_run.status : drop index, repasser en VARCHAR, remettre les valeurs en lower, recréer l'ancien CHECK ---
    with op.batch_alter_table("dispatch_run", schema=None) as batch_op:
        batch_op.drop_index("ix_dispatch_run_company_status_day")
        batch_op.alter_column(
            "status",
            existing_type=sa.Enum("PENDING", "RUNNING", "COMPLETED", "FAILED",
                                  name="dispatch_status", native_enum=False),
            type_=sa.VARCHAR(length=20),
            existing_nullable=False
        )

    # Valeurs de nouveau en lowercase pour satisfaire le vieux CHECK
    op.execute("UPDATE dispatch_run SET status = LOWER(status) WHERE status IS NOT NULL;")

    # Remet l'ancien CHECK (lowercase)
    op.create_check_constraint(
        "ck_dispatch_run_status",
        "dispatch_run",
        "status IN ('pending','running','completed','failed')"
    )

    # --- assignment.dispatch_run_id : NOT NULL + FK CASCADE (état antérieur) ---
    with op.batch_alter_table("assignment", schema=None) as batch_op:
        batch_op.drop_constraint("fk_assignment_dispatch_run_id", type_="foreignkey")
        batch_op.create_foreign_key(
            "assignment_dispatch_run_id_fkey",
            "dispatch_run",
            ["dispatch_run_id"],
            ["id"],
            ondelete="CASCADE"
        )
        batch_op.alter_column("dispatch_run_id",
                              existing_type=sa.INTEGER(),
                              nullable=False)
