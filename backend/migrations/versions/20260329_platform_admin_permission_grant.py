"""Table platform_admin_permission_grant (RBAC plateforme V1.1).

Revision ID: 20260329_plat_admin_perm
Revises: 20260329_plat_gov_tbls
Create Date: 2026-03-29

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

revision = "20260329_plat_admin_perm"
down_revision = "20260329_plat_gov_tbls"
branch_labels = None
depends_on = None

PERMS = (
    "observe.tenant.read",
    "governance.tenant.suspend",
    "policy.explain",
    "operate.runbooks.execute",
)


def upgrade() -> None:
    op.create_table(
        "platform_admin_permission_grant",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("permission", sa.String(length=128), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
            name=op.f("fk_platform_admin_perm_user"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_platform_admin_permission_grant")),
        sa.UniqueConstraint(
            "user_id",
            "permission",
            name="uq_platform_admin_perm_user_perm",
        ),
    )
    op.create_index(
        op.f("ix_platform_admin_permission_grant_user_id"),
        "platform_admin_permission_grant",
        ["user_id"],
        unique=False,
    )

    bind = op.get_bind()
    rows = bind.execute(
        text("SELECT id FROM \"user\" WHERE role::text = 'ADMIN'")
    ).fetchall()
    for (uid,) in rows:
        for perm in PERMS:
            bind.execute(
                text(
                    """
                    INSERT INTO platform_admin_permission_grant (user_id, permission)
                    VALUES (:uid, :p)
                    ON CONFLICT (user_id, permission) DO NOTHING
                    """
                ),
                {"uid": uid, "p": perm},
            )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_platform_admin_permission_grant_user_id"),
        table_name="platform_admin_permission_grant",
    )
    op.drop_table("platform_admin_permission_grant")
