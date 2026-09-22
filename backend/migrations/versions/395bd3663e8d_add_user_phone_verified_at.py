"""add_user_phone_verified_at

Revision ID: 395bd3663e8d
Revises: 7575a80bda48
Create Date: 2026-09-22 13:36:15.838798

Alembic a aussi détecté du drift d'index hors scope : seul
``user.phone_verified_at`` est conservé (AUTH-SMS-02).
"""

from alembic import op
import sqlalchemy as sa

from services.auth.portal_account_promotion_sql import (
    COUNT_PORTAL_PROMOTION_CLIENTS_SQL,
    COUNT_PORTAL_PROMOTION_USERS_SQL,
    PROMOTE_PORTAL_CLIENTS_SQL,
    PROMOTE_PORTAL_USERS_SQL,
)


revision = "395bd3663e8d"
down_revision = "7575a80bda48"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("user", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("phone_verified_at", sa.DateTime(timezone=True), nullable=True)
        )

    conn = op.get_bind()
    users_before = conn.execute(sa.text(COUNT_PORTAL_PROMOTION_USERS_SQL)).scalar()
    clients_before = conn.execute(sa.text(COUNT_PORTAL_PROMOTION_CLIENTS_SQL)).scalar()
    print(
        f"[AUTH-SMS-02] promotion avant: users={users_before} clients={clients_before}"
    )

    # Compatibilité : e-mail déjà confirmé → compte PORTAL activable, téléphone inchangé.
    op.execute(sa.text(PROMOTE_PORTAL_USERS_SQL))
    op.execute(sa.text(PROMOTE_PORTAL_CLIENTS_SQL))

    users_after = conn.execute(sa.text(COUNT_PORTAL_PROMOTION_USERS_SQL)).scalar()
    print(f"[AUTH-SMS-02] promotion après: users_restants={users_after}")


def downgrade():
    with op.batch_alter_table("user", schema=None) as batch_op:
        batch_op.drop_column("phone_verified_at")
