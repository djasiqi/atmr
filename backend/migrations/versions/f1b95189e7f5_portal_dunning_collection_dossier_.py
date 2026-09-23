"""portal dunning collection dossier snapshot

Revision ID: f1b95189e7f5
Revises: 4dff22f12b4f
Create Date: 2026-09-23 23:48:17.463810

Colonnes générées par ``flask db revision --autogenerate``.
Écarts d'index hors périmètre retirés.
``ADD COLUMN`` direct : ``batch_alter_table`` recréerait la table.
"""

import sqlalchemy as sa
from alembic import op

revision = "f1b95189e7f5"
down_revision = "4dff22f12b4f"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "portal_receivable_dunning_event",
        sa.Column("dossier_snapshot", sa.Text(), nullable=True),
    )
    op.add_column(
        "portal_receivable_dunning_event",
        sa.Column("dossier_snapshot_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "portal_receivable_dunning_event",
        sa.Column("formal_notice_event_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_portal_dunning_formal_notice_event",
        "portal_receivable_dunning_event",
        "portal_receivable_dunning_event",
        ["formal_notice_event_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade():
    op.drop_constraint(
        "fk_portal_dunning_formal_notice_event",
        "portal_receivable_dunning_event",
        type_="foreignkey",
    )
    op.drop_column("portal_receivable_dunning_event", "formal_notice_event_id")
    op.drop_column("portal_receivable_dunning_event", "dossier_snapshot_hash")
    op.drop_column("portal_receivable_dunning_event", "dossier_snapshot")
