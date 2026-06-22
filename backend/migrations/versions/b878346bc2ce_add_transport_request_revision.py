"""add_transport_request_revision

Revision ID: b878346bc2ce
Revises: 20260619_dt_prov_uq
Create Date: 2026-06-22

Ajoute transport_requests.revision (version métier pour dedupe request_updated).
Rollback GATE 10 : alembic downgrade -1 ou DROP COLUMN revision.
"""

from alembic import op
import sqlalchemy as sa

revision = "b878346bc2ce"
down_revision = "20260619_dt_prov_uq"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "transport_requests",
        sa.Column(
            "revision",
            sa.Integer(),
            server_default="1",
            nullable=False,
            comment="Version métier incrémentée à chaque modification institution",
        ),
    )


def downgrade():
    op.drop_column("transport_requests", "revision")
