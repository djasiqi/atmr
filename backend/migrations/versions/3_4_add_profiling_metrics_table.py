"""✅ 3.4: Add profiling_metrics table for automatic CPU/memory profiling.

Revision ID: 3_4_profiling
Revises: d7e8f9a1b2c3
Create Date: 2025-10-29 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "3_4_profiling"
down_revision = "d2_encrypted_fields"  # Point vers dernière migration stable
branch_labels = None
depends_on = None


def upgrade():
    # ✅ 3.4: Table pour stocker les métriques de profiling automatique
    op.create_table(
        "profiling_metrics",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("profiling_date", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("top_functions", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("system_metrics_before", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("system_metrics_after", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("total_stats", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("report_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )

    # Index pour performance
    op.create_index("ix_profiling_metrics_profiling_date", "profiling_metrics", ["profiling_date"], unique=False)


def downgrade():
    op.drop_index("ix_profiling_metrics_profiling_date", table_name="profiling_metrics")
    op.drop_table("profiling_metrics")
