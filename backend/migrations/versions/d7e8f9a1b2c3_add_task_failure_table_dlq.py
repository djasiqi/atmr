"""A3: Add task_failure table for DLQ monitoring.

Revision ID: d7e8f9a1b2c3
Revises: f3a9c7b8d1e2
Create Date: 2025-01-27 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "d7e8f9a1b2c3"
down_revision = "f3a9c7b8d1e2"  # Point vers dernière migration
branch_labels = None
depends_on = None


def upgrade():
    # ✅ A3: Table pour stocker les tâches Celery échouées (DLQ)
    op.create_table(
        "task_failure",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.String(length=255), nullable=False),
        sa.Column("task_name", sa.String(length=255), nullable=False),
        sa.Column("exception", sa.Text(), nullable=False),
        sa.Column("traceback", sa.Text(), nullable=True),
        sa.Column("args", sa.String(length=2000), nullable=True),
        sa.Column("kwargs", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("first_seen", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("failure_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("worker_name", sa.String(length=255), nullable=True),
        sa.Column("hostname", sa.String(length=255), nullable=True),
        sa.Column("dispatch_run_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Indexes pour performance
    op.create_index("ix_task_failure_task_id", "task_failure", ["task_id"], unique=True)
    op.create_index("ix_task_failure_task_name", "task_failure", ["task_name"])
    op.create_index("ix_task_failure_dispatch_run_id", "task_failure", ["dispatch_run_id"])
    op.create_index("ix_task_failure_first_seen", "task_failure", ["first_seen"])
    op.create_index("ix_task_failure_last_seen", "task_failure", ["last_seen"])


def downgrade():
    # Supprimer les indexes
    op.drop_index("ix_task_failure_last_seen", table_name="task_failure")
    op.drop_index("ix_task_failure_first_seen", table_name="task_failure")
    op.drop_index("ix_task_failure_dispatch_run_id", table_name="task_failure")
    op.drop_index("ix_task_failure_task_name", table_name="task_failure")
    op.drop_index("ix_task_failure_task_id", table_name="task_failure")

    # Supprimer la table
    op.drop_table("task_failure")
