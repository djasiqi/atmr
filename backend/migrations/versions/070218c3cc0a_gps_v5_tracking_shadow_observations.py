"""gps_v5_tracking_shadow_observations

Revision ID: 070218c3cc0a
Revises: 5620ba1e6460
Create Date: 2026-07-27 02:57:46.741803

Table non autoritaire pour le comparateur shadow Phase 2.
(Autogenerate Docker puis trim du drift hors périmètre.)
"""

from alembic import op
import sqlalchemy as sa

revision = "070218c3cc0a"
down_revision = "5620ba1e6460"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tracking_shadow_observations",
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("location_event_id", sa.String(length=64), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=True),
        sa.Column(
            "fingerprint_schema_version",
            sa.Integer(),
            server_default="1",
            nullable=False,
        ),
        sa.Column("direct_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("direct_accept_status", sa.String(length=64), nullable=True),
        sa.Column("direct_accept_reason", sa.String(length=128), nullable=True),
        sa.Column("direct_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("shadow_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("shadow_accept_status", sa.String(length=64), nullable=True),
        sa.Column("shadow_accept_reason", sa.String(length=128), nullable=True),
        sa.Column("shadow_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("comparison_deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "comparison_state",
            sa.String(length=32),
            server_default="waiting_shadow",
            nullable=False,
        ),
        sa.Column("result", sa.String(length=64), nullable=True),
        sa.Column("compared_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("driver_id", "location_event_id"),
    )
    op.create_index(
        "ix_tracking_shadow_obs_deadline",
        "tracking_shadow_observations",
        ["comparison_deadline_at"],
        unique=False,
    )
    op.create_index(
        "ix_tracking_shadow_obs_expires",
        "tracking_shadow_observations",
        ["expires_at"],
        unique=False,
    )
    op.create_index(
        "ix_tracking_shadow_obs_state",
        "tracking_shadow_observations",
        ["comparison_state"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_tracking_shadow_obs_state", table_name="tracking_shadow_observations"
    )
    op.drop_index(
        "ix_tracking_shadow_obs_expires", table_name="tracking_shadow_observations"
    )
    op.drop_index(
        "ix_tracking_shadow_obs_deadline", table_name="tracking_shadow_observations"
    )
    op.drop_table("tracking_shadow_observations")
