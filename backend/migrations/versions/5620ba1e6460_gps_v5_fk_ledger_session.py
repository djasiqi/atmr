"""gps_v5_fk_ledger_session — FK ledger→sessions + enrichments→ledger.

Revision ID: 5620ba1e6460
Revises: 20260727_gps_v5
Create Date: 2026-07-27

Générée via ``alembic revision --autogenerate`` puis réduite aux seules
contraintes FK du plan Annexe A.2 / A.7 (bruit hors scope retiré).
"""

from __future__ import annotations

from alembic import op

revision = "5620ba1e6460"
down_revision = "20260727_gps_v5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # NULL tracking_session_id autorisé (lignes legacy) — FK PostgreSQL ignore NULL
    op.create_foreign_key(
        "fk_tracking_ingest_session",
        "tracking_ingest_events",
        "tracking_sessions",
        ["driver_id", "tracking_session_id"],
        ["driver_id", "tracking_session_id"],
    )
    op.create_foreign_key(
        "fk_dle_enrichment_ledger",
        "driver_location_enrichments",
        "tracking_ingest_events",
        ["driver_id", "location_event_id"],
        ["driver_id", "location_event_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_dle_enrichment_ledger",
        "driver_location_enrichments",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_tracking_ingest_session",
        "tracking_ingest_events",
        type_="foreignkey",
    )
