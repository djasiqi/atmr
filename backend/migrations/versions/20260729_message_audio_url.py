"""Ajoute message.audio_url pour les messages vocaux hub.

Revision ID: 20260729_msg_audio
Revises: 070218c3cc0a
Create Date: 2026-07-29

Colonne additive pour persister les vocaux canal équipe / hub
(upload + REST / socket). À re-valider via autogenerate quand Docker est dispo.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260729_msg_audio"
down_revision = "070218c3cc0a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("message", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("audio_url", sa.String(length=500), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("message", schema=None) as batch_op:
        batch_op.drop_column("audio_url")
