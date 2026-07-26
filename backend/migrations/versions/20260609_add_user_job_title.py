"""Add job_title (fonction/metier) to user.

Champ descriptif independant du role LIRIE (institution_role).
N'accorde aucune permission ; sert aux exports, audits et statistiques.

Revision ID: 20260609_user_job_title
Revises: 20260608_global_username
"""

from __future__ import annotations

from alembic import op

revision = "20260609_user_job_title"
down_revision = "20260608_global_username"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('ALTER TABLE "user" ADD COLUMN IF NOT EXISTS job_title VARCHAR(120)')


def downgrade() -> None:
    op.execute('ALTER TABLE "user" DROP COLUMN IF EXISTS job_title')
