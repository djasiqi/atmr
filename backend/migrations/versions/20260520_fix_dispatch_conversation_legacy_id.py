"""Corrige legacy_thread_id dispatch dupliqué sur canal privé chauffeur.

Revision ID: 20260520_fix_dispatch_legacy
"""

from __future__ import annotations

from alembic import op

revision = "20260520_fix_dispatch_legacy"
down_revision = "20260520_msg_idem_uq"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Canal privé exploitation↔chauffeur (context_id = driver.id) ne doit pas
    # partager legacy_thread_id='dispatch' avec le canal Dispatch partagé.
    op.execute(
        """
        UPDATE conversation
        SET legacy_thread_id = 'company_driver:' || context_id::text
        WHERE conversation_type = 'COMPANY'
          AND legacy_thread_id = 'dispatch'
          AND context_id IS NOT NULL
          AND context_id != company_id
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE conversation
        SET legacy_thread_id = 'dispatch'
        WHERE legacy_thread_id LIKE 'company_driver:%'
          AND conversation_type = 'COMPANY'
          AND context_id IS NOT NULL
          AND context_id != company_id
        """
    )
