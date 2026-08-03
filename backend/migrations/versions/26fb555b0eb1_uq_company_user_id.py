"""uq_company_user_id — contrainte UNIQUE déterministe (PR1 Partenaires).

Revision ID: 26fb555b0eb1
Revises: e4f273565844
Create Date: 2026-08-03

Échoue explicitement si des collisions ``company.user_id`` existent.
Aucune fusion ni suppression automatique.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "26fb555b0eb1"
down_revision: Union[str, Sequence[str], None] = "e4f273565844"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    rows = conn.execute(
        sa.text(
            """
            SELECT user_id, COUNT(*) AS cnt,
                   ARRAY_AGG(id ORDER BY id) AS company_ids
            FROM company
            GROUP BY user_id
            HAVING COUNT(*) > 1
            """
        )
    ).fetchall()
    if rows:
        details = "; ".join(
            f"user_id={r.user_id} count={r.cnt} company_ids={list(r.company_ids)}"
            for r in rows
        )
        raise RuntimeError(
            "Impossible d'ajouter uq_company_user_id : collisions détectées. "
            f"Résoudre manuellement avant migration. Détails: {details}"
        )

    op.create_unique_constraint("uq_company_user_id", "company", ["user_id"])


def downgrade() -> None:
    op.drop_constraint("uq_company_user_id", "company", type_="unique")
