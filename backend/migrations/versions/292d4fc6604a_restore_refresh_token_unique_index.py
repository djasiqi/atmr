"""restore refresh_token unique index

Revision ID: 292d4fc6604a
Revises: 49ff432aafec
Create Date: 2026-01-24 21:37:06.197842

Idempotent: drop_index/create_index uniquement si l'index existe / n'existe pas,
pour éviter "index ix_refresh_token_token_hash does not exist" sur DROP.
Intention: restore = unique=True en upgrade, unique=False en downgrade.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


revision = "292d4fc6604a"
down_revision = "49ff432aafec"
branch_labels = None
depends_on = None

IDX_NAME = "ix_refresh_token_token_hash"
TABLE_NAME = "refresh_token"


def _index_exists(conn, index_name: str, schema: str = "public") -> bool:
    """Vrai si l'index existe (PostgreSQL pg_indexes)."""
    q = sa.text(
        "SELECT 1 FROM pg_indexes "
        "WHERE schemaname = :schema AND indexname = :index_name LIMIT 1"
    )
    return conn.execute(q, {"schema": schema, "index_name": index_name}).first() is not None


def upgrade():
    conn = op.get_bind()
    if _index_exists(conn, IDX_NAME):
        op.drop_index(IDX_NAME, table_name=TABLE_NAME)
    if not _index_exists(conn, IDX_NAME):
        op.create_index(
            IDX_NAME, TABLE_NAME, ["token_hash"], unique=True
        )


def downgrade():
    conn = op.get_bind()
    if _index_exists(conn, IDX_NAME):
        op.drop_index(IDX_NAME, table_name=TABLE_NAME)
    if not _index_exists(conn, IDX_NAME):
        op.create_index(
            IDX_NAME, TABLE_NAME, ["token_hash"], unique=False
        )

