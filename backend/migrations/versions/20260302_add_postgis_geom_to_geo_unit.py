"""add postgis geom to geo_unit

Revision ID: 20260302_geo_unit_geom
Revises: 20260226_platform_zone_sets
Create Date: 2026-03-02 09:00:00.000000
"""

from alembic import op
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = "20260302_geo_unit_geom"
down_revision = "20260226_platform_zone_sets"
branch_labels = None
depends_on = None


def _postgis_packaged_on_server(bind) -> bool:
    """True si le serveur expose l'extension postgis (fichier .control présent).

    Évite complètement CREATE EXTENSION sur Postgres CI vanilla : pas d'exception,
    pas de dépendance au wrapping SQLAlchemy/psycopg2.
    """
    try:
        row = bind.execute(
            text(
                "SELECT EXISTS ("
                "SELECT 1 FROM pg_available_extensions WHERE name = 'postgis'"
                ")"
            )
        ).scalar()
        return bool(row)
    except Exception:
        return False


def _postgis_extend_failed(exc: BaseException) -> bool:
    """CREATE EXTENSION postgis peut échouer sans fichier .control (CI postgres vanilla)."""
    parts: list[str] = [str(exc)]
    if getattr(exc, "__cause__", None) is not None:
        parts.append(str(exc.__cause__))
    if getattr(exc, "orig", None) is not None:
        parts.append(str(exc.orig))
    raw = " ".join(parts).lower()
    return (
        "postgis" in raw
        or "extension control file" in raw
        or "featurenotsupported" in raw.replace(" ", "")
        or ("extension" in raw and "not available" in raw)
    )


def upgrade():
    bind = op.get_bind()

    # Index portail zone : sans PostGIS (GitHub Actions, etc.).
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_platform_zone_membership_zone_set_zone "
        "ON platform_zone_membership (zone_set_id, zone_id)"
    )

    if bind.dialect.name != "postgresql":
        return

    if not _postgis_packaged_on_server(bind):
        # Postgres sans paquet postgis (ex. postgres:alpine en CI) : pg_available_extensions
        # ne liste pas 'postgis', pas besoin d'appeler CREATE EXTENSION.
        return

    try:
        op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    except Exception as exc:
        # CI / Postgres vanilla : psycopg2 peut lever FeatureNotSupported →
        # sqlalchemy.exc.NotSupportedError (pas toujours classé comme DatabaseError selon flux).
        if _postgis_extend_failed(exc):
            return
        raise

    op.execute(
        "ALTER TABLE geo_unit ADD COLUMN IF NOT EXISTS geom geometry(MultiPolygon, 4326)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_geo_unit_geom ON geo_unit USING GIST (geom)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_platform_zone_membership_zone_set_zone")
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("DROP INDEX IF EXISTS ix_geo_unit_geom")
    op.execute("ALTER TABLE geo_unit DROP COLUMN IF EXISTS geom")
