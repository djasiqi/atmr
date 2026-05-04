"""Diagnostic: révision Alembic en base vs tables sensibles (49ff, etc.).

Compare la révision enregistrée (alembic_version) avec l'existence des tables
créées par des migrations plus récentes (password_history, billing_audit_logs, …).
Utile pour détecter des divergences (tables existantes mais migration pas appliquée).

Usage (depuis la racine backend ou via Docker) :
  python scripts/diagnose_alembic_schema.py
  flask run --with-shell  # puis depuis le shell: from scripts.diagnose_alembic_schema import run; run()

Ou avec l'app Flask (connexion via app) :
  python -c "
  import sys; sys.path.insert(0, '.')
  from app import create_app
  from scripts.diagnose_alembic_schema import run_diagnostic
  app = create_app()
  with app.app_context():
      run_diagnostic()
  "
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Pattern repo: permettre imports depuis la racine backend
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Tables créées par 49ff ou des migrations en aval dans la chaîne
TABLES_49FF_OR_DOWNSTREAM = [
    "password_history",
    "billing_audit_logs",
    "company_billing_profile",
    "device_tokens",
    "transport_vouchers",
    "transport_voucher_files",
]

# Colonnes minimales attendues si la table existe (schéma incomplet → INCOMPLET)
EXPECTED_COLUMNS = {
    "billing_audit_logs": ["company_id", "booking_id", "action", "created_at"],
    "password_history": ["user_id", "password_hash", "created_at"],
}


def _get_alembic_heads():
    """Retourne la liste des révisions head (fichiers migrations) ou [] si échec."""
    try:
        import alembic.config
        from alembic.script import ScriptDirectory

        script_dir = _ROOT / "migrations"
        cfg = alembic.config.Config(str(script_dir / "alembic.ini"))
        cfg.set_main_option("script_location", str(script_dir))
        script = ScriptDirectory.from_config(cfg)
        return [h.revision for h in script.get_heads()]
    except Exception:
        return []


def run_diagnostic(bind=None):
    """Affiche révision Alembic + existence des tables sensibles.

    Si bind est None, tente create_app() + db.get_engine().
    """
    from sqlalchemy import text

    if bind is None:
        from app import create_app
        from ext import db

        app = create_app()
        with app.app_context():
            bind = db.get_engine().connect()
            try:
                _do_run(bind)
            finally:
                bind.close()
    else:
        _do_run(bind)


def _column_exists(conn, schema, table, column):
    from sqlalchemy import text

    q = text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = :s AND table_name = :t AND column_name = :c)"
    )
    return conn.execute(q, {"s": schema, "t": table, "c": column}).scalar()


def _do_run(conn):
    from sqlalchemy import text

    schema = "public"

    # 1) Révision en base (alembic_version) et heads (fichiers)
    try:
        r = conn.execute(text("SELECT version_num FROM alembic_version"))
        rows = r.fetchall()
        versions = [row[0] for row in rows] if rows else []
    except Exception as e:
        print("alembic_version: ERREUR", e)
        versions = []

    if len(versions) == 0:
        print("alembic_version (current): (aucune révision)")
    else:
        print("alembic_version (current):", ", ".join(versions))

    heads = _get_alembic_heads()
    if heads:
        print("alembic_heads (fichiers):", ", ".join(heads))
    else:
        print("alembic_heads (fichiers): (indisponible — lancer 'flask db heads')")

    # 2) Existence des tables sensibles + colonnes minimales si pertinent
    print(f"\nTables (schema={schema}):")
    for table in TABLES_49FF_OR_DOWNSTREAM:
        q = text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = :s AND table_name = :t)"
        )
        r = conn.execute(q, {"s": schema, "t": table})
        exists = r.scalar()
        if exists and table in EXPECTED_COLUMNS:
            missing = [
                c
                for c in EXPECTED_COLUMNS[table]
                if not _column_exists(conn, schema, table, c)
            ]
            if missing:
                print(
                    f"  {table}: OUI mais INCOMPLET (colonnes manquantes: {', '.join(missing)})"
                )
            else:
                print(f"  {table}: OUI")
        else:
            print(f"  {table}: {'OUI' if exists else 'non'}")
    print()


if __name__ == "__main__":
    # Connexion directe si DATABASE_URL dispo, sinon via Flask
    url = os.environ.get("DATABASE_URL") or os.environ.get("SQLALCHEMY_DATABASE_URI")
    if url:
        from sqlalchemy import create_engine

        engine = create_engine(url)
        with engine.connect() as conn:
            _do_run(conn)
    else:
        run_diagnostic(bind=None)
