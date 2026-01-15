#!/usr/bin/env python3
"""
Script pour synchroniser automatiquement le schéma de la base de données
avec les modèles SQLAlchemy sans passer par les migrations Alembic.
"""

import sys
import os

# Ajouter le backend au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy import inspect, text
from app import create_app, db


def get_missing_columns():
    """Compare le schéma DB actuel avec les modèles SQLAlchemy"""
    app = create_app("production")

    with app.app_context():
        inspector = inspect(db.engine)
        missing = []

        # Pour chaque modèle SQLAlchemy
        for table_name, table in db.metadata.tables.items():
            # Colonnes dans le modèle
            model_columns = {col.name: col for col in table.columns}

            # Colonnes dans la DB
            try:
                db_columns = {
                    col["name"]: col for col in inspector.get_columns(table_name)
                }
            except Exception:
                # Table n'existe pas
                continue

            # Trouver les colonnes manquantes
            for col_name, col in model_columns.items():
                if col_name not in db_columns:
                    missing.append(
                        {
                            "table": table_name,
                            "column": col_name,
                            "type": str(col.type),
                            "nullable": col.nullable,
                            "default": col.server_default,
                        }
                    )

        return missing


def generate_alter_statements(missing_columns):
    """Génère les commandes ALTER TABLE"""
    statements = []

    for col in missing_columns:
        nullable = "NULL" if col["nullable"] else "NOT NULL"
        default = f" DEFAULT {col['default']}" if col["default"] else ""

        stmt = f"ALTER TABLE {col['table']} ADD COLUMN IF NOT EXISTS {col['column']} {col['type']} {nullable}{default};"
        statements.append(stmt)

    return statements


def main():
    missing = get_missing_columns()

    if not missing:
        print("✅ Aucune colonne manquante détectée")
        return 0

    print(f"⚠️  {len(missing)} colonne(s) manquante(s) détectée(s):\n")

    statements = generate_alter_statements(missing)

    for stmt in statements:
        print(stmt)

    print(f"\n📝 {len(statements)} commande(s) SQL générée(s)")
    print("\nPour appliquer ces modifications, exécutez :")
    print(
        "  docker compose exec -T postgres psql -U atmr -d atmr < /tmp/sync_schema.sql"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
