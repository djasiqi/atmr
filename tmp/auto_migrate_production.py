#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration automatique du schéma en production
Compare les modèles SQLAlchemy avec le schéma DB et applique les différences
"""

import sys
import os

# Ajouter le backend au path
sys.path.insert(0, "/app")

from sqlalchemy import inspect, text, MetaData
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
                print(f"⚠️  Table {table_name} n'existe pas en DB")
                continue

            # Trouver les colonnes manquantes
            for col_name, col in model_columns.items():
                if col_name not in db_columns:
                    # Déterminer le type SQL
                    col_type = str(col.type)

                    # Gérer les types spéciaux
                    if "VARCHAR" in col_type:
                        sql_type = col_type
                    elif "INTEGER" in col_type:
                        sql_type = "INTEGER"
                    elif "BOOLEAN" in col_type:
                        sql_type = "BOOLEAN"
                    elif "TIMESTAMP" in col_type:
                        sql_type = "TIMESTAMP WITH TIME ZONE"
                    elif "NUMERIC" in col_type:
                        sql_type = col_type
                    elif "TEXT" in col_type:
                        sql_type = "TEXT"
                    else:
                        sql_type = col_type

                    missing.append(
                        {
                            "table": table_name,
                            "column": col_name,
                            "type": sql_type,
                            "nullable": col.nullable,
                            "default": col.server_default,
                        }
                    )

        return missing


def apply_missing_columns(missing_columns):
    """Applique les colonnes manquantes"""
    app = create_app("production")

    with app.app_context():
        for col in missing_columns:
            nullable = "NULL" if col["nullable"] else "NOT NULL"
            default = ""

            if col["default"]:
                default_val = str(col["default"].arg)
                if default_val:
                    default = f" DEFAULT {default_val}"

            stmt = f'ALTER TABLE "{col["table"]}" ADD COLUMN IF NOT EXISTS "{col["column"]}" {col["type"]} {nullable}{default};'

            try:
                print(f"🔧 {stmt}")
                db.session.execute(text(stmt))
                db.session.commit()
                print(f"   ✅ Colonne {col['table']}.{col['column']} ajoutée")
            except Exception as e:
                print(f"   ❌ Erreur: {e}")
                db.session.rollback()


def main():
    print("🔍 Détection des colonnes manquantes...")
    missing = get_missing_columns()

    if not missing:
        print("✅ Aucune colonne manquante détectée")
        return 0

    print(f"\n⚠️  {len(missing)} colonne(s) manquante(s) détectée(s):\n")

    for col in missing:
        print(f"  - {col['table']}.{col['column']} ({col['type']})")

    print(f"\n🔧 Application des modifications...\n")
    apply_missing_columns(missing)

    print(f"\n✅ Migration automatique terminée !")
    return 0


if __name__ == "__main__":
    sys.exit(main())
