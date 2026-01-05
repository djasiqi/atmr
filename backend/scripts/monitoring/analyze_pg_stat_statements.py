#!/usr/bin/env python3
"""
Script pour analyser pg_stat_statements et identifier les requêtes lentes.

Usage:
    python scripts/monitoring/analyze_pg_stat_statements.py

Ce script nécessite que l'extension pg_stat_statements soit activée dans PostgreSQL.
"""

import os
import sys
from pathlib import Path

# Ajouter le répertoire backend au PYTHONPATH
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Imports après modification du PYTHONPATH (nécessaire pour ce script)
from flask import Flask  # noqa: E402
from sqlalchemy import text  # noqa: E402

from ext import db  # noqa: E402

# Configuration Flask minimale pour utiliser SQLAlchemy
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL", "postgresql://atmr:atmr@localhost:5432/atmr"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)


def analyze_slow_queries(limit: int = 20) -> None:
    """Analyse les requêtes lentes depuis pg_stat_statements."""
    print("=" * 80)
    print("ANALYSE DES REQUÊTES LENTES (pg_stat_statements)")
    print("=" * 80)
    print()

    with app.app_context():
        # Vérifier que l'extension est activée
        result = db.session.execute(
            text(
                """
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
                ) AS extension_exists;
                """
            )
        ).fetchone()

        if not result or not result[0]:
            print("❌ ERREUR: L'extension pg_stat_statements n'est pas activée.")
            print("   Exécutez: CREATE EXTENSION IF NOT EXISTS pg_stat_statements;")
            return

        # Requêtes lentes (temps moyen > 1s)
        print("📊 TOP 20 REQUÊTES LENTES (temps moyen > 1s):")
        print("-" * 80)

        slow_queries = db.session.execute(
            text(
                """
                SELECT
                    round((total_exec_time / 1000)::numeric, 2) AS total_time_seconds,
                    calls,
                    round((mean_exec_time / 1000)::numeric, 2) AS mean_time_seconds,
                    round((max_exec_time / 1000)::numeric, 2) AS max_time_seconds,
                    round(100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0), 2) AS cache_hit_ratio,
                    left(query, 200) AS query_preview
                FROM pg_stat_statements
                WHERE mean_exec_time > 1000
                ORDER BY total_exec_time DESC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        ).fetchall()

        if not slow_queries:
            print("✅ Aucune requête lente détectée (temps moyen < 1s)")
        else:
            for i, row in enumerate(slow_queries, 1):
                print(
                    f"\n{i}. Total: {row[0]}s | Appels: {row[1]} | Moyenne: {row[2]}s | Max: {row[3]}s"
                )
                print(f"   Cache Hit Ratio: {row[4]}%")
                print(f"   Requête: {row[5]}...")

        print()
        print("=" * 80)
        print("📊 TOP 20 REQUÊTES LES PLUS FRÉQUENTES:")
        print("-" * 80)

        frequent_queries = db.session.execute(
            text(
                """
                SELECT
                    calls,
                    round((total_exec_time / 1000)::numeric, 2) AS total_time_seconds,
                    round((mean_exec_time / 1000)::numeric, 2) AS mean_time_seconds,
                    round((max_exec_time / 1000)::numeric, 2) AS max_time_seconds,
                    left(query, 200) AS query_preview
                FROM pg_stat_statements
                WHERE calls > 100
                ORDER BY calls DESC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        ).fetchall()

        if not frequent_queries:
            print("✅ Aucune requête très fréquente détectée")
        else:
            for i, row in enumerate(frequent_queries, 1):
                print(
                    f"\n{i}. Appels: {row[0]} | Total: {row[1]}s | Moyenne: {row[2]}s | Max: {row[3]}s"
                )
                print(f"   Requête: {row[4]}...")

        print()
        print("=" * 80)
        print("📊 REQUÊTES AVEC MAUVAIS CACHE HIT RATIO (< 80%):")
        print("-" * 80)

        poor_cache_queries = db.session.execute(
            text(
                """
                SELECT
                    calls,
                    round((total_exec_time / 1000)::numeric, 2) AS total_time_seconds,
                    round((mean_exec_time / 1000)::numeric, 2) AS mean_time_seconds,
                    round(100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0), 2) AS cache_hit_ratio,
                    shared_blks_read,
                    left(query, 200) AS query_preview
                FROM pg_stat_statements
                WHERE shared_blks_read > 100
                  AND (100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0)) < 80
                ORDER BY shared_blks_read DESC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        ).fetchall()

        if not poor_cache_queries:
            print("✅ Toutes les requêtes ont un bon cache hit ratio (> 80%)")
        else:
            for i, row in enumerate(poor_cache_queries, 1):
                print(
                    f"\n{i}. Appels: {row[0]} | Total: {row[1]}s | Moyenne: {row[2]}s"
                )
                print(f"   Cache Hit Ratio: {row[3]}% | Lectures disque: {row[4]}")
                print(f"   Requête: {row[5]}...")

        print()
        print("=" * 80)
        print("✅ Analyse terminée")
        print("=" * 80)


def reset_statistics() -> None:
    """Réinitialise les statistiques de pg_stat_statements."""
    print("🔄 Réinitialisation des statistiques pg_stat_statements...")

    with app.app_context():
        db.session.execute(text("SELECT pg_stat_statements_reset();"))
        db.session.commit()

        print("✅ Statistiques réinitialisées")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyser pg_stat_statements pour identifier les requêtes lentes"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Nombre de requêtes à afficher (défaut: 20)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Réinitialiser les statistiques avant l'analyse",
    )

    args = parser.parse_args()

    if args.reset:
        reset_statistics()
        print()

    analyze_slow_queries(limit=args.limit)
