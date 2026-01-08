#!/usr/bin/env python3
"""✅ Phase 1 N+1: Script de profilage pour détecter requêtes N+1 dans dispatch.

Usage:
    python backend/scripts/profiling/profile_dispatch_queries.py --company-id 1
    --for-date 2025-01-15

Options:
    --company-id: ID de l'entreprise
    --for-date: Date au format YYYY-MM-DD (optionnel, utilise demain par défaut)
    --enable-profiling: Activer le profilage DB (défaut: true)
    --output: Fichier de sortie pour le rapport (optionnel)
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flask import Flask

from config import config
from ext import db, setup_db_profiler
from shared.db_profiler import get_db_profiler, is_profiling_enabled, profile_db_context

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Créer application Flask pour contexte
app = Flask(__name__)
app.config.from_object(config["development"])

# Initialiser extensions
db.init_app(app)
setup_db_profiler(app)


def run_dispatch_with_profiling(company_id: int, for_date: str | None = None):
    """Exécute un dispatch avec profilage des requêtes SQL.

    Args:
        company_id: ID de l'entreprise
        for_date: Date au format YYYY-MM-DD (optionnel)

    Returns:
        Dict avec statistiques de profilage
    """
    from services.unified_dispatch.dispatch_run import dispatch_run  # type: ignore[reportMissingImports]  # noqa: I001

    # Utiliser demain si date non fournie
    if not for_date:
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        for_date = tomorrow

    logger.info("=" * 80)
    logger.info("PROFILAGE DISPATCH - REQUÊTES N+1")
    logger.info("=" * 80)
    logger.info("Company ID: %s", company_id)
    logger.info("Date: %s", for_date)
    logger.info("")

    # Activer profilage si pas déjà activé
    if not is_profiling_enabled():
        os.environ["ENABLE_DB_PROFILING"] = "true"
        logger.info("✅ Profilage DB activé via ENABLE_DB_PROFILING=true")

    profiler = get_db_profiler()

    with app.app_context():
        # Réinitialiser profiler
        profiler.reset()

        # Exécuter dispatch avec profilage
        logger.info("🚀 Démarrage du dispatch...")
        start_time = datetime.now()

        try:
            with profile_db_context("dispatch_run"):
                result = dispatch_run(
                    company_id=company_id,
                    for_date=for_date,
                    mode="heuristic_only",
                )
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Récupérer statistiques
            stats = profiler.get_stats()

            logger.info("")
            logger.info("=" * 80)
            logger.info("RÉSULTATS DU PROFILAGE")
            logger.info("=" * 80)
            logger.info("Durée totale: %.2fs", duration)
            logger.info("Nombre de requêtes SQL: %s", stats["query_count"])
            logger.info("Temps total DB: %sms", stats["total_time_ms"])
            logger.info("Temps moyen par requête: %sms", stats["avg_time_ms"])
            logger.info("Temps min: %sms", stats["min_time_ms"])
            logger.info("Temps max: %sms", stats["max_time_ms"])
            logger.info("")

            # Détecter N+1
            n_plus_1_detected = profiler.detect_n_plus_1(threshold=10)
            if n_plus_1_detected:
                logger.error("🚨 PATTERN N+1 DÉTECTÉ!")
                logger.error(
                    "   Action recommandée: Vérifier eager loading avec joinedload()"
                )
            else:
                logger.info("✅ Aucun pattern N+1 détecté")

            # Avertissement si trop de requêtes
            if stats["query_count"] > 20:
                logger.warning(
                    "⚠️ ATTENTION: %s requêtes détectées (suspect N+1?)",
                    stats["query_count"],
                )

            # Avertissement si requêtes lentes
            if stats["max_time_ms"] > 1000:
                logger.warning(
                    "⚠️ ATTENTION: Requête lente détectée (%sms)", stats["max_time_ms"]
                )

            # Afficher dernières requêtes
            if stats.get("queries"):
                logger.info("")
                logger.info("Dernières requêtes SQL:")
                for i, query in enumerate(stats["queries"][-5:], 1):
                    logger.info("  %s. %s...", i, query[:150])

            # Générer rapport complet
            report = profiler.generate_report()
            logger.info("")
            logger.info(report)

            return {
                "success": True,
                "duration_seconds": duration,
                "stats": stats,
                "n_plus_1_detected": n_plus_1_detected,
                "result": result,
            }

        except Exception as e:
            logger.exception("❌ Erreur lors du dispatch: %s", e)
            stats = profiler.get_stats()
            return {
                "success": False,
                "error": str(e),
                "stats": stats,
            }


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Profile les requêtes SQL d'un dispatch pour détecter N+1"
    )
    parser.add_argument(
        "--company-id",
        type=int,
        required=True,
        help="ID de l'entreprise",
    )
    parser.add_argument(
        "--for-date",
        type=str,
        default=None,
        help="Date au format YYYY-MM-DD (défaut: demain)",
    )
    parser.add_argument(
        "--enable-profiling",
        type=str,
        default="true",
        help="Activer le profilage DB (défaut: true)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Fichier de sortie pour le rapport (optionnel)",
    )

    args = parser.parse_args()

    # Activer profilage si demandé
    if args.enable_profiling.lower() in ("true", "1", "yes"):
        os.environ["ENABLE_DB_PROFILING"] = "true"

    # Exécuter profilage
    result = run_dispatch_with_profiling(args.company_id, args.for_date)

    # Sauvegarder rapport si fichier de sortie spécifié
    if args.output and result.get("success"):
        profiler = get_db_profiler()
        report = profiler.generate_report()
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(report)
        logger.info("📄 Rapport sauvegardé dans %s", args.output)

    # Code de sortie
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
