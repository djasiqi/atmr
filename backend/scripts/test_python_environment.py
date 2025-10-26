#!/usr/bin/env python3
"""Script simple pour tester l'environnement Python dans Docker.

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import logging
import os
import sys

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_python_environment():
    """Test l'environnement Python."""
    logger.info("🐍 Test de l'environnement Python")
    logger.info("=" * 50)

    # Test 1: Version Python
    logger.info("✅ Version Python: %s", sys.version)
    logger.info("✅ Version Python (info): %s", sys.version_info)

    # Test 2: Répertoire de travail
    logger.info("✅ Répertoire de travail: %s", os.getcwd())

    # Test 3: Variables d'environnement Python
    logger.info("✅ PYTHONPATH: %s", os.environ.get("PYTHONPATH", "Non défini"))
    logger.info("✅ PYTHONDONTWRITEBYTECODE: %s", os.environ.get("PYTHONDONTWRITEBYTECODE", "Non défini"))
    logger.info("✅ PYTHONUNBUFFERED: %s", os.environ.get("PYTHONUNBUFFERED", "Non défini"))

    # Test 4: Modules disponibles
    logger.info("\n📦 Test des modules disponibles:")

    modules_to_test = [
        "numpy",
        "torch",
        "gymnasium",
        "flask",
        "sqlalchemy",
        "redis",
        "celery"
    ]

    for module in modules_to_test:
        try:
            __import__(module)
            logger.info("  ✅ %s: Disponible", module)
        except ImportError as e:
            logger.warning("  ❌ %s: Non disponible (%s)", module, e)

    # Test 5: Modules RL
    logger.info("\n🤖 Test des modules RL:")

    rl_modules = [
        "services.rl.dispatch_env",
        "services.rl.improved_dqn_agent",
        "services.rl.rl_logger"
    ]

    for module in rl_modules:
        try:
            __import__(module)
            logger.info("  ✅ %s: Disponible", module)
        except ImportError as e:
            logger.warning("  ❌ %s: Non disponible (%s)", module, e)

    logger.info("\n🎉 Test de l'environnement Python terminé!")
    return True

if __name__ == "__main__":
    test_python_environment()
