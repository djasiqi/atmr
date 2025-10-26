#!/usr/bin/env python3

# Constantes pour éviter les valeurs magiques
from pathlib import Path

RETURNCODE_ZERO = 0

"""Script de vérification globale de tous les scripts de validation.

Vérifie que tous les scripts de validation fonctionnent correctement
dans l'environnement Docker.

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import logging
import os
import subprocess
import sys

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_script_exists(script_path):
    """Vérifie qu'un script existe."""
    return Path(script_path).exists()

def run_script_in_docker(script_path):
    """Exécute un script dans le conteneur Docker."""
    try:
        # Exécuter le script dans le conteneur Docker
        result = subprocess.run(
            ["docker", "exec", "atmr-api-1", "python", script_path],
            check=False, capture_output=True,
            text=True,
            timeout=0.300  # 5 minutes timeout
        )

        return {
            "success": result.returncode == RETURNCODE_ZERO,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Timeout après 5 minutes"
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e)
        }

def verify_all_validation_scripts():
    """Vérifie tous les scripts de validation."""
    logger.info("🔍 Vérification globale de tous les scripts de validation")
    logger.info("=" * 70)

    # Liste des scripts à vérifier
    scripts_to_check = [
        "scripts/validate_action_masking_docker.py",
        "scripts/validate_action_masking_fixes.py",
        "scripts/validate_rl_logger_implementation.py",
        "scripts/validate_safety_guards_implementation.py",
        "scripts/validate_step4alerts.py",
        "scripts/validate_step5final.py",
        "scripts/validate_step5n_step.py",
        "scripts/validate_step6dueling.py",
        "scripts/validate_step6manual.py",
        "scripts/validate_step7hyperparameter_tuning.py",
        "scripts/validate_step8shadow_mode.py",
        "scripts/validate_step9docker_hardening.py",
        "scripts/validate_step10complete_final.py",
        "scripts/validate_step10final_complete.py",
        "scripts/validate_step10final.py",
        "scripts/validate_step10simple.py",
        "scripts/validate_step10test_coverage.py",
        "scripts/validate_step11noisy_networks.py",
        "scripts/validate_step12distributional_rl.py",
        "scripts/validate_step13final.py",
        "scripts/validate_step13mlops.py",
        "scripts/validate_step15final.py",
        "scripts/verify_semaine4complete.py"
    ]

    results = {
        "total_scripts": len(scripts_to_check),
        "existing_scripts": 0,
        "successful_scripts": 0,
        "failed_scripts": 0,
        "missing_scripts": 0,
        "details": {}
    }

    # Vérifier chaque script
    for script_path in scripts_to_check:
        logger.info("\n📄 Vérification de %s...", script_path)

        # Vérifier l'existence du fichier
        if not check_script_exists(script_path):
            logger.warning("  ❌ Script manquant: %s", script_path)
            results["missing_scripts"] += 1
            results["details"][script_path] = {
                "status": "missing",
                "error": "Fichier non trouvé"
            }
            continue

        results["existing_scripts"] += 1
        logger.info("  ✅ Script trouvé: %s", script_path)

        # Exécuter le script dans Docker
        logger.info("  🐳 Exécution dans Docker...")
        result = run_script_in_docker(script_path)

        if result["success"]:
            logger.info("  ✅ Script exécuté avec succès")
            results["successful_scripts"] += 1
            results["details"][script_path] = {
                "status": "success",
                "returncode": result["returncode"],
                "stdout_lines": len(result["stdout"].split("\n")),
                "stderr_lines": len(result["stderr"].split("\n"))
            }
        else:
            logger.error("  ❌ Script échoué (code: %s)", result["returncode"])
            results["failed_scripts"] += 1
            results["details"][script_path] = {
                "status": "failed",
                "returncode": result["returncode"],
                "error": result["stderr"][:200] + "..." if len(result["stderr"]) > 200 else result["stderr"]
            }

    # Résumé des résultats
    logger.info("\n" + "=" * 70)
    logger.info("📊 RÉSUMÉ DE LA VÉRIFICATION GLOBALE")
    logger.info("=" * 70)

    logger.info("📄 Scripts totaux: %s", results["total_scripts"])
    logger.info("✅ Scripts existants: %s", results["existing_scripts"])
    logger.info("❌ Scripts manquants: %s", results["missing_scripts"])
    logger.info("🎯 Scripts réussis: %s", results["successful_scripts"])
    logger.info("💥 Scripts échoués: %s", results["failed_scripts"])

    # Calculer le pourcentage de succès
    if results["existing_scripts"] > 0:
        success_rate = (results["successful_scripts"] / results["existing_scripts"]) * 100
        logger.info("📈 Taux de succès: %.1f%%", success_rate)

    # Détails des échecs
    if results["failed_scripts"] > 0:
        logger.info("\n💥 DÉTAILS DES ÉCHECS:")
        for script_path, details in results["details"].items():
            if details["status"] == "failed":
                logger.error("  ❌ %s: %s", script_path, details["error"])

    # Scripts manquants
    if results["missing_scripts"] > 0:
        logger.info("\n📄 SCRIPTS MANQUANTS:")
        for script_path, details in results["details"].items():
            if details["status"] == "missing":
                logger.warning("  ⚠️ %s", script_path)

    # Conclusion
    if results["successful_scripts"] == results["existing_scripts"]:
        logger.info("\n🎉 TOUS LES SCRIPTS EXISTANTS FONCTIONNENT PARFAITEMENT !")
        return True
    logger.warning("\n⚠️ Certains scripts nécessitent des corrections.")
    return False

if __name__ == "__main__":
    success = verify_all_validation_scripts()
    sys.exit(0 if success else 1)
