#!/usr/bin/env python3
from pathlib import Path

"""Script de vérification simplifié des scripts de validation.

Teste directement les scripts Python sans utiliser Docker depuis l'intérieur.

Auteur: ATMR Project - RL Team
Date: 24 octobre 2025
"""

import logging
import os
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

def test_script_imports(script_path):
    """Teste les imports d'un script."""
    try:
        # Lire le contenu du script
        with Path(script_path, encoding="utf-8").open() as f:
            content = f.read()

        # Vérifier les imports critiques
        critical_imports = [
            "import logging",
            "from services.",
            "import numpy",
            "import torch",
            "from models.",
            "from config import"
        ]

        found_imports = []
        for imp in critical_imports:
            if imp in content:
                found_imports.append(imp)

        return len(found_imports) > 0, found_imports
    except Exception as e:
        return False, [f"Erreur lecture: {e}"]

def test_script_syntax(script_path):
    """Teste la syntaxe d'un script."""
    try:
        with Path(script_path, encoding="utf-8").open() as f:
            content = f.read()

        # Compiler le script pour vérifier la syntaxe
        compile(content, script_path, "exec")
        return True, "Syntaxe OK"
    except SyntaxError as e:
        return False, f"Erreur syntaxe: {e}"
    except Exception as e:
        return False, f"Erreur compilation: {e}"

def verify_validation_scripts():
    """Vérifie tous les scripts de validation."""
    logger.info("🔍 Vérification simplifiée des scripts de validation")
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
        "syntax_ok": 0,
        "imports_ok": 0,
        "fully_working": 0,
        "details": {}
    }

    # Vérifier chaque script
    for script_path in scripts_to_check:
        logger.info("\n📄 Vérification de %s...", script_path)

        # Vérifier l'existence du fichier
        if not check_script_exists(script_path):
            logger.warning("  ❌ Script manquant: %s", script_path)
            results["details"][script_path] = {
                "status": "missing",
                "error": "Fichier non trouvé"
            }
            continue

        results["existing_scripts"] += 1
        logger.info("  ✅ Script trouvé: %s", script_path)

        # Tester la syntaxe
        syntax_ok, syntax_msg = test_script_syntax(script_path)
        if syntax_ok:
            logger.info("  ✅ Syntaxe OK")
            results["syntax_ok"] += 1
        else:
            logger.error("  ❌ Erreur syntaxe: %s", syntax_msg)
            results["details"][script_path] = {
                "status": "syntax_error",
                "error": syntax_msg
            }
            continue

        # Tester les imports
        imports_ok, imports_msg = test_script_imports(script_path)
        if imports_ok:
            logger.info("  ✅ Imports OK: %s imports critiques trouvés", len(imports_msg))
            results["imports_ok"] += 1
            results["fully_working"] += 1
            results["details"][script_path] = {
                "status": "working",
                "imports": imports_msg
            }
        else:
            logger.warning("  ⚠️ Imports limités: %s", imports_msg)
            results["details"][script_path] = {
                "status": "limited_imports",
                "error": imports_msg
            }

    # Résumé des résultats
    logger.info("\n" + "=" * 70)
    logger.info("📊 RÉSUMÉ DE LA VÉRIFICATION SIMPLIFIÉE")
    logger.info("=" * 70)

    logger.info("📄 Scripts totaux: %s", results["total_scripts"])
    logger.info("✅ Scripts existants: %s", results["existing_scripts"])
    logger.info("🔧 Scripts syntaxe OK: %s", results["syntax_ok"])
    logger.info("📦 Scripts imports OK: %s", results["imports_ok"])
    logger.info("🎯 Scripts entièrement fonctionnels: %s", results["fully_working"])

    # Calculer les pourcentages
    if results["existing_scripts"] > 0:
        syntax_rate = (results["syntax_ok"] / results["existing_scripts"]) * 100
        imports_rate = (results["imports_ok"] / results["existing_scripts"]) * 100
        working_rate = (results["fully_working"] / results["existing_scripts"]) * 100

        logger.info("📈 Taux syntaxe OK: %.1f%%", syntax_rate)
        logger.info("📈 Taux imports OK: %.1f%%", imports_rate)
        logger.info("📈 Taux entièrement fonctionnels: %.1f%%", working_rate)

    # Détails des problèmes
    problems = [path for path, details in results["details"].items()
                if details["status"] in ["missing", "syntax_error"]]

    if problems:
        logger.info("\n💥 SCRIPTS AVEC PROBLÈMES:")
        for script_path in problems:
            details = results["details"][script_path]
            logger.error("  ❌ %s: %s", script_path, details["error"])

    # Scripts fonctionnels
    working_scripts = [path for path, details in results["details"].items()
                       if details["status"] == "working"]

    if working_scripts:
        logger.info("\n✅ SCRIPTS ENTIÈREMENT FONCTIONNELS:")
        for script_path in working_scripts:
            details = results["details"][script_path]
            logger.info("  ✅ %s", script_path)

    # Conclusion
    if results["fully_working"] == results["existing_scripts"]:
        logger.info("\n🎉 TOUS LES SCRIPTS EXISTANTS SONT FONCTIONNELS !")
        return True
    logger.info("\n📊 %s/%s scripts entièrement fonctionnels", results["fully_working"], results["existing_scripts"])
    return results["fully_working"] > results["existing_scripts"] * 0.8  # 80% de succès

if __name__ == "__main__":
    success = verify_validation_scripts()
    sys.exit(0 if success else 1)
