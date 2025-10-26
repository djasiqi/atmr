#!/usr/bin/env python3
"""Test rapide de l'Étape 9 - Hardening Docker/Prod.

Vérifie rapidement que tous les composants sont en place
et fonctionnels.
"""

import os
import subprocess
import sys
from pathlib import Path


def test_file_exists(file_path: str) -> bool:
    """Test si un fichier existe."""
    return Path(file_path).exists()


def test_file_executable(file_path: str) -> bool:
    """Test si un fichier est exécutable."""
    return os.access(file_path, os.X_OK)


def test_python_syntax(file_path: str) -> bool:
    """Test la syntaxe Python d'un fichier."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", file_path],
            check=False, capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def test_bash_syntax(file_path: str) -> bool:
    """Test la syntaxe Bash d'un fichier."""
    try:
        result = subprocess.run(
            ["bash", "-n", file_path],
            check=False, capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    """Fonction principale."""
    print("🧪 Test rapide de l'Étape 9 - Hardening Docker/Prod")
    print("=" * 60)
    
    # Fichiers à vérifier
    files_to_check = [
        ("backend/Dockerfile.production", "Dockerfile multi-stage"),
        ("backend/docker-entrypoint.sh", "Script d'entrée Docker"),
        ("backend/scripts/warmup_models.py", "Script de warmup des modèles"),
        ("backend/scripts/docker_smoke_tests.py", "Tests de smoke Docker"),
        ("backend/scripts/build-docker.sh", "Script de build Docker"),
        ("backend/scripts/validate_step9_docker_hardening.py", "Validation complète"),
        ("backend/scripts/deploy_step9_docker_hardening.py", "Déploiement automatisé"),
        ("backend/scripts/step9_final_summary.py", "Résumé final"),
        ("docker-compose.production.yml", "Docker Compose production"),
    ]
    
    passed_tests = 0
    total_tests = len(files_to_check)
    
    for file_path, _description in files_to_check:
        print("\n🔍 Test: {description}")
        
        # Test d'existence
        if not test_file_exists(file_path):
            print("  ❌ Fichier non trouvé: {file_path}")
            continue
        
        print("  ✅ Fichier trouvé: {file_path}")
        
        # Test de syntaxe selon le type de fichier
        if file_path.endswith(".py"):
            if test_python_syntax(file_path):
                print("  ✅ Syntaxe Python valide")
                passed_tests += 1
            else:
                print("  ❌ Syntaxe Python invalide")
        elif file_path.endswith(".sh"):
            if test_bash_syntax(file_path):
                print("  ✅ Syntaxe Bash valide")
                passed_tests += 1
            else:
                print("  ❌ Syntaxe Bash invalide")
        elif file_path.endswith((".yml", ".yaml")):
            # Pour les fichiers YAML, on vérifie juste l'existence
            print("  ✅ Fichier YAML présent")
            passed_tests += 1
        else:
            # Pour les autres fichiers, on vérifie juste l'existence
            print("  ✅ Fichier présent")
            passed_tests += 1
    
    # Test des permissions d'exécution
    print("\n🔍 Test des permissions d'exécution")
    executable_files = [
        "backend/docker-entrypoint.sh",
        "backend/scripts/build-docker.sh"
    ]
    
    for file_path in executable_files:
        if test_file_exists(file_path):
            if test_file_executable(file_path):
                print("  ✅ {file_path} exécutable")
            else:
                print("  ⚠️  {file_path} non exécutable")
        else:
            print("  ❌ {file_path} non trouvé")
    
    # Résumé des tests
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS RAPIDES")
    print("=" * 60)
    
    print("Tests réussis: {passed_tests}/{total_tests}")
    
    success_rate = passed_tests / total_tests
    
    if success_rate >= 0.9:
        print("\n🎉 TOUS LES TESTS RAPIDES RÉUSSIS!")
        print("✅ L'Étape 9 est prête pour la validation complète")
        return 0
    if success_rate >= 0.7:
        print("\n⚠️  TESTS RAPIDES PARTIELLEMENT RÉUSSIS")
        print("⚠️  Certains fichiers nécessitent une attention")
        return 1
    print("\n❌ TESTS RAPIDES ÉCHOUÉS")
    print("❌ L'Étape 9 nécessite des corrections")
    return 1


if __name__ == "__main__":
    sys.exit(main())
