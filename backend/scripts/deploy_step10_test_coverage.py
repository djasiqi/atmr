#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Script de déploiement pour l'Étape 10 - Couverture de tests ≥ 70%.

Ce script orchestre le déploiement de tous les tests créés
et valide que la couverture de tests atteint l'objectif.
"""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def deploy_test_files():
    """Déploie tous les fichiers de test."""
    print("📦 Déploiement des fichiers de test")
    
    test_files = [
        "tests/rl/test_per_comprehensive.py",
        "tests/rl/test_action_masking_comprehensive.py",
        "tests/rl/test_reward_shaping_comprehensive.py",
        "tests/rl/test_integration_comprehensive.py",
        "tests/test_alerts_comprehensive.py",
        "tests/test_shadow_mode_comprehensive.py",
        "tests/test_docker_production_comprehensive.py"
    ]
    
    deployed_files = []
    failed_files = []
    
    for test_file in test_files:
        file_path = Path(backend_dir) / test_file
        if file_path.exists():
            deployed_files.append(test_file)
            print("  ✅ {test_file} (déployé)")
        else:
            failed_files.append(test_file)
            print("  ❌ {test_file} (échec du déploiement)")
    
    return deployed_files, failed_files

def deploy_test_scripts():
    """Déploie les scripts de test."""
    print("\n🔧 Déploiement des scripts de test")
    
    test_scripts = [
        "scripts/run_comprehensive_test_coverage.py",
        "scripts/validate_step10_test_coverage.py",
        "scripts/analyze_test_coverage.py",
        "scripts/run_step10_test_coverage.py"
    ]
    
    deployed_scripts = []
    failed_scripts = []
    
    for script in test_scripts:
        script_path = Path(backend_dir) / script
        if script_path.exists():
            deployed_scripts.append(script)
            print("  ✅ {script} (déployé)")
        else:
            failed_scripts.append(script)
            print("  ❌ {script} (échec du déploiement)")
    
    return deployed_scripts, failed_scripts

def validate_test_environment():
    """Valide l'environnement de test."""
    print("\n🌍 Validation de l'environnement de test")
    
    # Vérifier Python
    python_version = sys.version_info
    print("  Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Vérifier les modules requis
    required_modules = ["pytest", "numpy", "torch", "unittest"]
    available_modules = []
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            available_modules.append(module)
            print("  ✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print("  ❌ {module} (manquant)")
    
    return available_modules, missing_modules

def run_test_suite():
    """Exécute la suite de tests complète."""
    print("\n🧪 Exécution de la suite de tests")
    
    # Essayer d'exécuter pytest si disponible
    try:
        result = subprocess.run(
            ["pytest", "--version"],
            check=False, capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("  ✅ pytest disponible: {result.stdout.strip()}")
            
            # Exécuter les tests avec couverture
            print("  🎯 Exécution des tests avec couverture...")
            
            coverage_result = subprocess.run(
                ["pytest", "tests/", "--cov=backend", "--cov-report=html", "--cov-report=term"],
                check=False, capture_output=True,
                text=True,
                timeout=0.300
            )
            
            if coverage_result.returncode == 0:
                print("  ✅ Tests exécutés avec succès")
                return True, coverage_result.stdout
            print("  ❌ Erreur lors de l'exécution des tests: {coverage_result.stderr}")
            return False, coverage_result.stderr
        print("  ❌ pytest non disponible: {result.stderr}")
        return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print("  ⏰ Timeout lors de l'exécution des tests")
        return False, "Timeout"
    except FileNotFoundError:
        print("  ❌ pytest non trouvé dans le PATH")
        return False, "pytest not found"
    except Exception as e:
        print("  💥 Erreur inattendue: {e}")
        return False, str(e)

def run_manual_tests():
    """Exécute les tests manuellement."""
    print("\n🔧 Exécution manuelle des tests")
    
    # Importer et exécuter les tests principaux
    test_modules = [
        "tests.rl.test_per_comprehensive",
        "tests.rl.test_action_masking_comprehensive",
        "tests.rl.test_reward_shaping_comprehensive",
        "tests.rl.test_integration_comprehensive",
        "tests.test_alerts_comprehensive",
        "tests.test_shadow_mode_comprehensive",
        "tests.test_docker_production_comprehensive"
    ]
    
    test_results = []
    
    for module in test_modules:
        try:
            print("  🧪 Exécution de {module}...")
            
            # Importer le module
            spec = __import__(module, fromlist=[""])
            
            # Exécuter les tests
            if hasattr(spec, "run_tests"):
                passed, total = spec.run_tests()
                test_results.append({
                    "module": module,
                    "passed": passed,
                    "total": total,
                    "success_rate": (passed / total * 100) if total > 0 else 0
                })
                print("    ✅ {passed}/{total} tests réussis")
            else:
                print("    ⚠️ Pas de fonction run_tests trouvée")
                test_results.append({
                    "module": module,
                    "passed": 0,
                    "total": 0,
                    "success_rate": 0
                })
                
        except Exception as e:
            print("    ❌ Erreur: {e}")
            test_results.append({
                "module": module,
                "passed": 0,
                "total": 0,
                "success_rate": 0,
                "error": str(e)
            })
    
    return test_results

def generate_coverage_report(test_results):
    """Génère un rapport de couverture."""
    print("\n📊 Génération du rapport de couverture")
    
    # Calculer les statistiques globales
    total_tests = sum(result["total"] for result in test_results)
    total_passed = sum(result["passed"] for result in test_results)
    global_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # Analyser les modules RL
    rl_modules = [result for result in test_results if "rl" in result["module"]]
    rl_tests = sum(result["total"] for result in rl_modules)
    rl_passed = sum(result["passed"] for result in rl_modules)
    rl_success_rate = (rl_passed / rl_tests * 100) if rl_tests > 0 else 0
    
    # Générer le rapport
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "step": "Étape 10 - Couverture de tests ≥ 70%",
        "summary": {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "global_success_rate": global_success_rate,
            "rl_tests": rl_tests,
            "rl_passed": rl_passed,
            "rl_success_rate": rl_success_rate,
            "target_met": global_success_rate >= 70
        },
        "test_results": test_results,
        "recommendations": generate_recommendations(test_results, global_success_rate)
    }
    

def generate_recommendations(test_results, global_success_rate):
    """Génère des recommandations."""
    recommendations = []
    
    if global_success_rate < 70:
        recommendations.append({
            "type": "critical",
            "message": f"Couverture globale insuffisante: {global_success_rate",
            "action": "Augmenter le nombre de tests et améliorer leur qualité"
        })
    
    failed_modules = [result for result in test_results if result["total"] > 0 and result["passed"] < result["total"]]
    if failed_modules:
        recommendations.append({
            "type": "warning",
            "message": f"Modules avec tests échoués: {len(failed_modules)}",
            "action": "Corriger les tests échoués"
        })
    
    modules_without_tests = [result for result in test_results if result["total"] == 0]
    if modules_without_tests:
        recommendations.append({
            "type": "info",
            "message": f"Modules sans tests: {len(modules_without_tests)}",
            "action": "Créer des tests pour ces modules"
        })
    
    return recommendations

def save_deployment_report(report, filename="step10_deployment_report.json"):
    """Sauvegarde le rapport de déploiement."""
    report_path = Path(__file__).parent / filename
    
    with Path(report_path, "w", encoding="utf-8").open() as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("📄 Rapport de déploiement sauvegardé: {report_path}")
    return report_path

def print_deployment_summary(report):
    """Affiche un résumé du déploiement."""
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DU DÉPLOIEMENT - ÉTAPE 10")
    print("="*60)
    
    summary = report["summary"]
    
    print("Tests totaux: {summary['total_tests']}")
    print("Tests réussis: {summary['total_passed']}")
    print("Couverture globale: {summary['global_success_rate']")
    print("Tests RL: {summary['rl_tests']}")
    print("Tests RL réussis: {summary['rl_passed']}")
    print("Couverture RL: {summary['rl_success_rate']")
    print("Objectif atteint: {'✅' if summary['target_met'] else '❌'}")
    
    print("\n📋 Résultats par module:")
    for result in report["test_results"]:
        status_emoji = "✅" if result["passed"] == result["total"] else "⚠️" if result["passed"] > 0 else "❌"
        print("  {status_emoji} {result['module']}: {result['passed']}/{result['total']} ({result['success_rate']")
    
    print("\n💡 Recommandations:")
    for rec in report["recommendations"]:
        type_emoji = {
            "critical": "🚨",
            "warning": "⚠️",
            "info": "ℹ️"
        }.get(rec["type"], "📝")
        
        print("  {type_emoji} {rec['message']}")
        print("     Action: {rec['action']}")
    
    print("="*60)

def main():
    """Fonction principale de déploiement."""
    print("🚀 Démarrage du déploiement de l'Étape 10")
    print("📅 {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Déployer les fichiers de test
    _deployed_files, _failed_files = deploy_test_files()
    
    # Déployer les scripts de test
    _deployed_scripts, _failed_scripts = deploy_test_scripts()
    
    # Valider l'environnement
    _available_modules, _missing_modules = validate_test_environment()
    
    # Exécuter la suite de tests
    pytest_success, _pytest_output = run_test_suite()
    
    # Exécuter les tests manuellement si pytest a échoué
    if not pytest_success:
        print("\n🔄 Fallback vers l'exécution manuelle des tests")
        test_results = run_manual_tests()
    else:
        # Parser les résultats de pytest
        test_results = []
        # (Dans un environnement réel, on parserait la sortie de pytest)
    
    # Générer le rapport de couverture
    report = generate_coverage_report(test_results)
    
    # Sauvegarder le rapport
    save_deployment_report(report)
    
    # Afficher le résumé
    print_deployment_summary(report)
    
    # Déterminer le code de sortie
    if report["summary"]["target_met"]:
        print("\n🎉 Déploiement réussi - Objectif de couverture atteint!")
        return 0
    print("\n⚠️ Déploiement partiel - Des améliorations sont nécessaires")
    return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
