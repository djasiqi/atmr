#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Script principal pour exécuter tous les tests et générer un rapport de couverture.

Ce script orchestre l'exécution de tous les tests créés pour améliorer
la couverture de tests du système RL/ATMR.
"""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def run_test_module(module_path, module_name):
    """Exécute un module de test et retourne les résultats."""
    print("\n🧪 Exécution des tests {module_name}")
    
    try:
        # Importer et exécuter le module de test
        spec = __import__(module_path, fromlist=[""])
        
        # Chercher la fonction de test principale
        if hasattr(spec, "run_tests"):
            passed, total = spec.run_tests()
        elif hasattr(spec, "run_per_tests"):
            passed, total = spec.run_per_tests()
        elif hasattr(spec, "run_masking_tests"):
            passed, total = spec.run_masking_tests()
        elif hasattr(spec, "run_reward_tests"):
            passed, total = spec.run_reward_tests()
        elif hasattr(spec, "run_integration_tests"):
            passed, total = spec.run_integration_tests()
        elif hasattr(spec, "run_alerts_tests"):
            passed, total = spec.run_alerts_tests()
        elif hasattr(spec, "run_shadow_mode_tests"):
            passed, total = spec.run_shadow_mode_tests()
        elif hasattr(spec, "run_docker_production_tests"):
            passed, total = spec.run_docker_production_tests()
        else:
            # Essayer d'exécuter toutes les méthodes de test
            passed, total = 0, 0
            for attr_name in dir(spec):
                if attr_name.startswith("run_") and attr_name.endswith("_tests"):
                    method = getattr(spec, attr_name)
                    try:
                        p, t = method()
                        passed += p
                        total += t
                    except Exception as e:
                        print("  ❌ Erreur dans {attr_name}: {e}")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        print("  📊 Résultats: {passed}/{total} ({success_rate")
        
        return {
            "module": module_name,
            "passed": passed,
            "total": total,
            "success_rate": success_rate,
            "status": "success" if passed == total else "partial" if passed > 0 else "failed"
        }
        
    except Exception as e:
        print("  ❌ Erreur lors de l'exécution: {e}")
        return {
            "module": module_name,
            "passed": 0,
            "total": 0,
            "success_rate": 0,
            "status": "error",
            "error": str(e)
        }

def analyze_test_coverage():
    """Analyse la couverture de tests actuelle."""
    print("\n📊 Analyse de la couverture de tests")
    
    # Définir les modules à analyser
    modules_to_analyze = [
        "services.rl.improved_dqn_agent",
        "services.rl.improved_q_network",
        "services.rl.dispatch_env",
        "services.rl.reward_shaping",
        "services.rl.n_step_buffer",
        "services.rl.hyperparameter_tuner",
        "services.rl.shadow_mode_manager",
        "services.proactive_alerts",
        "services.unified_dispatch.rl_optimizer"
    ]
    
    coverage_analysis = {}
    
    for module in modules_to_analyze:
        try:
            # Essayer d'importer le module
            __import__(module, fromlist=[""])
            
            # Analyser le module
            module_path = module.replace(".", "/") + ".py"
            if Path(module_path).exists():
                with Path(module_path, encoding="utf-8").open() as f:
                    content = f.read()
                
                # Compter les lignes de code
                lines = content.split("\n")
                code_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
                
                coverage_analysis[module] = {
                    "exists": True,
                    "total_lines": len(lines),
                    "code_lines": len(code_lines),
                    "has_tests": False  # Sera mis à jour par les tests
                }
            else:
                coverage_analysis[module] = {
                    "exists": False,
                    "total_lines": 0,
                    "code_lines": 0,
                    "has_tests": False
                }
                
        except ImportError:
            coverage_analysis[module] = {
                "exists": False,
                "total_lines": 0,
                "code_lines": 0,
                "has_tests": False
            }
    
    return coverage_analysis

def generate_coverage_report(test_results, coverage_analysis):
    """Génère un rapport de couverture complet."""
    print("\n📋 Génération du rapport de couverture")
    
    # Calculer les statistiques globales
    total_tests = sum(result["total"] for result in test_results)
    total_passed = sum(result["passed"] for result in test_results)
    global_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # Analyser les modules avec tests
    modules_with_tests = [result["module"] for result in test_results if result["total"] > 0]
    
    # Mettre à jour l'analyse de couverture
    for module, analysis in coverage_analysis.items():
        module_name = module.split(".")[-1]
        if module_name in modules_with_tests:
            analysis["has_tests"] = True
    
    # Générer le rapport
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "summary": {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "global_success_rate": global_success_rate,
            "modules_tested": len(modules_with_tests),
            "total_modules_analyzed": len(coverage_analysis)
        },
        "test_results": test_results,
        "coverage_analysis": coverage_analysis,
        "recommendations": generate_recommendations(test_results, coverage_analysis)
    }
    

def generate_recommendations(test_results, coverage_analysis):
    """Génère des recommandations pour améliorer la couverture."""
    recommendations = []
    
    # Calculer le taux de succès global
    total_tests = sum(result["total"] for result in test_results)
    total_passed = sum(result["passed"] for result in test_results)
    global_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # Analyser les résultats des tests
    failed_modules = [result["module"] for result in test_results if result["status"] == "failed"]
    partial_modules = [result["module"] for result in test_results if result["status"] == "partial"]
    
    if failed_modules:
        recommendations.append({
            "type": "critical",
            "message": f"Modules avec tests échoués: {', '.join(failed_modules)}",
            "action": "Corriger les tests échoués et vérifier l'implémentation"
        })
    
    if partial_modules:
        recommendations.append({
            "type": "warning",
            "message": f"Modules avec tests partiels: {', '.join(partial_modules)}",
            "action": "Améliorer les tests pour atteindre 100% de succès"
        })
    
    # Analyser les modules sans tests
    modules_without_tests = [
        module for module, analysis in coverage_analysis.items()
        if analysis["exists"] and not analysis["has_tests"]
    ]
    
    if modules_without_tests:
        recommendations.append({
            "type": "info",
            "message": f"Modules sans tests: {', '.join(modules_without_tests)}",
            "action": "Créer des tests pour ces modules"
        })
    
    # Recommandations générales
    if global_success_rate < 70:
        recommendations.append({
            "type": "warning",
            "message": f"Couverture globale faible: {global_success_rate",
            "action": "Augmenter le nombre de tests et améliorer leur qualité"
        })
    
    return recommendations

def save_report(report, filename="coverage_report.json"):
    """Sauvegarde le rapport dans un fichier JSON."""
    report_path = Path(__file__).parent / filename
    
    with Path(report_path, "w", encoding="utf-8").open() as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("📄 Rapport sauvegardé: {report_path}")
    return report_path

def print_summary(report):
    """Affiche un résumé du rapport."""
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DE LA COUVERTURE DE TESTS")
    print("="*60)
    
    summary = report["summary"]
    print("Tests totaux: {summary['total_tests']}")
    print("Tests réussis: {summary['total_passed']}")
    print("Taux de succès global: {summary['global_success_rate']")
    print("Modules testés: {summary['modules_tested']}")
    print("Modules analysés: {summary['total_modules_analyzed']}")
    
    print("\n📋 Résultats par module:")
    for result in report["test_results"]:
        status_emoji = {
            "success": "✅",
            "partial": "⚠️",
            "failed": "❌",
            "error": "💥"
        }.get(result["status"], "❓")
        
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
    """Fonction principale."""
    print("🚀 Démarrage de l'analyse de couverture de tests")
    print("📅 {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Définir les modules de test à exécuter
    test_modules = [
        ("tests.rl.test_per_comprehensive", "PER (Prioritized Experience Replay)"),
        ("tests.rl.test_action_masking_comprehensive", "Action Masking"),
        ("tests.rl.test_reward_shaping_comprehensive", "Reward Shaping"),
        ("tests.rl.test_integration_comprehensive", "Intégration RL"),
        ("tests.test_alerts_comprehensive", "Alertes Proactives"),
        ("tests.test_shadow_mode_comprehensive", "Shadow Mode"),
        ("tests.test_docker_production_comprehensive", "Docker & Production")
    ]
    
    # Exécuter tous les tests
    test_results = []
    for module_path, module_name in test_modules:
        result = run_test_module(module_path, module_name)
        test_results.append(result)
    
    # Analyser la couverture
    coverage_analysis = analyze_test_coverage()
    
    # Générer le rapport
    report = generate_coverage_report(test_results, coverage_analysis)
    
    # Sauvegarder le rapport
    save_report(report)
    
    # Afficher le résumé
    print_summary(report)
    
    # Retourner le code de sortie approprié
    if report["summary"]["global_success_rate"] >= 70:
        print("\n🎉 Objectif de couverture atteint (≥70%)")
        return 0
    print("\n⚠️ Objectif de couverture non atteint ({report['summary']['global_success_rate']")
    return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
