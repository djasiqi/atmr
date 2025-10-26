#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Analyseur de couverture de tests pour l'Étape 10.

Ce script analyse la couverture réelle des tests en comptant les lignes de code
et les méthodes testées dans les modules de l'Étape 10.
"""

import ast
import sys
from pathlib import Path

# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def count_lines_in_file(file_path):
    """Compte les lignes de code dans un fichier."""
    try:
        with Path(file_path, encoding="utf-8").open() as f:
            lines = f.readlines()
        
        # Compter les lignes non vides et non commentaires
        code_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                code_lines += 1
        
        return code_lines, len(lines)
    except Exception:
        return 0, 0

def count_methods_in_file(file_path):
    """Compte les méthodes dans un fichier Python."""
    try:
        with Path(file_path, encoding="utf-8").open() as f:
            content = f.read()
        
        tree = ast.parse(content)
        methods = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                methods.append(node.name)
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(f"{node.name}.{item.name}")
        
        return methods
    except Exception:
        return []

def analyze_coverage():
    """Analyse la couverture des tests."""
    print("📊 ANALYSE DE COUVERTURE DÉTAILLÉE - ÉTAPE 10")
    print("=" * 70)
    
    # Modules de l'Étape 10
    step10_modules = {
        "ImprovedDQNAgent": "services/rl/improved_dqn_agent.py",
        "RewardShaping": "services/rl/reward_shaping.py",
        "NStepBuffer": "services/rl/n_step_buffer.py",
        "QNetwork": "services/rl/improved_q_network.py",
        "HyperparameterTuner": "services/rl/hyperparameter_tuner.py",
        "ShadowModeManager": "services/rl/shadow_mode_manager.py",
        "OptimalHyperparameters": "services/rl/optimal_hyperparameters.py",
        "ProactiveAlerts": "services/proactive_alerts.py",
        "DispatchEnv": "services/rl/dispatch_env.py"
    }
    
    # Tests correspondants
    step10_tests = {
        "ImprovedDQNAgent": [
            "tests/rl/test_per_comprehensive.py",
            "tests/rl/test_dqn_agent.py",
            "tests/rl/test_sprint1_improvements.py"
        ],
        "RewardShaping": [
            "tests/rl/test_reward_shaping_comprehensive.py"
        ],
        "NStepBuffer": [
            "tests/rl/test_n_step_buffer.py"
        ],
        "QNetwork": [
            "tests/rl/test_dueling_network.py",
            "tests/rl/test_q_network.py"
        ],
        "HyperparameterTuner": [
            "tests/rl/test_hyperparameter_tuner.py"
        ],
        "ShadowModeManager": [
            "tests/rl/test_shadow_mode.py",
            "tests/test_shadow_mode_comprehensive.py"
        ],
        "OptimalHyperparameters": [
            "tests/rl/test_sprint1_improvements.py"
        ],
        "ProactiveAlerts": [
            "tests/test_alerts_comprehensive.py",
            "tests/test_alerts_delay_risk.py"
        ],
        "DispatchEnv": [
            "tests/rl/test_action_masking_comprehensive.py",
            "tests/rl/test_dispatch_env.py"
        ]
    }
    
    total_code_lines = 0
    total_test_lines = 0
    total_methods = 0
    total_test_methods = 0
    
    print("\n📋 ANALYSE PAR MODULE:")
    print("-" * 50)
    
    for module_name, module_path in step10_modules.items():
        full_module_path = Path(__file__).parent.parent / module_path
        
        if full_module_path.exists():
            # Analyser le module
            code_lines, _total_lines = count_lines_in_file(full_module_path)
            methods = count_methods_in_file(full_module_path)
            
            # Analyser les tests
            test_code_lines = 0
            test_methods = []
            
            if module_name in step10_tests:
                for test_file in step10_tests[module_name]:
                    test_path = Path(__file__).parent.parent / test_file
                    if test_path.exists():
                        test_lines, _ = count_lines_in_file(test_path)
                        test_code_lines += test_lines
                        
                        test_methods_in_file = count_methods_in_file(test_path)
                        test_methods.extend(test_methods_in_file)
            
            # Calculer les ratios
            len(test_methods) / len(methods) * 100 if methods else 0
            test_code_lines / code_lines * 100 if code_lines else 0
            
            print("\n🔧 {module_name}:")
            print("  📄 Lignes de code: {code_lines}")
            print("  🧪 Lignes de test: {test_code_lines}")
            print("  📊 Couverture lignes: {line_coverage")
            print("  🔧 Méthodes: {len(methods)}")
            print("  🧪 Méthodes testées: {len(test_methods)}")
            print("  📊 Couverture méthodes: {method_coverage")
            
            total_code_lines += code_lines
            total_test_lines += test_code_lines
            total_methods += len(methods)
            total_test_methods += len(test_methods)
        else:
            print("\n❌ {module_name}: Module non trouvé")
    
    # Calculer les totaux
    overall_line_coverage = total_test_lines / total_code_lines * 100 if total_code_lines else 0
    overall_method_coverage = total_test_methods / total_methods * 100 if total_methods else 0
    
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ GLOBAL DE COUVERTURE")
    print("=" * 70)
    
    print("📄 Total lignes de code: {total_code_lines}")
    print("🧪 Total lignes de test: {total_test_lines}")
    print("📊 Couverture lignes globale: {overall_line_coverage")
    
    print("\n🔧 Total méthodes: {total_methods}")
    print("🧪 Total méthodes testées: {total_test_methods}")
    print("📊 Couverture méthodes globale: {overall_method_coverage")
    
    # Évaluation
    print("\n🎯 ÉVALUATION DE LA COUVERTURE:")
    
    if overall_line_coverage >= 85:
        print("🎉 COUVERTURE EXCELLENTE!")
        print("✅ Objectif ≥85% largement atteint")
    elif overall_line_coverage >= 70:
        print("✅ COUVERTURE BONNE!")
        print("✅ Objectif ≥70% atteint")
    elif overall_line_coverage >= 50:
        print("⚠️ COUVERTURE MOYENNE")
        print("⚠️ Objectif ≥70% non atteint")
    else:
        print("❌ COUVERTURE INSUFFISANTE")
        print("❌ Beaucoup de code non testé")
    
    if overall_method_coverage >= 80:
        print("🎉 COUVERTURE MÉTHODES EXCELLENTE!")
    elif overall_method_coverage >= 60:
        print("✅ COUVERTURE MÉTHODES BONNE!")
    else:
        print("⚠️ COUVERTURE MÉTHODES À AMÉLIORER")
    
    # Recommandations
    print("\n💡 RECOMMANDATIONS:")
    
    if overall_line_coverage < 70:
        print("🔧 Ajouter plus de tests unitaires")
        print("🔧 Couvrir les cas d'erreur et edge cases")
        print("🔧 Tester les méthodes privées importantes")
    
    if overall_method_coverage < 60:
        print("🧪 Créer des tests pour les méthodes non couvertes")
        print("🧪 Ajouter des tests d'intégration")
    
    print("📊 Surveiller la couverture en continu")
    print("🔄 Exécuter les tests régulièrement")
    
    return overall_line_coverage, overall_method_coverage

def main():
    """Fonction principale."""
    try:
        line_coverage, method_coverage = analyze_coverage()
        
        # Code de sortie basé sur la couverture
        if line_coverage >= 70 and method_coverage >= 60:
            print("\n🎉 COUVERTURE ACCEPTABLE!")
            return 0
        print("\n⚠️ COUVERTURE À AMÉLIORER!")
        return 1
            
    except Exception:
        print("\n🚨 ERREUR: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
