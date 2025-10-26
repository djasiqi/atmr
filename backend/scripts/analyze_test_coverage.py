#!/usr/bin/env python3
"""Analyseur de couverture de tests pour l'Étape 10.

Identifie les modules avec faible couverture et génère
des recommandations pour améliorer la couverture de tests.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


class CoverageAnalyzer:
    """Analyseur de couverture de tests."""

    def __init__(self):
        """Initialise l'analyseur."""
        self.backend_dir = Path("backend")
        self.target_modules = [
            "services/rl",
            "services/unified_dispatch",
            "services/proactive_alerts",
            "routes/proactive_alerts",
            "routes/shadow_mode_routes",
            "sockets/proactive_alerts",
            "models",
            "tasks"
        ]
        
        # Modules RL spécifiques à cibler
        self.rl_modules = [
            "services/rl/improved_dqn_agent.py",
            "services/rl/improved_q_network.py",
            "services/rl/replay_buffer.py",
            "services/rl/n_step_buffer.py",
            "services/rl/reward_shaping.py",
            "services/rl/optimal_hyperparameters.py",
            "services/rl/hyperparameter_tuner.py",
            "services/rl/shadow_mode_manager.py",
            "services/rl/dispatch_env.py"
        ]

    def analyze_file_coverage(self, ____________________________________________________________________________________________________file_path: Path) -> Dict[str, Any]:
        """Analyse la couverture d'un fichier spécifique.
        
        Args:
            file_path: Chemin vers le fichier à analyser
            
        Returns:
            Dictionnaire avec les informations de couverture

        """
        if not file_path.exists():
            return {
                "exists": False,
                "lines": 0,
                "functions": 0,
                "classes": 0,
                "coverage_estimate": 0
            }
        
        try:
            with Path(file_path, encoding="utf-8").open() as f:
                content = f.read()
            
            # Compter les lignes de code (approximatif)
            lines = len([line for line in content.split("\n")
                        if line.strip() and not line.strip().startswith("#")])
            
            # Compter les fonctions
            functions = content.count("def ")
            
            # Compter les classes
            classes = content.count("class ")
            
            # Estimation de couverture basée sur la complexité
            # (plus il y a de fonctions/classes, plus c'est complexe à tester)
            complexity_score = functions + classes * 2
            coverage_estimate = max(0, min(100, 100 - complexity_score * 2))
            
            return {
                "exists": True,
                "lines": lines,
                "functions": functions,
                "classes": classes,
                "coverage_estimate": coverage_estimate,
                "complexity_score": complexity_score
            }
            
        except Exception as e:
            return {
                "exists": True,
                "error": str(e),
                "coverage_estimate": 0
            }

    def find_test_files(self) -> List[Path]:
        """Trouve tous les fichiers de test."""
        test_files = []
        
        # Chercher dans le répertoire tests
        tests_dir = self.backend_dir / "tests"
        if tests_dir.exists():
            for test_file in tests_dir.rglob("test_*.py"):
                test_files.append(test_file)
        
        return test_files

    def find_source_files(self) -> List[Path]:
        """Trouve tous les fichiers source Python."""
        source_files = []
        
        for module_path in self.target_modules:
            full_path = self.backend_dir / module_path
            if full_path.exists():
                for py_file in full_path.rglob("*.py"):
                    if not py_file.name.startswith("__"):
                        source_files.append(py_file)
        
        return source_files

    def analyze_module_coverage(self) -> Dict[str, Any]:
        """Analyse la couverture de tous les modules."""
        print("🔍 Analyse de la couverture de tests...")
        
        # Trouver les fichiers source et de test
        source_files = self.find_source_files()
        test_files = self.find_test_files()
        
        print("📁 Fichiers source trouvés: {len(source_files)}")
        print("🧪 Fichiers de test trouvés: {len(test_files)}")
        
        # Analyser chaque fichier source
        module_analysis = {}
        total_lines = 0
        total_functions = 0
        total_classes = 0
        weighted_coverage = 0
        
        for source_file in source_files:
            relative_path = source_file.relative_to(self.backend_dir)
            analysis = self.analyze_file_coverage(source_file)
            
            module_analysis[str(relative_path)] = analysis
            
            if analysis["exists"] and "error" not in analysis:
                total_lines += analysis["lines"]
                total_functions += analysis["functions"]
                total_classes += analysis["classes"]
                
                # Ponderer par la complexité
                weight = analysis["complexity_score"] + 1
                weighted_coverage += analysis["coverage_estimate"] * weight
        
        # Calculer la couverture globale estimée
        total_weight = sum(analysis.get("complexity_score", 0) + 1
                          for analysis in module_analysis.values()
                          if analysis.get("exists", False) and "error" not in analysis)
        
        global_coverage = weighted_coverage / total_weight if total_weight > 0 else 0
        
        return {
            "global_coverage": global_coverage,
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "source_files_count": len(source_files),
            "test_files_count": len(test_files),
            "module_analysis": module_analysis
        }

    def identify_low_coverage_modules(self, ____________________________________________________________________________________________________analysis: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Identifie les modules avec faible couverture."""
        low_coverage = []
        
        for module_path, module_data in analysis["module_analysis"].items():
            if (module_data.get("exists", False) and
                "error" not in module_data and
                module_data.get("coverage_estimate", 0) < 70):
                low_coverage.append((module_path, module_data))
        
        # Trier par couverture (plus faible en premier)
        low_coverage.sort(key=lambda x: x[1].get("coverage_estimate", 0))
        
        return low_coverage

    def generate_test_recommendations(self, ____________________________________________________________________________________________________analysis: Dict[str, Any]) -> List[str]:
        """Génère des recommandations pour améliorer la couverture."""
        recommendations = []
        
        # Analyser les modules RL spécifiquement
        rl_coverage = []
        for module_path in self.rl_modules:
            full_path = self.backend_dir / module_path
            if full_path.exists():
                relative_path = str(full_path.relative_to(self.backend_dir))
                if relative_path in analysis["module_analysis"]:
                    module_data = analysis["module_analysis"][relative_path]
                    rl_coverage.append((relative_path, module_data.get("coverage_estimate", 0)))
        
        # Recommandations générales
        if analysis["global_coverage"] < 70:
            recommendations.append(f"🎯 Couverture globale actuelle: {analysis['global_coverage']")
            recommendations.append(f"📈 Objectif: Atteindre ≥70% (manque {70 - analysis['global_coverage']")
        
        # Recommandations pour les modules RL
        if rl_coverage:
            avg_rl_coverage = sum(cov for _, cov in rl_coverage) / len(rl_coverage)
            recommendations.append(f"🤖 Couverture RL moyenne: {avg_rl_coverage")
            recommendations.append(f"🎯 Objectif RL: Atteindre ≥85% (manque {85 - avg_rl_coverage")
        
        # Recommandations spécifiques par module
        low_coverage_modules = self.identify_low_coverage_modules(analysis)
        
        recommendations.append("\n📋 Modules nécessitant plus de tests:")
        for module_path, module_data in low_coverage_modules[:10]:  # Top 10
            coverage = module_data.get("coverage_estimate", 0)
            functions = module_data.get("functions", 0)
            classes = module_data.get("classes", 0)
            recommendations.append(f"  • {module_path}: {coverage")
        
        return recommendations

    def generate_test_plan(self) -> List[str]:
        """Génère un plan de tests détaillé."""
        return [
            "🧪 PLAN DE TESTS POUR ÉTAPE 10",
            "=" * 50,
            "",
            "1. TESTS PER (Prioritized Experience Replay)",
            "   • test_prioritized_replay_buffer.py",
            "   • Test d'ajout de transitions avec priorités",
            "   • Test d'échantillonnage pondéré",
            "   • Test de mise à jour des priorités",
            "   • Test de l'importance sampling",
            "",
            "2. TESTS ACTION MASKING",
            "   • test_action_masking.py",
            "   • Test de génération de masques valides",
            "   • Test de sélection d'actions avec masques",
            "   • Test de contraintes de fenêtres temporelles",
            "   • Test de calcul de temps de trajet",
            "",
            "3. TESTS REWARD SHAPING",
            "   • test_reward_shaping.py",
            "   • Test de calcul de récompenses de ponctualité",
            "   • Test de calcul de récompenses de distance",
            "   • Test de calcul de récompenses d'équité",
            "   • Test de configuration des poids",
            "",
            "4. TESTS DUELING DQN",
            "   • test_dueling_network.py",
            "   • Test de séparation valeur/avantage",
            "   • Test d'agrégation Q-values",
            "   • Test de formes de sortie",
            "   • Test d'initialisation des poids",
            "",
            "5. TESTS N-STEP LEARNING",
            "   • test_n_step_buffer.py",
            "   • Test de calcul de retours N-step",
            "   • Test de buffer N-step prioritisé",
            "   • Test d'intégration avec l'agent",
            "   • Test d'efficacité d'échantillonnage",
            "",
            "6. TESTS ALERTES PROACTIVES",
            "   • test_proactive_alerts.py",
            "   • Test de prédiction de retard",
            "   • Test de système de debounce",
            "   • Test d'explicabilité RL",
            "   • Test d'intégration Socket.IO",
            "",
            "7. TESTS SHADOW MODE",
            "   • test_shadow_mode.py",
            "   • Test de comparaison de décisions",
            "   • Test de calcul de KPIs",
            "   • Test de génération de rapports",
            "   • Test d'export de données",
            "",
            "8. TESTS INTÉGRATION",
            "   • test_integration_rl.py",
            "   • Test d'intégration complète RL",
            "   • Test de workflow de dispatch",
            "   • Test de performance end-to-end",
            "   • Test de robustesse système",
        ]
        

    def run_analysis(self) -> Dict[str, Any]:
        """Exécute l'analyse complète."""
        print("🚀 Analyse de couverture de tests - Étape 10")
        print("=" * 60)
        
        # Analyser la couverture
        analysis = self.analyze_module_coverage()
        
        # Générer les recommandations
        recommendations = self.generate_test_recommendations(analysis)
        
        # Générer le plan de tests
        test_plan = self.generate_test_plan()
        
        # Résumé
        print("\n📊 RÉSUMÉ DE L'ANALYSE")
        print("=" * 60)
        print("Couverture globale estimée: {analysis['global_coverage']")
        print("Fichiers source analysés: {analysis['source_files_count']}")
        print("Fichiers de test existants: {analysis['test_files_count']}")
        print("Lignes de code totales: {analysis['total_lines']}")
        print("Fonctions totales: {analysis['total_functions']}")
        print("Classes totales: {analysis['total_classes']}")
        
        print("\n📋 RECOMMANDATIONS")
        print("=" * 60)
        for recommendation in recommendations:
            print(recommendation)
        
        print("\n🧪 PLAN DE TESTS")
        print("=" * 60)
        for line in test_plan:
            print(line)
        
        return {
            "analysis": analysis,
            "recommendations": recommendations,
            "test_plan": test_plan
        }


def main():
    """Fonction principale."""
    analyzer = CoverageAnalyzer()
    
    try:
        results = analyzer.run_analysis()
        
        # Sauvegarder les résultats
        import json
        results_file = "coverage_analysis_results.json"
        with Path(results_file, "w", encoding="utf-8").open() as f:
            json.dump({
                "analysis": results["analysis"],
                "recommendations": results["recommendations"],
                "test_plan": results["test_plan"]
            }, f, indent=2, ensure_ascii=False)
        
        print("\n💾 Résultats sauvegardés: {results_file}")
        
        return 0
        
    except Exception as e:
        print("\n❌ Erreur lors de l'analyse: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
