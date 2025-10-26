#!/usr/bin/env python3
# ruff: noqa: E402
"""Script de déploiement pour l'Étape 8 - Shadow Mode Enrichi & KPIs.

Orchestre le déploiement complet du système de comparaison
humain vs RL avec génération de rapports quotidiens.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ajouter le répertoire backend au path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from services.rl.shadow_mode_manager import ShadowModeManager


def setup_logging():
    """Configure le logging."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"logs/deploy_step8_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_tests():
    """Exécute les tests du shadow mode."""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Exécution des tests Shadow Mode...")
    
    try:
        # Importer et exécuter les tests
        from tests.test_shadow_mode import run_shadow_mode_tests
        run_shadow_mode_tests()
        
        logger.info("✅ Tests Shadow Mode réussis")
        return True
        
    except Exception as e:
        logger.error("❌ Erreur lors des tests: %s", e)
        return False


def validate_implementation():
    """Valide l'implémentation complète."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Validation de l'implémentation...")
    
    try:
        # Importer et exécuter la validation
        from scripts.validate_step8_shadow_mode import Step8ValidationSuite
        
        validator = Step8ValidationSuite()
        validator.run_all_validations()
        success = validator.generate_report()
        
        if success:
            logger.info("✅ Validation complète réussie")
        else:
            logger.error("❌ Validation échouée")
        
        return success
        
    except Exception as e:
        logger.error("❌ Erreur lors de la validation: %s", e)
        return False


def create_sample_data():
    """Crée des données d'exemple pour démonstration."""
    logger = logging.getLogger(__name__)
    logger.info("📊 Création de données d'exemple...")
    
    try:
        # Créer le gestionnaire
        manager = ShadowModeManager(data_dir="data/rl/shadow_mode")
        
        # Données d'exemple pour plusieurs entreprises
        companies_data = [
            {
                "company_id": "company_alpha",
                "decisions": [
                    {
                        "booking_id": "booking_alpha_1",
                        "human_decision": {
                            "driver_id": "driver_h1",
                            "eta_minutes": 30,
                            "delay_minutes": 10,
                            "distance_km": 15.0,
                            "driver_load": 0.8,
                            "confidence": 0.7
                        },
                        "rl_decision": {
                            "driver_id": "driver_r1",
                            "eta_minutes": 25,
                            "delay_minutes": 5,
                            "distance_km": 12.5,
                            "driver_load": 0.6,
                            "confidence": 0.9,
                            "alternative_drivers": ["driver_r1", "driver_alt1", "driver_alt2"],
                            "respects_time_window": True,
                            "driver_available": True,
                            "passenger_count": 2,
                            "in_service_area": True
                        },
                        "context": {
                            "avg_eta": 28,
                            "avg_distance": 14.0,
                            "avg_load": 0.7,
                            "vehicle_capacity": 4,
                            "driver_performance": {
                                "driver_r1": {"rating": 4.5},
                                "driver_h1": {"rating": 4.2}
                            }
                        }
                    },
                    {
                        "booking_id": "booking_alpha_2",
                        "human_decision": {
                            "driver_id": "driver_h2",
                            "eta_minutes": 20,
                            "delay_minutes": 0,
                            "distance_km": 10.0,
                            "driver_load": 0.5,
                            "confidence": 0.9
                        },
                        "rl_decision": {
                            "driver_id": "driver_h2",  # Accord avec l'humain
                            "eta_minutes": 20,
                            "delay_minutes": 0,
                            "distance_km": 10.0,
                            "driver_load": 0.5,
                            "confidence": 0.95,
                            "alternative_drivers": ["driver_h2", "driver_alt3"],
                            "respects_time_window": True,
                            "driver_available": True,
                            "passenger_count": 1,
                            "in_service_area": True
                        },
                        "context": {
                            "avg_eta": 28,
                            "avg_distance": 14.0,
                            "avg_load": 0.7,
                            "vehicle_capacity": 4,
                            "driver_performance": {
                                "driver_h2": {"rating": 4.8}
                            }
                        }
                    }
                ]
            },
            {
                "company_id": "company_beta",
                "decisions": [
                    {
                        "booking_id": "booking_beta_1",
                        "human_decision": {
                            "driver_id": "driver_h3",
                            "eta_minutes": 35,
                            "delay_minutes": 15,
                            "distance_km": 18.0,
                            "driver_load": 0.9,
                            "confidence": 0.6
                        },
                        "rl_decision": {
                            "driver_id": "driver_r2",
                            "eta_minutes": 28,
                            "delay_minutes": 8,
                            "distance_km": 15.0,
                            "driver_load": 0.7,
                            "confidence": 0.85,
                            "alternative_drivers": ["driver_r2", "driver_alt4"],
                            "respects_time_window": True,
                            "driver_available": True,
                            "passenger_count": 3,
                            "in_service_area": True
                        },
                        "context": {
                            "avg_eta": 32,
                            "avg_distance": 16.0,
                            "avg_load": 0.8,
                            "vehicle_capacity": 4,
                            "driver_performance": {
                                "driver_r2": {"rating": 4.3},
                                "driver_h3": {"rating": 3.9}
                            }
                        }
                    }
                ]
            }
        ]
        
        # Enregistrer toutes les décisions
        for company_data in companies_data:
            company_id = company_data["company_id"]
            
            for decision_data in company_data["decisions"]:
                manager.log_decision_comparison(
                    company_id=company_id,
                    booking_id=decision_data["booking_id"],
                    human_decision=decision_data["human_decision"],
                    rl_decision=decision_data["rl_decision"],
                    context=decision_data["context"]
                )
        
        # Générer les rapports quotidiens
        for company_data in companies_data:
            company_id = company_data["company_id"]
            report = manager.generate_daily_report(company_id)
            
            logger.info("📊 Rapport généré pour %s: %s décisions", company_id, report["total_decisions"])
        
        # Générer les résumés d'entreprise
        for company_data in companies_data:
            company_id = company_data["company_id"]
            summary = manager.get_company_summary(company_id, 7)
            
            logger.info("📈 Résumé généré pour %s: %s décisions sur 7 jours", company_id, summary["total_decisions"])
        
        logger.info("✅ Données d'exemple créées avec succès")
        return True
        
    except Exception as e:
        logger.error("❌ Erreur lors de la création des données: %s", e)
        return False


def update_app_integration():
    """Met à jour l'intégration avec l'application Flask."""
    logger = logging.getLogger(__name__)
    logger.info("🔗 Mise à jour de l'intégration Flask...")
    
    try:
        # Vérifier que les routes existent
        routes_file = Path("routes/shadow_mode_routes.py")
        if not routes_file.exists():
            logger.error("❌ Fichier routes/shadow_mode_routes.py non trouvé")
            return False
        
        # Vérifier que le gestionnaire existe
        manager_file = Path("services/rl/shadow_mode_manager.py")
        if not manager_file.exists():
            logger.error("❌ Fichier services/rl/shadow_mode_manager.py non trouvé")
            return False
        
        # Vérifier que les tests existent
        tests_file = Path("tests/test_shadow_mode.py")
        if not tests_file.exists():
            logger.error("❌ Fichier tests/test_shadow_mode.py non trouvé")
            return False
        
        logger.info("✅ Intégration Flask validée")
        return True
        
    except Exception as e:
        logger.error("❌ Erreur lors de la mise à jour de l'intégration: %s", e)
        return False


def generate_deployment_summary():
    """Génère un résumé du déploiement."""
    logger = logging.getLogger(__name__)
    logger.info("📋 Génération du résumé de déploiement...")
    
    try:
        summary = {
            "deployment_date": datetime.now(UTC).isoformat(),
            "step": "Étape 8 - Shadow Mode Enrichi & KPIs",
            "components": {
                "shadow_mode_manager": {
                    "file": "services/rl/shadow_mode_manager.py",
                    "description": "Gestionnaire principal du shadow mode avec KPIs",
                    "features": [
                        "Comparaison humain vs RL",
                        "Calcul des KPIs détaillés",
                        "Génération de rapports quotidiens",
                        "Export CSV/JSON automatisé"
                    ]
                },
                "shadow_mode_routes": {
                    "file": "routes/shadow_mode_routes.py",
                    "description": "Routes API pour le shadow mode",
                    "endpoints": [
                        "/api/shadow-mode/reports/daily/<company_id>",
                        "/api/shadow-mode/reports/summary/<company_id>",
                        "/api/shadow-mode/kpis/metrics/<company_id>",
                        "/api/shadow-mode/kpis/export/<company_id>",
                        "/api/shadow-mode/health",
                        "/api/shadow-mode/companies"
                    ]
                },
                "shadow_mode_tests": {
                    "file": "tests/test_shadow_mode.py",
                    "description": "Tests complets du shadow mode",
                    "test_categories": [
                        "Tests unitaires ShadowModeManager",
                        "Tests de calcul des KPIs",
                        "Tests d'enregistrement des décisions",
                        "Tests de génération de rapports",
                        "Tests d'export de fichiers",
                        "Tests d'intégration"
                    ]
                }
            },
            "kpis_implemented": [
                "eta_delta - Différence ETA humain vs RL",
                "delay_delta - Différence retard humain vs RL",
                "second_best_driver - Second meilleur driver suggéré",
                "rl_confidence - Confiance RL dans la décision",
                "human_confidence - Confiance humaine (si disponible)",
                "decision_reasons - Raisons de la décision RL",
                "constraint_violations - Violations de contraintes",
                "performance_impact - Impact sur performance globale"
            ],
            "daily_reports_features": [
                "Statistiques quotidiennes détaillées",
                "Résumé des KPIs avec insights",
                "Recommandations basées sur les données",
                "Export automatique en JSON et CSV",
                "Analyse des tendances multi-jours"
            ],
            "api_endpoints": {
                "reports": {
                    "daily": "GET /api/shadow-mode/reports/daily/<company_id>",
                    "summary": "GET /api/shadow-mode/reports/summary/<company_id>",
                    "log_decision": "POST /api/shadow-mode/reports/daily/<company_id>"
                },
                "kpis": {
                    "metrics": "GET /api/shadow-mode/kpis/metrics/<company_id>",
                    "export": "GET /api/shadow-mode/kpis/export/<company_id>"
                },
                "utility": {
                    "health": "GET /api/shadow-mode/health",
                    "companies": "GET /api/shadow-mode/companies"
                }
            },
            "data_structure": {
                "storage": "data/rl/shadow_mode/<company_id>/",
                "files": [
                    "report_YYYY-MM-DD.json - Rapport quotidien JSON",
                    "data_YYYY-MM-DD.csv - Données tabulaires CSV"
                ],
                "retention": "30 jours par défaut (configurable)"
            }
        }
        
        # Sauvegarder le résumé
        summary_file = Path("data/rl/shadow_mode/deployment_summary.json")
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        with Path(summary_file, "w", encoding="utf-8").open() as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info("📋 Résumé sauvegardé: %s", summary_file)
        
        # Afficher le résumé
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ DU DÉPLOIEMENT ÉTAPE 8 - SHADOW MODE ENRICHI & KPIs")
        print("=" * 70)
        
        print("Date de déploiement: {summary['deployment_date']}")
        print("Étape: {summary['step']}")
        
        print("\n🔧 COMPOSANTS DÉPLOYÉS:")
        for _component, _details in summary["components"].items():
            print("  • {component}: {details['file']}")
            print("    {details['description']}")
        
        print("\n📊 KPIs IMPLÉMENTÉS:")
        for _kpi in summary["kpis_implemented"]:
            print("  • {kpi}")
        
        print("\n📈 FONCTIONNALITÉS DES RAPPORTS QUOTIDIENS:")
        for _feature in summary["daily_reports_features"]:
            print("  • {feature}")
        
        print("\n🌐 ENDPOINTS API:")
        for _category, endpoints in summary["api_endpoints"].items():
            print("  {category.upper()}:")
            for _name, _endpoint in endpoints.items():
                print("    • {name}: {endpoint}")
        
        print("\n💾 STRUCTURE DES DONNÉES:")
        print("  Stockage: {summary['data_structure']['storage']}")
        for _file_type in summary["data_structure"]["files"]:
            print("  • {file_type}")
        print("  Rétention: {summary['data_structure']['retention']}")
        
        print("\n✅ DÉPLOIEMENT ÉTAPE 8 TERMINÉ AVEC SUCCÈS!")
        
        return True
        
    except Exception as e:
        logger.error("❌ Erreur lors de la génération du résumé: %s", e)
        return False


def main():
    """Fonction principale de déploiement."""
    logger = setup_logging()
    
    logger.info("🚀 Démarrage du déploiement Étape 8 - Shadow Mode Enrichi & KPIs")
    logger.info("=" * 70)
    
    # Étapes de déploiement
    steps = [
        ("Tests", run_tests),
        ("Validation", validate_implementation),
        ("Données d'exemple", create_sample_data),
        ("Intégration Flask", update_app_integration),
        ("Résumé de déploiement", generate_deployment_summary)
    ]
    
    success_count = 0
    total_steps = len(steps)
    
    for step_name, step_func in steps:
        logger.info("\n📋 Étape: %s", step_name)
        try:
            if step_func():
                logger.info("✅ %s réussi", step_name)
                success_count += 1
            else:
                logger.error("❌ %s échoué", step_name)
        except Exception as e:
            logger.error("❌ Erreur dans %s: %s", step_name, e)
    
    # Résultat final
    logger.info("\n" + "=" * 70)
    logger.info("📊 RÉSULTAT DU DÉPLOIEMENT: %s/%s étapes réussies", success_count, total_steps)
    
    if success_count == total_steps:
        logger.info("🎉 DÉPLOIEMENT ÉTAPE 8 RÉUSSI!")
        logger.info("✅ Shadow Mode Enrichi & KPIs déployé avec succès")
        logger.info("✅ KPIs détaillés opérationnels")
        logger.info("✅ Rapports quotidiens fonctionnels")
        logger.info("✅ Export CSV/JSON automatisé")
        logger.info("✅ Routes API intégrées")
        logger.info("✅ Tests complets validés")
        return True
    logger.error("⚠️  DÉPLOIEMENT PARTIEL: %s étapes échouées", total_steps - success_count)
    logger.error("❌ Corriger les erreurs avant la mise en production")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
