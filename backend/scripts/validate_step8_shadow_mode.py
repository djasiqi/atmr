#!/usr/bin/env python3
from pathlib import Path

"""Script de validation pour l'Étape 8 - Shadow Mode Enrichi & KPIs.

Valide l'implémentation complète du système de comparaison
humain vs RL avec génération de rapports quotidiens.
"""

import json
import sys
import tempfile

from services.rl.shadow_mode_manager import ShadowModeManager


class Step8ValidationSuite:
    """Suite de validation pour l'Étape 8."""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ShadowModeManager(data_dir=self.temp_dir)
        self.results = {}

    def validate_shadow_mode_manager(self):
        """Valide le ShadowModeManager."""
        print("\n🧪 Validation du ShadowModeManager...")
        
        try:
            # Test initialisation
            assert isinstance(self.manager, ShadowModeManager)
            assert self.manager.data_dir.exists()
            
            # Test structure KPIs
            expected_metrics = [
                "eta_delta", "delay_delta", "second_best_driver",
                "rl_confidence", "human_confidence", "decision_reasons",
                "constraint_violations", "performance_impact"
            ]
            for metric in expected_metrics:
                assert metric in self.manager.kpi_metrics
            
            print("   ✅ ShadowModeManager initialisé correctement")
            self.results["shadow_mode_manager"] = True
            
        except Exception:
            print("   ❌ Erreur ShadowModeManager: {e}")
            self.results["shadow_mode_manager"] = False

    def validate_kpi_calculation(self):
        """Valide le calcul des KPIs."""
        print("\n🧪 Validation du calcul des KPIs...")
        
        try:
            # Données de test
            human_decision = {
                "driver_id": "driver_human",
                "eta_minutes": 25,
                "delay_minutes": 5,
                "distance_km": 12.5,
                "driver_load": 0.7,
                "confidence": 0.8
            }
            
            rl_decision = {
                "driver_id": "driver_rl",
                "eta_minutes": 22,
                "delay_minutes": 2,
                "distance_km": 11.0,
                "driver_load": 0.6,
                "confidence": 0.9,
                "alternative_drivers": ["driver_rl", "driver_alt1", "driver_alt2"],
                "respects_time_window": True,
                "driver_available": True,
                "passenger_count": 2,
                "in_service_area": True
            }
            
            context = {
                "avg_eta": 24,
                "avg_distance": 12.0,
                "avg_load": 0.65,
                "vehicle_capacity": 4,
                "driver_performance": {
                    "driver_rl": {"rating": 4.5}
                }
            }
            
            # Calculer les KPIs
            kpis = self.manager._calculate_kpis(human_decision, rl_decision, context)
            
            # Vérifier les KPIs calculés
            assert kpis["eta_delta"] == -3  # RL meilleur de 3 minutes
            assert kpis["delay_delta"] == -3  # RL meilleur de 3 minutes
            assert kpis["second_best_driver"] == "driver_alt1"
            assert kpis["rl_confidence"] == 0.9
            assert kpis["human_confidence"] == 0.8
            assert isinstance(kpis["decision_reasons"], list)
            assert isinstance(kpis["constraint_violations"], list)
            assert isinstance(kpis["performance_impact"], dict)
            
            print("   ✅ Calcul des KPIs validé")
            self.results["kpi_calculation"] = True
            
        except Exception:
            print("   ❌ Erreur calcul KPIs: {e}")
            self.results["kpi_calculation"] = False

    def validate_decision_logging(self):
        """Valide l'enregistrement des décisions."""
        print("\n🧪 Validation de l'enregistrement des décisions...")
        
        try:
            company_id = "test_company_123"
            booking_id = "booking_456"
            
            human_decision = {
                "driver_id": "driver_human",
                "eta_minutes": 25,
                "delay_minutes": 5,
                "distance_km": 12.5,
                "driver_load": 0.7,
                "confidence": 0.8
            }
            
            rl_decision = {
                "driver_id": "driver_rl",
                "eta_minutes": 22,
                "delay_minutes": 2,
                "distance_km": 11.0,
                "driver_load": 0.6,
                "confidence": 0.9,
                "alternative_drivers": ["driver_rl", "driver_alt1", "driver_alt2"],
                "respects_time_window": True,
                "driver_available": True,
                "passenger_count": 2,
                "in_service_area": True
            }
            
            context = {
                "avg_eta": 24,
                "avg_distance": 12.0,
                "avg_load": 0.65,
                "vehicle_capacity": 4,
                "driver_performance": {
                    "driver_rl": {"rating": 4.5}
                }
            }
            
            # Enregistrer la décision
            kpis = self.manager.log_decision_comparison(
                company_id=company_id,
                booking_id=booking_id,
                human_decision=human_decision,
                rl_decision=rl_decision,
                context=context
            )
            
            # Vérifier l'enregistrement
            assert len(self.manager.decision_metadata["timestamp"]) == 1
            assert self.manager.decision_metadata["company_id"][0] == company_id
            assert self.manager.decision_metadata["booking_id"][0] == booking_id
            
            # Vérifier les KPIs retournés
            assert isinstance(kpis, dict)
            assert "eta_delta" in kpis
            
            print("   ✅ Enregistrement des décisions validé")
            self.results["decision_logging"] = True
            
        except Exception:
            print("   ❌ Erreur enregistrement décisions: {e}")
            self.results["decision_logging"] = False

    def validate_daily_report_generation(self):
        """Valide la génération des rapports quotidiens."""
        print("\n🧪 Validation de la génération des rapports quotidiens...")
        
        try:
            company_id = "test_company_reports"
            
            # Enregistrer plusieurs décisions
            for i in range(5):
                human_decision = {
                    "driver_id": f"driver_human_{i}",
                    "eta_minutes": 25 + i,
                    "delay_minutes": 5 + i,
                    "distance_km": 12.5 + i,
                    "driver_load": 0.7,
                    "confidence": 0.8
                }
                
                rl_decision = {
                    "driver_id": f"driver_rl_{i}",
                    "eta_minutes": 22 + i,
                    "delay_minutes": 2 + i,
                    "distance_km": 11.0 + i,
                    "driver_load": 0.6,
                    "confidence": 0.9,
                    "alternative_drivers": [f"driver_rl_{i}", f"driver_alt_{i}"],
                    "respects_time_window": True,
                    "driver_available": True,
                    "passenger_count": 2,
                    "in_service_area": True
                }
                
                context = {
                    "avg_eta": 24,
                    "avg_distance": 12.0,
                    "avg_load": 0.65,
                    "vehicle_capacity": 4,
                    "driver_performance": {
                        f"driver_rl_{i}": {"rating": 4.5}
                    }
                }
                
                self.manager.log_decision_comparison(
                    company_id=company_id,
                    booking_id=f"booking_{i}",
                    human_decision=human_decision,
                    rl_decision=rl_decision,
                    context=context
                )
            
            # Générer le rapport quotidien
            report = self.manager.generate_daily_report(company_id)
            
            # Vérifier la structure du rapport
            required_fields = [
                "company_id", "date", "total_decisions",
                "statistics", "kpis_summary", "top_insights", "recommendations"
            ]
            for field in required_fields:
                assert field in report
            
            # Vérifier les valeurs
            assert report["company_id"] == company_id
            assert report["total_decisions"] == 5
            
            # Vérifier les statistiques
            stats = report["statistics"]
            assert "eta_delta" in stats
            assert "delay_delta" in stats
            assert "rl_confidence" in stats
            assert "agreement_rate" in stats
            
            # Vérifier le résumé KPIs
            kpis_summary = report["kpis_summary"]
            assert "eta_improvement_rate" in kpis_summary
            assert "total_violations" in kpis_summary
            assert "top_reasons" in kpis_summary
            
            print("   ✅ Génération des rapports quotidiens validée")
            self.results["daily_report_generation"] = True
            
        except Exception:
            print("   ❌ Erreur génération rapports: {e}")
            self.results["daily_report_generation"] = False

    def validate_file_export(self):
        """Valide l'export des fichiers."""
        print("\n🧪 Validation de l'export des fichiers...")
        
        try:
            company_id = "test_company_export"
            
            # Enregistrer une décision
            human_decision = {
                "driver_id": "driver_human",
                "eta_minutes": 25,
                "delay_minutes": 5,
                "distance_km": 12.5,
                "driver_load": 0.7,
                "confidence": 0.8
            }
            
            rl_decision = {
                "driver_id": "driver_rl",
                "eta_minutes": 22,
                "delay_minutes": 2,
                "distance_km": 11.0,
                "driver_load": 0.6,
                "confidence": 0.9,
                "alternative_drivers": ["driver_rl", "driver_alt1"],
                "respects_time_window": True,
                "driver_available": True,
                "passenger_count": 2,
                "in_service_area": True
            }
            
            context = {
                "avg_eta": 24,
                "avg_distance": 12.0,
                "avg_load": 0.65,
                "vehicle_capacity": 4,
                "driver_performance": {
                    "driver_rl": {"rating": 4.5}
                }
            }
            
            self.manager.log_decision_comparison(
                company_id=company_id,
                booking_id="booking_export",
                human_decision=human_decision,
                rl_decision=rl_decision,
                context=context
            )
            
            # Générer le rapport
            report = self.manager.generate_daily_report(company_id)
            
            # Vérifier que les fichiers sont créés
            company_dir = self.manager.data_dir / company_id
            assert company_dir.exists()
            
            # Vérifier les fichiers JSON et CSV
            date_str = report["date"]
            json_path = company_dir / f"report_{date_str}.json"
            csv_path = company_dir / f"data_{date_str}.csv"
            
            assert json_path.exists()
            assert csv_path.exists()
            
            # Vérifier le contenu JSON
            with Path(json_path, encoding="utf-8").open() as f:
                saved_report = json.load(f)
            assert saved_report["company_id"] == company_id
            assert saved_report["total_decisions"] == 1
            
            # Vérifier le contenu CSV
            import pandas as pd
            df = pd.read_csv(csv_path)
            assert len(df) == 1
            assert df["company_id"].iloc[0] == company_id
            
            print("   ✅ Export des fichiers validé")
            self.results["file_export"] = True
            
        except Exception:
            print("   ❌ Erreur export fichiers: {e}")
            self.results["file_export"] = False

    def validate_company_summary(self):
        """Valide la génération du résumé d'entreprise."""
        print("\n🧪 Validation du résumé d'entreprise...")
        
        try:
            company_id = "test_company_summary"
            
            # Enregistrer des décisions sur plusieurs jours simulés
            for i in range(3):
                human_decision = {
                    "driver_id": f"driver_human_{i}",
                    "eta_minutes": 25 + i,
                    "delay_minutes": 5 + i,
                    "distance_km": 12.5 + i,
                    "driver_load": 0.7,
                    "confidence": 0.8
                }
                
                rl_decision = {
                    "driver_id": f"driver_rl_{i}",
                    "eta_minutes": 22 + i,
                    "delay_minutes": 2 + i,
                    "distance_km": 11.0 + i,
                    "driver_load": 0.6,
                    "confidence": 0.9,
                    "alternative_drivers": [f"driver_rl_{i}", f"driver_alt_{i}"],
                    "respects_time_window": True,
                    "driver_available": True,
                    "passenger_count": 2,
                    "in_service_area": True
                }
                
                context = {
                    "avg_eta": 24,
                    "avg_distance": 12.0,
                    "avg_load": 0.65,
                    "vehicle_capacity": 4,
                    "driver_performance": {
                        f"driver_rl_{i}": {"rating": 4.5}
                    }
                }
                
                self.manager.log_decision_comparison(
                    company_id=company_id,
                    booking_id=f"booking_{i}",
                    human_decision=human_decision,
                    rl_decision=rl_decision,
                    context=context
                )
            
            # Générer le résumé
            summary = self.manager.get_company_summary(company_id, 7)
            
            # Vérifier la structure du résumé
            required_fields = [
                "company_id", "period_days", "total_decisions",
                "avg_decisions_per_day", "avg_agreement_rate",
                "avg_eta_improvement", "trend_analysis"
            ]
            for field in required_fields:
                assert field in summary
            
            # Vérifier les valeurs
            assert summary["company_id"] == company_id
            assert summary["period_days"] == 7
            assert summary["total_decisions"] == 3
            
            print("   ✅ Résumé d'entreprise validé")
            self.results["company_summary"] = True
            
        except Exception:
            print("   ❌ Erreur résumé entreprise: {e}")
            self.results["company_summary"] = False

    def validate_routes_integration(self):
        """Valide l'intégration des routes."""
        print("\n🧪 Validation de l'intégration des routes...")
        
        try:
            # Importer les routes
            from routes.shadow_mode_routes import register_shadow_mode_routes, shadow_mode_bp
            
            # Vérifier que le blueprint est créé
            assert shadow_mode_bp is not None
            assert shadow_mode_bp.name == "shadow_mode"
            assert shadow_mode_bp.url_prefix == "/api/shadow-mode"
            
            # Vérifier que la fonction d'enregistrement existe
            assert callable(register_shadow_mode_routes)
            
            print("   ✅ Intégration des routes validée")
            self.results["routes_integration"] = True
            
        except Exception:
            print("   ❌ Erreur intégration routes: {e}")
            self.results["routes_integration"] = False

    def validate_kpis_completeness(self):
        """Valide la complétude des KPIs."""
        print("\n🧪 Validation de la complétude des KPIs...")
        
        try:
            # Vérifier que tous les KPIs requis sont présents
            required_kpis = [
                "eta_delta",           # Différence ETA humain vs RL
                "delay_delta",         # Différence retard humain vs RL
                "second_best_driver",  # Second meilleur driver suggéré
                "rl_confidence",      # Confiance RL dans la décision
                "human_confidence",    # Confiance humaine (si disponible)
                "decision_reasons",    # Raisons de la décision RL
                "constraint_violations", # Violations de contraintes
                "performance_impact"   # Impact sur performance globale
            ]
            
            for kpi in required_kpis:
                assert kpi in self.manager.kpi_metrics
            
            # Vérifier que les KPIs sont calculés correctement
            human_decision = {
                "driver_id": "driver_human",
                "eta_minutes": 25,
                "delay_minutes": 5,
                "distance_km": 12.5,
                "driver_load": 0.7,
                "confidence": 0.8
            }
            
            rl_decision = {
                "driver_id": "driver_rl",
                "eta_minutes": 22,
                "delay_minutes": 2,
                "distance_km": 11.0,
                "driver_load": 0.6,
                "confidence": 0.9,
                "alternative_drivers": ["driver_rl", "driver_alt1", "driver_alt2"],
                "respects_time_window": True,
                "driver_available": True,
                "passenger_count": 2,
                "in_service_area": True
            }
            
            context = {
                "avg_eta": 24,
                "avg_distance": 12.0,
                "avg_load": 0.65,
                "vehicle_capacity": 4,
                "driver_performance": {
                    "driver_rl": {"rating": 4.5}
                }
            }
            
            kpis = self.manager._calculate_kpis(human_decision, rl_decision, context)
            
            # Vérifier que tous les KPIs sont calculés
            for kpi in required_kpis:
                assert kpi in kpis
            
            # Vérifier les types de données
            assert isinstance(kpis["eta_delta"], (int, float))
            assert isinstance(kpis["delay_delta"], (int, float))
            assert isinstance(kpis["decision_reasons"], list)
            assert isinstance(kpis["constraint_violations"], list)
            assert isinstance(kpis["performance_impact"], dict)
            
            print("   ✅ Complétude des KPIs validée")
            self.results["kpis_completeness"] = True
            
        except Exception:
            print("   ❌ Erreur complétude KPIs: {e}")
            self.results["kpis_completeness"] = False

    def run_all_validations(self):
        """Exécute toutes les validations."""
        print("🚀 Démarrage de la validation Étape 8 - Shadow Mode Enrichi & KPIs")
        print("=" * 70)
        
        validations = [
            ("ShadowModeManager", self.validate_shadow_mode_manager),
            ("Calcul des KPIs", self.validate_kpi_calculation),
            ("Enregistrement des décisions", self.validate_decision_logging),
            ("Génération rapports quotidiens", self.validate_daily_report_generation),
            ("Export des fichiers", self.validate_file_export),
            ("Résumé d'entreprise", self.validate_company_summary),
            ("Intégration des routes", self.validate_routes_integration),
            ("Complétude des KPIs", self.validate_kpis_completeness),
        ]
        
        for name, validation_func in validations:
            try:
                validation_func()
            except Exception:
                print("❌ Erreur dans {name}: {e}")
                self.results[name.lower().replace(" ", "_")] = False

    def generate_report(self):
        """Génère un rapport de validation."""
        print("\n" + "=" * 70)
        print("📊 RAPPORT DE VALIDATION ÉTAPE 8 - SHADOW MODE ENRICHI & KPIs")
        print("=" * 70)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        
        print("Tests réussis: {passed_tests}/{total_tests}")
        
        # Détails par test
        for _test_name, _result in self.results.items():
            print("  {test_name}: {status}")
        
        # Recommandations
        print("\n🎯 RECOMMANDATIONS:")
        
        if self.results.get("shadow_mode_manager", False):
            print("  ✅ ShadowModeManager fonctionnel")
        else:
            print("  ❌ Corriger ShadowModeManager")
        
        if self.results.get("kpi_calculation", False):
            print("  ✅ Calcul des KPIs validé")
        else:
            print("  ❌ Corriger le calcul des KPIs")
        
        if self.results.get("decision_logging", False):
            print("  ✅ Enregistrement des décisions validé")
        else:
            print("  ❌ Corriger l'enregistrement des décisions")
        
        if self.results.get("daily_report_generation", False):
            print("  ✅ Génération des rapports quotidiens validée")
        else:
            print("  ❌ Corriger la génération des rapports")
        
        if self.results.get("file_export", False):
            print("  ✅ Export des fichiers validé")
        else:
            print("  ❌ Corriger l'export des fichiers")
        
        if self.results.get("company_summary", False):
            print("  ✅ Résumé d'entreprise validé")
        else:
            print("  ❌ Corriger le résumé d'entreprise")
        
        if self.results.get("routes_integration", False):
            print("  ✅ Intégration des routes validée")
        else:
            print("  ❌ Corriger l'intégration des routes")
        
        if self.results.get("kpis_completeness", False):
            print("  ✅ Complétude des KPIs validée")
        else:
            print("  ❌ Corriger la complétude des KPIs")
        
        # Conclusion
        if passed_tests == total_tests:
            print("\n🎉 VALIDATION COMPLÈTE RÉUSSIE!")
            print("✅ L'Étape 8 - Shadow Mode Enrichi & KPIs est prête")
            print("✅ KPIs détaillés implémentés")
            print("✅ Rapports quotidiens fonctionnels")
            print("✅ Export CSV/JSON automatisé")
            print("✅ Routes API intégrées")
            print("✅ Tests complets validés")
        else:
            print("\n⚠️  {total_tests - passed_tests} tests ont échoué")
            print("❌ Corriger les erreurs avant le déploiement")
        
        return passed_tests == total_tests


def main():
    """Fonction principale."""
    # Créer la suite de validation
    validator = Step8ValidationSuite()
    
    # Exécuter toutes les validations
    validator.run_all_validations()
    
    # Générer le rapport
    return validator.generate_report()
    


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
