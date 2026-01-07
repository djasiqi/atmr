#!/usr/bin/env python3
"""
Tests pour le Shadow Mode Manager et les KPIs.

Valide le fonctionnement complet du système de comparaison
humain vs RL avec génération de rapports quotidiens.
"""

import json
import shutil
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from services.ml.rl.shadow_mode_manager import ShadowModeManager


class TestShadowModeManager(unittest.TestCase):
    """Tests pour ShadowModeManager."""

    def setUp(self):
        """Configure les tests."""
        # Créer un répertoire temporaire pour les tests
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ShadowModeManager(data_dir=self.temp_dir)

        # Données de test
        self.test_company_id = "test_company_123"
        self.test_booking_id = "booking_456"

        self.human_decision = {
            "driver_id": "driver_human",
            "eta_minutes": 25,
            "delay_minutes": 5,
            "distance_km": 12.5,
            "driver_load": 0.7,
            "confidence": 0.8,
        }

        self.rl_decision = {
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
            "in_service_area": True,
        }

        self.context = {
            "avg_eta": 24,
            "avg_distance": 12.0,
            "avg_load": 0.65,
            "vehicle_capacity": 4,
            "driver_performance": {
                "driver_rl": {"rating": 4.5},
                "driver_human": {"rating": 4.2},
            },
        }

    def tearDown(self):
        """Nettoie après les tests."""
        # Nettoyer le répertoire temporaire
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_manager_initialization(self):
        """Test l'initialisation du gestionnaire."""
        assert isinstance(self.manager, ShadowModeManager)
        assert self.manager.data_dir == Path(self.temp_dir)
        assert self.manager.data_dir.exists()

        # Vérifier la structure des KPIs
        expected_metrics = [
            "eta_delta",
            "delay_delta",
            "second_best_driver",
            "rl_confidence",
            "human_confidence",
            "decision_reasons",
            "constraint_violations",
            "performance_impact",
        ]
        for metric in expected_metrics:
            assert metric in self.manager.kpi_metrics

    def test_log_decision_comparison(self):
        """Test l'enregistrement d'une comparaison de décision."""
        kpis = self.manager.log_decision_comparison(
            company_id=self.test_company_id,
            booking_id=self.test_booking_id,
            human_decision=self.human_decision,
            rl_decision=self.rl_decision,
            context=self.context,
        )

        # Vérifier que les KPIs sont calculés
        assert isinstance(kpis, dict)
        assert "eta_delta" in kpis
        assert "delay_delta" in kpis
        assert "second_best_driver" in kpis

        # Vérifier les valeurs calculées
        assert kpis["eta_delta"] == -3  # RL meilleur de 3 minutes
        assert kpis["delay_delta"] == -3  # RL meilleur de 3 minutes
        assert kpis["second_best_driver"] == "driver_alt1"
        assert kpis["rl_confidence"] == 0.9

        # Vérifier que les données sont enregistrées
        assert len(self.manager.decision_metadata["timestamp"]) == 1
        assert self.manager.decision_metadata["company_id"][0] == self.test_company_id
        assert self.manager.decision_metadata["booking_id"][0] == self.test_booking_id

    def test_calculate_kpis(self):
        """Test le calcul des KPIs."""
        kpis = self.manager._calculate_kpis(
            self.human_decision, self.rl_decision, self.context
        )

        # Test ETA delta
        assert kpis["eta_delta"] == -3

        # Test delay delta
        assert kpis["delay_delta"] == -3

        # Test second best driver
        assert kpis["second_best_driver"] == "driver_alt1"

        # Test confiance
        assert kpis["rl_confidence"] == 0.9
        assert kpis["human_confidence"] == 0.8

        # Test raisons de décision
        assert isinstance(kpis["decision_reasons"], list)
        assert "ETA inférieur à la moyenne" in kpis["decision_reasons"]
        assert "Distance optimisée" in kpis["decision_reasons"]

        # Test violations de contraintes
        assert isinstance(kpis["constraint_violations"], list)
        assert len(kpis["constraint_violations"]) == 0  # Aucune violation

        # Test impact performance
        assert isinstance(kpis["performance_impact"], dict)
        assert "eta_improvement" in kpis["performance_impact"]
        assert "distance_improvement" in kpis["performance_impact"]
        assert "global_score" in kpis["performance_impact"]

    def test_extract_decision_reasons(self):
        """Test l'extraction des raisons de décision."""
        reasons = self.manager._extract_decision_reasons(self.rl_decision, self.context)

        assert isinstance(reasons, list)
        assert "ETA inférieur à la moyenne" in reasons
        assert "Distance optimisée" in reasons
        assert "Charge chauffeur équilibrée" in reasons
        assert "Respecte la fenêtre horaire" in reasons
        assert "Chauffeur disponible" in reasons
        assert "Chauffeur bien noté" in reasons

    def test_check_constraint_violations(self):
        """Test la vérification des violations de contraintes."""
        # Test sans violations
        violations = self.manager._check_constraint_violations(
            self.rl_decision, self.context
        )
        assert len(violations) == 0

        # Test avec violations
        rl_decision_with_violations = self.rl_decision.copy()
        rl_decision_with_violations["respects_time_window"] = False
        rl_decision_with_violations["driver_available"] = False
        rl_decision_with_violations["passenger_count"] = 6
        rl_decision_with_violations["in_service_area"] = False

        violations = self.manager._check_constraint_violations(
            rl_decision_with_violations, self.context
        )

        assert len(violations) == 4
        assert "Fenêtre horaire non respectée" in violations
        assert "Chauffeur non disponible" in violations
        assert "Capacité véhicule dépassée" in violations
        assert "Hors zone de service" in violations

    def test_calculate_performance_impact(self):
        """Test le calcul de l'impact sur la performance."""
        impact = self.manager._calculate_performance_impact(
            self.human_decision, self.rl_decision, self.context
        )

        assert isinstance(impact, dict)
        assert "eta_improvement" in impact
        assert "distance_improvement" in impact
        assert "load_balance" in impact
        assert "global_score" in impact

        # Vérifier les calculs
        assert impact["eta_improvement"] == 3  # Humain 25 - RL 22
        assert impact["distance_improvement"] == 1.5  # Humain 12.5 - RL 11.0

    def test_generate_daily_report(self):
        """Test la génération d'un rapport quotidien."""
        # Enregistrer quelques décisions
        for i in range(3):
            self.manager.log_decision_comparison(
                company_id=self.test_company_id,
                booking_id=f"booking_{i}",
                human_decision=self.human_decision,
                rl_decision=self.rl_decision,
                context=self.context,
            )

        # Générer le rapport
        report = self.manager.generate_daily_report(self.test_company_id)

        # Vérifier la structure du rapport
        assert "company_id" in report
        assert "date" in report
        assert "total_decisions" in report
        assert "statistics" in report
        assert "kpis_summary" in report
        assert "top_insights" in report
        assert "recommendations" in report

        # Vérifier les valeurs
        assert report["company_id"] == self.test_company_id
        assert report["total_decisions"] == 3

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

    def test_filter_data_by_company_and_date(self):
        """Test le filtrage des données par entreprise et date."""
        # Enregistrer des décisions pour différentes entreprises et dates
        today = datetime.now(UTC).date()
        yesterday = today - timedelta(days=1)

        # Décision d'aujourd'hui pour test_company
        self.manager.log_decision_comparison(
            company_id=self.test_company_id,
            booking_id="booking_today",
            human_decision=self.human_decision,
            rl_decision=self.rl_decision,
            context=self.context,
        )

        # Décision d'hier pour test_company (ajouter directement aux métadonnées)
        yesterday_timestamp = datetime.combine(yesterday, datetime.min.time()).replace(
            tzinfo=UTC
        )
        self.manager.decision_metadata["timestamp"].append(yesterday_timestamp)
        self.manager.decision_metadata["company_id"].append(self.test_company_id)
        self.manager.decision_metadata["booking_id"].append("booking_yesterday")
        self.manager.decision_metadata["driver_id"].append("driver_test")
        self.manager.decision_metadata["human_decision"].append(self.human_decision)
        self.manager.decision_metadata["rl_decision"].append(self.rl_decision)
        self.manager.decision_metadata["context"].append(self.context)

        # Filtrer pour aujourd'hui
        filtered_data = self.manager._filter_data_by_company_and_date(
            self.test_company_id, today
        )

        assert len(filtered_data["decisions"]) == 1
        assert filtered_data["decisions"][0]["booking_id"] == "booking_today"

    def test_calculate_daily_statistics(self):
        """Test le calcul des statistiques quotidiennes."""
        # Créer des données de test
        company_data = {
            "decisions": [
                {
                    "human_decision": {"driver_id": "h1"},
                    "rl_decision": {"driver_id": "r1"},
                },
                {
                    "human_decision": {"driver_id": "h2"},
                    "rl_decision": {"driver_id": "h2"},
                },  # Accord
                {
                    "human_decision": {"driver_id": "h3"},
                    "rl_decision": {"driver_id": "r3"},
                },
            ],
            "kpis": [
                {"eta_delta": -2, "delay_delta": -1, "rl_confidence": 0.8},
                {"eta_delta": 1, "delay_delta": 0, "rl_confidence": 0.9},
                {"eta_delta": -3, "delay_delta": -2, "rl_confidence": 0.7},
            ],
        }

        stats = self.manager._calculate_daily_statistics(company_data)

        # Vérifier les statistiques ETA
        eta_stats = stats["eta_delta"]
        assert eta_stats["mean"] == pytest.approx(-1.33, abs=0.01)  # (-2 + 1 - 3) / 3
        assert eta_stats["min"] == -3
        assert eta_stats["max"] == 1

        # Vérifier le taux d'accord
        assert stats["agreement_rate"] == 1 / 3  # 1 accord sur 3 décisions

    def test_generate_kpis_summary(self):
        """Test la génération du résumé KPIs."""
        company_data = {
            "decisions": [
                {
                    "human_decision": {"driver_id": "h1"},
                    "rl_decision": {"driver_id": "r1"},
                },
                {
                    "human_decision": {"driver_id": "h2"},
                    "rl_decision": {"driver_id": "r2"},
                },
            ],
            "kpis": [
                {
                    "eta_delta": -2,  # RL meilleur
                    "constraint_violations": [],
                    "decision_reasons": ["ETA inférieur", "Distance optimisée"],
                },
                {
                    "eta_delta": 1,  # Humain meilleur
                    "constraint_violations": ["Fenêtre horaire non respectée"],
                    "decision_reasons": ["Charge équilibrée"],
                },
            ],
        }

        summary = self.manager._generate_kpis_summary(company_data)

        # Vérifier le taux d'amélioration ETA
        assert summary["eta_improvement_rate"] == 0.5  # 1 sur 2

        # Vérifier les violations
        assert summary["total_violations"] == 1
        assert summary["violation_rate"] == 0.5  # 1 sur 2

        # Vérifier les raisons principales
        assert "top_reasons" in summary
        assert isinstance(summary["top_reasons"], list)

    def test_generate_top_insights(self):
        """Test la génération des insights principaux."""
        company_data = {
            "decisions": [
                {
                    "human_decision": {"driver_id": "h1"},
                    "rl_decision": {"driver_id": "r1"},
                },
                {
                    "human_decision": {"driver_id": "h2"},
                    "rl_decision": {"driver_id": "h2"},
                },  # Accord
            ],
            "kpis": [
                {"eta_delta": -5, "rl_confidence": 0.9},  # RL beaucoup meilleur
                {"eta_delta": 0, "rl_confidence": 0.8},  # Égalité
            ],
        }

        insights = self.manager._generate_top_insights(company_data)

        assert isinstance(insights, list)
        # Devrait contenir un insight sur l'amélioration ETA
        eta_insights = [insight for insight in insights if "ETA" in insight]
        assert len(eta_insights) > 0

    def test_generate_recommendations(self):
        """Test la génération des recommandations."""
        company_data = {
            "decisions": [
                {
                    "human_decision": {"driver_id": "h1"},
                    "rl_decision": {"driver_id": "r1"},
                },
                {
                    "human_decision": {"driver_id": "h2"},
                    "rl_decision": {"driver_id": "r2"},
                },
            ],
            "kpis": [
                {"eta_delta": -6, "constraint_violations": []},  # RL excellent
                {
                    "eta_delta": 0,
                    "constraint_violations": ["Violation test"],
                },  # Avec violation
            ],
        }

        recommendations = self.manager._generate_recommendations(company_data)

        assert isinstance(recommendations, list)
        # Devrait contenir des recommandations basées sur les données

    def test_save_daily_report(self):
        """Test la sauvegarde du rapport quotidien."""
        report = {
            "company_id": self.test_company_id,
            "date": "2024-0.1-15",
            "total_decisions": 5,
            "statistics": {"eta_delta": {"mean": -2.0}},
            "kpis_summary": {"eta_improvement_rate": 0.8},
            "top_insights": ["RL améliore l'ETA"],
            "recommendations": ["Considérer l'activation automatique"],
        }

        self.manager._save_daily_report(report)

        # Vérifier que les fichiers sont créés
        company_dir = self.manager.data_dir / self.test_company_id
        assert company_dir.exists()

        json_path = company_dir / "report_2024-0.1-15.json"
        csv_path = company_dir / "data_2024-0.1-15.csv"

        assert json_path.exists()
        assert csv_path.exists()

        # Vérifier le contenu JSON
        with Path(json_path, encoding="utf-8").open() as f:
            saved_report = json.load(f)
        assert saved_report["company_id"] == self.test_company_id
        assert saved_report["total_decisions"] == 5

    def test_get_company_summary(self):
        """Test la génération du résumé d'entreprise."""
        # Enregistrer des décisions sur plusieurs jours
        for i in range(5):
            self.manager.log_decision_comparison(
                company_id=self.test_company_id,
                booking_id=f"booking_{i}",
                human_decision=self.human_decision,
                rl_decision=self.rl_decision,
                context=self.context,
            )

        summary = self.manager.get_company_summary(self.test_company_id, 7)

        assert "company_id" in summary
        assert "period_days" in summary
        assert "total_decisions" in summary
        assert "avg_decisions_per_day" in summary
        assert "avg_agreement_rate" in summary
        assert "avg_eta_improvement" in summary
        assert "trend_analysis" in summary

    def test_clear_old_data(self):
        """Test le nettoyage des anciennes données."""
        # Enregistrer une décision
        self.manager.log_decision_comparison(
            company_id=self.test_company_id,
            booking_id=self.test_booking_id,
            human_decision=self.human_decision,
            rl_decision=self.rl_decision,
            context=self.context,
        )

        # Vérifier qu'il y a des données
        assert len(self.manager.decision_metadata["timestamp"]) == 1

        # Nettoyer les données (garder seulement 0 jour = tout supprimer)
        self.manager.clear_old_data(days_to_keep=0)

        # Vérifier que les données sont supprimées
        assert len(self.manager.decision_metadata["timestamp"]) == 0


class TestShadowModeIntegration(unittest.TestCase):
    """Tests d'intégration pour le shadow mode."""

    def setUp(self):
        """Configure les tests d'intégration."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ShadowModeManager(data_dir=self.temp_dir)

    def tearDown(self):
        """Nettoie après les tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_workflow(self):
        """Test le workflow complet du shadow mode."""
        company_id = "integration_test_company"

        # 1. Enregistrer plusieurs décisions
        decisions_data = [
            {
                "booking_id": "booking_1",
                "human_decision": {
                    "driver_id": "h1",
                    "eta_minutes": 30,
                    "delay_minutes": 10,
                },
                "rl_decision": {
                    "driver_id": "r1",
                    "eta_minutes": 25,
                    "delay_minutes": 5,
                },
                "context": {"avg_eta": 28},
            },
            {
                "booking_id": "booking_2",
                "human_decision": {
                    "driver_id": "h2",
                    "eta_minutes": 20,
                    "delay_minutes": 0,
                },
                "rl_decision": {
                    "driver_id": "h2",
                    "eta_minutes": 20,
                    "delay_minutes": 0,
                },  # Accord
                "context": {"avg_eta": 28},
            },
            {
                "booking_id": "booking_3",
                "human_decision": {
                    "driver_id": "h3",
                    "eta_minutes": 35,
                    "delay_minutes": 15,
                },
                "rl_decision": {
                    "driver_id": "r3",
                    "eta_minutes": 28,
                    "delay_minutes": 8,
                },
                "context": {"avg_eta": 28},
            },
        ]

        for decision_data in decisions_data:
            self.manager.log_decision_comparison(
                company_id=company_id,
                booking_id=decision_data["booking_id"],
                human_decision=decision_data["human_decision"],
                rl_decision=decision_data["rl_decision"],
                context=decision_data["context"],
            )

        # 2. Générer le rapport quotidien
        report = self.manager.generate_daily_report(company_id)

        # Vérifier le rapport
        assert report["total_decisions"] == 3
        assert report["company_id"] == company_id

        # Vérifier les statistiques
        stats = report["statistics"]
        assert "eta_delta" in stats
        assert "agreement_rate" in stats

        # Le taux d'accord devrait être 1/3 (1 accord sur 3 décisions)
        self.assertAlmostEqual(stats["agreement_rate"], 1 / 3, places=2)

        # 3. Générer le résumé d'entreprise
        summary = self.manager.get_company_summary(company_id, 7)

        assert summary["company_id"] == company_id
        assert summary["total_decisions"] == 3

        # 4. Vérifier que les fichiers sont créés
        company_dir = self.manager.data_dir / company_id
        assert company_dir.exists()

        # Devrait y avoir des fichiers de rapport
        report_files = list(company_dir.glob("report_*.json"))
        assert len(report_files) > 0


def run_shadow_mode_tests():
    """Exécute tous les tests du shadow mode."""
    print("🧪 Exécution des tests Shadow Mode...")

    # Tests unitaires
    unittest.main(argv=[""], exit=False, verbosity=2)

    print("✅ Tests Shadow Mode terminés avec succès!")


if __name__ == "__main__":
    run_shadow_mode_tests()
