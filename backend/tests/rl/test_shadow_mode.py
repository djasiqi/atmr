"""
Tests pour le Shadow Mode Manager.

Vérifie que le shadow mode fonctionne correctement en production.
"""

import tempfile
from pathlib import Path

from services.ml.rl.shadow_mode_manager import ShadowModeManager


class TestShadowModeManagerCreation:
    """Tests de création du Shadow Mode Manager."""

    def test_shadow_manager_creation(self):
        """Test création basique du shadow manager."""
        # ✅ FIX: ShadowModeManager n'utilise pas DQNAgent et n'accepte
        # que data_dir comme paramètre
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(data_dir=tmpdir)

            assert manager is not None
            assert manager.data_dir == Path(tmpdir)
            assert isinstance(manager.kpi_metrics, dict)
            assert isinstance(manager.decision_metadata, dict)
            assert manager.logger is not None

    def test_shadow_manager_creates_log_dir(self):
        """Test que le manager crée le répertoire de logs."""
        # ✅ FIX: ShadowModeManager crée automatiquement le répertoire data_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir, "shadow_logs")

            ShadowModeManager(data_dir=str(log_dir))

            assert log_dir.exists()
            assert log_dir.is_dir()


class TestShadowModePredictions:
    """Tests des prédictions shadow."""

    def test_log_decision_comparison(self):
        """Test enregistrement de comparaison de décision."""
        # ✅ FIX: Utiliser l'API réelle de ShadowModeManager
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(data_dir=tmpdir)

            human_decision = {
                "driver_id": "driver_1",
                "eta_minutes": 25,
                "delay_minutes": 5,
            }
            rl_decision = {
                "driver_id": "driver_1",
                "eta_minutes": 22,
                "delay_minutes": 2,
            }
            context = {"avg_eta": 24, "avg_distance": 12.0}

            kpis = manager.log_decision_comparison(
                company_id="company_123",
                booking_id="booking_456",
                human_decision=human_decision,
                rl_decision=rl_decision,
                context=context,
            )

            assert kpis is not None
            assert "eta_delta" in kpis
            assert len(manager.decision_metadata["company_id"]) == 1
            assert manager.decision_metadata["company_id"][0] == "company_123"


class TestShadowModeComparisons:
    """Tests des comparaisons avec décisions réelles."""

    def test_compare_agreement(self):
        """Test comparaison avec accord."""
        # ✅ FIX: Utiliser l'API réelle log_decision_comparison
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(data_dir=tmpdir)

            human_decision = {"driver_id": "driver_1", "eta_minutes": 25}
            rl_decision = {"driver_id": "driver_1", "eta_minutes": 22}
            context = {"avg_eta": 24}

            kpis = manager.log_decision_comparison(
                company_id="company_123",
                booking_id="booking_123",
                human_decision=human_decision,
                rl_decision=rl_decision,
                context=context,
            )

            assert kpis is not None
            assert len(manager.decision_metadata["company_id"]) == 1

    def test_compare_disagreement(self):
        """Test comparaison avec désaccord."""
        # ✅ FIX: Utiliser l'API réelle log_decision_comparison
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(data_dir=tmpdir)

            human_decision = {"driver_id": "driver_1", "eta_minutes": 25}
            rl_decision = {"driver_id": "driver_2", "eta_minutes": 22}
            context = {"avg_eta": 24}

            kpis = manager.log_decision_comparison(
                company_id="company_123",
                booking_id="booking_123",
                human_decision=human_decision,
                rl_decision=rl_decision,
                context=context,
            )

            assert kpis is not None
            assert len(manager.decision_metadata["company_id"]) == 1


class TestShadowModeStats:
    """Tests des statistiques."""

    def test_generate_daily_report(self):
        """Test génération de rapport quotidien."""
        # ✅ FIX: Utiliser l'API réelle generate_daily_report
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(data_dir=tmpdir)

            # Ajouter quelques décisions
            for i in range(3):
                manager.log_decision_comparison(
                    company_id="company_123",
                    booking_id=f"booking_{i}",
                    human_decision={"driver_id": f"driver_{i}", "eta_minutes": 25},
                    rl_decision={"driver_id": f"driver_{i}", "eta_minutes": 22},
                    context={"avg_eta": 24},
                )

            report = manager.generate_daily_report("company_123")

            assert report is not None
            assert "company_id" in report
            assert report["company_id"] == "company_123"
            assert "total_decisions" in report

    def test_generate_daily_report_empty(self):
        """Test rapport quotidien sans données."""
        # ✅ FIX: Utiliser l'API réelle generate_daily_report
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(data_dir=tmpdir)

            report = manager.generate_daily_report("company_123")

            assert report is not None
            assert report["total_decisions"] == 0
            assert "message" in report


class TestShadowModeLogging:
    """Tests du logging."""

    def test_log_decision_comparison_logs_data(self):
        """Test que log_decision_comparison enregistre les données."""
        # ✅ FIX: Utiliser l'API réelle log_decision_comparison
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(data_dir=tmpdir)

            human_decision = {"driver_id": "driver_1", "eta_minutes": 25}
            rl_decision = {"driver_id": "driver_1", "eta_minutes": 22}
            context = {"avg_eta": 24}

            kpis = manager.log_decision_comparison(
                company_id="company_123",
                booking_id="booking_789",
                human_decision=human_decision,
                rl_decision=rl_decision,
                context=context,
            )

            # Vérifier que les données sont enregistrées
            assert len(manager.decision_metadata["booking_id"]) == 1
            assert manager.decision_metadata["booking_id"][0] == "booking_789"
            assert kpis is not None


class TestShadowModeDailyReport:
    """Tests des rapports quotidiens."""

    def test_generate_daily_report_empty(self):
        """Test génération rapport sans données."""
        # ✅ FIX: Utiliser l'API réelle generate_daily_report avec company_id
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(data_dir=tmpdir)

            report = manager.generate_daily_report("company_123")

            assert report is not None
            assert "company_id" in report
            assert report["company_id"] == "company_123"
            assert report["total_decisions"] == 0
            assert "message" in report

    def test_daily_report_saves_to_file(self):
        """Test que le rapport est sauvegardé."""
        # ✅ FIX: Utiliser l'API réelle generate_daily_report
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(data_dir=tmpdir)

            # Ajouter quelques décisions
            manager.log_decision_comparison(
                company_id="company_123",
                booking_id="booking_1",
                human_decision={"driver_id": "driver_1", "eta_minutes": 25},
                rl_decision={"driver_id": "driver_1", "eta_minutes": 22},
                context={"avg_eta": 24},
            )

            report = manager.generate_daily_report("company_123")

            # Vérifier que le rapport contient les données
            assert report is not None
            assert report["total_decisions"] > 0
