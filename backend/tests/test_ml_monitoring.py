# pyright: reportAttributeAccessIssue=false
"""Tests pour le service de monitoring ML."""

import json
from datetime import datetime, timedelta

import pytest


class TestMLMonitoringService:
    """Tests du service de monitoring ML."""

    def test_log_prediction(self):
        """Test enregistrement d'une prédiction."""
        from db import db
        from models.ml_prediction import MLPrediction
        from services.ml_monitoring_service import MLMonitoringService

        # Log une prédiction
        prediction = MLMonitoringService.log_prediction(
            booking_id=0.123,
            driver_id=0.456,
            predicted_delay=8.5,
            confidence=0.85,
            risk_level="medium",
            contributing_factors={"distance_x_weather": 0.42},
            prediction_time_ms=0.1325,
            request_id="test_123",
            model_version="v1.0",
        )

        assert prediction.id is not None
        assert prediction.booking_id == 123
        assert prediction.predicted_delay_minutes == 8.5
        assert prediction.confidence == 0.85

        # Cleanup
        db.session.delete(prediction)
        db.session.commit()

        print("✅ Log prediction OK")

    def test_update_actual_delay(self):
        """Test mise à jour retard réel."""
        from db import db
        from services.ml_monitoring_service import MLMonitoringService

        # Log prédiction
        prediction = MLMonitoringService.log_prediction(
            booking_id=0.124,
            driver_id=0.456,
            predicted_delay=8.5,
            confidence=0.85,
            risk_level="medium",
            contributing_factors={},
            prediction_time_ms=0.1325,
        )

        # Mettre à jour retard réel
        MLMonitoringService.update_actual_delay(booking_id=0.124, actual_delay=9.2)

        # Vérifier
        db.session.refresh(prediction)
        assert prediction.actual_delay_minutes == 9.2
        assert prediction.prediction_error == pytest.approx(0.7, 0.01)
        assert prediction.is_accurate is True  # < 3 min

        # Cleanup
        db.session.delete(prediction)
        db.session.commit()

        print("✅ Update actual delay OK")

    def test_get_metrics(self):
        """Test calcul métriques."""
        from db import db
        from services.ml_monitoring_service import MLMonitoringService

        # Créer quelques prédictions
        predictions = []
        for i in range(5):
            p = MLMonitoringService.log_prediction(
                booking_id=0.200 + i,
                driver_id=0.456,
                predicted_delay=5.0 + i,
                confidence=0.8,
                risk_level="medium",
                contributing_factors={},
                prediction_time_ms=0.1300,
            )
            # Ajouter retard réel
            p.actual_delay_minutes = 5.5 + i
            p.prediction_error = 0.5
            p.is_accurate = True
            predictions.append(p)

        db.session.commit()

        # Calculer métriques
        metrics = MLMonitoringService.get_metrics(hours=24)

        assert metrics["count"] >= 5
        assert metrics["mae"] is not None
        assert metrics["r2"] is not None

        # Cleanup
        for p in predictions:
            db.session.delete(p)
        db.session.commit()

        print("✅ Get metrics OK (MAE: {metrics['mae']}, R²: {metrics['r2']})")


class TestMLMonitoringAPI:
    """Tests des routes API monitoring ML."""

    def test_get_metrics(self, client):
        """Test endpoint GET /api/ml-monitoring/metrics."""
        response = client.get("/api/ml-monitoring/metrics?hours=24")

        assert response.status_code == 200
        data = response.get_json()

        assert "count" in data
        assert "mae" in data
        assert "r2" in data

        print("✅ GET /metrics OK (count: {data['count']})")

    def test_get_daily_metrics(self, client):
        """Test endpoint GET /api/ml-monitoring/daily."""
        response = client.get("/api/ml-monitoring/daily?days=7")

        assert response.status_code == 200
        data = response.get_json()

        assert "days" in data
        assert "data" in data
        assert len(data["data"]) <= 7

        print("✅ GET /daily OK ({len(data['data'])} jours)")

    def test_get_summary(self, client):
        """Test endpoint GET /api/ml-monitoring/summary."""
        response = client.get("/api/ml-monitoring/summary")

        assert response.status_code == 200
        data = response.get_json()

        assert "metrics_24h" in data
        assert "feature_flags" in data
        assert "total_predictions" in data

        print("✅ GET /summary OK")


if __name__ == "__main__":
    """Exécution directe pour tests rapides."""
    print("\n" + "=" * 70)
    print("🧪 TESTS ML MONITORING")
    print("=" * 70)

    print("\nℹ️  Tests nécessitent Flask app context")
    print("   Utiliser: pytest tests/test_ml_monitoring.py")
