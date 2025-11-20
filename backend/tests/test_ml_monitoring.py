# pyright: reportAttributeAccessIssue=false
"""Tests pour le service de monitoring ML."""

import json
from datetime import datetime, timedelta

import pytest


class TestMLMonitoringService:
    """Tests du service de monitoring ML."""

    def test_log_prediction(self, app, sample_booking):
        """Test enregistrement d'une prédiction."""
        from db import db
        from models.ml_prediction import MLPrediction
        from services.ml_monitoring_service import MLMonitoringService

        with app.app_context():
            # Log une prédiction avec un booking réel
            prediction = MLMonitoringService.log_prediction(
                booking_id=sample_booking.id,
                driver_id=0,  # driver_id peut être 0 ou None pour les tests
                predicted_delay=8.5,
                confidence=0.85,
                risk_level="medium",
                contributing_factors={"distance_x_weather": 0.42},
                prediction_time_ms=0.1325,
                request_id="test_123",
                model_version="v1.0",
            )

            assert prediction.id is not None
            assert prediction.booking_id == sample_booking.id
            assert prediction.predicted_delay_minutes == 8.5
            assert prediction.confidence == 0.85

            # Cleanup
            db.session.delete(prediction)
            db.session.commit()

        print("✅ Log prediction OK")

    def test_update_actual_delay(self, app, sample_booking):
        """Test mise à jour retard réel."""
        from db import db
        from services.ml_monitoring_service import MLMonitoringService

        with app.app_context():
            # Log prédiction avec un booking réel
            prediction = MLMonitoringService.log_prediction(
                booking_id=sample_booking.id,
                driver_id=0,  # driver_id peut être 0 ou None pour les tests
                predicted_delay=8.5,
                confidence=0.85,
                risk_level="medium",
                contributing_factors={},
                prediction_time_ms=0.1325,
            )

            # Mettre à jour retard réel
            MLMonitoringService.update_actual_delay(booking_id=sample_booking.id, actual_delay=9.2)

            # Vérifier
            db.session.refresh(prediction)
            assert prediction.actual_delay_minutes == 9.2
            assert prediction.prediction_error == pytest.approx(0.7, 0.01)
            assert prediction.is_accurate is True  # < 3 min

            # Cleanup
            db.session.delete(prediction)
            db.session.commit()

        print("✅ Update actual delay OK")

    def test_get_metrics(self, app, sample_booking, db):
        """Test calcul métriques."""
        from models.booking import Booking
        from models.enums import BookingStatus
        from services.ml_monitoring_service import MLMonitoringService

        with app.app_context():
            # Créer plusieurs bookings pour les tests
            bookings = []
            for i in range(5):
                booking = Booking()
                booking.customer_name = f"Test Customer {i}"
                booking.pickup_location = f"Rue de Test {i}, 1000 Lausanne"
                booking.dropoff_location = f"Rue de Test {i + 1}, 1000 Lausanne"
                booking.pickup_lat = 46.2044
                booking.pickup_lon = 6.1432
                booking.dropoff_lat = 46.2100
                booking.dropoff_lon = 6.1500
                booking.booking_type = "standard"
                booking.amount = 50.0
                booking.status = BookingStatus.PENDING
                booking.user_id = sample_booking.user_id
                booking.client_id = sample_booking.client_id
                booking.company_id = sample_booking.company_id
                booking.duration_seconds = 1800
                booking.distance_meters = 5000
                db.session.add(booking)
                bookings.append(booking)

            db.session.flush()

            # Créer quelques prédictions avec des bookings réels
            predictions = []
            for i, booking in enumerate(bookings):
                p = MLMonitoringService.log_prediction(
                    booking_id=booking.id,
                    driver_id=0,  # driver_id peut être 0 ou None pour les tests
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
            for booking in bookings:
                db.session.delete(booking)
            db.session.commit()

        print(f"✅ Get metrics OK (MAE: {metrics['mae']}, R²: {metrics['r2']})")


class TestMLMonitoringAPI:
    """Tests des routes API monitoring ML."""

    def test_get_metrics(self, client, auth_headers):
        """Test endpoint GET /api/ml-monitoring/metrics."""
        response = client.get("/api/ml-monitoring/metrics?hours=24", headers=auth_headers)

        assert response.status_code == 200
        data = response.get_json()

        assert "count" in data
        assert "mae" in data
        assert "r2" in data

        print("✅ GET /metrics OK (count: {data['count']})")

    def test_get_daily_metrics(self, client, auth_headers):
        """Test endpoint GET /api/ml-monitoring/daily."""
        response = client.get("/api/ml-monitoring/daily?days=7", headers=auth_headers)

        assert response.status_code == 200
        data = response.get_json()

        assert "days" in data
        assert "data" in data
        assert len(data["data"]) <= 7

        print("✅ GET /daily OK ({len(data['data'])} jours)")

    def test_get_summary(self, client, auth_headers):
        """Test endpoint GET /api/ml-monitoring/summary."""
        response = client.get("/api/ml-monitoring/summary", headers=auth_headers)

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
