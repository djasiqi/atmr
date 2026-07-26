# pyright: reportAttributeAccessIssue=false
"""Tests pour le service de monitoring ML."""

import pytest


class TestMLMonitoringService:
    """Tests du service de monitoring ML."""

    def test_log_prediction(self, app, sample_booking, db):
        """Test enregistrement d'une prédiction."""
        from services.ml.monitoring import MLMonitoringService

        # NOTE: Le fixture `db` maintient déjà un app context actif pendant le test.
        # Éviter de ré-ouvrir un app_context ici, car cela peut créer une nouvelle
        # session SQLAlchemy (scoped) et rendre les fixtures flushées invisibles,
        # entraînant FK errors ou blocages (locks) lors de flush().
        booking_id = sample_booking.id
        assert booking_id is not None, "Le booking doit avoir un ID"

        prediction = MLMonitoringService.log_prediction(
            booking_id=booking_id,
            driver_id=None,
            predicted_delay=8.5,
            confidence=0.85,
            risk_level="medium",
            contributing_factors={"distance_x_weather": 0.42},
            prediction_time_ms=0.1325,
            request_id="test_123",
            model_version="v1.0",
        )

        db.session.flush()

        assert prediction.id is not None
        assert prediction.booking_id == booking_id
        assert prediction.predicted_delay_minutes == 8.5
        assert prediction.confidence == 0.85

        db.session.delete(prediction)

        print("✅ Log prediction OK")

    def test_update_actual_delay(self, app, sample_booking, db):
        """Test mise à jour retard réel."""
        from services.ml.monitoring import MLMonitoringService

        booking_id = sample_booking.id

        prediction = MLMonitoringService.log_prediction(
            booking_id=booking_id,
            driver_id=None,
            predicted_delay=8.5,
            confidence=0.85,
            risk_level="medium",
            contributing_factors={},
            prediction_time_ms=0.1325,
        )

        db.session.flush()

        MLMonitoringService.update_actual_delay(booking_id=booking_id, actual_delay=9.2)

        db.session.refresh(prediction)
        assert prediction.actual_delay_minutes == 9.2
        assert prediction.prediction_error == pytest.approx(0.7, 0.01)
        assert prediction.is_accurate is True  # < 3 min

        db.session.delete(prediction)
        db.session.flush()

        print("✅ Update actual delay OK")

    def test_get_metrics(self, app, sample_booking, db):
        """Test calcul métriques."""
        from models.booking import Booking
        from models.enums import BookingStatus
        from services.ml.monitoring import MLMonitoringService

        # Le fixture `db` maintient déjà un app context actif (voir note plus haut).
        db.session.flush()
        booking_ref = sample_booking

        bookings: list[Booking] = []
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
            booking.user_id = booking_ref.user_id
            booking.client_id = booking_ref.client_id
            booking.company_id = booking_ref.company_id
            booking.duration_seconds = 1800
            booking.distance_meters = 5000
            db.session.add(booking)
            bookings.append(booking)

        db.session.flush()
        booking_ids = [b.id for b in bookings]
        assert all(bid is not None for bid in booking_ids)

        predictions = []
        for i, booking_id in enumerate(booking_ids):
            p = MLMonitoringService.log_prediction(
                booking_id=int(booking_id),
                driver_id=None,
                predicted_delay=5.0 + i,
                confidence=0.8,
                risk_level="medium",
                contributing_factors={},
                prediction_time_ms=0.1300,
            )
            p.actual_delay_minutes = 5.5 + i
            p.prediction_error = 0.5
            p.is_accurate = True
            predictions.append(p)

        db.session.flush()

        metrics = MLMonitoringService.get_metrics(hours=24)

        assert metrics["count"] >= 5
        assert metrics["mae"] is not None
        assert metrics["r2"] is not None

        for p in predictions:
            db.session.delete(p)
        for booking in bookings:
            db.session.delete(booking)
        db.session.flush()

        print(f"✅ Get metrics OK (MAE: {metrics['mae']}, R²: {metrics['r2']})")


class TestMLMonitoringAPI:
    """Tests des routes API monitoring ML (ADMIN requis — F-05)."""

    def test_company_forbidden(self, client, auth_headers):
        response = client.get(
            "/api/ml-monitoring/metrics?hours=24", headers=auth_headers
        )
        assert response.status_code == 403

    def test_get_metrics(self, client, admin_headers):
        """Test endpoint GET /api/ml-monitoring/metrics."""
        response = client.get(
            "/api/ml-monitoring/metrics?hours=24", headers=admin_headers
        )
        assert response.status_code == 200
        data = response.get_json()
        assert "count" in data
        assert "mae" in data
        assert "r2" in data

    def test_get_daily_metrics(self, client, admin_headers):
        """Test endpoint GET /api/ml-monitoring/daily."""
        response = client.get("/api/ml-monitoring/daily?days=7", headers=admin_headers)
        assert response.status_code == 200
        data = response.get_json()
        assert "days" in data
        assert "data" in data
        assert len(data["data"]) <= 7

    def test_get_summary(self, client, admin_headers):
        """Test endpoint GET /api/ml-monitoring/summary."""
        response = client.get("/api/ml-monitoring/summary", headers=admin_headers)
        assert response.status_code == 200
        data = response.get_json()
        assert "metrics_24h" in data
        assert "feature_flags" in data
        assert "total_predictions" in data


if __name__ == "__main__":
    """Exécution directe pour tests rapides."""
    print("\n" + "=" * 70)
    print("🧪 TESTS ML MONITORING")
    print("=" * 70)

    print("\nℹ️  Tests nécessitent Flask app context")
    print("   Utiliser: pytest tests/test_ml_monitoring.py")
