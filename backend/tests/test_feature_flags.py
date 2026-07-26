"""Tests pour le système de feature flags ML."""


class TestFeatureFlags:
    """Tests du système de feature flags."""

    def test_default_configuration(self):
        """Test configuration par défaut."""
        from feature_flags import FeatureFlags

        # Reset stats
        FeatureFlags.reset_stats()

        # Vérifier fallback activé par défaut
        assert FeatureFlags.should_fallback_on_error() is True

        print("✅ Configuration par défaut OK")

    def test_enable_disable_ml(self):
        """Test activation/désactivation ML."""
        from feature_flags import FeatureFlags

        FeatureFlags.reset_stats()

        # Activer ML
        FeatureFlags.set_ml_enabled(True)
        FeatureFlags.set_ml_traffic_percentage(100)

        # Vérifier activation
        enabled = FeatureFlags.is_ml_enabled(request_id="test_1")
        assert enabled is True

        # Désactiver ML
        FeatureFlags.set_ml_enabled(False)

        # Vérifier désactivation
        enabled = FeatureFlags.is_ml_enabled(request_id="test_2")
        assert enabled is False

        print("✅ Activation/désactivation OK")

    def test_traffic_percentage(self):
        """Test pourcentage de trafic."""
        from feature_flags import FeatureFlags

        FeatureFlags.reset_stats()
        FeatureFlags.set_ml_enabled(True)
        FeatureFlags.set_ml_traffic_percentage(50)

        # ✅ FIX: Utiliser should_use_ml() au lieu de is_ml_enabled()
        # car should_use_ml() prend en compte le pourcentage de trafic
        # Tester sur 100 requêtes
        enabled_count = 0
        for _ in range(100):
            if FeatureFlags.should_use_ml():
                enabled_count += 1

        # Vérifier proportion (avec tolérance)
        assert 30 <= enabled_count <= 70  # ~50% ±20%

        print(f"✅ Trafic percentage OK ({enabled_count}% activé sur 100 requêtes)")

    def test_stats_recording(self):
        """Test enregistrement statistiques."""
        from feature_flags import FeatureFlags

        FeatureFlags.reset_stats()
        FeatureFlags.set_ml_enabled(True)
        FeatureFlags.set_ml_traffic_percentage(100)

        # Simuler requêtes
        FeatureFlags.is_ml_enabled(request_id="test_1")
        FeatureFlags.record_ml_success()

        FeatureFlags.is_ml_enabled(request_id="test_2")
        FeatureFlags.record_ml_success()

        FeatureFlags.is_ml_enabled(request_id="test_3")
        FeatureFlags.record_ml_failure()

        # Vérifier stats
        stats = FeatureFlags.get_stats()

        assert stats["ml_requests"] == 3
        assert stats["ml_successes"] == 2
        assert stats["ml_failures"] == 1
        assert stats["ml_success_rate"] == 2 / 3

        print("✅ Stats recording OK (success rate: {stats['ml_success_rate']")

    def test_get_stats(self):
        """Test récupération statistiques complètes."""
        from feature_flags import FeatureFlags

        FeatureFlags.reset_stats()

        stats = FeatureFlags.get_stats()

        assert "ml_enabled" in stats
        assert "ml_traffic_percentage" in stats
        assert "total_requests" in stats
        assert "ml_success_rate" in stats

        print("✅ Get stats OK ({len(stats)} metrics)")


class TestFeatureFlagsAPI:
    """Tests des routes API feature flags (ADMIN requis — F-04)."""

    def test_company_forbidden(self, client, auth_headers):
        response = client.get("/api/feature-flags/status", headers=auth_headers)
        assert response.status_code == 403

    def test_get_status(self, client, admin_headers):
        """Test endpoint GET /api/feature-flags/status."""
        response = client.get("/api/feature-flags/status", headers=admin_headers)
        assert response.status_code == 200
        data = response.get_json()
        assert "config" in data
        assert "stats" in data

    def test_enable_ml(self, client, admin_headers):
        """Test endpoint POST /api/feature-flags/ml/enable."""
        response = client.post(
            "/api/feature-flags/ml/enable",
            json={"percentage": 25},
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "ML activé" in data["message"]
        assert data["status"]["config"]["ML_ENABLED"] is True
        assert data["status"]["config"]["ML_TRAFFIC_PERCENTAGE"] == 25

    def test_disable_ml(self, client, admin_headers):
        """Test endpoint POST /api/feature-flags/ml/disable."""
        response = client.post("/api/feature-flags/ml/disable", headers=admin_headers)
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "désactivé" in data["message"]
        assert data["status"]["config"]["ML_ENABLED"] is False

    def test_set_percentage(self, client, admin_headers):
        """Test endpoint POST /api/feature-flags/ml/percentage."""
        response = client.post(
            "/api/feature-flags/ml/percentage",
            json={"percentage": 75},
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["status"]["config"]["ML_TRAFFIC_PERCENTAGE"] == 75

    def test_set_invalid_percentage(self, client, admin_headers):
        """Test validation pourcentage invalide."""
        response = client.post(
            "/api/feature-flags/ml/percentage",
            json={"percentage": 150},
            headers=admin_headers,
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_reset_stats(self, client, admin_headers):
        """Test endpoint POST /api/feature-flags/reset-stats."""
        client.post(
            "/api/feature-flags/ml/enable",
            json={"percentage": 100},
            headers=admin_headers,
        )
        response = client.post("/api/feature-flags/reset-stats", headers=admin_headers)
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["status"]["stats"]["total_requests"] == 0

    def test_ml_health(self, client, admin_headers):
        """Test endpoint GET /api/feature-flags/ml/health."""
        response = client.get("/api/feature-flags/ml/health", headers=admin_headers)
        assert response.status_code in (200, 503)
        data = response.get_json()
        assert "status" in data
        assert "healthy" in data
        assert "success_rate" in data
        assert data.get("error") != "ml_control_plane_disabled"


if __name__ == "__main__":
    """Exécution directe pour tests rapides."""
    print("\n" + "=" * 70)
    print("🧪 TESTS FEATURE FLAGS")
    print("=" * 70)

    # Tests unitaires
    print("\n1. Tests unitaires feature flags...")
    test = TestFeatureFlags()
    try:
        test.test_default_configuration()
        test.test_enable_disable_ml()
        test.test_traffic_percentage()
        test.test_stats_recording()
        test.test_get_stats()
    except Exception:
        print("❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
        import sys

        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ TESTS UNITAIRES RÉUSSIS !")
    print("=" * 70 + "\n")

    print("ℹ️ Pour tester les routes API:")
    print("   pytest tests/test_feature_flags.py::TestFeatureFlagsAPI")
