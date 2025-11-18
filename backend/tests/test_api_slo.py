"""Tests pour le module api_slo (Service Level Objectives)."""

import unittest
from unittest.mock import MagicMock, patch

from services.api_slo import (
    API_SLOS,
    APISLOTarget,
    get_slo_target,
    normalize_endpoint,
    record_slo_metric,
)


class TestAPISLO(unittest.TestCase):
    """Tests pour les SLO API."""

    def test_slo_targets_exist(self):
        """Vérifie que les SLO critiques sont définis."""
        critical_endpoints = [
            "/api/bookings",
            "/api/bookings/:id",
            "/api/companies/me",
            "/api/auth/login",
            "/api/health",
        ]

        for endpoint in critical_endpoints:
            slo = get_slo_target(endpoint)
            self.assertIsNotNone(slo, f"SLO manquant pour {endpoint}")
            self.assertIsInstance(slo, APISLOTarget)

    def test_get_slo_target_exact_match(self):
        """Test récupération SLO par correspondance exacte."""
        slo = get_slo_target("/api/bookings")
        self.assertIsNotNone(slo)
        self.assertEqual(slo.endpoint, "/api/bookings")
        self.assertEqual(slo.latency_p95_max_ms, 500)
        self.assertEqual(slo.error_rate_max, 0.01)
        self.assertEqual(slo.availability_min, 0.99)

    def test_get_slo_target_prefix_match(self):
        """Test récupération SLO par correspondance de préfixe."""
        # /api/bookings/:id devrait matcher /api/bookings/123
        normalized = normalize_endpoint("/api/bookings/123")
        self.assertEqual(normalized, "/api/bookings/:id")

        slo = get_slo_target(normalized)
        self.assertIsNotNone(slo)
        self.assertEqual(slo.latency_p95_max_ms, 300)  # SLO pour :id

    def test_get_slo_target_not_found(self):
        """Test récupération SLO pour endpoint inexistant."""
        slo = get_slo_target("/api/nonexistent")
        self.assertIsNone(slo)

    def test_normalize_endpoint(self):
        """Test normalisation d'endpoints."""
        test_cases = [
            ("/api/bookings/123", "/api/bookings/:id"),
            ("/api/bookings/456", "/api/bookings/:id"),
            ("/api/drivers/789", "/api/drivers/:id"),
            ("/api/bookings", "/api/bookings"),  # Pas d'ID
            ("/api/bookings/123/details", "/api/bookings/:id/details"),
        ]

        for input_endpoint, expected in test_cases:
            with self.subTest(endpoint=input_endpoint):
                result = normalize_endpoint(input_endpoint)
                self.assertEqual(result, expected)

    def test_normalize_endpoint_long_path(self):
        """Test normalisation d'endpoint très long."""
        long_endpoint = "/api/" + "very/" * 30 + "long/path/123"
        normalized = normalize_endpoint(long_endpoint)
        self.assertLessEqual(len(normalized), 103)  # 100 + "..."
        self.assertTrue(normalized.endswith("...") or len(normalized) <= 100)

    @patch("services.api_slo.PROMETHEUS_AVAILABLE", True)
    @patch("services.api_slo.SLO_LATENCY_BREACH")
    @patch("services.api_slo.SLO_LATENCY_HISTOGRAM")
    def test_record_slo_metric_no_breach(self, mock_histogram, mock_counter):
        """Test enregistrement métrique SLO sans violation."""
        # Durée < seuil (300ms pour /api/bookings/:id)
        record_slo_metric(
            endpoint="/api/bookings/:id",
            duration_seconds=0.200,  # 200ms < 300ms
            status_code=200,
            method="GET",
        )

        # Pas de breach de latence
        mock_counter.inc.assert_not_called()
        # Mais histogram doit être mis à jour
        if mock_histogram:
            mock_histogram.labels().observe.assert_called_once()

    @patch("services.api_slo.PROMETHEUS_AVAILABLE", True)
    @patch("services.api_slo.SLO_LATENCY_BREACH")
    @patch("services.api_slo.SLO_ERROR_BREACH")
    def test_record_slo_metric_latency_breach(self, mock_error, mock_latency):
        """Test enregistrement violation latence SLO."""
        # Durée > seuil (500ms pour /api/bookings)
        record_slo_metric(
            endpoint="/api/bookings",
            duration_seconds=0.750,  # 750ms > 500ms
            status_code=200,
            method="GET",
        )

        # Breach de latence doit être enregistré
        if mock_latency:
            mock_latency.labels.assert_called_once()
            mock_latency.labels().inc.assert_called_once()

    @patch("services.api_slo.PROMETHEUS_AVAILABLE", True)
    @patch("services.api_slo.SLO_ERROR_BREACH")
    def test_record_slo_metric_error_breach(self, mock_error):
        """Test enregistrement violation taux d'erreurs SLO."""
        # Erreur serveur (5xx)
        record_slo_metric(
            endpoint="/api/bookings",
            duration_seconds=0.100,
            status_code=500,
            method="GET",
        )

        # Breach d'erreur doit être enregistré
        if mock_error:
            mock_error.labels.assert_called_once()
            mock_error.labels().inc.assert_called_once()

    @patch("services.api_slo.PROMETHEUS_AVAILABLE", False)
    def test_record_slo_metric_no_prometheus(self):
        """Test que l'enregistrement fonctionne même sans Prometheus."""
        # Ne doit pas lever d'exception
        record_slo_metric(
            endpoint="/api/bookings",
            duration_seconds=0.5,
            status_code=200,
            method="GET",
        )

    def test_slo_values_reasonable(self):
        """Vérifie que les valeurs SLO sont raisonnables."""
        for endpoint, slo in API_SLOS.items():
            with self.subTest(endpoint=endpoint):
                # Latence doit être > 0
                self.assertGreater(slo.latency_p95_max_ms, 0)
                self.assertLess(slo.latency_p95_max_ms, 60000)  # < 60s

                # Taux d'erreurs entre 0 et 1
                self.assertGreaterEqual(slo.error_rate_max, 0)
                self.assertLessEqual(slo.error_rate_max, 1)

                # Disponibilité entre 0 et 1
                self.assertGreaterEqual(slo.availability_min, 0)
                self.assertLessEqual(slo.availability_min, 1)
                self.assertGreater(slo.availability_min, 0.9)  # Au moins 90%


if __name__ == "__main__":
    unittest.main()
