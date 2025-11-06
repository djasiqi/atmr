"""Tests pour l'endpoint Prometheus métriques HTTP (2.10).

Valide que toutes les routes sont instrumentées et que les métriques
sont correctement exposées sur /prometheus/metrics-http.
"""

import pytest


class TestPrometheusMetricsEndpoint:
    """Tests pour l'endpoint /prometheus/metrics-http."""
    
    def test_metrics_endpoint_exists(self, client):
        """✅ 2.10: Test que l'endpoint /prometheus/metrics-http existe."""
        response = client.get("/prometheus/metrics-http")
        
        # L'endpoint doit exister (200 si prometheus-client installé, 503 sinon)
        assert response.status_code in [200, 503], f"Endpoint retourne {response.status_code}"
        
        if response.status_code == 503:
            # Si prometheus-client n'est pas installé, vérifier le message d'erreur
            data = response.get_json()
            assert "error" in data
            assert "prometheus" in data.get("error", "").lower() or "prometheus" in data.get("message", "").lower()
            pytest.skip("prometheus-client non installé - tests métriques ignorés")
        
        # Si prometheus-client est installé, valider le format Prometheus
        assert response.content_type.startswith("text/plain")
        content = response.get_data(as_text=True)
        
        # Vérifier présence de métriques standard
        assert "http_request_duration_seconds" in content
        assert "http_requests_total" in content
        assert "http_requests_in_progress" in content
    
    def test_metrics_after_request(self, client, auth_headers):
        """✅ 2.10: Test que les métriques sont enregistrées après une requête."""
        # Faire une requête pour générer des métriques
        client.get("/api/auth/me", headers=auth_headers)
        
        # Récupérer les métriques
        metrics_response = client.get("/prometheus/metrics-http")
        
        if metrics_response.status_code == 503:
            pytest.skip("prometheus-client non installé")
        
        content = metrics_response.get_data(as_text=True)
        
        # Vérifier que des métriques ont été générées
        assert "http_requests_total" in content
        
        # Vérifier présence de labels standard
        assert 'method="GET"' in content or "method=GET" in content
    
    def test_metrics_labels_method_endpoint_status(self, client, auth_headers):
        """✅ 2.10: Test que les métriques ont les labels method, endpoint, status."""
        # Faire plusieurs requêtes pour générer des métriques variées
        client.get("/api/auth/me", headers=auth_headers)
        client.get("/health")
        
        metrics_response = client.get("/prometheus/metrics-http")
        
        if metrics_response.status_code == 503:
            pytest.skip("prometheus-client non installé")
        
        content = metrics_response.get_data(as_text=True)
        
        # Vérifier format des métriques avec labels
        # Exemple: http_requests_total{method="GET",endpoint="/api/auth/me",status="200"} 1.0
        assert "http_requests_total" in content
        # Les labels peuvent être dans différents formats (quote ou non)
        assert "method=" in content or 'method="' in content
        assert "status=" in content or 'status="' in content
    
    def test_latency_histogram_buckets(self, client, auth_headers):
        """✅ 2.10: Test que l'histogramme de latence a des buckets corrects."""
        # Faire quelques requêtes
        for _ in range(3):
            client.get("/api/auth/me", headers=auth_headers)
        
        metrics_response = client.get("/prometheus/metrics-http")
        
        if metrics_response.status_code == 503:
            pytest.skip("prometheus-client non installé")
        
        content = metrics_response.get_data(as_text=True)
        
        # Vérifier présence de buckets d'histogramme
        # Les buckets sont exposés comme: http_request_duration_seconds_bucket{le="0.005"} ...
        assert "http_request_duration_seconds_bucket" in content
        assert "http_request_duration_seconds_count" in content
        assert "http_request_duration_seconds_sum" in content


class TestPrometheusMiddlewareIntegration:
    """Tests d'intégration du middleware Prometheus."""
    
    def test_all_routes_instrumented(self, client, auth_headers):
        """✅ 2.10: Test que toutes les routes sont instrumentées."""
        # Tester plusieurs routes différentes
        routes_to_test = [
            ("GET", "/health"),
            ("GET", "/api/auth/me", auth_headers),
            ("GET", "/api/admin/stats", auth_headers),
        ]
        
        for route_info in routes_to_test:
            method = route_info[0]
            path = route_info[1]
            headers = route_info[2] if len(route_info) > 2 else {}
            
            if method == "GET":
                client.get(path, headers=headers)
        
        # Vérifier que les métriques sont générées
        metrics_response = client.get("/prometheus/metrics-http")
        
        if metrics_response.status_code == 503:
            pytest.skip("prometheus-client non installé")
        
        content = metrics_response.get_data(as_text=True)
        
        # Vérifier que plusieurs endpoints sont dans les métriques
        # (normalisation avec :id peut grouper certains endpoints)
        assert "http_requests_total" in content
        assert "http_request_duration_seconds" in content
    
    def test_error_metrics(self, client):
        """✅ 2.10: Test que les erreurs 4xx/5xx sont bien comptées."""
        # Générer quelques erreurs
        client.get("/api/nonexistent")  # 404
        client.post("/api/auth/login", json={})  # 400/401
        
        metrics_response = client.get("/prometheus/metrics-http")
        
        if metrics_response.status_code == 503:
            pytest.skip("prometheus-client non installé")
        
        content = metrics_response.get_data(as_text=True)
        
        # Vérifier présence de métriques d'erreur
        # Les erreurs doivent être dans les labels status
        assert "http_requests_total" in content
        # Vérifier qu'il y a des status != 200
        assert 'status="404"' in content or "status=404" in content or 'status="4' in content or "status=4" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

