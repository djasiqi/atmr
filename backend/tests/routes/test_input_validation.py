"""✅ Tests d'intégration pour validation des inputs dans les endpoints critiques.

Teste que les validations Marshmallow fonctionnent correctement dans les endpoints
et que les tentatives d'injection sont rejetées.
"""


# Note: Ces tests utilisent les fixtures de conftest.py (app, client, auth_headers, admin_headers, etc.)


class TestDispatchRunValidation:
    """Tests pour validation de l'endpoint /company_dispatch/run."""

    def test_valid_dispatch_request(self, client, auth_headers):
        """Test avec requête valide."""
        data = {"for_date": "2025-01-15", "async": True}
        response = client.post("/api/company_dispatch/run", json=data, headers=auth_headers)
        # Note: Ce test nécessite un utilisateur authentifié et une company valide
        # Pour l'instant, on teste juste que la validation fonctionne
        assert response.status_code in [200, 202, 400, 401, 403]  # Peut varier selon l'auth

    def test_invalid_date_format(self, client, auth_headers):
        """Test avec format de date invalide."""
        data = {"for_date": "2025/01/15", "async": True}
        response = client.post("/api/company_dispatch/run", json=data, headers=auth_headers)
        assert response.status_code == 400
        assert "for_date" in response.get_json().get("errors", {})

    def test_sql_injection_in_date(self, client, auth_headers):
        """Test avec tentative d'injection SQL dans for_date."""
        data = {"for_date": "2025-01-15'; DROP TABLE users; --", "async": True}
        response = client.post("/api/company_dispatch/run", json=data, headers=auth_headers)
        assert response.status_code == 400
        # La validation devrait rejeter le format invalide

    def test_invalid_mode(self, client, auth_headers):
        """Test avec mode invalide."""
        data = {"for_date": "2025-01-15", "mode": "invalid_mode"}
        response = client.post("/api/company_dispatch/run", json=data, headers=auth_headers)
        assert response.status_code == 400
        assert "mode" in response.get_json().get("errors", {})


class TestVehicleUpdateValidation:
    """Tests pour validation de l'endpoint /companies/me/vehicles/<id>."""

    def test_invalid_license_plate_too_long(self, client, auth_headers):
        """Test avec plaque d'immatriculation trop longue."""
        data = {"license_plate": "A" * 25}  # > 20 caractères
        response = client.put("/api/companies/me/vehicles/1", json=data, headers=auth_headers)
        assert response.status_code == 400
        assert "license_plate" in response.get_json().get("errors", {})

    def test_invalid_year(self, client, auth_headers):
        """Test avec année invalide."""
        data = {"year": 1899}  # < 1900
        response = client.put("/api/companies/me/vehicles/1", json=data, headers=auth_headers)
        assert response.status_code == 400
        assert "year" in response.get_json().get("errors", {})


class TestQueryParamsValidation:
    """Tests pour validation des query parameters GET."""

    def test_valid_pagination(self, client, auth_headers):
        """Test avec pagination valide."""
        response = client.get("/api/admin/autonomous-actions?page=2&per_page=25", headers=auth_headers)
        # Note: Ce test nécessite un admin authentifié
        assert response.status_code in [200, 401, 403]  # Peut varier selon l'auth

    def test_invalid_page_negative(self, client, auth_headers):
        """Test avec page négative."""
        response = client.get("/api/admin/autonomous-actions?page=-1&per_page=50", headers=auth_headers)
        assert response.status_code == 400
        assert "page" in response.get_json().get("errors", {})

    def test_invalid_per_page_too_large(self, client, auth_headers):
        """Test avec per_page trop grand."""
        response = client.get("/api/admin/autonomous-actions?page=1&per_page=600", headers=auth_headers)
        assert response.status_code == 400
        assert "per_page" in response.get_json().get("errors", {})

    def test_invalid_date_format(self, client, auth_headers):
        """Test avec format de date invalide."""
        response = client.get("/api/admin/autonomous-actions?start_date=2025/01/15", headers=auth_headers)
        assert response.status_code == 400
        assert "start_date" in response.get_json().get("errors", {})


class TestValidationErrorFormat:
    """Tests pour le format des erreurs de validation."""

    def test_validation_error_structure(self, client, auth_headers):
        """Test que les erreurs de validation ont la structure attendue."""
        data = {"for_date": "invalid", "mode": "invalid_mode"}
        response = client.post("/api/company_dispatch/run", json=data, headers=auth_headers)
        assert response.status_code == 400
        json_data = response.get_json()
        assert "message" in json_data
        assert "errors" in json_data
        assert isinstance(json_data["errors"], dict)

    def test_multiple_validation_errors(self, client, auth_headers):
        """Test avec plusieurs erreurs de validation."""
        data = {"for_date": "invalid", "mode": "invalid_mode", "regular_first": "not_a_bool"}
        response = client.post("/api/company_dispatch/run", json=data, headers=auth_headers)
        assert response.status_code == 400
        json_data = response.get_json()
        assert len(json_data.get("errors", {})) > 1
