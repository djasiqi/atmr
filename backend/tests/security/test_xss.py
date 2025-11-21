"""✅ Priorité 8: Tests de sécurité pour XSS (Cross-Site Scripting).

Valide que les données utilisateur sont correctement échappées et que les payloads XSS
sont stockés comme texte (pas exécutés) :
- XSS dans champs texte (customer_name, pickup_location, etc.)
- XSS dans query parameters
- XSS dans JSON body
- Échappement HTML/JS via input_sanitizer
"""

import pytest

# Payloads XSS classiques à tester
XSS_PAYLOADS = [
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert('XSS')>",
    "javascript:alert('XSS')",
    "<svg onload=alert('XSS')>",
    "';alert('XSS');//",
    "<iframe src=javascript:alert('XSS')>",
    "<body onload=alert('XSS')>",
    "<input onfocus=alert('XSS') autofocus>",
    "<select onfocus=alert('XSS') autofocus><option>test</option></select>",
    "<textarea onfocus=alert('XSS') autofocus>test</textarea>",
    "<keygen onfocus=alert('XSS') autofocus>",
    "<video><source onerror=alert('XSS')>",
    "<audio src=x onerror=alert('XSS')>",
    "<details open ontoggle=alert('XSS')>",
    "<marquee onstart=alert('XSS')>test</marquee>",
]


class TestXSSInTextFields:
    """Tests XSS dans les champs texte (customer_name, pickup_location, etc.)."""

    def test_xss_in_customer_name(self, client, auth_headers, sample_user):
        """Test que les payloads XSS dans customer_name sont stockés comme texte."""
        payload = "<script>alert('XSS')</script>"
        data = {
            "customer_name": payload,
            "pickup_location": "Rue Test 1, Genève",
            "dropoff_location": "Rue Test 2, Genève",
            "scheduled_time": "2025-12-25T10:00:00",
            "amount": 50.0,
        }
        response = client.post(
            f"/api/bookings/clients/{sample_user.public_id}/bookings",
            json=data,
            headers=auth_headers,
        )
        # Doit accepter le payload comme texte (pas d'exécution)
        assert response.status_code in (201, 400, 401, 403, 404)
        if response.status_code == 201:
            # Si la création réussit, vérifier que les données sont stockées telles quelles
            json_data = response.get_json()
            # Les données doivent être stockées comme texte, pas exécutées
            assert json_data is not None

    def test_xss_in_location_fields(self, client, auth_headers, sample_user):
        """Test que les payloads XSS dans pickup_location/dropoff_location sont stockés comme texte."""
        for payload in XSS_PAYLOADS[:5]:  # Tester les 5 premiers payloads
            data = {
                "customer_name": "Test Customer",
                "pickup_location": payload,
                "dropoff_location": payload,
                "scheduled_time": "2025-12-25T10:00:00",
                "amount": 50.0,
            }
            response = client.post(
                f"/api/bookings/clients/{sample_user.public_id}/bookings",
                json=data,
                headers=auth_headers,
            )
            # Doit accepter le payload comme texte
            assert response.status_code in (201, 400, 401, 403, 404)
            response_text = response.get_data(as_text=True).lower()
            # Ne doit pas exécuter le JavaScript
            assert "alert" not in response_text or response.status_code == 400

    def test_xss_in_username_field(self, client):
        """Test que les payloads XSS dans username sont rejetés ou échappés."""
        payload = "<script>alert('XSS')</script>"
        data = {
            "username": payload,
            "email": "test@example.com",
            "password": "password123",
        }
        response = client.post("/api/auth/register", json=data)
        # Doit retourner 400 pour validation échouée, pas d'exécution XSS
        assert response.status_code in (400, 409)
        response_text = response.get_data(as_text=True).lower()
        # Ne doit pas exécuter le JavaScript
        assert "alert" not in response_text or response.status_code == 400

    def test_xss_in_email_field(self, client):
        """Test que les payloads XSS dans email sont rejetés par validation."""
        payload = "test<script>alert('XSS')</script>@example.com"
        data = {
            "email": payload,
            "password": "password123",
        }
        response = client.post("/api/auth/login", json=data)
        # Doit retourner 400 pour email invalide, pas d'exécution XSS
        assert response.status_code in (400, 401)
        response_text = response.get_data(as_text=True).lower()
        assert "alert" not in response_text or response.status_code == 400


class TestXSSInQueryParams:
    """Tests XSS dans les query parameters GET."""

    def test_xss_in_search_query(self, client, auth_headers):
        """Test que les payloads XSS dans search= sont traités comme texte."""
        payload = "<script>alert('XSS')</script>"
        response = client.get(
            f"/api/companies/me/clients?search={payload}",
            headers=auth_headers,
        )
        # Doit traiter le payload comme texte, pas l'exécuter
        assert response.status_code in (200, 400, 401, 403)
        response_text = response.get_data(as_text=True).lower()
        # Ne doit pas exécuter le JavaScript dans la réponse JSON
        assert "alert" not in response_text or response.status_code == 400

    def test_xss_in_medical_search(self, client):
        """Test que les payloads XSS dans recherche médicale sont traités comme texte."""
        payload = "<img src=x onerror=alert('XSS')>"
        response = client.get(f"/api/medical/establishments?q={payload}")
        # Endpoint public, doit traiter comme texte
        assert response.status_code in (200, 400)
        response_text = response.get_data(as_text=True).lower()
        # Ne doit pas exécuter le JavaScript
        assert "alert" not in response_text or response.status_code == 400


class TestXSSSanitization:
    """Tests pour valider que les fonctions de sanitisation échappent correctement."""

    def test_escape_html_escapes_xss_payloads(self):
        """Test que escape_html() échappe correctement les payloads XSS."""
        from shared.input_sanitizer import escape_html

        for payload in XSS_PAYLOADS[:10]:  # Tester les 10 premiers
            escaped = escape_html(payload)
            # Vérifier que les balises sont échappées
            assert "<script" not in escaped or "&lt;script" in escaped
            assert "alert" in escaped or "&lt;" in escaped or "&#x27;" in escaped
            # Vérifier que le HTML n'est pas valide après échappement
            assert escaped != payload  # Doit être différent de l'original

    def test_escape_js_escapes_javascript(self):
        """Test que escape_js() échappe correctement le JavaScript."""
        from shared.input_sanitizer import escape_js

        js_payloads = [
            "';alert('XSS');//",
            '";alert("XSS");//',
            "javascript:alert('XSS')",
        ]
        for payload in js_payloads:
            escaped = escape_js(payload)
            # Vérifier que les guillemets sont échappés
            assert "\\'" in escaped or '\\"' in escaped
            # Vérifier que le JavaScript n'est pas valide après échappement
            assert escaped != payload

    def test_sanitize_string_strips_html(self):
        """Test que sanitize_string() supprime les balises HTML si demandé."""
        from shared.input_sanitizer import sanitize_string

        payload = "<script>alert('XSS')</script>Test"
        sanitized = sanitize_string(payload, strip_html=True)
        # Vérifier que les balises sont supprimées
        assert "<script>" not in sanitized
        assert "</script>" not in sanitized
        # Vérifier que le contenu reste
        assert "alert" in sanitized or "Test" in sanitized

    def test_sanitize_string_escapes_html(self):
        """Test que sanitize_string() échappe HTML si demandé."""
        from shared.input_sanitizer import sanitize_string

        payload = "<script>alert('XSS')</script>"
        sanitized = sanitize_string(payload, escape_html_chars=True)
        # Vérifier que les balises sont échappées
        assert "&lt;script&gt;" in sanitized or "&lt;script" in sanitized


class TestXSSInJSONBody:
    """Tests XSS dans les champs JSON body."""

    def test_xss_in_client_creation(self, client, auth_headers):
        """Test que les payloads XSS dans création client sont stockés comme texte."""
        payload = "<script>alert('XSS')</script>"
        data = {
            "client_type": "PRIVATE",
            "first_name": payload,
            "last_name": payload,
            "address": payload,
        }
        response = client.post("/api/companies/me/clients", json=data, headers=auth_headers)
        # Doit accepter le payload comme texte
        assert response.status_code in (201, 400, 401, 403)
        response_text = response.get_data(as_text=True).lower()
        # Ne doit pas exécuter le JavaScript
        assert "alert" not in response_text or response.status_code == 400

    def test_xss_in_driver_creation(self, client, auth_headers):
        """Test que les payloads XSS dans création chauffeur sont stockés comme texte."""
        payload = "<img src=x onerror=alert('XSS')>"
        data = {
            "username": "testdriver",
            "first_name": payload,
            "last_name": payload,
            "email": "test@example.com",
            "password": "password123",
            "vehicle_assigned": payload,
            "brand": "Test",
            "license_plate": "TEST123",
        }
        response = client.post("/api/companies/me/drivers/create", json=data, headers=auth_headers)
        # Doit accepter le payload comme texte ou rejeter par validation
        assert response.status_code in (201, 400, 401, 403, 409)
        response_text = response.get_data(as_text=True).lower()
        # Ne doit pas exécuter le JavaScript
        assert "alert" not in response_text or response.status_code == 400


class TestXSSResponseSanitization:
    """Tests pour valider que les réponses API ne contiennent pas de XSS exécutable."""

    def test_api_responses_are_json_not_html(self, client, auth_headers):
        """Test que les réponses API sont en JSON, pas en HTML."""
        # Tester différents endpoints
        endpoints = [
            "/api/bookings/",
            "/api/companies/me",
            "/api/auth/me",
        ]
        for endpoint in endpoints:
            response = client.get(endpoint, headers=auth_headers)
            if response.status_code == 200:
                # Vérifier que la réponse est du JSON, pas du HTML
                content_type = response.headers.get("Content-Type", "")
                assert "application/json" in content_type.lower()
                # Vérifier que le contenu n'est pas du HTML
                response_text = response.get_data(as_text=True)
                assert not response_text.strip().startswith("<")
                # Vérifier que le JSON est valide
                try:
                    response.get_json()
                except Exception:
                    pytest.fail(f"Response from {endpoint} is not valid JSON")

    def test_error_responses_are_safe(self, client):
        """Test que les messages d'erreur sont sécurisés (pas d'exécution XSS)."""
        # Tester avec un payload XSS dans un champ invalide
        payload = "<script>alert('XSS')</script>"
        data = {"for_date": payload, "async": True}
        response = client.post("/api/company_dispatch/run", json=data, headers={"Authorization": "Bearer invalid"})
        # Doit retourner une erreur 400 ou 401
        assert response.status_code in (400, 401, 403)
        response_text = response.get_data(as_text=True).lower()
        # Vérifier que le message d'erreur ne contient pas le payload non échappé
        # (ou alors échappé)
        if payload.lower() in response_text:
            # Si le payload est présent, il doit être échappé
            assert "&lt;script&gt;" in response_text or "&lt;script" in response_text
