"""✅ 2.8: Tests pour le rate limiting des endpoints critiques.

Vérifie que les limites sont respectées et que HTTP 429 est retourné.
"""




class TestRateLimitingBookings:
    """Tests rate limiting pour les endpoints bookings."""

    def test_create_booking_rate_limit(self, client, auth_headers):
        """Test que la création de réservation est limitée à 50/heure."""
        # Désactiver temporairement le limiter global pour ce test spécifique
        # Flask-Limiter utilise get_remote_address par défaut donc tous les appels
        # depuis le même client apparaissent comme le même IP

        # Faire 51 tentatives de création (devrait échouer à la 51ème)
        for i in range(51):
            response = client.post(
                "/api/bookings/clients/test_public_id/bookings",
                json={
                    "customer_name": f"Test {i}",
                    "pickup_location": "Rue Test 1, Genève",
                    "dropoff_location": "Rue Test 2, Genève",
                    "scheduled_time": "2024-12-25T10:00:00",
                    "amount": 50.0,
                },
                headers=auth_headers,
            )

            if i < 50:
                # Les 50 premières devraient passer (ou échouer pour autre raison)
                assert response.status_code in (201, 400, 403, 404)  # Pas de 429
            else:
                # La 51ème devrait retourner 429
                assert response.status_code == 429, f"Expected 429, got {response.status_code}"

    def test_list_bookings_rate_limit(self, client, auth_headers):
        """Test que la liste des réservations est limitée à 300/heure."""
        # Faire 301 tentatives
        for i in range(301):
            response = client.get("/api/bookings/", headers=auth_headers)

            if i < 300:
                assert response.status_code in (200, 401, 403)  # Pas de 429
            else:
                assert response.status_code == 429, f"Expected 429, got {response.status_code}"


class TestRateLimitingAdmin:
    """Tests rate limiting pour les endpoints admin."""

    def test_admin_stats_rate_limit(self, client, admin_headers):
        """Test que les stats admin sont limitées à 100/heure."""
        # Faire 101 tentatives
        for i in range(101):
            response = client.get("/api/admin/stats", headers=admin_headers)

            if i < 100:
                assert response.status_code in (200, 401, 403)  # Pas de 429
            else:
                assert response.status_code == 429, f"Expected 429, got {response.status_code}"

    def test_reset_password_rate_limit(self, client, admin_headers):
        """Test que le reset password est limité à 10/heure (sécurité)."""
        # Faire 11 tentatives
        for i in range(11):
            response = client.post(
                "/api/admin/users/1/reset-password",
                headers=admin_headers,
            )

            if i < 10:
                assert response.status_code in (200, 404, 401, 403)  # Pas de 429
            else:
                assert response.status_code == 429, f"Expected 429, got {response.status_code}"


class TestRateLimitingCompanies:
    """Tests rate limiting pour les endpoints companies."""

    def test_create_driver_rate_limit(self, client, auth_headers):
        """Test que la création de chauffeur est limitée à 20/heure."""
        # Faire 21 tentatives
        for i in range(21):
            response = client.post(
                "/api/companies/me/drivers/create",
                json={
                    "username": f"testdriver{i}",
                    "first_name": "Test",
                    "last_name": f"Driver{i}",
                    "email": f"test{i}@example.com",
                    "password": "password123",
                    "vehicle_assigned": "Test Vehicle",
                    "brand": "Test Brand",
                    "license_plate": f"TEST{i}",
                },
                headers=auth_headers,
            )

            if i < 20:
                assert response.status_code in (201, 400, 401, 403, 409)  # Pas de 429
            else:
                assert response.status_code == 429, f"Expected 429, got {response.status_code}"

    def test_create_client_rate_limit(self, client, auth_headers):
        """Test que la création de client est limitée à 50/heure."""
        # Faire 51 tentatives
        for i in range(51):
            response = client.post(
                "/api/companies/me/clients",
                json={
                    "client_type": "PRIVATE",
                    "first_name": f"Test{i}",
                    "last_name": f"Client{i}",
                    "address": f"Rue Test {i}, Genève",
                },
                headers=auth_headers,
            )

            if i < 50:
                assert response.status_code in (201, 400, 401, 403)  # Pas de 429
            else:
                assert response.status_code == 429, f"Expected 429, got {response.status_code}"


class TestRateLimitingAuth:
    """Tests rate limiting pour les endpoints auth (déjà existants)."""

    def test_login_rate_limit(self, client):
        """Test que le login est limité à 5/minute (déjà testé ailleurs)."""
        # Faire 6 tentatives
        for i in range(6):
            response = client.post(
                "/api/auth/login",
                json={
                    "email": "test@example.com",
                    "password": "wrongpassword",
                },
            )

            if i < 5:
                assert response.status_code in (401, 404)  # Pas de 429
            else:
                assert response.status_code == 429, f"Expected 429, got {response.status_code}"
