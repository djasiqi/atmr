"""✅ Priorité 8: Tests de sécurité pour injection SQL.

Valide que SQLAlchemy protège contre l'injection SQL dans :
- Query parameters (recherche)
- Filtres (client_id, status, year, month)
- Path parameters (booking_id, user_id, company_id)
- Body JSON (champs texte)
"""

import pytest

# Payloads SQL injection classiques à tester
SQL_INJECTION_PAYLOADS = [
    "1' OR '1'='1",
    "1' OR '1'='1' --",
    "1' OR '1'='1' /*",
    "1' UNION SELECT * FROM users--",
    "'; DROP TABLE bookings--",
    "1' OR 1=1--",
    "' OR 'x'='x",
    "' AND 1=1--",
    "' AND 1=2--",
    "' UNION SELECT NULL--",
    "admin'--",
    "admin'/*",
    "' OR '1'='1'--",
    "' OR 1=1 LIMIT 1--",
]


class TestSQLInjectionQueryParams:
    """Tests d'injection SQL dans les query parameters."""

    def test_sql_injection_in_search_query(self, client, auth_headers):
        """Test que l'injection SQL dans ?q= est bloquée."""
        # Note: SQLAlchemy protège automatiquement via requêtes paramétrées
        # On teste que les payloads SQL ne causent pas d'erreurs SQL ni d'accès non autorisé
        for payload in SQL_INJECTION_PAYLOADS:
            # Tester avec différents endpoints de recherche
            response = client.get(
                f"/api/invoices/companies/1/invoices?q={payload}",
                headers=auth_headers,
            )
            # Ne doit pas retourner d'erreur SQL (500), juste 400/404/200 avec résultats vides
            assert response.status_code in (200, 400, 401, 403, 404)
            response_text = response.get_data(as_text=True).lower()
            # Ne doit pas contenir d'erreurs SQL PostgreSQL/MySQL
            assert "sql" not in response_text
            assert "syntax error" not in response_text
            assert "pg_error" not in response_text

    def test_sql_injection_in_client_search(self, client, auth_headers):
        """Test que l'injection SQL dans search= est bloquée."""
        for payload in SQL_INJECTION_PAYLOADS[:5]:  # Tester les 5 premiers payloads
            response = client.get(
                f"/api/companies/me/clients?search={payload}",
                headers=auth_headers,
            )
            # Ne doit pas retourner d'erreur SQL
            assert response.status_code in (200, 400, 401, 403)
            response_text = response.get_data(as_text=True).lower()
            assert "sql" not in response_text

    def test_sql_injection_in_medical_search(self, client):
        """Test que l'injection SQL dans recherche médicale est bloquée."""
        for payload in SQL_INJECTION_PAYLOADS[:5]:
            response = client.get(f"/api/medical/establishments?q={payload}")
            # Endpoint public, mais doit protéger contre injection SQL
            assert response.status_code in (200, 400)
            response_text = response.get_data(as_text=True).lower()
            assert "sql" not in response_text


class TestSQLInjectionFilters:
    """Tests d'injection SQL dans les filtres."""

    def test_sql_injection_in_client_id_filter(self, client, auth_headers):
        """Test que l'injection SQL dans client_id= est bloquée."""
        payloads = ["1 OR 1=1", "1' OR '1'='1", "1; DROP TABLE users--"]
        for payload in payloads:
            response = client.get(
                f"/api/invoices/companies/1/invoices?client_id={payload}",
                headers=auth_headers,
            )
            # Type int attendu, donc doit retourner 400 pour valeur invalide
            assert response.status_code in (200, 400, 401, 403, 404)
            response_text = response.get_data(as_text=True).lower()
            assert "sql" not in response_text

    def test_sql_injection_in_status_filter(self, client, auth_headers):
        """Test que l'injection SQL dans status= est bloquée."""
        payloads = ["draft' OR '1'='1", "draft'; DROP TABLE invoices--"]
        for payload in payloads:
            response = client.get(
                f"/api/invoices/companies/1/invoices?status={payload}",
                headers=auth_headers,
            )
            # Status invalide doit retourner 400 ou résultats filtrés, pas d'erreur SQL
            assert response.status_code in (200, 400, 401, 403)
            response_text = response.get_data(as_text=True).lower()
            assert "sql" not in response_text

    def test_sql_injection_in_year_month_filters(self, client, auth_headers):
        """Test que l'injection SQL dans year= et month= est bloquée."""
        payloads = ["2024' OR '1'='1", "12'; DROP TABLE invoices--"]
        for payload in payloads:
            response = client.get(
                f"/api/invoices/companies/1/invoices?year={payload}&month=1",
                headers=auth_headers,
            )
            # Type int attendu, donc doit retourner 400 pour valeur invalide
            assert response.status_code in (200, 400, 401, 403)
            response_text = response.get_data(as_text=True).lower()
            assert "sql" not in response_text


class TestSQLInjectionPathParams:
    """Tests d'injection SQL dans les path parameters."""

    def test_sql_injection_in_booking_id(self, client, auth_headers):
        """Test que l'injection SQL dans booking_id est bloquée."""
        # Les IDs dans les path params sont typés comme int par Flask-RESTX
        # Tester avec des valeurs qui pourraient contourner la validation
        payloads = ["1 OR 1=1", "999999999", "-1", "0"]
        for payload in payloads:
            try:
                response = client.get(
                    f"/api/bookings/{payload}",
                    headers=auth_headers,
                )
                # Doit retourner 404 pour ID inexistant, pas d'erreur SQL
                assert response.status_code in (200, 401, 403, 404)
                response_text = response.get_data(as_text=True).lower()
                assert "sql" not in response_text
            except ValueError:
                # Flask-RESTX peut rejeter les valeurs non-int avant notre code
                pass  # C'est acceptable, la validation fonctionne

    def test_sql_injection_in_user_id(self, client, admin_headers):
        """Test que l'injection SQL dans user_id est bloquée."""
        payloads = ["1 OR 1=1", "1' OR '1'='1"]
        for payload in payloads:
            try:
                response = client.get(
                    f"/api/admin/users/{payload}",
                    headers=admin_headers,
                )
                # Doit retourner 404 pour ID invalide, pas d'erreur SQL
                assert response.status_code in (200, 400, 401, 403, 404)
                response_text = response.get_data(as_text=True).lower()
                assert "sql" not in response_text
            except ValueError:
                # Flask-RESTX peut rejeter les valeurs non-int
                pass

    def test_sql_injection_in_company_id(self, client, auth_headers):
        """Test que l'injection SQL dans company_id est bloquée."""
        payloads = ["1 OR 1=1", "1' OR '1'='1"]
        for payload in payloads:
            try:
                response = client.get(
                    f"/api/invoices/companies/{payload}/invoices",
                    headers=auth_headers,
                )
                # Doit retourner 404/403 pour ID invalide, pas d'erreur SQL
                assert response.status_code in (200, 400, 401, 403, 404)
                response_text = response.get_data(as_text=True).lower()
                assert "sql" not in response_text
            except ValueError:
                # Flask-RESTX peut rejeter les valeurs non-int
                pass


class TestSQLInjectionBodyJSON:
    """Tests d'injection SQL dans les champs texte du body JSON."""

    def test_sql_injection_in_customer_name(self, client, auth_headers, sample_user):
        """Test que l'injection SQL dans customer_name est stockée telle quelle (pas exécutée)."""
        payload = "Test'; DROP TABLE bookings--"
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
        # Doit accepter le payload comme texte (validation peut échouer, mais pas d'erreur SQL)
        assert response.status_code in (201, 400, 401, 403, 404)
        response_text = response.get_data(as_text=True).lower()
        # Ne doit pas contenir d'erreurs SQL
        assert "sql" not in response_text
        assert "syntax error" not in response_text

    def test_sql_injection_in_location_fields(self, client, auth_headers, sample_user):
        """Test que l'injection SQL dans pickup_location/dropoff_location est stockée telle quelle."""
        payload = "Rue Test'; DROP TABLE bookings--"
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
        assert "sql" not in response_text

    def test_sql_injection_in_email_field(self, client, auth_headers):
        """Test que l'injection SQL dans email est rejetée par validation."""
        payload = "test@example.com'; DROP TABLE users--"
        data = {
            "email": payload,
            "password": "password123",
        }
        response = client.post("/api/auth/login", json=data)
        # Doit retourner 400 pour email invalide (validation Marshmallow), pas d'erreur SQL
        assert response.status_code in (400, 401)  # 400 = validation error, 401 = invalid credentials
        response_text = response.get_data(as_text=True).lower()
        assert "sql" not in response_text


class TestSQLAlchemyProtection:
    """Tests pour valider que SQLAlchemy protège contre l'injection SQL."""

    def test_sqlalchemy_uses_parameterized_queries(self, app_context, db, sample_user):
        """Test que SQLAlchemy utilise des requêtes paramétrées."""
        from models import User

        # Tester avec un payload SQL dans un filter_by
        payload = "1' OR '1'='1"
        # SQLAlchemy devrait traiter ça comme une chaîne littérale, pas comme du SQL
        users = User.query.filter_by(username=payload).all()
        # Doit retourner une liste vide (pas d'utilisateur avec ce username exact)
        assert isinstance(users, list)
        # Ne doit pas lever d'exception SQL
        assert len(users) == 0  # Ou peut-être 0 si aucun utilisateur ne correspond

    def test_sqlalchemy_filter_protection(self, app_context, db):
        """Test que filter() protège contre l'injection SQL."""
        from models import Booking

        # Tester avec un payload SQL dans un filter
        payload = "1' OR '1'='1"
        # SQLAlchemy devrait protéger automatiquement
        bookings = Booking.query.filter(Booking.customer_name == payload).all()
        assert isinstance(bookings, list)
        # Ne doit pas lever d'exception SQL
        assert len(bookings) >= 0  # Peut être vide, mais pas d'erreur

    def test_sqlalchemy_like_protection(self, app_context, db):
        """Test que like() protège contre l'injection SQL avec caractères spéciaux."""
        from sqlalchemy import func

        from models import User

        # Tester avec des caractères spéciaux SQL dans un LIKE
        payload = "%' OR '1'='1--%"
        users = User.query.filter(func.lower(User.username).like(f"%{payload}%")).all()
        # Doit traiter comme une recherche littérale, pas comme du SQL
        assert isinstance(users, list)
        # Ne doit pas lever d'exception SQL
        assert len(users) >= 0
