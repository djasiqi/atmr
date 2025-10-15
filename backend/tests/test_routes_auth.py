"""
Tests des routes d'authentification
"""
from flask_jwt_extended import create_access_token, create_refresh_token
from models import User, Client, UserRole

class TestLoginRoute:
    """Tests pour POST /api/auth/login"""
    
    def test_login_success(self, test_client, client_user):
        """Test login avec credentials valides"""
        _, user = client_user
        
        response = test_client.post('/api/auth/login', json={
            'email': 'client@test.com',
            'password': 'testpass123'
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'token' in data
        assert 'refresh_token' in data
        assert data['user']['email'] == 'client@test.com'
        assert data['user']['role'] == 'CLIENT'
    
    def test_login_invalid_email(self, test_client):
        """Test login avec email inexistant"""
        response = test_client.post('/api/auth/login', json={
            'email': 'notfound@test.com',
            'password': 'anypass'
        })
        
        assert response.status_code == 401
        assert 'error' in response.get_json()
    
    def test_login_invalid_password(self, test_client, client_user):
        """Test login avec mauvais mot de passe"""
        response = test_client.post('/api/auth/login', json={
            'email': 'client@test.com',
            'password': 'wrongpassword'
        })
        
        assert response.status_code == 401
        data = response.get_json()
        assert 'error' in data
    
    def test_login_missing_fields(self, test_client):
        """Test login avec champs manquants"""
        response = test_client.post('/api/auth/login', json={
            'email': 'client@test.com'
            # password manquant
        })
        
        assert response.status_code == 400
        assert 'error' in response.get_json()
    
    def test_login_rate_limit(self, test_client, client_user):
        """Test rate limiting (5 per minute)"""
        # 6 tentatives rapides
        for i in range(6):
            response = test_client.post('/api/auth/login', json={
                'email': 'client@test.com',
                'password': 'wrongpass'
            })
            if i < 5:
                assert response.status_code in (401, 200)
            else:
                # 6ème requête devrait être rate-limited
                assert response.status_code == 429


class TestRefreshToken:
    """Tests pour POST /api/auth/refresh-token"""
    
    def test_refresh_success(self, test_client, client_user, app):
        """Test refresh avec token valide"""
        _, user = client_user
        
        with app.app_context():
            refresh_token = create_refresh_token(identity=user.public_id)
        
        response = test_client.post('/api/auth/refresh-token', headers={
            'Authorization': f'Bearer {refresh_token}'
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'access_token' in data
    
    def test_refresh_with_access_token(self, test_client, client_user, app):
        """Test refresh avec access token (devrait échouer)"""
        _, user = client_user
        
        with app.app_context():
            access_token = create_access_token(identity=user.public_id)
        
        response = test_client.post('/api/auth/refresh-token', headers={
            'Authorization': f'Bearer {access_token}'
        })
        
        assert response.status_code in (401, 422)
    
    def test_refresh_missing_token(self, test_client):
        """Test refresh sans token"""
        response = test_client.post('/api/auth/refresh-token')
        assert response.status_code == 401


class TestRegister:
    """Tests pour POST /api/auth/register"""
    
    def test_register_success(self, test_client, app):
        """Test inscription nouveau client"""
        response = test_client.post('/api/auth/register', json={
            'username': 'newuser',
            'email': 'newuser@test.com',
            'password': 'securepass123',
            'first_name': 'Jane',
            'last_name': 'Smith',
            'phone': '+41221234567'
        })
        
        assert response.status_code == 201
        data = response.get_json()
        assert data['username'] == 'newuser'
        
        # Vérifier que le client a été créé
        with app.app_context():
            user = User.query.filter_by(email='newuser@test.com').first()
            assert user is not None
            assert user.role == UserRole.CLIENT
            
            client = Client.query.filter_by(user_id=user.id).first()
            assert client is not None
            assert client.is_active is True
    
    def test_register_duplicate_email(self, test_client, client_user):
        """Test inscription avec email existant"""
        response = test_client.post('/api/auth/register', json={
            'username': 'duplicate',
            'email': 'client@test.com',  # Déjà utilisé
            'password': 'pass123'
        })
        
        assert response.status_code == 409
        assert 'error' in response.get_json()
    
    def test_register_invalid_email(self, test_client):
        """Test inscription avec email invalide"""
        response = test_client.post('/api/auth/register', json={
            'username': 'invalid',
            'email': 'not-an-email',
            'password': 'pass123'
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

