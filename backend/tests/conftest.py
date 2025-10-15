"""
Fixtures pytest partagées pour tous les tests backend
"""
import pytest
from models import db, User, Client, Company, UserRole
from app import create_app

@pytest.fixture(scope='session')
def app():
    """Application Flask configurée pour les tests"""
    app = create_app('testing')
    
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture(scope='function')
def test_client(app):
    """Client de test Flask"""
    return app.test_client()

@pytest.fixture(scope='function')
def db_session(app):
    """Session DB transactionnelle (rollback après chaque test)"""
    with app.app_context():
        connection = db.engine.connect()
        transaction = connection.begin()
        
        # Bind session à cette transaction
        options = dict(bind=connection, binds={})
        session = db.create_scoped_session(options=options)
        db.session = session
        
        yield session
        
        # Rollback transaction
        transaction.rollback()
        connection.close()
        session.remove()

@pytest.fixture
def client_user(app, db_session):
    """Crée un utilisateur client de test"""
    user = User(
        username='testclient',
        email='client@test.com',
        role=UserRole.CLIENT,
        first_name='John',
        last_name='Doe'
    )
    user.set_password('testpass123')
    db_session.add(user)
    db_session.flush()
    
    client = Client(user_id=user.id, is_active=True)
    db_session.add(client)
    db_session.commit()
    
    return client, user

@pytest.fixture
def company_user(app, db_session):
    """Crée une entreprise de test"""
    user = User(
        username='testcompany',
        email='company@test.com',
        role=UserRole.COMPANY
    )
    user.set_password('companypass123')
    db_session.add(user)
    db_session.flush()
    
    company = Company(
        user_id=user.id,
        name='Test Transport SA',
        is_approved=True,
        dispatch_enabled=True
    )
    db_session.add(company)
    db_session.commit()
    
    return company, user

