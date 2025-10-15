"""
Fixtures pytest pour les tests backend ATMR.
"""
import os

# Mock JSONB → JSON AVANT tout import (SQLite ne supporte pas JSONB)
from sqlalchemy import JSON
from sqlalchemy.dialects import postgresql

postgresql.JSONB = JSON

import pytest
from flask import Flask

# Forcer environnement de test avant d'importer l'app
os.environ['FLASK_ENV'] = 'testing'
os.environ['PDF_BASE_URL'] = 'http://localhost:5000'  # Valeur factice pour tests

from app import create_app
from ext import db as _db
from models import Company, User, UserRole


@pytest.fixture(scope='session')
def app() -> Flask:
    """Crée une instance Flask en mode test."""

    app = create_app()
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'WTF_CSRF_ENABLED': False,
        'JWT_SECRET_KEY': 'test-secret-key',
        'SECRET_KEY': 'test-secret-key',
    })
    return app


@pytest.fixture(scope='function')
def db(app):
    """Crée une DB propre pour chaque test."""
    with app.app_context():
        _db.create_all()
        yield _db
        _db.session.remove()
        _db.drop_all()


@pytest.fixture
def client(app, db):
    """Client de test Flask."""
    return app.test_client()


@pytest.fixture
def sample_company(db, sample_user):
    """Crée une entreprise de test."""
    company = Company(
        name="Test Transport SA",
        address="Rue de Test 1, 1000 Lausanne",
        contact_phone="0211234567",
        contact_email="contact@test-transport.ch",
        user_id=sample_user.id
    )
    db.session.add(company)
    db.session.commit()
    return company


@pytest.fixture
def sample_user(db):
    """Crée un utilisateur de test (rôle company)."""
    from ext import bcrypt
    user = User(
        username='testuser',
        email='test@example.com',
        role=UserRole.company
    )
    user.password = bcrypt.generate_password_hash('password123').decode('utf-8')
    db.session.add(user)
    db.session.commit()
    return user


@pytest.fixture
def auth_headers(client, sample_user):
    """Génère un token JWT valide pour l'utilisateur test."""
    response = client.post('/api/auth/login', json={
        'email': 'test@example.com',
        'password': 'password123'
    })
    data = response.get_json()
    if not data or 'token' not in data:
        pytest.fail(f"Login failed: {response.get_json()}")

    token = data['token']
    return {'Authorization': f'Bearer {token}'}

