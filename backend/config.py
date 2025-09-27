# backend/config.py
import os
from datetime import timedelta

# Il est préférable de ne charger les variables d'environnement que si nécessaire
# from dotenv import load_dotenv
# load_dotenv()

base_dir = os.path.abspath(os.path.dirname(__file__))

class Config:
    """
    Configuration de base partagée.
    Ne contient AUCUNE clé secrète pour être sûr en toute circonstance.
    """
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 1800,
    }

    # --- JWT (délais par défaut, surclassables par env) ---
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(
        seconds=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES_SECONDS", str(60 * 60)))
    )
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(
        seconds=int(os.getenv("JWT_REFRESH_TOKEN_EXPIRES_SECONDS", str(30 * 24 * 3600)))
    )

    # --- Redis / Socket.IO ---
    REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    # Liste d'origines autorisées pour Socket.IO (séparées par des virgules).
    # Exemple: SOCKETIO_CORS_ORIGINS="https://app.example.com,https://admin.example.com,http://localhost:3000"
    SOCKETIO_CORS_ORIGINS = os.getenv("SOCKETIO_CORS_ORIGINS", "")

    # Ratelimit on/off (tests = off plus bas)
    RATELIMIT_ENABLED = True

    # Logique d'initialisation commune (optionnel)
    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    """Configuration pour le développement local."""
    DEBUG = True
    # Les clés sont chargées depuis .env pour le développement
    SECRET_KEY = os.getenv('SECRET_KEY')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
    # Supporte DATABASE_URL (standard) ou fallback local SQLite
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL') or os.getenv('DATABASE_URI', f"sqlite:///{base_dir}/development.db")
    
    # CORRECTION: Ajout des configurations de cookies pour le développement
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_SECURE = False
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = "Lax"

class ProductionConfig(Config):
    """Configuration pour la production."""
    DEBUG = False
    # En production, ces variables DOIVENT être fournies par l'environnement du serveur
    SECRET_KEY = os.getenv('SECRET_KEY')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')  # Ex: PostgreSQL, MySQL, etc.
    # Cookies plus stricts en prod (peuvent être ajustés via env si reverse proxy HTTP)
    SESSION_COOKIE_SECURE = os.getenv('SESSION_COOKIE_SECURE', 'true').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = os.getenv('SESSION_COOKIE_SAMESITE', 'Lax')
    REMEMBER_COOKIE_SECURE = os.getenv('REMEMBER_COOKIE_SECURE', 'true').lower() == 'true'
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = os.getenv('REMEMBER_COOKIE_SAMESITE', 'Lax')

class TestingConfig(Config):
    TESTING = True
    SECRET_KEY = 'test-secret-key'
    JWT_SECRET_KEY = 'test-jwt-key'
    MAIL_PASSWORD = 'test-mail-password'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False
    RATELIMIT_ENABLED = False


# C'est ce que votre "Application Factory" utilisera
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}