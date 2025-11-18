# backend/config.py
import os
from datetime import timedelta
from pathlib import Path
from typing import ClassVar

# Il est préférable de ne charger les variables d'environnement que si nécessaire
# from dotenv import load_dotenv
# load_dotenv()

# ✅ 4.1: Import Vault client (optionnel)
try:
    from shared.vault_client import get_vault_client as _get_vault_client
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False
    _get_vault_client = None

base_dir = Path(__file__).resolve().parent


def _get_secret_from_vault_or_env(
    vault_path: str,
    vault_key: str,
    env_key: str,
    default: str | None = None,
    required: bool = False,
) -> str | None:
    """✅ 4.1: Récupère un secret depuis Vault ou variable d'environnement.
    
    Essaie d'abord Vault, puis fallback vers .env.
    
    Args:
        vault_path: Chemin Vault (ex: "dev/flask/secret_key")
        vault_key: Clé dans Vault (généralement "value")
        env_key: Nom de la variable d'environnement
        default: Valeur par défaut si non trouvée
        required: Si True, lève une exception si non trouvé
        
    Returns:
        Valeur du secret ou None
    """
    if VAULT_AVAILABLE and _get_vault_client:
        try:
            vault = _get_vault_client()
            value = vault.get_secret(vault_path, vault_key, env_fallback=env_key, default=default)
            if value:
                return value
        except Exception:
            # Fallback silencieux vers .env en cas d'erreur Vault
            pass
    
    # Fallback vers variable d'environnement
    value = os.getenv(env_key, default)
    if value:
        return value
    
    if required:
        raise RuntimeError(f"Secret requis non trouvé: {env_key} (Vault path: {vault_path})")
    
    return None

class Config:
    """Configuration de base partagée.
    Ne contient AUCUNE clé secrète pour être sûr en toute circonstance.
    """

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Options de base compatibles avec toutes les bases de données
    SQLALCHEMY_ENGINE_OPTIONS: ClassVar[dict[str, int | bool]] = {
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
    REDIS_URL = os.getenv("REDIS_URL", "redis://127.00.1:6379/0")
    # Liste d'origines autorisées pour Socket.IO (séparées par des virgules).
    # Exemple: SOCKETIO_CORS_ORIGINS="https://app.example.com,https://admin.example.com,http://localhost:3000"
    SOCKETIO_CORS_ORIGINS = os.getenv("SOCKETIO_CORS_ORIGINS", "")

    # Ratelimit on/off (tests = off plus bas)
    RATELIMIT_ENABLED = True

    # ✅ URLs dynamiques pour PDFs/uploads
    PDF_BASE_URL = os.getenv("PDF_BASE_URL", "http://localhost:5000")
    UPLOADS_PUBLIC_BASE = os.getenv("UPLOADS_PUBLIC_BASE", "/uploads")

    # Logique d'initialisation commune (optionnel)
    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    """Configuration pour le développement local (PostgreSQL via Docker)."""

    DEBUG = True
    # ✅ 4.1: Chargement depuis Vault (si configuré) ou .env (fallback)
    SECRET_KEY = _get_secret_from_vault_or_env(
        vault_path="dev/flask/secret_key",
        vault_key="value",
        env_key="SECRET_KEY",
    )
    JWT_SECRET_KEY = _get_secret_from_vault_or_env(
        vault_path="dev/jwt/secret_key",
        vault_key="value",
        env_key="JWT_SECRET_KEY",
    )
    MAIL_PASSWORD = _get_secret_from_vault_or_env(
        vault_path="dev/mail/password",
        vault_key="value",
        env_key="MAIL_PASSWORD",
    )
    # PostgreSQL via Docker (DATABASE_URL doit être défini)
    # ✅ 4.1: Support dynamic secrets Database (via Vault) ou DATABASE_URL (fallback)
    SQLALCHEMY_DATABASE_URI = _get_secret_from_vault_or_env(
        vault_path="dev/database/url",
        vault_key="value",
        env_key="DATABASE_URL",
        default=os.getenv("DATABASE_URI"),
    )

    # ✅ PostgreSQL-specific options pour développement
    SQLALCHEMY_ENGINE_OPTIONS: ClassVar[dict[str, int | bool | dict[str, str]]] = {  # pyright: ignore[reportIncompatibleVariableOverride]
        **Config.SQLALCHEMY_ENGINE_OPTIONS,
        "pool_size": 10,        # ✅ PERF: Connection pooling (PostgreSQL uniquement)
        "max_overflow": 20,     # ✅ PERF: Max connections overflow (PostgreSQL uniquement)
        "connect_args": {"client_encoding": "utf8"}
    }

    # CORRECTION: Ajout des configurations de cookies pour le développement
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_SECURE = False
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = "Lax"

    # Dev: PDFs accessibles en local
    PDF_BASE_URL = os.getenv("PDF_BASE_URL", "http://localhost:5000")
    UPLOADS_PUBLIC_BASE = "/uploads"


class ProductionConfig(Config):
    """Configuration pour la production (PostgreSQL)."""

    DEBUG = False
    # ✅ 4.1: En production, utiliser Vault (recommandé) ou variables d'environnement
    SECRET_KEY = _get_secret_from_vault_or_env(
        vault_path="prod/flask/secret_key",
        vault_key="value",
        env_key="SECRET_KEY",
        required=True,  # Requis en production
    )
    JWT_SECRET_KEY = _get_secret_from_vault_or_env(
        vault_path="prod/jwt/secret_key",
        vault_key="value",
        env_key="JWT_SECRET_KEY",
        required=True,  # Requis en production
    )
    MAIL_PASSWORD = _get_secret_from_vault_or_env(
        vault_path="prod/mail/password",
        vault_key="value",
        env_key="MAIL_PASSWORD",
    )
    # ✅ 4.1: Support dynamic secrets Database (via Vault) pour rotation automatique
    SQLALCHEMY_DATABASE_URI = _get_secret_from_vault_or_env(
        vault_path="prod/database/url",
        vault_key="value",
        env_key="DATABASE_URL",
        required=True,  # Requis en production
    )

    # ✅ PostgreSQL-specific options
    SQLALCHEMY_ENGINE_OPTIONS: ClassVar[dict[str, int | bool | dict[str, str]]] = {  # pyright: ignore[reportIncompatibleVariableOverride]
        **Config.SQLALCHEMY_ENGINE_OPTIONS,
        "pool_size": 10,        # ✅ PERF: Connection pooling (PostgreSQL uniquement)
        "max_overflow": 20,     # ✅ PERF: Max connections overflow (PostgreSQL uniquement)
        "connect_args": {"client_encoding": "utf8"}
    }
    # Cookies plus stricts en prod (peuvent être ajustés via env si reverse proxy HTTP)
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "true").lower() == "true"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
    REMEMBER_COOKIE_SECURE = os.getenv("REMEMBER_COOKIE_SECURE", "true").lower() == "true"
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = os.getenv("REMEMBER_COOKIE_SAMESITE", "Lax")

    # ✅ Prod: URL backend publique (depuis env)
    PDF_BASE_URL = os.getenv("PDF_BASE_URL")
    if not PDF_BASE_URL:
        pass  # PDF_BASE_URL sera validé au runtime si nécessaire

class TestingConfig(Config):
    TESTING = True
    SECRET_KEY = "test-secret-key"
    JWT_SECRET_KEY = "test-jwt-key"
    MAIL_PASSWORD = "test-mail-password"
    # Utiliser DATABASE_URL si disponible (PostgreSQL), sinon SQLite en mémoire
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///:memory:")
    # Options de base uniquement (pas de pool_size/max_overflow pour compatibilité SQLite)
    SQLALCHEMY_ENGINE_OPTIONS: ClassVar[dict[str, int | bool]] = {
        "pool_pre_ping": True,
        "pool_recycle": 1800,
    }
    WTF_CSRF_ENABLED = False
    RATELIMIT_ENABLED = False
    PDF_BASE_URL = "http://testserver"
    UPLOADS_PUBLIC_BASE = "/uploads"
    
    @staticmethod
    def init_app(app):
        """Ajuste les options d'engine selon le type de base de données."""
        # Utiliser DATABASE_URL de l'environnement si disponible (priorité sur valeur par défaut)
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            app.config["SQLALCHEMY_DATABASE_URI"] = database_url
        
        # Ajouter options PostgreSQL uniquement si DATABASE_URL pointe vers PostgreSQL
        db_uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
        if db_uri and db_uri.startswith("postgresql"):
            app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
                **app.config.get("SQLALCHEMY_ENGINE_OPTIONS", {}),
                "pool_size": 5,        # Pool plus petit pour les tests
                "max_overflow": 10,     # Overflow plus petit pour les tests
            }


# C'est ce que votre "Application Factory" utilisera
config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}
