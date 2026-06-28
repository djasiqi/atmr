# backend/config.py
import os
from datetime import timedelta
from pathlib import Path
from typing import Callable, ClassVar
from urllib.parse import quote_plus

# Options SQLAlchemy create_engine ; connect_args inclut int (ex. prepare_threshold) pour PgBouncer.
_SqlalchemyEngineOptions = dict[str, int | bool | dict[str, str | int | bool]]

# Il est préférable de ne charger les variables d'environnement que si nécessaire
# from dotenv import load_dotenv
# load_dotenv()

# ✅ 4.1: Import Vault client (optionnel)
_get_vault_client: Callable[[], "VaultClient"] | None = None
VAULT_AVAILABLE = False

try:
    from shared.vault_client import VaultClient, get_vault_client

    _get_vault_client = get_vault_client
    VAULT_AVAILABLE = True
except ImportError:
    pass

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
    if VAULT_AVAILABLE and _get_vault_client is not None:
        try:
            vault = _get_vault_client()
            value = vault.get_secret(
                vault_path, vault_key, env_fallback=env_key, default=default
            )
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
        raise RuntimeError(
            f"Secret requis non trouvé: {env_key} (Vault path: {vault_path})"
        )

    return None


def _is_psycopg3_dialect_in_url(db_url: str | None) -> bool:
    """True si l'URL SQLAlchemy cible le dialecte psycopg (v3), pas psycopg2.

    prepare_threshold (PgBouncer mode transaction) n'est supporté que par
    psycopg>=3. Avec le dialecte par défaut postgresql:// (psycopg2), l'inclure
    dans connect_args provoque: invalid connection option "prepare_threshold".
    """
    if not db_url:
        return False
    head = db_url.split("://", 1)[0].lower()
    if head.endswith("+psycopg2") or head.endswith("+pypostgresql"):
        return False
    return "+psycopg" in head


def _is_internal_database_host(host: str) -> bool:
    """Détecte si l'hôte de base de données est sur le réseau Docker interne.

    Args:
        host: Nom d'hôte ou adresse IP de la base de données

    Returns:
        True si l'hôte est sur le réseau Docker interne (pas besoin de SSL)
    """
    # Hôtes Docker internes (noms de service)
    internal_hosts = ["postgres", "localhost", "127.0.0.1", "0.0.0.0"]
    if host in internal_hosts:
        return True

    # IPs privées (RFC 1918)
    if (
        host.startswith("10.")
        or host.startswith("172.16.")
        or host.startswith("192.168.")
    ):
        return True

    # IPs Docker (172.17.0.0/16, 172.18.0.0/16, etc.)
    return host.startswith("172.17.") or host.startswith("172.18.")


def _build_database_url_safe() -> str:
    """Construit DATABASE_URL depuis variables individuelles avec échappement URL.

    Si DATABASE_URL existe déjà dans l'environnement (depuis Vault ou env),
    il est utilisé tel quel. Sinon, l'URL est construite depuis les variables
    individuelles POSTGRES_* avec échappement automatique du mot de passe.

    Pour les hôtes Docker internes, SSL est automatiquement désactivé (sslmode=disable).

    Returns:
        URL de connexion PostgreSQL avec mot de passe échappé si nécessaire.
    """
    # Vérifier si DATABASE_URL existe déjà
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        # Si DATABASE_URL contient déjà sslmode, on le garde tel quel
        # Sinon, on détecte si c'est un hôte interne et on ajoute sslmode=disable
        if "sslmode=" not in db_url:
            from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

            parsed = urlparse(db_url)
            host = parsed.hostname or ""
            if _is_internal_database_host(host):
                # Ajouter sslmode=disable pour hôtes internes
                query_params = parse_qs(parsed.query)
                query_params["sslmode"] = ["disable"]
                new_query = urlencode(query_params, doseq=True)
                new_parsed = parsed._replace(query=new_query)
                return urlunparse(new_parsed)
        return db_url

    # Construire depuis variables individuelles
    # ✅ CORRECTION: Valeurs par défaut alignées sur PostgreSQL réel (atmr/atmr)
    user = os.getenv("POSTGRES_USER", "atmr")
    password = os.getenv("POSTGRES_PASSWORD", "atmr_password")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "atmr")

    # Échapper le mot de passe pour URL (caractères spéciaux comme !, @, #, etc.)
    password_escaped = quote_plus(password)

    # Détecter si c'est un hôte interne et ajouter sslmode=disable si nécessaire
    sslmode = os.getenv("POSTGRES_SSLMODE")
    if not sslmode and _is_internal_database_host(host):
        sslmode = "disable"

    base_url = f"postgresql://{user}:{password_escaped}@{host}:{port}/{db}"
    if sslmode:
        return f"{base_url}?sslmode={sslmode}"

    return base_url


def _parse_replica_database_urls() -> list[str]:
    """Parse la liste de replicas depuis l'environnement.

    Sources supportées (ordre de priorité):
    - REPLICA_DATABASE_URLS: CSV d'URLs
    - REPLICA_DATABASE_URL: URL unique
    """
    replica_urls_csv = (os.getenv("REPLICA_DATABASE_URLS") or "").strip()
    if replica_urls_csv:
        return [url.strip() for url in replica_urls_csv.split(",") if url.strip()]

    replica_url_single = (os.getenv("REPLICA_DATABASE_URL") or "").strip()
    if replica_url_single:
        return [replica_url_single]

    return []


def _build_sqlalchemy_binds(primary_url: str | None) -> dict[str, str]:
    """Construit les binds SQLAlchemy primary/replica pour Phase 4.

    Note:
    - Le bind `replica` pointe sur la première URL replica (si fournie).
    - Le round-robin multi-replicas est géré au niveau applicatif
      par un routeur dédié, pas par SQLAlchemy_Binds.
    """
    binds: dict[str, str] = {}
    if primary_url:
        binds["primary"] = primary_url

    replicas = _parse_replica_database_urls()
    if replicas:
        binds["replica"] = replicas[0]

    return binds


class Config:
    """Configuration de base partagée.
    Ne contient AUCUNE clé secrète pour être sûr en toute circonstance.
    """

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # ✅ Phase 1 N+1: Configuration profilage SQLAlchemy
    # Activer SQLALCHEMY_ECHO pour logger toutes les requêtes SQL
    # (développement uniquement)
    SQLALCHEMY_ECHO = os.getenv("SQLALCHEMY_ECHO", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    # Activer SQLALCHEMY_RECORD_QUERIES pour Flask-SQLAlchemy query tracking
    SQLALCHEMY_RECORD_QUERIES = os.getenv(
        "SQLALCHEMY_RECORD_QUERIES", "false"
    ).lower() in (
        "true",
        "1",
        "yes",
    )
    # Timeout pour requêtes SQL (secondes)
    SQLALCHEMY_QUERY_TIMEOUT = float(os.getenv("SQLALCHEMY_QUERY_TIMEOUT", "5.0"))
    # Options de base compatibles avec toutes les bases de données
    SQLALCHEMY_ENGINE_OPTIONS: ClassVar[_SqlalchemyEngineOptions] = {
        "pool_pre_ping": True,
        "pool_recycle": 1800,
    }

    # --- JWT (délais par défaut, surclassables par env) ---
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(
        seconds=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES_SECONDS", str(60 * 60)))
    )
    # Mobile natif (iOS/Android): fenêtre plus longue pour réduire les
    # déconnexions perçues lors des reprises réseau/app en arrière-plan.
    JWT_MOBILE_ACCESS_TOKEN_EXPIRES = timedelta(
        seconds=int(
            os.getenv(
                "JWT_MOBILE_ACCESS_TOKEN_EXPIRES_SECONDS",
                str(24 * 60 * 60),
            )
        )
    )
    # Tolérance d'horloge pour limiter les faux "token expired" (drift client/serveur).
    JWT_DECODE_LEEWAY = int(os.getenv("JWT_DECODE_LEEWAY_SECONDS", "120"))
    # ✅ PHASE 4 : Augmentation de la durée du refresh token de 30 à 90 jours
    # Améliore l'expérience utilisateur en réduisant les déconnexions
    # ⚠️ SÉCURITÉ : Les refresh tokens sont stockés dans Redis avec rotation et révocation
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(
        seconds=int(os.getenv("JWT_REFRESH_TOKEN_EXPIRES_SECONDS", str(90 * 24 * 3600)))
    )
    # Validation de l'audience JWT (prévention token replay)
    # Désactivé pour permettre plusieurs audiences :
    # - "atmr-api" pour web/chauffeur mobile
    # - "atmr-mobile-enterprise" pour entreprise mobile
    # La validation est faite manuellement dans les routes concernées
    JWT_DECODE_AUDIENCE = None  # None = désactive la validation automatique
    # Algorithme de signature JWT (HS256 = symétrique, secret partagé via Vault)
    JWT_ALGORITHM = "HS256"
    # Sprint 2 H14: rotation JWT versionnée par kid.
    JWT_ACTIVE_KID = os.getenv("JWT_ACTIVE_KID", "active")
    JWT_ACCEPTED_KIDS = os.getenv("JWT_ACCEPTED_KIDS", "")
    # Compat tokens legacy sans kid pendant transition.
    JWT_LEGACY_DEFAULT_KID = os.getenv("JWT_LEGACY_DEFAULT_KID", JWT_ACTIVE_KID)

    # --- Cookies httpOnly pour tokens JWT ---
    # ✅ Migration localStorage → cookies httpOnly pour protection XSS
    # Configuration des cookies pour les tokens JWT
    COOKIE_SECURE = os.getenv("COOKIE_SECURE", "true").lower() == "true"
    COOKIE_SAME_SITE = os.getenv("COOKIE_SAME_SITE", "Strict")
    COOKIE_HTTP_ONLY = os.getenv("COOKIE_HTTP_ONLY", "true").lower() == "true"
    COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN", None)  # Optionnel (pour sous-domaines)
    COOKIE_PATH = os.getenv("COOKIE_PATH", "/")  # Chemin des cookies
    # Noms des cookies
    COOKIE_ACCESS_TOKEN_NAME = "access_token"
    COOKIE_REFRESH_TOKEN_NAME = "refresh_token"

    # --- Flask-JWT-Extended : Configuration pour lire depuis cookies ---
    # ✅ Migration localStorage → cookies httpOnly
    # JWT_TOKEN_LOCATION spécifie où chercher les tokens (cookies, headers, etc.)
    # On supporte à la fois cookies (web) et headers (mobile)
    JWT_TOKEN_LOCATION: ClassVar[list[str]] = [
        "cookies",
        "headers",
    ]  # Priorité : cookies puis headers
    JWT_COOKIE_SECURE = COOKIE_SECURE  # HTTPS uniquement si Secure=true
    JWT_COOKIE_HTTP_ONLY = COOKIE_HTTP_ONLY  # Protection XSS
    JWT_COOKIE_SAMESITE = COOKIE_SAME_SITE  # Protection CSRF
    # NOTE (CSRF) :
    # The application already enforces CSRF protection via the custom middleware
    # (backend/services/security/csrf.py) using X-CSRF-Token headers.
    #
    # Flask-JWT-Extended provides its own cookie CSRF mechanism
    # (JWT_COOKIE_CSRF_PROTECT), but enabling it would introduce a second,
    # incompatible CSRF validation layer unless login endpoints explicitly set
    # the JWT CSRF cookies and integration tests are updated accordingly.
    #
    # For now we keep JWT_COOKIE_CSRF_PROTECT = False and rely on the existing
    # application-level CSRF middleware which protects both web cookie flows
    # and API header-based authentication.
    #
    # If future migration to native Flask-JWT CSRF is desired:
    # 1. ensure login endpoints set CSRF cookies
    # 2. align frontend header expectations
    # 3. add integration tests for cookie-auth mutations
    JWT_COOKIE_CSRF_PROTECT = False  # intentionally False — see NOTE above
    JWT_ACCESS_COOKIE_NAME = COOKIE_ACCESS_TOKEN_NAME
    JWT_REFRESH_COOKIE_NAME = COOKIE_REFRESH_TOKEN_NAME
    JWT_COOKIE_DOMAIN = COOKIE_DOMAIN
    JWT_COOKIE_PATH = COOKIE_PATH

    # Portail client (ClientType PORTAL) : entreprise de référence pour
    # POST /clients/me/bookings/preview (grille de tarification entreprise).
    # 0 = le backend utilise la grille indicatif plateforme
    # (``PlatformClientIndicativeFareConfig``), alignee sur
    # ``POST /clients/me/indicative-fare/estimate`` — meme ordre de grandeur
    # que l'indicatif affiche dans l'app. Sinon, ID d'une company dont le
    # PricingProfile sert de prévisualisation contractuelle.
    PORTAL_CLIENT_PREVIEW_COMPANY_ID: int = int(
        (os.getenv("PORTAL_CLIENT_PREVIEW_COMPANY_ID") or "0").strip() or "0"
    )

    # --- Redis / Socket.IO ---
    REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    # Liste d'origines autorisées pour Socket.IO (séparées par des virgules).
    # Exemple: SOCKETIO_CORS_ORIGINS="https://app.example.com,
    # https://admin.example.com,http://localhost:3000"
    SOCKETIO_CORS_ORIGINS = os.getenv("SOCKETIO_CORS_ORIGINS", "")

    # Ratelimit on/off (tests = off plus bas)
    RATELIMIT_ENABLED = True

    # ✅ Reinforcement Learning / ML lourd
    # En production, RL_ENABLED=false (pas de torch, gymnasium, optuna)
    # En environnement RL, RL_ENABLED=true (image Dockerfile.rl)
    RL_ENABLED = os.getenv("RL_ENABLED", "false").lower() in ("true", "1", "yes")
    WITH_RL = os.getenv("WITH_RL", "false").lower() in ("true", "1", "yes")

    # ✅ URLs dynamiques pour PDFs/uploads
    # PDF_PUBLIC_BASE_URL = URL écrite dans les PDFs (utilisée par le navigateur)
    # - Dev local Windows/Docker: http://127.0.0.1:5000 (évite IPv6 localhost + ERR_CONNECTION_RESET)
    # - Docker interne: http://backend:5000 (si backend s'appelle lui-même)
    # - Prod: https://api.lirie.ch (ou domaine configuré)
    # ⚠️ Pas de fallback "localhost" — source unique de vérité via env var
    PDF_PUBLIC_BASE_URL: str = os.getenv("PDF_PUBLIC_BASE_URL") or os.getenv(
        "PDF_BASE_URL", "http://127.0.0.1:5000"
    )
    # Alias pour rétrocompatibilité
    PDF_BASE_URL: str = PDF_PUBLIC_BASE_URL
    UPLOADS_PUBLIC_BASE = os.getenv("UPLOADS_PUBLIC_BASE", "/uploads")
    # URL publique de base pour les emails (logos en mode URL). Priorité: PUBLIC_BASE_URL > PDF_BASE_URL
    # Prod-grade: PUBLIC_BASE_URL doit pointer vers le domaine public (pas localhost).
    # /uploads/company_logos/... doit être accessible sans auth avec Content-Type image/*.
    PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL") or os.getenv(
        "PDF_BASE_URL", "http://127.0.0.1:5000"
    )
    # --- Gateway auth local unifié (Sprint 3) ---
    # API app locale (même service)
    GATEWAY_APP_AUTH_URL = os.getenv(
        "GATEWAY_APP_AUTH_URL", "http://127.0.0.1:5000/api/v1/auth/login"
    )
    GATEWAY_APP_ME_URL = os.getenv(
        "GATEWAY_APP_ME_URL", "http://127.0.0.1:5000/api/v1/auth/me"
    )
    # API demo locale (par défaut même backend unifié en prod)
    GATEWAY_DEMO_AUTH_URL = os.getenv(
        "GATEWAY_DEMO_AUTH_URL", "http://127.0.0.1:5000/api/v1/auth/login"
    )
    GATEWAY_DEMO_ME_URL = os.getenv(
        "GATEWAY_DEMO_ME_URL", "http://127.0.0.1:5000/api/v1/auth/me"
    )
    # --- Admin Ops / Platform (GET /api/v1/platform/status) ---
    # Base URL sans suffixe /api/v1 (ex. https://api.lirie.ch). Vide = non monitoré.
    PLATFORM_API_URL_PROD = (os.getenv("PLATFORM_API_URL_PROD") or "").strip() or None
    PLATFORM_API_URL_DEMO = (os.getenv("PLATFORM_API_URL_DEMO") or "").strip() or None
    PLATFORM_LINK_GRAFANA = (os.getenv("PLATFORM_LINK_GRAFANA") or "").strip() or None
    PLATFORM_LINK_PROMETHEUS = (
        os.getenv("PLATFORM_LINK_PROMETHEUS") or ""
    ).strip() or None
    PLATFORM_LINK_ALERTMANAGER = (
        os.getenv("PLATFORM_LINK_ALERTMANAGER") or ""
    ).strip() or None
    PLATFORM_STATUS_TIMEOUT_SECONDS = float(
        os.getenv("PLATFORM_STATUS_TIMEOUT_SECONDS", "2.5")
    )
    # Si true en production : ADMIN_IP_WHITELIST doit être non vide (fail-closed pour platform/status).
    ADMIN_IP_WHITELIST_REQUIRED: bool = os.getenv(
        "ADMIN_IP_WHITELIST_REQUIRED", "false"
    ).strip().lower() in ("true", "1", "yes")
    # Mode envoi email factures/rappels: brevo_api (URL logo) | brevo_smtp (CID MIME multipart/related)
    # SMTP: BREVO_SMTP_PASSWORD obligatoire (pas de fallback API_KEY). Port 587 sortant + DKIM/SPF.
    EMAIL_PROVIDER_MODE: str = (
        (os.getenv("EMAIL_PROVIDER_MODE", "brevo_api") or "brevo_api").strip().lower()
    )

    # --- Rate Limiting Configuration ---
    # ✅ Configuration adaptative selon l'environnement
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    # Stratégie de rate limiting
    # - "fixed-window": Fenêtre fixe (plus rapide, moins précis)
    # - "moving-window": Fenêtre glissante (plus précis, plus coûteux en CPU)
    RATELIMIT_STRATEGY = os.getenv("RATELIMIT_STRATEGY", "moving-window")

    # Rate limits par défaut (conservateurs pour la production)
    # Ces valeurs seront surclassées dans DevelopmentConfig/ProductionConfig
    RATELIMIT_DEFAULT_LIMITS: ClassVar[list[str]] = ["1000 per hour"]
    RATELIMIT_DISPATCH_RUN = "30 per hour"
    RATELIMIT_DISPATCH_TRIGGER = "50 per hour"
    RATELIMIT_COMPANY_DISPATCH_RUN = "10 per minute"
    RATELIMIT_COMPANY_DISPATCH_OPTIMIZER = "10 per minute"

    # Version de configuration pour versioning des clés Redis
    # Incrémenter manuellement lors de changements majeurs de rate limits
    RATELIMIT_CONFIG_VERSION = os.getenv("RATELIMIT_CONFIG_VERSION", "v1")

    # ✅ C2 Phase 2: TTL automatique sur les clés de rate limit
    # Durée de vie par défaut: 2 heures (7200 secondes)
    # Les clés expirent automatiquement, évitant l'accumulation en mémoire Redis
    # Valeur recommandée: 2-4 heures (7200-14400 secondes)
    RATELIMIT_KEY_TTL = int(os.getenv("RATELIMIT_KEY_TTL", "7200"))

    # Logique d'initialisation commune (optionnel)
    @staticmethod
    def init_app(app):
        pass


def validate_production_security(app) -> None:
    """Valide les paramètres de sécurité critiques en production.

    Cette fonction vérifie que les paramètres de sécurité des cookies
    sont correctement configurés en production. Elle lève une RuntimeError
    si la configuration est dangereuse.

    Args:
        app: Instance Flask

    Raises:
        RuntimeError: Si la configuration de sécurité est invalide en production
    """
    # Vérifier si on est en production
    flask_env = (
        app.config.get("FLASK_ENV")
        or os.getenv("FLASK_ENV")
        or os.getenv("FLASK_CONFIG")
    )

    if flask_env == "production":
        # Vérifier les secrets critiques
        secret_key = app.config.get("SECRET_KEY") or os.getenv("SECRET_KEY")
        jwt_secret_key = app.config.get("JWT_SECRET_KEY") or os.getenv("JWT_SECRET_KEY")

        if not secret_key:
            raise RuntimeError(
                "❌ SECURITÉ: SECRET_KEY doit être défini en production. "
                + "Placez la valeur dans Vault ou dans la variable "
                + "d'environnement SECRET_KEY."
            )
        if not jwt_secret_key:
            raise RuntimeError(
                "❌ SECURITÉ: JWT_SECRET_KEY doit être défini en production. "
                + "Placez la valeur dans Vault ou dans la variable "
                + "d'environnement JWT_SECRET_KEY."
            )

        # Vérifier COOKIE_SECURE
        cookie_secure = app.config.get("COOKIE_SECURE")
        if not cookie_secure:
            raise RuntimeError(
                "❌ SECURITÉ: COOKIE_SECURE doit être True en production. "
                + "Définissez COOKIE_SECURE=true dans les variables d'environnement."
            )

        # Vérifier COOKIE_HTTP_ONLY
        cookie_httponly = app.config.get("COOKIE_HTTP_ONLY")
        if not cookie_httponly:
            raise RuntimeError(
                "❌ SECURITÉ: COOKIE_HTTP_ONLY doit être True en production."
            )

        # Vérifier COOKIE_SAME_SITE
        samesite = app.config.get("COOKIE_SAME_SITE")
        if samesite not in ["Strict", "Lax"]:
            raise RuntimeError(
                "❌ SECURITÉ: COOKIE_SAME_SITE doit être 'Strict' "
                + f"ou 'Lax' en production, actuellement: {samesite}"
            )

        # ✅ Verrouillage démo: avertir si GATEWAY_* utilisent des fallbacks fragiles en prod
        import logging

        _logger = logging.getLogger(__name__)
        for key in ("GATEWAY_APP_AUTH_URL", "GATEWAY_DEMO_AUTH_URL"):
            url = app.config.get(key) or os.getenv(key, "")
            if url and (
                "127.0.0.1" in url
                or "localhost" in url
                or "host.docker.internal" in url
            ):
                _logger.warning(
                    "[config] %s=%s : en prod Docker, utiliser http://backend:5000/... (définir explicitement dans .env.production)",
                    key,
                    url,
                )


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
    # ✅ S3: Support legacy keys JWT (chargement depuis Vault ou env)
    # Format env: "key1,key2,key3" (séparées par virgule)
    # Format Vault: liste dans "keys" (retournée comme liste ou JSON string)
    _jwt_legacy_keys_raw = _get_secret_from_vault_or_env(
        vault_path="dev/jwt/legacy_secret_keys",
        vault_key="keys",
        env_key="JWT_LEGACY_SECRET_KEYS",
    )
    if _jwt_legacy_keys_raw:
        # Si depuis Vault, peut être une liste ou une string JSON
        import json

        try:
            if isinstance(_jwt_legacy_keys_raw, list):
                # Liste directement depuis Vault
                JWT_LEGACY_SECRET_KEYS = ",".join(_jwt_legacy_keys_raw)
            else:
                # À ce point, _jwt_legacy_keys_raw est forcément une str
                # (pas None car vérifié dans le if)
                legacy_keys_str: str = str(_jwt_legacy_keys_raw)
                if legacy_keys_str.startswith("["):
                    # Format JSON string depuis Vault
                    JWT_LEGACY_SECRET_KEYS = ",".join(json.loads(legacy_keys_str))
                else:
                    # Format string depuis env (déjà formaté "key1,key2")
                    JWT_LEGACY_SECRET_KEYS = legacy_keys_str
        except Exception:
            JWT_LEGACY_SECRET_KEYS = str(_jwt_legacy_keys_raw)
    else:
        JWT_LEGACY_SECRET_KEYS = os.getenv("JWT_LEGACY_SECRET_KEYS", "")
    MAIL_PASSWORD = _get_secret_from_vault_or_env(
        vault_path="dev/mail/password",
        vault_key="value",
        env_key="MAIL_PASSWORD",
    )
    # Supprime l'envoi SMTP réel en dev sauf si MAIL_SUPPRESS_SEND=false explicite.
    # Évite ConnectionRefusedError sur localhost:25 (défaut Flask-Mail sans MAIL_SERVER).
    MAIL_SUPPRESS_SEND: bool = (
        os.getenv("MAIL_SUPPRESS_SEND", "true").lower() != "false"
    )
    # PostgreSQL via Docker (DATABASE_URL doit être défini)
    # ✅ 4.1: Support dynamic secrets Database (via Vault) ou DATABASE_URL (fallback)
    SQLALCHEMY_DATABASE_URI = _get_secret_from_vault_or_env(
        vault_path="dev/database/url",
        vault_key="value",
        env_key="DATABASE_URL",
        default=os.getenv("DATABASE_URI"),
    )
    SQLALCHEMY_BINDS = _build_sqlalchemy_binds(SQLALCHEMY_DATABASE_URI)

    # ✅ PostgreSQL-specific options pour développement
    # prepare_threshold=0 : requis avec PgBouncer en mode transaction (psycopg3
    # uniquement ; exclu si postgresql:// → psycopg2).
    _dev_db_uri = SQLALCHEMY_DATABASE_URI
    _dev_connect_args: ClassVar[dict[str, str | int]] = {
        "client_encoding": "utf8",
    }
    if _is_psycopg3_dialect_in_url(_dev_db_uri):
        _dev_connect_args["prepare_threshold"] = 0
    SQLALCHEMY_ENGINE_OPTIONS: ClassVar[_SqlalchemyEngineOptions] = {
        **Config.SQLALCHEMY_ENGINE_OPTIONS,
        "pool_size": 10,  # ✅ PERF: Connection pooling (PostgreSQL uniquement)
        "max_overflow": 20,  # ✅ PERF: Max connections overflow (PostgreSQL uniquement)
        "connect_args": _dev_connect_args,
    }

    # CORRECTION: Ajout des configurations de cookies pour le développement
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_SECURE = False
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = "Lax"

    # ✅ Cookies JWT pour développement (Secure=false pour localhost HTTP)
    COOKIE_SECURE = False  # HTTP en dev
    COOKIE_SAME_SITE = "Lax"  # Plus permissif en dev

    # Dev: PDFs accessibles en local — 127.0.0.1 évite IPv6 localhost + ERR_CONNECTION_RESET
    PDF_PUBLIC_BASE_URL = os.getenv("PDF_PUBLIC_BASE_URL") or os.getenv(
        "PDF_BASE_URL", "http://127.0.0.1:5000"
    )
    PDF_BASE_URL = PDF_PUBLIC_BASE_URL
    UPLOADS_PUBLIC_BASE = "/uploads"

    # ✅ Phase 1 N+1: Profilage activé par défaut en développement
    SQLALCHEMY_ECHO = os.getenv("SQLALCHEMY_ECHO", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    SQLALCHEMY_RECORD_QUERIES = os.getenv(
        "SQLALCHEMY_RECORD_QUERIES", "false"
    ).lower() in (
        "true",
        "1",
        "yes",
    )

    # ✅ Rate Limits Development: Très permissifs pour faciliter le développement
    ENVIRONMENT = "development"
    RATELIMIT_DEFAULT_LIMITS: ClassVar[list[str]] = [
        os.getenv("RATELIMIT_DEFAULT_LIMITS", "100000 per hour")
    ]
    RATELIMIT_DISPATCH_RUN = os.getenv("RATELIMIT_DISPATCH_RUN", "10000 per hour")
    RATELIMIT_DISPATCH_TRIGGER = os.getenv(
        "RATELIMIT_DISPATCH_TRIGGER", "10000 per hour"
    )
    RATELIMIT_COMPANY_DISPATCH_RUN = os.getenv(
        "RATELIMIT_COMPANY_DISPATCH_RUN", "10000 per hour"
    )
    RATELIMIT_COMPANY_DISPATCH_OPTIMIZER = os.getenv(
        "RATELIMIT_COMPANY_DISPATCH_OPTIMIZER", "10000 per hour"
    )


class ProductionConfig(Config):
    """Configuration pour la production (PostgreSQL)."""

    DEBUG = False
    # ✅ 4.1: En production, utiliser Vault (recommandé)
    # ou variables d'environnement
    # Note: required=False pour permettre l'import en CI,
    # validation au runtime dans validate_production_security
    SECRET_KEY = _get_secret_from_vault_or_env(
        vault_path="prod/flask/secret_key",
        vault_key="value",
        env_key="SECRET_KEY",
        required=False,
    )
    JWT_SECRET_KEY = _get_secret_from_vault_or_env(
        vault_path="prod/jwt/secret_key",
        vault_key="value",
        env_key="JWT_SECRET_KEY",
        required=False,
    )
    # ✅ S3: Support legacy keys JWT (chargement depuis Vault ou env)
    _jwt_legacy_keys_raw = _get_secret_from_vault_or_env(
        vault_path="prod/jwt/legacy_secret_keys",
        vault_key="keys",
        env_key="JWT_LEGACY_SECRET_KEYS",
    )
    if _jwt_legacy_keys_raw:
        import json

        try:
            if isinstance(_jwt_legacy_keys_raw, list):
                JWT_LEGACY_SECRET_KEYS = ",".join(_jwt_legacy_keys_raw)
            else:
                # À ce point, _jwt_legacy_keys_raw est forcément une str
                # (pas None car vérifié dans le if)
                legacy_keys_str: str = str(_jwt_legacy_keys_raw)
                if legacy_keys_str.startswith("["):
                    JWT_LEGACY_SECRET_KEYS = ",".join(json.loads(legacy_keys_str))
                else:
                    JWT_LEGACY_SECRET_KEYS = legacy_keys_str
        except Exception:
            JWT_LEGACY_SECRET_KEYS = str(_jwt_legacy_keys_raw)
    else:
        JWT_LEGACY_SECRET_KEYS = os.getenv("JWT_LEGACY_SECRET_KEYS", "")
    MAIL_PASSWORD = _get_secret_from_vault_or_env(
        vault_path="prod/mail/password",
        vault_key="value",
        env_key="MAIL_PASSWORD",
    )
    # ✅ 4.1: Support dynamic secrets Database (via Vault)
    # pour rotation automatique
    # Construire DATABASE_URL depuis Vault/env,
    # ou depuis variables individuelles avec échappement
    _db_url_from_secret = _get_secret_from_vault_or_env(
        vault_path="prod/database/url",
        vault_key="value",
        env_key="DATABASE_URL",
        required=False,  # Pas requis si variables individuelles disponibles
    )
    SQLALCHEMY_DATABASE_URI = (
        _db_url_from_secret if _db_url_from_secret else _build_database_url_safe()
    )
    SQLALCHEMY_BINDS = _build_sqlalchemy_binds(SQLALCHEMY_DATABASE_URI)
    # Validation explicite pour éviter les erreurs
    # lors de l'initialisation Flask-SQLAlchemy
    if not SQLALCHEMY_DATABASE_URI:
        error_msg = (
            "SQLALCHEMY_DATABASE_URI must be set. "
            "Provide either DATABASE_URL or "
            "POSTGRES_USER/POSTGRES_PASSWORD/POSTGRES_DB"
        )
        raise RuntimeError(error_msg)

    # ✅ PostgreSQL-specific options
    # Détecter automatiquement si SSL est nécessaire selon l'hôte
    _db_uri = SQLALCHEMY_DATABASE_URI
    _db_host = ""
    if _db_uri:
        from urllib.parse import urlparse

        parsed = urlparse(_db_uri)
        _db_host = parsed.hostname or ""

    # Pour les hôtes Docker internes, désactiver SSL automatiquement
    # Pour les DB externes (managed), SSL reste requis par défaut
    _default_sslmode = "disable" if _is_internal_database_host(_db_host) else "require"
    _sslmode = os.getenv("POSTGRES_SSLMODE", _default_sslmode)

    # Phase 4 : pool_size augmenté à 50 pour absorber 100k chauffeurs.
    # Calcul : 12 workers x 50 = 600 connexions max -> PgBouncer multiplex vers 200 vraies.
    # Configurable via DB_POOL_SIZE pour ajuster selon la RAM du serveur DB.
    _prod_pool_size = int(os.getenv("DB_POOL_SIZE", "50"))
    _prod_pool_max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    _prod_pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    _prod_pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "1800"))
    SQLALCHEMY_ENGINE_OPTIONS: ClassVar[_SqlalchemyEngineOptions] = {
        **Config.SQLALCHEMY_ENGINE_OPTIONS,
        "pool_size": _prod_pool_size,
        "max_overflow": _prod_pool_max_overflow,
        "pool_timeout": _prod_pool_timeout,
        "pool_pre_ping": True,
        "pool_recycle": _prod_pool_recycle,
        "connect_args": {
            "client_encoding": "utf8",
            # ✅ SECURITY: SSL désactivé pour Docker interne,
            # requis pour DB externes
            "sslmode": _sslmode,
            **(
                {"prepare_threshold": 0} if _is_psycopg3_dialect_in_url(_db_uri) else {}
            ),
        },
    }
    # Cookies plus stricts en prod (peuvent être ajustés via env si reverse proxy HTTP)
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "true").lower() == "true"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
    REMEMBER_COOKIE_SECURE = (
        os.getenv("REMEMBER_COOKIE_SECURE", "true").lower() == "true"
    )
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = os.getenv("REMEMBER_COOKIE_SAMESITE", "Lax")

    # ✅ Cookies JWT pour production (Secure=true, Strict pour sécurité maximale)
    COOKIE_SECURE = True  # HTTPS uniquement en prod
    COOKIE_SAME_SITE = "Strict"  # Protection CSRF maximale

    # ✅ Prod: URL backend publique (depuis env)
    # En production, PDF_BASE_URL est REQUIS et doit être HTTPS
    # Exception: localhost/127.0.0.1 autorisé en développement local
    # ⚠️ Ne vérifier que si on est réellement en production
    # (pas lors de l'import du module)
    _flask_config = os.getenv("FLASK_CONFIG", default=None)
    _pdf_base_url = os.getenv("PDF_BASE_URL", default=None)
    if _flask_config == "production":
        if not _pdf_base_url:
            raise RuntimeError(
                "PDF_BASE_URL est requis en production. "
                + "Définissez cette variable d'environnement avec une URL HTTPS valide."
            )
        # ✅ Exception pour développement local (localhost/127.0.0.1)
        _is_localhost = _pdf_base_url.startswith(
            "http://localhost"
        ) or _pdf_base_url.startswith("http://127.0.0.1")
        if not _pdf_base_url.startswith("https://") and not _is_localhost:
            raise RuntimeError(
                "PDF_BASE_URL doit commencer par 'https://' en production. "
                + f"Valeur actuelle: {_pdf_base_url}"
            )
        PDF_BASE_URL = _pdf_base_url
    else:
        # Pour les autres environnements, utiliser une valeur par défaut
        # si non définie — 127.0.0.1 en dev (évite IPv6 localhost)
        PDF_BASE_URL = _pdf_base_url or "http://127.0.0.1:5000"

    # ✅ Prod: Validation SOCKETIO_CORS_ORIGINS en production
    # En production, SOCKETIO_CORS_ORIGINS doit être défini, non vide,
    # et ne pas contenir "*"
    # ⚠️ Ne vérifier que si on est réellement en production
    # (pas lors de l'import du module)
    _socketio_cors_origins = os.getenv("SOCKETIO_CORS_ORIGINS", "").strip()
    if _flask_config == "production":
        if not _socketio_cors_origins:
            raise RuntimeError(
                "SOCKETIO_CORS_ORIGINS est requis en production. "
                + "Définissez cette variable d'environnement avec une liste "
                + "d'origines HTTPS séparées par des virgules."
            )
        if _socketio_cors_origins == "*":
            raise RuntimeError(
                "SOCKETIO_CORS_ORIGINS ne peut pas être '*' en production "
                + "pour des raisons de sécurité. Définissez une liste "
                + "explicite d'origines HTTPS."
            )
        # Valider que toutes les origines sont HTTPS (sauf localhost pour tests)
        for origin in (origin.strip() for origin in _socketio_cors_origins.split(",")):
            if origin and not origin.startswith(
                ("https://", "http://localhost", "http://127.0.0.1")
            ):
                msg = (
                    f"SOCKETIO_CORS_ORIGINS contient une origine non sécurisée "
                    f"en production: {origin}. Toutes les origines doivent être "
                )
                raise RuntimeError(msg + "HTTPS (sauf localhost pour tests).")
        SOCKETIO_CORS_ORIGINS = _socketio_cors_origins
    else:
        # Pour les autres environnements, utiliser une valeur par défaut si non définie
        SOCKETIO_CORS_ORIGINS = _socketio_cors_origins or "*"

    # ✅ Rate Limits Production: Conservateurs pour protéger contre l'abus
    ENVIRONMENT = "production"
    RATELIMIT_DEFAULT_LIMITS: ClassVar[list[str]] = [
        os.getenv("RATELIMIT_DEFAULT_LIMITS", "1000 per hour")
    ]
    # Dispatch endpoints: limites restrictives
    RATELIMIT_DISPATCH_RUN = os.getenv("RATELIMIT_DISPATCH_RUN", "30 per hour")
    RATELIMIT_DISPATCH_TRIGGER = os.getenv("RATELIMIT_DISPATCH_TRIGGER", "50 per hour")
    RATELIMIT_COMPANY_DISPATCH_RUN = os.getenv(
        "RATELIMIT_COMPANY_DISPATCH_RUN", "10 per minute"
    )
    RATELIMIT_COMPANY_DISPATCH_OPTIMIZER = os.getenv(
        "RATELIMIT_COMPANY_DISPATCH_OPTIMIZER", "10 per minute"
    )


class TestingConfig(Config):
    TESTING = True
    SECRET_KEY = "test-secret-key"
    JWT_SECRET_KEY = "test-jwt-key"
    MAIL_PASSWORD = "test-mail-password"
    MAIL_SUPPRESS_SEND = True
    # Utiliser DATABASE_URL si disponible (PostgreSQL),
    # sinon SQLite en mémoire
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///:memory:")
    SQLALCHEMY_BINDS = _build_sqlalchemy_binds(SQLALCHEMY_DATABASE_URI)
    # Options de base uniquement
    # (pas de pool_size/max_overflow pour compatibilité SQLite)
    SQLALCHEMY_ENGINE_OPTIONS: ClassVar[_SqlalchemyEngineOptions] = {
        "pool_pre_ping": True,
        "pool_recycle": 1800,
    }
    WTF_CSRF_ENABLED = False
    RATELIMIT_ENABLED = False
    PDF_BASE_URL = "http://testserver"
    UPLOADS_PUBLIC_BASE = "/uploads"

    @staticmethod
    def init_app(app):  # pyright: ignore[reportImplicitOverride]
        """Ajuste les options d'engine selon le type de base de données."""
        # Utiliser DATABASE_URL de l'environnement si disponible
        # (priorité sur valeur par défaut)
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            app.config["SQLALCHEMY_DATABASE_URI"] = database_url

        # Ajouter options PostgreSQL uniquement si DATABASE_URL pointe vers PostgreSQL
        db_uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
        if db_uri and db_uri.startswith("postgresql"):
            prior = app.config.get("SQLALCHEMY_ENGINE_OPTIONS") or {}
            prior_connect = (
                prior.get("connect_args", {}) if isinstance(prior, dict) else {}
            )
            if not isinstance(prior_connect, dict):
                prior_connect = {}
            _tc_connect = {
                **prior_connect,
                "client_encoding": prior_connect.get("client_encoding", "utf8"),
            }
            if _is_psycopg3_dialect_in_url(db_uri):
                _tc_connect["prepare_threshold"] = 0
            else:
                _tc_connect.pop("prepare_threshold", None)
            app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
                **prior,
                "pool_size": 5,  # Pool plus petit pour les tests
                "max_overflow": 10,  # Overflow plus petit pour les tests
                "connect_args": _tc_connect,
            }


class DemoConfig(DevelopmentConfig):
    """Configuration dédiée à la stack de démonstration isolée."""

    DEBUG = False
    ENVIRONMENT = "demo"
    RATELIMIT_DEFAULT_LIMITS: ClassVar[list[str]] = [
        os.getenv("RATELIMIT_DEFAULT_LIMITS", "300 per hour")
    ]


# C'est ce que votre "Application Factory" utilisera
config = {
    "development": DevelopmentConfig,
    "demo": DemoConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
