"""✅ 4.1: Client Vault pour gestion centralisée des secrets.

Fournit une interface Python pour accéder aux secrets depuis HashiCorp Vault
avec cache, fallback .env, et gestion d'erreurs.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

try:
    import hvac  # type: ignore[import-untyped]  # Dépendance optionnelle
    HVAC_AVAILABLE = True
except ImportError:
    hvac = None
    HVAC_AVAILABLE = False

logger = logging.getLogger(__name__)

# ✅ 4.1: Configuration par défaut
DEFAULT_VAULT_ADDR = "http://localhost:8200"
VAULT_TOKEN_ENV = "VAULT_TOKEN"
VAULT_ADDR_ENV = "VAULT_ADDR"
VAULT_ROLE_ID_ENV = "VAULT_ROLE_ID"
VAULT_SECRET_ID_ENV = "VAULT_SECRET_ID"
USE_VAULT_ENV = "USE_VAULT"

# Cache pour éviter appels répétés (TTL 5 minutes par défaut)
_cache: dict[str, tuple[Any, float]] = {}
_cache_ttl = 300  # 5 minutes


class VaultClientError(Exception):
    """Exception levée lors d'erreurs avec Vault."""

    pass


class VaultClient:
    """✅ 4.1: Client Vault avec cache et fallback .env.
    
    Utilise HashiCorp Vault pour stocker et récupérer les secrets.
    En cas d'erreur ou si Vault n'est pas configuré, fallback vers variables d'environnement.
    """

    def __init__(
        self,
        vault_addr: Optional[str] = None,
        vault_token: Optional[str] = None,
        role_id: Optional[str] = None,
        secret_id: Optional[str] = None,
        use_vault: Optional[bool] = None,
        cache_ttl: int = 300,
    ):
        """Initialise le client Vault.
        
        Args:
            vault_addr: Adresse Vault (par défaut: VAULT_ADDR ou http://localhost:8200)
            vault_token: Token Vault (dev uniquement)
            role_id: Role ID pour authentification AppRole (production)
            secret_id: Secret ID pour authentification AppRole (production)
            use_vault: Forcer l'utilisation de Vault (par défaut: auto-détection)
            cache_ttl: TTL du cache en secondes (défaut: 5 minutes)
        """
        super().__init__()
        self.vault_addr = vault_addr or os.getenv(VAULT_ADDR_ENV, DEFAULT_VAULT_ADDR)
        self.vault_token = vault_token or os.getenv(VAULT_TOKEN_ENV)
        self.role_id = role_id or os.getenv(VAULT_ROLE_ID_ENV)
        self.secret_id = secret_id or os.getenv(VAULT_SECRET_ID_ENV)
        
        # Détection automatique : utiliser Vault si VAULT_ADDR est défini
        if use_vault is None:
            use_vault = bool(os.getenv(USE_VAULT_ENV, "").lower() == "true" or self.vault_addr)
        
        self.use_vault = use_vault and HVAC_AVAILABLE
        self.cache_ttl = cache_ttl
        self._client: Optional[Any] = None
        
        if self.use_vault:
            self._init_client()
            logger.info("[4.1 Vault] Client Vault initialisé (addr=%s)", self.vault_addr)
        else:
            logger.info("[4.1 Vault] Vault désactivé, utilisation des variables d'environnement")

    def _init_client(self) -> None:
        """Initialise le client hvac."""
        if not HVAC_AVAILABLE:
            logger.warning("[4.1 Vault] hvac non installé, désactivation de Vault")
            self.use_vault = False
            return
        
        try:
            assert hvac is not None  # HVAC_AVAILABLE garanti que hvac est défini
            self._client = hvac.Client(url=self.vault_addr)
            
            # Authentification
            if self.vault_token:
                # Token direct (développement)
                self._client.token = self.vault_token
            elif self.role_id and self.secret_id:
                # AppRole (production)
                response = self._client.auth.approle.login(
                    role_id=self.role_id,
                    secret_id=self.secret_id
                )
                self._client.token = response["auth"]["client_token"]
            else:
                logger.warning("[4.1 Vault] Aucune authentification configurée, désactivation")
                self.use_vault = False
                return
            
            # Test de connexion
            if not self._client.is_authenticated():
                raise VaultClientError("Authentification Vault échouée")
            
            logger.debug("[4.1 Vault] Authentification Vault réussie")
            
        except Exception as e:
            logger.warning("[4.1 Vault] Erreur initialisation client: %s, fallback .env", e)
            self.use_vault = False
            self._client = None

    def get_secret(
        self,
        path: str,
        key: str,
        env_fallback: Optional[str] = None,
        default: Optional[str] = None,
        use_cache: bool = True,
    ) -> Optional[str]:
        """Récupère un secret depuis Vault ou variable d'environnement.
        
        Args:
            path: Chemin Vault (ex: "atmr/prod/flask/secret_key")
            key: Clé du secret dans Vault (ex: "value")
            env_fallback: Nom de la variable d'environnement en fallback
            default: Valeur par défaut si non trouvé
            use_cache: Utiliser le cache (défaut: True)
            
        Returns:
            Valeur du secret ou None
        """
        # Vérifier le cache
        if use_cache:
            cache_key = f"{path}:{key}"
            if cache_key in _cache:
                value, timestamp = _cache[cache_key]
                import time
                if time.time() - timestamp < self.cache_ttl:
                    logger.debug("[4.1 Vault] Secret récupéré depuis cache: %s", path)
                    return value
                del _cache[cache_key]
        
        # Essayer Vault
        if self.use_vault and self._client:
            try:
                # KV v2 path format: secret/data/path
                vault_path = f"atmr/data/{path}" if not path.startswith("atmr/") else f"{path.replace('atmr/', 'atmr/data/')}"
                
                response = self._client.secrets.kv.v2.read_secret_version(path=vault_path)
                value = response["data"]["data"].get(key)
                
                if value:
                    # Mettre en cache
                    if use_cache:
                        import time
                        cache_key = f"{path}:{key}"
                        _cache[cache_key] = (value, time.time())
                    
                    logger.debug("[4.1 Vault] Secret récupéré depuis Vault: %s", path)
                    return value
                    
            except Exception as e:
                logger.warning("[4.1 Vault] Erreur lecture secret %s: %s, fallback .env", path, e)
        
        # Fallback vers variable d'environnement
        if env_fallback:
            value = os.getenv(env_fallback)
            if value:
                logger.debug("[4.1 Vault] Secret récupéré depuis .env: %s", env_fallback)
                return value
        
        # Valeur par défaut
        if default is not None:
            logger.debug("[4.1 Vault] Utilisation valeur par défaut pour: %s", path)
            return default
        
        logger.warning("[4.1 Vault] Secret non trouvé: %s (path=%s, key=%s)", env_fallback or path, path, key)
        return None

    def get_secret_required(
        self,
        path: str,
        key: str,
        env_fallback: Optional[str] = None,
    ) -> str:
        """Récupère un secret requis (lance une exception si absent).
        
        Args:
            path: Chemin Vault
            key: Clé du secret
            env_fallback: Nom de la variable d'environnement en fallback
            
        Returns:
            Valeur du secret
            
        Raises:
            RuntimeError: Si le secret n'est pas trouvé
        """
        value = self.get_secret(path, key, env_fallback=env_fallback)
        if value is None:
            raise RuntimeError(f"Secret requis non trouvé: {path}:{key} (env: {env_fallback})")
        return value

    def get_database_credentials(self, role: str = "atmr-db-role") -> dict[str, str]:
        """Récupère des credentials database dynamiques.
        
        Args:
            role: Nom du rôle Vault pour database secrets
            
        Returns:
            Dict avec 'username' et 'password'
        """
        if self.use_vault and self._client:
            try:
                response = self._client.secrets.database.generate_credentials(role)
                return {
                    "username": response["data"]["username"],
                    "password": response["data"]["password"],
                }
            except Exception as e:
                logger.warning("[4.1 Vault] Erreur génération credentials DB: %s", e)
        
        # Fallback vers DATABASE_URL
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            # Parser DATABASE_URL (format: postgresql://user:pass@host:port/db)
            from urllib.parse import urlparse
            parsed = urlparse(database_url)
            return {
                "username": parsed.username or "atmr",
                "password": parsed.password or "",
            }
        
        raise RuntimeError("Credentials database non disponibles")

    def clear_cache(self) -> None:
        """Vide le cache des secrets."""
        _cache.clear()
        logger.debug("[4.1 Vault] Cache vidé")


# ✅ 4.1: Instance globale (singleton)
_vault_client: Optional[VaultClient] = None


def get_vault_client() -> VaultClient:
    """Récupère ou crée l'instance globale du client Vault."""
    global _vault_client  # noqa: PLW0603
    if _vault_client is None:
        _vault_client = VaultClient()
    return _vault_client


def reset_vault_client() -> None:
    """Réinitialise le client Vault (utile pour tests)."""
    global _vault_client  # noqa: PLW0603
    _vault_client = None

