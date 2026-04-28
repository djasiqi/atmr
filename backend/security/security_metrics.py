"""✅ Priorité 7: Métriques Prometheus de sécurité.

Métriques pour monitorer les événements de sécurité :
- Authentification (login, logout, token refresh)
- Actions sensibles (modifications utilisateurs, permissions)
- Événements de sécurité (tentatives échouées, accès refusés)
"""

from prometheus_client import Counter, Gauge  # pyright: ignore[reportMissingImports]

# ========================
# Métriques d'authentification
# ========================

# Compteur total des tentatives de login (succès + échecs)
security_login_attempts_total = Counter(
    "security_login_attempts_total",
    "Nombre total de tentatives de login",
    ["type"],  # type peut être "success" ou "failed"
)

# Compteur des échecs de login
security_login_failures_total = Counter(
    "security_login_failures_total",
    "Nombre total de tentatives de login échouées",
)

# Compteur des rafraîchissements de token
security_token_refreshes_total = Counter(
    "security_token_refreshes_total",
    "Nombre total de rafraîchissements de token",
)

# Vérification JWT par kid (rotation versionnée)
jwt_verify_kid_total = Counter(
    "jwt_verify_kid_total",
    "Nombre total de vérifications JWT par kid",
    ["kid"],
)

jwt_verify_legacy_key_total = Counter(
    "jwt_verify_legacy_key_total",
    "Nombre de vérifications JWT via clé legacy",
    ["kid"],
)

jwt_invalid_kid_total = Counter(
    "jwt_invalid_kid_total",
    "Nombre de tokens JWT avec kid invalide ou non accepté",
    ["kid"],
)

# Compteur des déconnexions
security_logout_total = Counter(
    "security_logout_total",
    "Nombre total de déconnexions",
)

# ========================
# Métriques d'actions sensibles
# ========================

# Compteur des actions sensibles (création/modification/suppression utilisateurs, etc.)
security_sensitive_actions_total = Counter(
    "security_sensitive_actions_total",
    "Nombre total d'actions sensibles",
    [
        "action_type"
    ],  # action_type: "user_created", "user_updated", "permission_changed", etc.
)

# Compteur des changements de permissions
security_permission_changes_total = Counter(
    "security_permission_changes_total",
    "Nombre total de changements de permissions",
)

# Compteur des accès aux données sensibles
security_data_access_total = Counter(
    "security_data_access_total",
    "Nombre total d'accès aux données sensibles",
    ["data_type"],  # data_type: "user", "booking", "driver", etc.
)

# ========================
# Métriques de sécurité générales
# ========================

# Compteur des autorisations refusées
security_failed_authorizations_total = Counter(
    "security_failed_authorizations_total",
    "Nombre total d'autorisations refusées",
    ["resource_type"],  # resource_type: "endpoint", "data", etc.
)

# Compteur des hits de rate limiting
security_rate_limit_hits_total = Counter(
    "security_rate_limit_hits_total",
    "Nombre total de hits de rate limiting",
    ["endpoint"],  # endpoint: "/auth/login", "/api/...", etc.
)

# Compteur des rejets de tokens avec audience invalide
security_invalid_audience_total = Counter(
    "security_invalid_audience_total",
    "Nombre total de tokens rejetés pour audience invalide",
    ["reason"],  # reason: "missing", "wrong_audience"
)

# ✅ S3: Compteur des invalidations de tokens
security_token_invalidations_total = Counter(
    "security_token_invalidations_total",
    "Nombre total d'invalidations de tokens",
    ["reason"],  # reason: "logout", "password_change", "revoke_all", "admin_revoke"
)

# ✅ S3: Compteur des tentatives d'accès non autorisé (401/403)
security_unauthorized_access_total = Counter(
    "security_unauthorized_access_total",
    "Nombre total de tentatives d'accès non autorisé",
    ["status_code", "endpoint"],  # status_code: "401", "403", endpoint: "/api/..."
)

# ✅ S3: Gauge pour suivre les IPs suspectes (tentatives répétées)
security_suspicious_ips = Gauge(
    "security_suspicious_ips",
    "Nombre d'IPs suspectes détectées",
)

# ========================
# ✅ PHASE 3: Métriques de révocation et rotation
# ========================

# Compteur des tokens révoqués (par type)
tokens_revoked_total = Counter(
    "security_tokens_revoked_total",
    "Nombre total de tokens révoqués",
    ["token_type"],  # token_type: "access_token" ou "refresh_token"
)

# Compteur des rotations de refresh tokens
tokens_rotation_total = Counter(
    "security_tokens_rotated_total",
    "Nombre total de rotations de refresh tokens",
)

# Compteur des échecs de validation CSRF
csrf_validation_failures_total = Counter(
    "security_csrf_validation_failures_total",
    "Nombre total d'échecs de validation CSRF",
)

# ========================
# ✅ C2: Métriques avancées de rate limiting
# ========================

# Compteur des limites atteintes par endpoint spécifique
rate_limit_exceeded_total = Counter(
    "rate_limit_exceeded_total",
    "Nombre total de dépassements de rate limit par endpoint",
    ["endpoint", "user_type"],  # user_type: "authenticated", "anonymous"
)

# Gauge du nombre de clés actives de rate limit dans Redis
rate_limit_active_keys = Gauge(
    "rate_limit_active_keys",
    "Nombre de clés actives de rate limit dans Redis",
)

# Compteur des flushes de rate limit (via endpoint admin)
rate_limit_flushes_total = Counter(
    "rate_limit_flushes_total",
    "Nombre total de flushes de rate limits (via admin endpoint)",
    ["admin_user_id"],  # ID de l'admin qui a effectué le flush
)
