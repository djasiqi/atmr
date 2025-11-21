"""✅ Priorité 7: Métriques Prometheus de sécurité.

Métriques pour monitorer les événements de sécurité :
- Authentification (login, logout, token refresh)
- Actions sensibles (modifications utilisateurs, permissions)
- Événements de sécurité (tentatives échouées, accès refusés)
"""

from prometheus_client import Counter

# ========================
# Métriques d'authentification
# ========================

# Compteur total des tentatives de login (succès + échecs)
security_login_attempts_total = Counter(
    "security_login_attempts_total",
    "Nombre total de tentatives de login",
    ["type"],  # type: "success" ou "failed"
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
    ["action_type"],  # action_type: "user_created", "user_updated", "permission_changed", etc.
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
