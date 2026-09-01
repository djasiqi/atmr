"""Constantes centralisées pour éviter les valeurs magiques dans le code.

Ce module regroupe les constantes utilisées dans différents domaines de l'application,
améliorant la lisibilité, la maintenabilité et évitant les incohérences.
"""

# ============================================================================
# CONSTANTES GÉOGRAPHIQUES
# ============================================================================


class GeoConstants:
    """Constantes pour les coordonnées géographiques."""

    # Limites des coordonnées
    LATITUDE_MIN = -90.0
    LATITUDE_MAX = 90.0
    LONGITUDE_MIN = -180.0
    LONGITUDE_MAX = 180.0

    # Coordonnées par défaut (Genève, Suisse)
    FALLBACK_COORD_DEFAULT = (46.2044, 6.1432)

    # Seuils de qualité de coordonnées
    LOW_COORD_QUALITY_THRESHOLD = 0.65


# ============================================================================
# CONSTANTES DE PAGINATION
# ============================================================================


class PaginationConstants:
    """Constantes pour la pagination."""

    PAGE_ONE = 1
    PAGE_DEFAULT = 1


# ============================================================================
# CONSTANTES NUMÉRIQUES COMMUNES
# ============================================================================


class NumericConstants:
    """Constantes numériques communes utilisées dans plusieurs contextes."""

    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3


# ============================================================================
# CONSTANTES DE DISPATCH - HEURISTICS
# ============================================================================


class DispatchHeuristicsConstants:
    """Constantes pour les heuristiques de dispatch."""

    # Vitesse et distance
    AVG_KMH_ZERO = 0
    DIST_KM_ONE = 1

    # Seuils temporels (en minutes)
    MINS_THRESHOLD = 20
    TO_PICKUP_MIN_THRESHOLD = 5
    LATENESS_THRESHOLD_MIN = 15

    # Seuils de charge
    CURRENT_LOAD_THRESHOLD = 2
    DID_THRESHOLD = 3

    # Seuils pour trajets d'urgence (minutes depuis le bureau)
    EMERGENCY_PICKUP_NEAR_THRESHOLD = 10  # Pickup proche du bureau
    EMERGENCY_PICKUP_MEDIUM_THRESHOLD = 15  # Pickup moyen du bureau
    EMERGENCY_PICKUP_FAR_THRESHOLD = 20  # Pickup loin du bureau
    EMERGENCY_TRIP_SHORT_THRESHOLD = 15  # Trajet court
    EMERGENCY_TRIP_MEDIUM_THRESHOLD = 20  # Trajet moyen

    # Équité
    MAX_FAIRNESS_GAP = 2  # Écart maximum entre chauffeurs réguliers (équité stricte)
    PREFERRED_EXTRA_GAP = 1  # Marge supplémentaire autorisée pour le chauffeur préféré

    # Seuils de comptage
    CNT_ZERO = 0
    SC_ZERO = 0

    # Seuil pour parallélisation
    PARALLEL_MIN_BOOKINGS = 20


# ============================================================================
# CONSTANTES DE DISPATCH - DATA
# ============================================================================


class DispatchDataConstants:
    """Constantes pour la construction des données de dispatch."""

    # Seuils de validation
    SI_THRESHOLD = 5
    LAT_FLOAT_THRESHOLD = 90
    LON_MIN_THRESHOLD = -180
    LON_MAX_THRESHOLD = 180
    N_THRESHOLD = 2
    TW_THRESHOLD = 2
    SERVICE_TIMES_THRESHOLD = 2

    # Valeurs numériques communes
    N_ZERO = 0
    N_ONE = 1
    W_ZERO = 0
    W_ONE = 1

    # Seuils temporels
    DELAY_SECONDS_ZERO = 0
    MIN_TRAVEL_MINUTES = 2

    # Seuils de performance
    MAX_DRIVER_IDS_IN_LOG = 10  # Limite le nombre de driver IDs affichés dans les logs
    FAIRNESS_SLOW_QUERY_THRESHOLD_MS = (
        5000  # Seuil pour warning si fairness_counts > 5s
    )
    BUILD_MATRIX_SLOW_THRESHOLD_MS = (
        30000  # Seuil pour warning si build_time_matrix > 30s
    )

    # Seuils de taille
    LARGE_MATRIX_THRESHOLD = 999


# ============================================================================
# CONSTANTES DE DISPATCH - METRICS
# ============================================================================


class DispatchMetricsConstants:
    """Constantes pour les métriques de dispatch."""

    DELAY_MINUTES_THRESHOLD = 5
    DELAYED_ZERO = 0
    AVG_ZERO = 0
    POOLING_WINDOW_SECONDS = 600  # 10 minutes en secondes
    MIN_VALUES_FOR_GINI = 2  # Minimum drivers pour calculer Gini
    QUALITY_THRESHOLD = 70.0  # Seuil pour activer auto-apply RL


# ============================================================================
# CODES D'ERREUR API (machine-readable, stables)
# ============================================================================


class ErrorCodes:
    """Codes d'erreur API standardisés pour le frontend et les intégrations."""

    # Livraison matériel
    MATERIAL_DELIVERY_PRICE_NOT_CONFIGURED = "MATERIAL_DELIVERY_PRICE_NOT_CONFIGURED"
    MATERIAL_DELIVERY_DESCRIPTION_REQUIRED = "MATERIAL_DELIVERY_DESCRIPTION_REQUIRED"

    # Intégrité facturation A/R (InvoiceLine ↔ Booking)
    BILLING_INVOICE_LINE_LINK_INCOMPLETE = "BILLING_INVOICE_LINE_LINK_INCOMPLETE"


# ============================================================================
# CODES D'ERREUR AUTH (authentification / inscription / mot de passe)
# ============================================================================


class AuthErrorCodes:
    """Codes d'erreur pour l'authentification et la gestion des comptes.

    Ces codes sont utilisés avec api_error() / auth_error() de api_error_utils.py.
    Voir docs/api/errors.md pour la documentation complète.

    Format de réponse standard:
    {
        "error": "<code>",
        "message": "<message FR>",
        "details": {}  // optionnel
    }
    """

    # -------------------------------------------------------------------------
    # Inscription (409 Conflict)
    # -------------------------------------------------------------------------
    EMAIL_EXISTS = "email_exists"
    """Email déjà utilisé par un autre compte. HTTP 409."""

    USERNAME_EXISTS = "username_exists"
    """Nom d'utilisateur déjà pris. HTTP 409."""

    # -------------------------------------------------------------------------
    # Validation mot de passe (400 Bad Request)
    # -------------------------------------------------------------------------
    PASSWORD_POLICY_ERROR = "password_policy_error"
    """Mot de passe non conforme à la politique de sécurité. HTTP 400."""

    PASSWORD_TOO_SHORT = "password_too_short"
    """Mot de passe trop court (< 12 caractères). HTTP 400."""

    PASSWORD_NO_UPPERCASE = "password_no_uppercase"
    """Mot de passe sans majuscule. HTTP 400."""

    PASSWORD_NO_LOWERCASE = "password_no_lowercase"
    """Mot de passe sans minuscule. HTTP 400."""

    PASSWORD_NO_DIGIT = "password_no_digit"
    """Mot de passe sans chiffre. HTTP 400."""

    PASSWORD_NO_SPECIAL = "password_no_special"
    """Mot de passe sans caractère spécial. HTTP 400."""

    PASSWORD_COMPROMISED = "password_compromised"
    """Mot de passe trouvé dans une base de données de fuites (HIBP). HTTP 400."""

    PASSWORD_IN_HISTORY = "password_in_history"
    """Mot de passe déjà utilisé récemment. HTTP 400."""

    # -------------------------------------------------------------------------
    # Tokens (400 Bad Request)
    # -------------------------------------------------------------------------
    TOKEN_INVALID = "token_invalid"
    """Token invalide ou malformé. HTTP 400."""

    TOKEN_EXPIRED = "token_expired"
    """Token expiré. HTTP 400 ou 401."""

    TOKEN_ALREADY_USED = "token_already_used"
    """Token déjà utilisé (reset password, email verification). HTTP 400."""

    # -------------------------------------------------------------------------
    # Authentification (401 Unauthorized)
    # -------------------------------------------------------------------------
    INVALID_CREDENTIALS = "invalid_credentials"
    """Email ou mot de passe incorrect. HTTP 401."""

    EMAIL_NOT_FOUND = "email_not_found"
    """Email inexistant. HTTP 401."""

    INVALID_PASSWORD = "invalid_password"
    """Mot de passe incorrect. HTTP 401."""

    MISSING_TOKEN = "missing_token"
    """Token JWT manquant dans la requête. HTTP 401."""

    # -------------------------------------------------------------------------
    # Accès refusé (403 Forbidden)
    # -------------------------------------------------------------------------
    ACCOUNT_LOCKED = "account_locked"
    """Compte verrouillé après trop de tentatives. HTTP 403."""

    ACCOUNT_DISABLED = "account_disabled"
    """Compte désactivé par un administrateur. HTTP 403."""

    EMAIL_NOT_VERIFIED = "email_not_verified"
    """Email non vérifié, accès restreint. HTTP 403."""

    # -------------------------------------------------------------------------
    # Rate limiting (429 Too Many Requests)
    # -------------------------------------------------------------------------
    RATE_LIMITED = "rate_limited"
    """Trop de tentatives, veuillez patienter. HTTP 429."""

    # -------------------------------------------------------------------------
    # Erreurs génériques
    # -------------------------------------------------------------------------
    REGISTRATION_ERROR = "registration_error"
    """Erreur générique lors de l'inscription. HTTP 400."""

    LOGIN_ERROR = "login_error"
    """Erreur générique lors de la connexion. HTTP 400."""


# Mapping codes auth -> HTTP status (pour référence)
AUTH_ERROR_HTTP_STATUS = {
    # 409 Conflict
    AuthErrorCodes.EMAIL_EXISTS: 409,
    AuthErrorCodes.USERNAME_EXISTS: 409,
    # 400 Bad Request
    AuthErrorCodes.PASSWORD_POLICY_ERROR: 400,
    AuthErrorCodes.PASSWORD_TOO_SHORT: 400,
    AuthErrorCodes.PASSWORD_NO_UPPERCASE: 400,
    AuthErrorCodes.PASSWORD_NO_LOWERCASE: 400,
    AuthErrorCodes.PASSWORD_NO_DIGIT: 400,
    AuthErrorCodes.PASSWORD_NO_SPECIAL: 400,
    AuthErrorCodes.PASSWORD_COMPROMISED: 400,
    AuthErrorCodes.PASSWORD_IN_HISTORY: 400,
    AuthErrorCodes.TOKEN_INVALID: 400,
    AuthErrorCodes.TOKEN_EXPIRED: 401,
    AuthErrorCodes.TOKEN_ALREADY_USED: 400,
    AuthErrorCodes.REGISTRATION_ERROR: 400,
    AuthErrorCodes.LOGIN_ERROR: 400,
    # 401 Unauthorized
    AuthErrorCodes.INVALID_CREDENTIALS: 401,
    AuthErrorCodes.EMAIL_NOT_FOUND: 401,
    AuthErrorCodes.INVALID_PASSWORD: 401,
    AuthErrorCodes.MISSING_TOKEN: 401,
    # 403 Forbidden
    AuthErrorCodes.ACCOUNT_LOCKED: 403,
    AuthErrorCodes.ACCOUNT_DISABLED: 403,
    AuthErrorCodes.EMAIL_NOT_VERIFIED: 403,
    # 429 Too Many Requests
    AuthErrorCodes.RATE_LIMITED: 429,
}
