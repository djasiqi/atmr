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
