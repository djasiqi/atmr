"""
Pipeline de feature engineering pour production ML.

Réplique exactement le feature engineering utilisé pendant l'entraînement.
"""
# ruff: noqa: DTZ005
# pyright: reportArgumentType=false
# datetime sans tz intentionnel

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_base_features(booking: Any, driver: Any) -> dict[str, float]:
    """
    Extrait les features de base depuis booking et driver.

    Returns:
        Dict avec 15 features de base
    """
    scheduled_time = getattr(booking, 'scheduled_time', None) or datetime.now()

    # Time features
    time_of_day = float(scheduled_time.hour)
    day_of_week = float(scheduled_time.weekday())
    month = float(scheduled_time.month)

    # Distance
    try:
        from shared.geo_utils import haversine_distance

        pickup_lat_val = getattr(booking, 'pickup_lat', None)
        pickup_lon_val = getattr(booking, 'pickup_lon', None)
        dropoff_lat_val = getattr(booking, 'dropoff_lat', None)
        dropoff_lon_val = getattr(booking, 'dropoff_lon', None)

        pickup_lat = float(pickup_lat_val) if pickup_lat_val is not None else 0.0
        pickup_lon = float(pickup_lon_val) if pickup_lon_val is not None else 0.0
        dropoff_lat = float(dropoff_lat_val) if dropoff_lat_val is not None else 0.0
        dropoff_lon = float(dropoff_lon_val) if dropoff_lon_val is not None else 0.0

        if all([pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]):
            distance_km = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        else:
            distance_meters_val = getattr(booking, 'distance_meters', None)
            distance_km = distance_meters_val / 1000.0 if distance_meters_val else 0.0
    except Exception:
        distance_km = 0.0

    # Duration (estimation)
    duration_seconds_val = getattr(booking, 'duration_seconds', None)
    duration_seconds = float(duration_seconds_val) if duration_seconds_val else distance_km * 7 * 60

    # Booking characteristics
    medical_facility_val = getattr(booking, 'medical_facility', None)
    is_medical = 1.0 if medical_facility_val else 0.0
    is_urgent_val = getattr(booking, 'is_urgent', False)
    is_urgent = 1.0 if is_urgent_val else 0.0
    is_round_trip_val = getattr(booking, 'is_round_trip', False)
    is_round_trip = 1.0 if is_round_trip_val else 0.0

    booking_priority = 0.8 if (is_medical or is_urgent) else 0.5

    # Driver features
    driver_total_bookings = float(len(getattr(driver, 'assignments', [])) if hasattr(driver, 'assignments') else 0)

    # Traffic density (estimation basée sur l'heure)
    if time_of_day in [7, 8, 17, 18]:
        traffic_density = 0.8
    elif time_of_day in [6, 9, 16, 19]:
        traffic_density = 0.6
    elif time_of_day in [12, 13]:
        traffic_density = 0.5
    else:
        traffic_density = 0.3

    # Weather (données réelles via OpenWeatherMap)
    try:
        from services.weather_service import get_weather_factor

        pickup_lat_val = getattr(booking, 'pickup_lat', None)
        pickup_lon_val = getattr(booking, 'pickup_lon', None)

        pickup_lat = float(pickup_lat_val) if pickup_lat_val is not None else 0.0
        pickup_lon = float(pickup_lon_val) if pickup_lon_val is not None else 0.0

        weather_factor = get_weather_factor(pickup_lat, pickup_lon) if pickup_lat and pickup_lon else 0.5
    except Exception as e:
        logger.warning(f"[MLFeatures] Weather API failed, using neutral: {e}")
        weather_factor = 0.5

    return {
        'time_of_day': time_of_day,
        'day_of_week': day_of_week,
        'month': month,
        'distance_km': distance_km,
        'duration_seconds': duration_seconds,
        'is_medical': is_medical,
        'is_urgent': is_urgent,
        'is_round_trip': is_round_trip,
        'booking_priority': booking_priority,
        'driver_total_bookings': driver_total_bookings,
        'traffic_density': traffic_density,
        'weather_factor': weather_factor,
    }


def create_interaction_features(features: dict[str, float]) -> dict[str, float]:
    """Crée les features d'interaction."""
    interactions = {}

    # Interactions critiques (53.7% importance)
    interactions['distance_x_traffic'] = features['distance_km'] * features['traffic_density']
    interactions['distance_x_weather'] = features['distance_km'] * features['weather_factor']
    interactions['traffic_x_weather'] = features['traffic_density'] * features['weather_factor']
    interactions['medical_x_distance'] = features['is_medical'] * features['distance_km']
    interactions['urgent_x_traffic'] = features['is_urgent'] * features['traffic_density']

    return interactions


def create_temporal_features(features: dict[str, float]) -> dict[str, float]:
    """Crée les features temporelles avancées."""
    temporal = {}

    time_of_day = features['time_of_day']
    day_of_week = features['day_of_week']

    # Binaires
    temporal['is_rush_hour'] = 1.0 if time_of_day in [7, 8, 17, 18] else 0.0
    temporal['is_morning_peak'] = 1.0 if time_of_day in [7, 8] else 0.0
    temporal['is_evening_peak'] = 1.0 if time_of_day in [17, 18] else 0.0
    temporal['is_weekend'] = 1.0 if day_of_week >= 5 else 0.0
    temporal['is_lunch_time'] = 1.0 if time_of_day in [12, 13] else 0.0

    # Encodage cyclique
    temporal['hour_sin'] = np.sin(2 * np.pi * time_of_day / 24)
    temporal['hour_cos'] = np.cos(2 * np.pi * time_of_day / 24)
    temporal['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    temporal['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    return temporal


def create_aggregated_features(features: dict[str, float]) -> dict[str, float]:
    """
    Crée les features agrégées basées sur moyennes historiques.

    Note: En production, ces valeurs devraient être mises à jour régulièrement
    avec les données réelles.
    """
    aggregated = {}

    # Moyennes par heure (basées sur training data)
    # Ces valeurs devraient être chargées depuis un fichier de configuration
    hour_delays = {
        6: 6.16, 7: 7.45, 8: 7.68, 9: 5.97, 10: 5.12, 11: 5.34,
        12: 5.42, 13: 5.89, 14: 5.67, 15: 5.34, 16: 6.11, 17: 7.49,
        18: 7.31, 19: 6.38, 20: 5.78, 21: 5.45, 22: 5.23
    }
    aggregated['delay_by_hour'] = hour_delays.get(int(features['time_of_day']), 6.28)

    # Moyennes par jour
    day_delays = {0: 6.45, 1: 6.38, 2: 6.29, 3: 6.21, 4: 6.42, 5: 5.89, 6: 5.74}
    aggregated['delay_by_day'] = day_delays.get(int(features['day_of_week']), 6.28)

    # Niveau d'expérience driver
    driver_bookings = features['driver_total_bookings']
    if driver_bookings < 50:
        aggregated['driver_experience_level'] = 0.0  # novice
        aggregated['delay_by_driver_exp'] = 8.2
    elif driver_bookings < 200:
        aggregated['driver_experience_level'] = 1.0  # intermédiaire
        aggregated['delay_by_driver_exp'] = 6.1
    else:
        aggregated['driver_experience_level'] = 2.0  # expert
        aggregated['delay_by_driver_exp'] = 4.3

    # Catégorie de distance
    distance = features['distance_km']
    if distance < 5:
        aggregated['distance_category'] = 0.0
    elif distance < 10:
        aggregated['distance_category'] = 1.0
    elif distance < 20:
        aggregated['distance_category'] = 2.0
    else:
        aggregated['distance_category'] = 3.0

    # Niveau de trafic
    traffic = features['traffic_density']
    if traffic < 0.4:
        aggregated['traffic_level'] = 0.0
    elif traffic < 0.7:
        aggregated['traffic_level'] = 1.0
    else:
        aggregated['traffic_level'] = 2.0

    return aggregated


def create_polynomial_features(features: dict[str, float]) -> dict[str, float]:
    """Crée les features polynomiales."""
    polynomial = {}

    polynomial['distance_squared'] = features['distance_km'] ** 2
    polynomial['traffic_squared'] = features['traffic_density'] ** 2
    polynomial['driver_exp_log'] = np.log1p(features['driver_total_bookings'])

    return polynomial


def engineer_features(booking: Any, driver: Any) -> dict[str, float]:
    """
    Pipeline complet de feature engineering pour production.

    Args:
        booking: Objet Booking
        driver: Objet Driver

    Returns:
        Dict avec toutes les features (35 au total)
    """
    # 1. Features de base
    base_features = extract_base_features(booking, driver)

    # 2. Interactions
    interactions = create_interaction_features(base_features)

    # 3. Temporelles
    temporal = create_temporal_features(base_features)

    # 4. Agrégées
    aggregated = create_aggregated_features(base_features)

    # 5. Polynomiales
    polynomial = create_polynomial_features(base_features)

    # Combiner toutes les features
    all_features = {
        **base_features,
        **interactions,
        **temporal,
        **aggregated,
        **polynomial,
    }

    return all_features


def normalize_features(
    features: dict[str, float],
    scaler_params: dict[str, Any]
) -> dict[str, float]:
    """
    Normalise les features avec StandardScaler (paramètres pré-calculés).

    Args:
        features: Features à normaliser
        scaler_params: Paramètres du scaler (mean, scale)

    Returns:
        Features normalisées
    """
    normalized = features.copy()

    if 'columns' in scaler_params and 'mean' in scaler_params and 'scale' in scaler_params:
        columns = scaler_params['columns']
        means = scaler_params['mean']
        scales = scaler_params['scale']

        for i, col in enumerate(columns):
            if col in normalized:
                # Appliquer StandardScaler: (x - mean) / scale
                normalized[col] = (normalized[col] - means[i]) / scales[i]

    return normalized


def features_to_dataframe(
    features: dict[str, float],
    feature_order: list[str]
) -> pd.DataFrame:
    """
    Convertit dict de features en DataFrame avec bon ordre de colonnes.

    Args:
        features: Dict de features
        feature_order: Ordre des colonnes (du modèle entraîné)

    Returns:
        DataFrame 1 ligne avec features dans le bon ordre
    """
    # Créer DataFrame avec features dans l'ordre du modèle
    row_data = {col: features.get(col, 0.0) for col in feature_order}

    return pd.DataFrame([row_data])

