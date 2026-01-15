# backend/services/geolocation/historical_eta.py
"""Service d'apprentissage historique pour améliorer les estimations ETA.

Utilise l'historique des trajets réels (EtaAccuracyLog) pour calculer
des durées moyennes par route (coordonnées) et par heure de la journée.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Tuple

from sqlalchemy import and_

from ext import db

logger = logging.getLogger(__name__)

# Constantes pour l'apprentissage historique
MIN_HISTORICAL_TRIPS = 5  # Minimum de trajets pour utiliser l'historique
COORD_PRECISION = 3  # Précision des coordonnées (3 décimales ≈ 111m)
HOUR_WINDOW = 2  # Fenêtre horaire (±2h autour de l'heure cible)
MAX_HISTORICAL_DAYS = 90  # Nombre de jours d'historique à considérer
WEATHER_IMPACT_FACTOR = (
    0.25  # Impact météo maximum (+25% de durée en conditions très défavorables)
)


def round_coord(coord: float, precision: int = COORD_PRECISION) -> float:
    """Arrondit une coordonnée à la précision spécifiée.

    Args:
        coord: Coordonnée (lat ou lon)
        precision: Nombre de décimales (défaut: 3 ≈ 111m)

    Returns:
        Coordonnée arrondie
    """
    factor = 10**precision
    return round(coord * factor) / factor


def get_historical_duration(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    target_hour: int,
    *,
    min_trips: int = MIN_HISTORICAL_TRIPS,
    hour_window: int = HOUR_WINDOW,
    max_days: int = MAX_HISTORICAL_DAYS,
) -> float | None:
    """Calcule la durée moyenne historique pour une route à une heure donnée.

    Recherche dans l'historique (EtaAccuracyLog) les trajets similaires :
    - Même origine/destination (coordonnées arrondies)
    - Même heure de la journée (±hour_window heures)
    - Trajets terminés (actual_duration_seconds non NULL)

    Args:
        origin: Coordonnées origine (lat, lon)
        destination: Coordonnées destination (lat, lon)
        target_hour: Heure cible (0-23)
        min_trips: Nombre minimum de trajets requis (défaut: 5)
        hour_window: Fenêtre horaire (±heures autour de target_hour, défaut: 2)
        max_days: Nombre maximum de jours d'historique (défaut: 90)

    Returns:
        Durée moyenne en secondes si suffisamment de données, sinon None
    """
    try:
        from models.eta_accuracy_log import EtaAccuracyLog

        # Arrondir les coordonnées pour grouper les routes similaires
        origin_lat_rounded = round_coord(origin[0])
        origin_lon_rounded = round_coord(origin[1])
        dest_lat_rounded = round_coord(destination[0])
        dest_lon_rounded = round_coord(destination[1])

        # Date limite (max_days jours en arrière)
        cutoff_date = datetime.now(UTC) - timedelta(days=max_days)

        # Fenêtre horaire (target_hour ± hour_window)
        hour_min = (target_hour - hour_window) % 24
        hour_max = (target_hour + hour_window) % 24

        # Requête pour trouver les trajets similaires
        query = db.session.query(EtaAccuracyLog).filter(
            and_(
                # Coordonnées arrondies (tolérance de ±0.001 ≈ 111m)
                EtaAccuracyLog.origin_lat.between(
                    origin_lat_rounded - 0.001, origin_lat_rounded + 0.001
                ),
                EtaAccuracyLog.origin_lon.between(
                    origin_lon_rounded - 0.001, origin_lon_rounded + 0.001
                ),
                EtaAccuracyLog.dest_lat.between(
                    dest_lat_rounded - 0.001, dest_lat_rounded + 0.001
                ),
                EtaAccuracyLog.dest_lon.between(
                    dest_lon_rounded - 0.001, dest_lon_rounded + 0.001
                ),
                # Trajets terminés avec durée réelle (filtre > 0 côté Python)
                EtaAccuracyLog.actual_duration_seconds.isnot(None),
                # Historique récent (max_days jours)
                EtaAccuracyLog.created_at >= cutoff_date,
            )
        )

        # Récupérer les résultats (filtrer par heure côté Python pour compatibilité)
        logs = query.all()

        # Filtrer par heure de la journée (côté Python pour compatibilité tous SGBD)
        if hour_min > hour_max:  # Fenêtre qui traverse minuit (ex: 22h-2h)
            logs = [
                log
                for log in logs
                if log.created_at
                and (
                    (log.created_at.hour >= hour_min)
                    or (log.created_at.hour <= hour_max)
                )
            ]
        else:  # Fenêtre horaire normale (ex: 7h-11h)
            logs = [
                log
                for log in logs
                if log.created_at and (hour_min <= log.created_at.hour <= hour_max)
            ]

        # Vérifier si on a suffisamment de données
        if len(logs) < min_trips:
            logger.debug(
                "⚠️ Pas assez de données historiques pour route (%.3f, %.3f) -> (%.3f, %.3f) à %dh: %d/%d trajets",
                origin_lat_rounded,
                origin_lon_rounded,
                dest_lat_rounded,
                dest_lon_rounded,
                target_hour,
                len(logs),
                min_trips,
            )
            return None

        # Calculer la durée moyenne
        durations = [
            log.actual_duration_seconds
            for log in logs
            if log.actual_duration_seconds and log.actual_duration_seconds > 0
        ]

        if not durations:
            return None

        avg_duration = sum(durations) / len(durations)

        logger.debug(
            "✅ Durée historique moyenne pour route (%.3f, %.3f) -> (%.3f, %.3f) à %dh: %.0fs (%d trajets)",
            origin_lat_rounded,
            origin_lon_rounded,
            dest_lat_rounded,
            dest_lon_rounded,
            target_hour,
            avg_duration,
            len(durations),
        )

        return avg_duration

    except ImportError:
        logger.debug(
            "[HistoricalETA] EtaAccuracyLog non disponible (migration en cours)"
        )
        return None
    except Exception as e:
        logger.warning("[HistoricalETA] Erreur calcul durée historique: %s", e)
        return None


def get_improved_duration_estimate(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    osrm_duration_seconds: float,
    target_hour: int | None = None,
    *,
    traffic_factor: float = 1.9,
    use_weather: bool = True,
) -> Tuple[float, str]:
    """Améliore l'estimation OSRM en utilisant l'historique et la météo si disponible.

    Args:
        origin: Coordonnées origine (lat, lon)
        destination: Coordonnées destination (lat, lon)
        osrm_duration_seconds: Durée estimée par OSRM (secondes)
        target_hour: Heure cible (0-23, optionnel, défaut: heure actuelle)
        traffic_factor: Facteur de correction OSRM si pas d'historique (défaut: 1.9)
        use_weather: Utiliser OpenWeather pour ajuster selon la météo (défaut: True)

    Returns:
        Tuple (durée_améliorée_secondes, source)
        - source: "historical" si historique utilisé, "osrm_corrected" sinon
    """
    # Heure cible (actuelle si non fournie)
    if target_hour is None:
        target_hour = datetime.now().hour

    # Essayer d'obtenir la durée historique
    historical_duration = get_historical_duration(origin, destination, target_hour)

    # Durée de base (historique si disponible, sinon OSRM corrigé)
    if historical_duration is not None:
        base_duration = historical_duration
        source = "historical"
    else:
        # Fallback: OSRM avec facteur de correction
        base_duration = max(1, int(osrm_duration_seconds * traffic_factor))
        source = "osrm_corrected"

    # Ajuster selon les conditions météo
    if use_weather:
        try:
            from services.external.weather import get_weather_factor

            # Récupérer le facteur météo (point médian de la route)
            # Utiliser le point d'origine (ou destination) pour simplifier
            # (en pratique, on pourrait utiliser le point médian)
            weather_factor = get_weather_factor(origin[0], origin[1])

            # Appliquer l'ajustement météo : +0% à +25% selon weather_factor (0-1)
            # weather_factor = 0 (idéal) → +0%
            # weather_factor = 1 (très défavorable) → +25%
            weather_adjustment = 1 + (weather_factor * WEATHER_IMPACT_FACTOR)
            adjusted_duration = max(1, int(base_duration * weather_adjustment))

            logger.debug(
                "✅ Durée ajustée météo (factor=%.2f, adjustment=+%.1f%%) : %ds (base: %ds)",
                weather_factor,
                weather_factor * WEATHER_IMPACT_FACTOR * 100,
                adjusted_duration,
                base_duration,
            )

            return (adjusted_duration, source)
        except Exception as weather_error:
            # En cas d'erreur météo, retourner la durée sans ajustement
            logger.debug(
                "[HistoricalETA] Erreur récupération météo (durée sans ajustement): %s",
                weather_error,
            )

    return (base_duration, source)
