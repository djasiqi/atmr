"""Service pour récupérer les données météo en temps réel.

Utilise OpenWeatherMap API (gratuit jusqu'à 1,000 calls/jour).

Features météo:
- Température actuelle
- Conditions (pluie, neige, brouillard)
- Précipitations (mm)
- Vent (km/h)
- Visibilité (m)

Conversion en weather_factor (0 - 1):
- 0 = Conditions idéales
- 0.5 = Conditions normales
- 1 = Conditions très défavorables
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any

# Constantes pour éviter les valeurs magiques
SNOW_ZERO = 0
RAIN_ZERO = 0
WIND_SPEED_THRESHOLD = 50
VISIBILITY_THRESHOLD = 1000
CLOUDS_THRESHOLD = 80
TEMP_THRESHOLD = 35
TEMP_ZERO = 0
TEMP_EXTREME_MIN = -5
TEMP_MODERATE_MIN = 0
TEMP_MODERATE_MAX = 3

logger = logging.getLogger(__name__)

# Configuration
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Cache simple en mémoire (amélioration future: Redis)
_weather_cache: dict[str, dict[str, Any]] = {}
_cache_ttl_seconds = 3600  # 1 heure


class WeatherService:
    """Service centralié pour données météo."""

    @staticmethod
    def get_weather(lat: float, lon: float) -> dict[str, Any]:
        """Récupère les données météo pour des coordonnées.

        Args:
            lat: Latitude
            lon: Longitude
        Returns:
            Dict avec données météo + weather_factor calculé

        """
        if not OPENWEATHER_API_KEY:
            logger.warning("[Weather] API key not configured, using default factor")
            return WeatherService._get_default_weather()

        # Vérifier cache
        cache_key = f"{lat},{lon}"
        cached = WeatherService._get_from_cache(cache_key)
        if cached:
            logger.debug("[Weather] Using cached data for %s", cache_key)
            return cached

        # Appeler API
        try:
            import requests

            # Typer correctement params pour satisfaire mypy
            params: dict[str, str | float] = {
                "lat": lat,
                "lon": lon,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric",  # Celsius
                "lang": "fr",
            }

            response = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=5)
            response.raise_for_status()

            data = response.json()

            # Extraire infos pertinentes
            weather_data = WeatherService._parse_weather_response(data)

            # Calculer weather_factor
            weather_data["weather_factor"] = WeatherService._calculate_weather_factor(
                weather_data
            )

            # Mettre en cache
            WeatherService._put_in_cache(cache_key, weather_data)

            logger.info(
                "[Weather] Fetched for (%s,%s) temp=%s°C, conditions=%s, factor=%s",
                lat,
                lon,
                weather_data["temperature"],
                weather_data["main_condition"],
                weather_data["weather_factor"],
            )

            return weather_data

        except Exception as e:
            logger.error("[Weather] API call failed: %s", e)
            return WeatherService._get_default_weather()

    @staticmethod
    def _parse_weather_response(data: dict[str, Any]) -> dict[str, Any]:
        """Parse la réponse OpenWeatherMap.

        Args:
            data: Réponse JSON de l'API
        Returns:
            Dict avec données structurées

        """
        try:
            main = data.get("main", {})
            weather = data.get("weather", [{}])[0]
            wind = data.get("wind", {})
            rain = data.get("rain", {})
            snow = data.get("snow", {})

            return {
                "temperature": main.get("temp", 15),
                "feels_like": main.get("feels_like", 15),
                "humidity": main.get("humidity", 50),
                "main_condition": weather.get(
                    "main", "Clear"
                ),  # Clear, Rain, Snow, etc.
                "description": weather.get("description", ""),
                "wind_speed": wind.get("speed", 0) * 3.6,  # m/s → km/h
                "rain_1h": rain.get("1h", 0),  # mm
                "snow_1h": snow.get("1h", 0),  # mm
                "visibility": data.get("visibility", 10000),  # metres
                "clouds": data.get("clouds", {}).get("all", 0),  # %
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error("[Weather] Failed to parse response: %s", e)
            return WeatherService._get_default_weather()

    @staticmethod
    def _calculate_weather_factor(weather_data: dict[str, Any]) -> float:
        """Calcule le facteur météo (0 = idéal, 1 = très défavorable).
        Facteurs considérés:
        - Précipitations (pluie/neige)
        - Vent
        - Visibilité
        - Nuages
        - Température extrême
        Args:
            weather_data: Données météo parsées
        Returns:
            Float entre 0 et 1.
        """
        factor = 0

        # 1. Précipitations (40% du facteur)
        rain = weather_data.get("rain_1h", 0)
        snow = weather_data.get("snow_1h", 0)

        if snow > SNOW_ZERO:
            # Neige = très impactant
            factor += min(0.4, 0.2 + snow * 0.05)
        elif rain > RAIN_ZERO:
            # Pluie modérément impactant
            factor += min(0.3, 0.1 + rain * 0.02)

        # 2. Vent (20% du facteur)
        wind_speed = weather_data.get("wind_speed", 0)
        if wind_speed > WIND_SPEED_THRESHOLD:  # > WIND_SPEED_THRESHOLD km/h = fort
            factor += 0.2
        elif wind_speed > WIND_SPEED_THRESHOLD:  # WIND_SPEED_THRESHOLD-50 km/h = modéré
            factor += 0.1

        # 3. Visibilité (20% du facteur)
        visibility = weather_data.get("visibility", 10000)
        if visibility < VISIBILITY_THRESHOLD:  # < 1km = brouillard épais
            factor += 0.2
        elif visibility < VISIBILITY_THRESHOLD:  # 1-5km = visibilité réduite
            factor += 0.1

        # 4. Nuages (10% du facteur)
        clouds = weather_data.get("clouds", 0)
        if clouds > CLOUDS_THRESHOLD:  # Très couvert
            factor += 0.05

        # 5. Température extrême (10% du facteur)
        temp = weather_data.get("temperature", 15)
        if temp < TEMP_EXTREME_MIN or temp > TEMP_THRESHOLD:  # Extrême
            factor += 0.1
        elif temp < TEMP_MODERATE_MIN or temp > TEMP_MODERATE_MAX:  # Froid/chaud
            factor += 0.05

        # Limiter entre 0 et 1
        return max(0, min(1, factor))

    @staticmethod
    def _get_default_weather() -> dict[str, Any]:
        """Retourne des conditions météo par défaut (neutre).
        Utilisé en cas d'erreur API ou API key manquante.

        Returns:
            Dict avec conditions neutres

        """
        return {
            "temperature": 15,
            "feels_like": 15,
            "humidity": 50,
            "main_condition": "Clear",
            "description": "Conditions normales (par défaut)",
            "wind_speed": 10,
            "rain_1h": 0,
            "snow_1h": 0,
            "visibility": 10000,
            "clouds": 20,
            "weather_factor": 0.5,  # Neutre
            "timestamp": datetime.now().isoformat(),
            "is_default": True,
        }

    @staticmethod
    def _get_from_cache(key: str) -> dict[str, Any] | None:
        """Récupère depuis le cache si valide.

        Args:
            key: Clé du cache (lat,lon)

        Returns:
            Dict avec données ou None si expiré

        """
        if key not in _weather_cache:
            return None

        cached_data = _weather_cache[key]
        cached_at = datetime.fromisoformat(cached_data["cached_at"])

        # Vérifier expiration (1h)
        if datetime.now() - cached_at > timedelta(seconds=_cache_ttl_seconds):
            del _weather_cache[key]
            return None

        return cached_data["data"]

    @staticmethod
    def _put_in_cache(key: str, data: dict[str, Any]) -> None:
        """Met en cache les données météo.

        Args:
            key: Clé du cache (lat,lon)
            data: Données à cacher

        """
        _weather_cache[key] = {
            "data": data,
            "cached_at": datetime.now().isoformat(),
        }

    @staticmethod
    def clear_cache() -> None:
        """Vide le cache météo."""
        _weather_cache.clear()
        logger.info("[Weather] Cache cleared")

    @staticmethod
    def get_cache_stats() -> dict[str, Any]:
        """Statistiques du cache.

        Returns:
            Dict avec nombre d'entrées et taille

        """
        return {
            "entries": len(_weather_cache),
            "keys": list(_weather_cache.keys()),
        }


def get_weather_factor(lat: float, lon: float) -> float:
    """Helper pour obtenir rapidement le weather_factor.

    Args:
        lat: Latitude
        lon: Longitude
    Returns:
        Float entre 0 et 1 (facteur météo)

    """
    weather = WeatherService.get_weather(lat, lon)
    return weather.get("weather_factor", 0.5)
