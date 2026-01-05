"""Interfaces abstraites pour faciliter une future migration vers microservices.

Ces interfaces définissent les contrats entre services, permettant de :
1. Faciliter les tests unitaires (mocks)
2. Préparer une migration vers microservices (remplacement par API REST)
3. Améliorer la séparation des responsabilités
"""

from .geocoding_interface import GeocodingServiceInterface
from .notification_interface import NotificationServiceInterface
from .routing_interface import RoutingServiceInterface
from .weather_interface import WeatherServiceInterface

__all__ = [
    "GeocodingServiceInterface",
    "NotificationServiceInterface",
    "RoutingServiceInterface",
    "WeatherServiceInterface",
]
