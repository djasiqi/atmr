# backend/services/geofencing_service.py

"""✅ 3.3.4: Service de geofencing pour détection entrée/sortie zones.

Supporte :
- Cercles (rayon + centre)
- Polygones (liste de points)
- Détection arrivée pickup/dropoff
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

from models import Assignment, AssignmentStatus, Booking
from repositories.assignment_repository import AssignmentRepository
from repositories.booking_repository import BookingRepository
from shared.geo_utils import haversine_distance

logger = logging.getLogger(__name__)

# Constantes
DEFAULT_PICKUP_RADIUS_M = 50.0  # 50m pour détection arrivée pickup
DEFAULT_DROPOFF_RADIUS_M = 50.0  # 50m pour détection arrivée dropoff
MIN_POLYGON_POINTS = 3  # Minimum de points pour un polygone valide


@dataclass
class GeofenceCircle:
    """Geofence circulaire."""

    center_lat: float
    center_lon: float
    radius_m: float  # Rayon en mètres

    def contains(self, lat: float, lon: float) -> bool:
        """Vérifie si position dans le cercle."""
        distance_km = haversine_distance(self.center_lat, self.center_lon, lat, lon)
        distance_m = distance_km * 1000
        return distance_m <= self.radius_m

    def distance_to_edge(self, lat: float, lon: float) -> float:
        """Distance depuis le bord du cercle (négative si dedans)."""
        distance_km = haversine_distance(self.center_lat, self.center_lon, lat, lon)
        distance_m = distance_km * 1000
        return distance_m - self.radius_m


@dataclass
class GeofencePolygon:
    """Geofence polygonal (liste de points)."""

    points: List[Tuple[float, float]]  # Liste de (lat, lon)

    def contains(self, lat: float, lon: float) -> bool:
        """Vérifie si position dans le polygone (ray casting algorithm)."""
        if len(self.points) < MIN_POLYGON_POINTS:
            return False

        inside = False
        j = len(self.points) - 1

        for i in range(len(self.points)):
            xi, yi = self.points[i]
            xj, yj = self.points[j]

            if ((yi > lat) != (yj > lat)) and (
                lon < (xj - xi) * (lat - yi) / (yj - yi) + xi
            ):
                inside = not inside
            j = i

        return inside

    def distance_to_edge(self, lat: float, lon: float) -> float:
        """Distance minimale depuis le bord du polygone."""
        # Approximation: distance au point le plus proche
        min_distance = float("inf")
        for point in self.points:
            distance_km = haversine_distance(point[0], point[1], lat, lon)
            distance_m = distance_km * 1000
            min_distance = min(min_distance, distance_m)
        return min_distance


@dataclass
class GeofenceResult:
    """Résultat d'une vérification geofence."""

    inside: bool
    distance_to_edge_m: float  # Distance au bord (négative si dedans)
    geofence_type: str  # "circle" ou "polygon"


class GeofencingService:
    """Service de geofencing pour détection entrée/sortie zones."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        pickup_radius_m: float = DEFAULT_PICKUP_RADIUS_M,
        dropoff_radius_m: float = DEFAULT_DROPOFF_RADIUS_M,
    ):
        """Initialise le service de geofencing.

        Args:
            pickup_radius_m: Rayon pour détection arrivée pickup (mètres)
            dropoff_radius_m: Rayon pour détection arrivée dropoff (mètres)
        """
        self.pickup_radius_m = pickup_radius_m
        self.dropoff_radius_m = dropoff_radius_m

    def check_geofence(
        self,
        position: Tuple[float, float],
        geofence: GeofenceCircle | GeofencePolygon,
    ) -> GeofenceResult:
        """Vérifie si position dans geofence.

        Args:
            position: Position (lat, lon)
            geofence: Geofence à vérifier

        Returns:
            GeofenceResult avec inside et distance_to_edge
        """
        lat, lon = position

        if isinstance(geofence, GeofenceCircle):
            inside = geofence.contains(lat, lon)
            distance_to_edge = geofence.distance_to_edge(lat, lon)
            geofence_type = "circle"
        else:  # GeofencePolygon
            # Type narrowing: si ce n'est pas un cercle, c'est un polygone
            inside = geofence.contains(lat, lon)
            distance_to_edge = geofence.distance_to_edge(lat, lon)
            geofence_type = "polygon"

        return GeofenceResult(
            inside=inside,
            distance_to_edge_m=distance_to_edge,
            geofence_type=geofence_type,
        )

    def detect_pickup_arrival(
        self,
        driver_id: int,
        assignment_id: int,
        driver_lat: float,
        driver_lon: float,
    ) -> bool:
        """Détecte arrivée pickup (rayon 50m par défaut).

        Args:
            driver_id: ID du chauffeur
            assignment_id: ID de l'assignment
            driver_lat: Latitude du chauffeur
            driver_lon: Longitude du chauffeur

        Returns:
            True si arrivée détectée, False sinon
        """
        result_arrived = False
        try:
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            assignment_repo = AssignmentRepository()
            assignment_dto = assignment_repo.find_by_id(assignment_id)
            if not assignment_dto or assignment_dto.driver_id != driver_id:
                return False
            # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
            assignment = Assignment.query.get(assignment_dto.id)
            if not assignment:
                return False

            booking_repo = BookingRepository()
            booking_dto = booking_repo.find_by_id(assignment.booking_id)
            if (
                not booking_dto
                or not booking_dto.pickup_lat
                or not booking_dto.pickup_lon
            ):
                return False
            # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
            booking = Booking.query.get(booking_dto.id)
            if not booking:
                return False

            # Créer geofence circulaire autour du pickup
            pickup_geofence = GeofenceCircle(
                center_lat=booking.pickup_lat,
                center_lon=booking.pickup_lon,
                radius_m=self.pickup_radius_m,
            )

            result = self.check_geofence((driver_lat, driver_lon), pickup_geofence)

            if result.inside:
                msg = (
                    "[GeofencingService] Driver %d arrived at pickup "
                    "(assignment %d, distance=%.1fm)"
                )
                logger.info(
                    msg,
                    driver_id,
                    assignment_id,
                    -result.distance_to_edge_m,
                )
                result_arrived = True
        except Exception as e:
            logger.debug(
                "[GeofencingService] Pickup arrival detection failed: %s", str(e)
            )
        return result_arrived

    def detect_dropoff_arrival(
        self,
        driver_id: int,
        assignment_id: int,
        driver_lat: float,
        driver_lon: float,
    ) -> bool:
        """Détecte arrivée dropoff (rayon 50m par défaut).

        Args:
            driver_id: ID du chauffeur
            assignment_id: ID de l'assignment
            driver_lat: Latitude du chauffeur
            driver_lon: Longitude du chauffeur

        Returns:
            True si arrivée détectée, False sinon
        """
        result_arrived = False
        try:
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            assignment_repo = AssignmentRepository()
            assignment_dto = assignment_repo.find_by_id(assignment_id)
            if not assignment_dto or assignment_dto.driver_id != driver_id:
                return False
            # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
            assignment = Assignment.query.get(assignment_dto.id)
            if not assignment:
                return False

            booking_repo = BookingRepository()
            booking_dto = booking_repo.find_by_id(assignment.booking_id)
            if (
                not booking_dto
                or not booking_dto.dropoff_lat
                or not booking_dto.dropoff_lon
            ):
                return False
            # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
            booking = Booking.query.get(booking_dto.id)
            if not booking:
                return False

            # Créer geofence circulaire autour du dropoff
            dropoff_geofence = GeofenceCircle(
                center_lat=booking.dropoff_lat,
                center_lon=booking.dropoff_lon,
                radius_m=self.dropoff_radius_m,
            )

            result = self.check_geofence((driver_lat, driver_lon), dropoff_geofence)

            if result.inside:
                msg = (
                    "[GeofencingService] Driver %d arrived at dropoff "
                    "(assignment %d, distance=%.1fm)"
                )
                logger.info(
                    msg,
                    driver_id,
                    assignment_id,
                    -result.distance_to_edge_m,
                )
                result_arrived = True
        except Exception as e:
            logger.debug(
                "[GeofencingService] Dropoff arrival detection failed: %s", str(e)
            )
        return result_arrived

    def check_active_assignment_geofencing(
        self, driver_id: int, driver_lat: float, driver_lon: float
    ) -> List[str]:
        """Vérifie geofencing pour assignment actif du chauffeur.

        Args:
            driver_id: ID du chauffeur
            driver_lat: Latitude du chauffeur
            driver_lon: Longitude du chauffeur

        Returns:
            Liste d'events déclenchés (ex: ["arrived_at_pickup", "arrived_at_dropoff"])
        """
        events = []

        try:
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            assignment_repo = AssignmentRepository()
            assignment_dtos = assignment_repo.find_by_driver_id(driver_id)
            # Filtrer par statut IN_PROGRESS en mémoire
            in_progress_dtos = [
                dto for dto in assignment_dtos if dto.status == AssignmentStatus.ONBOARD
            ]
            # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
            assignment = (
                Assignment.query.get(in_progress_dtos[0].id)
                if in_progress_dtos
                else None
            )

            if not assignment:
                return events

            # Vérifier arrivée pickup
            if self.detect_pickup_arrival(
                driver_id, assignment.id, driver_lat, driver_lon
            ):
                events.append("arrived_at_pickup")

            # Vérifier arrivée dropoff
            if self.detect_dropoff_arrival(
                driver_id, assignment.id, driver_lat, driver_lon
            ):
                events.append("arrived_at_dropoff")

        except Exception as e:
            logger.debug(
                "[GeofencingService] Active assignment geofencing failed: %s", str(e)
            )

        return events


# Instance globale (singleton)
_geofencing_service_instance: GeofencingService | None = None


def get_geofencing_service() -> GeofencingService:
    """Retourne l'instance singleton du GeofencingService."""
    global _geofencing_service_instance  # noqa: PLW0603
    if _geofencing_service_instance is None:
        _geofencing_service_instance = GeofencingService()
    return _geofencing_service_instance
