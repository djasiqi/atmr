# backend/services/location_service.py

"""✅ 3.3.1: Service centralisé pour localisation avec snap, map-matching, geofencing et historique.

Ce service unifie toute la logique de localisation :
- Validation et snap OSRM nearest
- Map-matching avec ring buffer
- Stockage Redis + DB
- Diffusion Socket.IO
- Détection geofencing (pickup/dropoff)
- Log historique trajets
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Tuple

import requests  # pyright: ignore[reportMissingModuleSource]
from requests import (  # pyright: ignore[reportMissingModuleSource]
    RequestException,
    Timeout,
)
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import Session

from application.events.event_bus import publish_event
from domain.events.events import DriverLocationUpdatedEvent
from ext import db, redis_client
from models import Assignment, AssignmentStatus, Driver, TripTracking
from repositories.assignment_repository import AssignmentRepository
from repositories.driver_repository import DriverRepository
from services.geofencing_service import get_geofencing_service

logger = logging.getLogger(__name__)

# Constantes
LAT_THRESHOLD = 90.0
LON_THRESHOLD = 180.0
MIN_POINTS_FOR_MATCHING = 3  # Minimum de points pour map-matching
DEFAULT_OSRM_BASE_URL = os.getenv("UD_OSRM_BASE_URL", "http://osrm:5000")
DEFAULT_DRIVER_LOC_TTL_SEC = int(os.getenv("DRIVER_LOC_TTL_SEC", "600"))  # 10 min
DEFAULT_MATCH_WINDOW = int(os.getenv("DRIVER_LOC_MATCH_WINDOW", "5"))  # 5 points
DEFAULT_GEOFENCE_RADIUS_M = 50.0  # 50m pour détection arrivée pickup/dropoff
TRIP_TRACKING_MIN_INTERVAL_SEC = 30  # Minimum 30s entre logs pour performance


@dataclass
class LocationUpdateResult:
    """Résultat d'une mise à jour de localisation."""

    success: bool
    snapped_lat: float
    snapped_lon: float
    source: str  # "raw", "osrm_nearest", "osrm_match"
    geofence_events: list[str] = field(
        default_factory=list
    )  # Events geofencing déclenchés
    trip_logged: bool = False  # Si position loggée dans historique


class LocationService:
    """Service centralisé pour localisation avec snap, map-matching, geofencing et historique."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        osrm_base_url: str = DEFAULT_OSRM_BASE_URL,
        redis_client_instance: Any | None = None,
        match_window: int = DEFAULT_MATCH_WINDOW,
        geofence_radius_m: float = DEFAULT_GEOFENCE_RADIUS_M,
    ):
        """Initialise le service de localisation.

        Args:
            osrm_base_url: URL du serveur OSRM
            redis_client_instance: Client Redis (optionnel)
            match_window: Nombre de points pour map-matching
            geofence_radius_m: Rayon geofence pour détection arrivée (mètres)
        """
        self.osrm_base_url = osrm_base_url
        self.redis_client = redis_client_instance or redis_client
        self.match_window = match_window
        self.geofence_radius_m = geofence_radius_m

    def update_driver_location(
        self,
        driver_id: int,
        latitude: float,
        longitude: float,
        *,
        speed: float | None = None,
        heading: float | None = None,
        accuracy: float | None = None,
        source: str = "gps",  # noqa: ARG002
        timestamp: datetime | None = None,
        db_session: Session | None = None,
    ) -> LocationUpdateResult:
        """Met à jour la position d'un chauffeur (snap OSRM + map-matching + stockage).

        Args:
            driver_id: ID du chauffeur
            latitude: Latitude brute
            longitude: Longitude brute
            speed: Vitesse (m/s, optionnel)
            heading: Cap (degrés, optionnel)
            accuracy: Précision GPS (mètres, optionnel)
            source: Source de la position ("gps", "network", etc.)
            timestamp: Timestamp de la position (défaut: maintenant)
            db_session: Session DB (optionnel, pour transactions)

        Returns:
            LocationUpdateResult avec position snapée et métadonnées

        Raises:
            ValueError: si les coordonnées sont hors bornes ([-90, 90], [-180, 180]).

        Exemple:
            >>> service = LocationService(redis_client_instance=None)  # doctest: +SKIP
            >>> res = service.update_driver_location(driver_id=1, latitude=46.2, longitude=6.1)  # doctest: +SKIP
            >>> res.success  # doctest: +SKIP
            True
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        # 1. Validation
        if not (-LAT_THRESHOLD <= latitude <= LAT_THRESHOLD):
            raise ValueError(f"Latitude invalide: {latitude}")
        if not (-LON_THRESHOLD <= longitude <= LON_THRESHOLD):
            raise ValueError(f"Longitude invalide: {longitude}")

        snapped_lat, snapped_lon = latitude, longitude
        snap_source = "raw"

        # 2. Snap OSRM nearest
        try:
            snapped = self._snap_to_road(longitude, latitude)
            if snapped:
                snapped_lon, snapped_lat = snapped
                snap_source = "osrm_nearest"
        except (RequestException, Timeout, ConnectionError, OSError) as e:
            # Erreurs réseau attendues : OSRM indisponible, timeout
            logger.debug(
                "[LocationService] Snap OSRM failed (network error: %s): %s",
                type(e).__name__,
                str(e),
            )
        except (ValueError, TypeError, KeyError) as e:
            # Erreurs de validation attendues : réponse JSON invalide
            logger.debug(
                "[LocationService] Snap OSRM failed (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
        except Exception:
            # Erreur inattendue lors du snap OSRM
            logger.debug("[LocationService] Snap OSRM failed")

        # 3. Map-matching si ring buffer suffisant
        try:
            matched = self._map_match(driver_id, snapped_lon, snapped_lat)
            if matched:
                snapped_lon, snapped_lat = matched
                snap_source = "osrm_match"
        except (RequestException, Timeout, ConnectionError, OSError) as e:
            # Erreurs réseau attendues : OSRM indisponible, timeout
            logger.debug(
                "[LocationService] Map-matching failed (network error: %s): %s",
                type(e).__name__,
                str(e),
            )
        except (ValueError, TypeError, KeyError) as e:
            # Erreurs de validation attendues : réponse JSON invalide
            logger.debug(
                "[LocationService] Map-matching failed (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
        except Exception:
            # Erreur inattendue lors du map-matching
            logger.debug("[LocationService] Map-matching failed")

        # 4. Stockage Redis + DB
        self._store_location(
            driver_id=driver_id,
            latitude=snapped_lat,
            longitude=snapped_lon,
            speed=speed,
            heading=heading,
            accuracy=accuracy,
            source=snap_source,
            timestamp=timestamp,
            db_session=db_session,
        )

        # 5. Détection geofencing (pickup/dropoff)
        geofencing_service = get_geofencing_service()
        geofence_events = geofencing_service.check_active_assignment_geofencing(
            driver_id=driver_id,
            driver_lat=snapped_lat,
            driver_lon=snapped_lon,
        )

        # 6. Log historique si en trajet
        trip_logged = self._log_trip_tracking(
            driver_id=driver_id,
            latitude=snapped_lat,
            longitude=snapped_lon,
            speed=speed,
            heading=heading,
            accuracy=accuracy,
            timestamp=timestamp,
            db_session=db_session,
        )

        # Event métier (consommable par d'autres services) - sans changer le comportement actuel
        try:
            driver_repo = DriverRepository()
            driver_dto = driver_repo.find_by_id(driver_id)
            publish_event(
                DriverLocationUpdatedEvent(
                    driver_id=driver_id,
                    company_id=driver_dto.company_id if driver_dto else None,
                )
            )
        except (ValueError, TypeError, AttributeError, KeyError):
            # Erreurs de validation attendues : données invalides
            # Ne pas faire échouer l'endpoint de localisation pour un event
            pass
        except Exception:
            # Erreur inattendue lors de la publication d'événement
            # Ne pas faire échouer l'endpoint de localisation pour un event
            logger.debug("[LocationService] Event publish failed (ignored)")

        return LocationUpdateResult(
            success=True,
            snapped_lat=snapped_lat,
            snapped_lon=snapped_lon,
            source=snap_source,
            geofence_events=geofence_events,
            trip_logged=trip_logged,
        )

    def _snap_to_road(
        self, longitude: float, latitude: float
    ) -> Tuple[float, float] | None:
        """Snap une position GPS sur la chaussée la plus proche via OSRM nearest.

        Args:
            longitude: Longitude
            latitude: Latitude

        Returns:
            Tuple (lon, lat) snapée ou None si échec
        """
        try:
            url = f"{self.osrm_base_url}/nearest/v1/driving/{longitude},{latitude}"
            r = requests.get(url, params={"number": 1}, timeout=2)
            if r.ok:
                data = r.json()
                waypoints = data.get("waypoints", [])
                if waypoints and waypoints[0].get("location"):
                    loc = waypoints[0]["location"]
                    return (float(loc[0]), float(loc[1]))  # (lon, lat)
        except (RequestException, Timeout, ConnectionError, OSError) as e:
            # Erreurs réseau attendues : OSRM indisponible, timeout
            logger.debug(
                "[LocationService] OSRM nearest failed (network error: %s): %s",
                type(e).__name__,
                str(e),
            )
        except (ValueError, TypeError, KeyError) as e:
            # Erreurs de validation attendues : réponse JSON invalide
            logger.debug(
                "[LocationService] OSRM nearest failed (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
        except Exception:
            # Erreur inattendue lors de l'appel OSRM nearest
            logger.debug("[LocationService] OSRM nearest failed")
        return None

    def _map_match(
        self, driver_id: int, longitude: float, latitude: float
    ) -> Tuple[float, float] | None:
        """Map-matching avec ring buffer de positions récentes.

        Args:
            driver_id: ID du chauffeur
            longitude: Longitude
            latitude: Latitude

        Returns:
            Tuple (lon, lat) matchée ou None si échec
        """
        if not self.redis_client:
            return None

        try:
            # Ajouter point actuel au ring buffer
            ring_key = f"driver:{driver_id}:ring"
            point = {
                "ts": datetime.now(UTC).isoformat(),
                "lat": latitude,
                "lon": longitude,
            }
            self.redis_client.lpush(ring_key, json.dumps(point))
            self.redis_client.ltrim(ring_key, 0, self.match_window - 1)
            self.redis_client.expire(ring_key, DEFAULT_DRIVER_LOC_TTL_SEC)

            # Récupérer points du ring buffer
            lrange_result = self.redis_client.lrange(ring_key, 0, self.match_window - 1)
            if not lrange_result:
                return None

            # Convertir en liste (lrange peut retourner différents types selon implémentation Redis)
            pts_raw: list[bytes | str] = (
                list(lrange_result) if isinstance(lrange_result, (list, tuple)) else []
            )
            if not pts_raw:
                return None

            # Décoder bytes en strings si nécessaire
            pts = []
            for raw_item in pts_raw:
                decoded_item = (
                    raw_item.decode("utf-8")
                    if isinstance(raw_item, bytes)
                    else raw_item
                )
                pts.append(json.loads(decoded_item))
            if len(pts) < MIN_POINTS_FOR_MATCHING:
                return None

            # Construire chaîne de coordonnées pour OSRM match
            coords = ";".join(
                f"{pp['lon']:.6f},{pp['lat']:.6f}" for pp in reversed(pts)
            )

            # Appel OSRM match
            url = f"{self.osrm_base_url}/match/v1/driving/{coords}"
            r = requests.get(
                url, params={"tidy": "true", "overview": "false"}, timeout=3
            )

            if r.ok:
                data = r.json()
                matchings = data.get("matchings", [])
                tracepoints = data.get("tracepoints", [])

                if matchings and tracepoints:
                    # Prendre le dernier tracepoint (position actuelle)
                    tp = tracepoints[-1]
                    if tp and tp.get("location"):
                        loc = tp["location"]
                        return (float(loc[0]), float(loc[1]))  # (lon, lat)
        except (RequestException, Timeout, ConnectionError, OSError) as e:
            # Erreurs réseau attendues : OSRM indisponible, timeout
            logger.debug(
                "[LocationService] Map-matching failed (network error: %s): %s",
                type(e).__name__,
                str(e),
            )
        except (ValueError, TypeError, KeyError) as e:
            # Erreurs de validation attendues : réponse JSON invalide
            logger.debug(
                "[LocationService] Map-matching failed (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
        except Exception:
            # Erreur inattendue lors du map-matching
            logger.debug("[LocationService] Map-matching failed")
        return None

    def _store_location(
        self,
        driver_id: int,
        latitude: float,
        longitude: float,
        speed: float | None,
        heading: float | None,
        accuracy: float | None,
        source: str,
        timestamp: datetime,
        db_session: Session | None = None,
    ) -> None:
        """Stocke la position dans Redis et DB.

        Args:
            driver_id: ID du chauffeur
            latitude: Latitude snapée
            longitude: Longitude snapée
            speed: Vitesse (m/s)
            heading: Cap (degrés)
            accuracy: Précision GPS (mètres)
            source: Source de la position
            timestamp: Timestamp
            db_session: Session DB (optionnel)
        """
        ts_iso = timestamp.isoformat()

        # Redis
        if self.redis_client:
            try:
                key = f"driver:{driver_id}:loc"
                # ✅ Utilisation du repository pour découpler de SQLAlchemy
                driver_repo = DriverRepository()
                driver_dto = driver_repo.find_by_id(driver_id)
                company_id = driver_dto.company_id if driver_dto else None

                self.redis_client.hset(
                    key,
                    mapping={
                        "company_id": str(company_id) if company_id else "",
                        "lat": str(latitude),
                        "lon": str(longitude),
                        "speed": str(speed) if speed is not None else "",
                        "heading": str(heading) if heading is not None else "",
                        "accuracy": str(accuracy) if accuracy is not None else "",
                        "ts": ts_iso,
                        "source": source,
                    },
                )
                self.redis_client.expire(key, DEFAULT_DRIVER_LOC_TTL_SEC)
            except (ConnectionError, OSError, TimeoutError) as e:
                # Erreurs réseau attendues : Redis indisponible, timeout
                logger.warning(
                    "[LocationService] Redis store failed (network error: %s): %s",
                    type(e).__name__,
                    str(e),
                )
            except (ValueError, TypeError) as e:
                # Erreurs de validation attendues : données non sérialisables
                logger.warning(
                    "[LocationService] Redis store failed (validation error: %s): %s",
                    type(e).__name__,
                    str(e),
                )
            except Exception:
                # Erreur inattendue lors du stockage Redis
                logger.warning("[LocationService] Redis store failed")

        # DB
        session = db_session or db.session
        try:
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            driver_repo = DriverRepository()
            driver_dto = driver_repo.find_by_id(driver_id)
            if driver_dto:
                # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
                driver = Driver.query.get(driver_dto.id)
                if driver:
                    driver.latitude = latitude
                    driver.longitude = longitude
                    driver.last_position_update = timestamp
                    session.add(driver)
                if not db_session:
                    session.commit()
        except (OperationalError, DBAPIError) as e:
            # Erreurs DB attendues : connexion, timeout
            if not db_session:
                session.rollback()
            logger.warning(
                "[LocationService] DB store failed (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
        except (ValueError, TypeError, AttributeError) as e:
            # Erreurs de validation attendues : données invalides
            if not db_session:
                session.rollback()
            logger.warning(
                "[LocationService] DB store failed (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
        except Exception:
            # Erreur inattendue lors du stockage DB
            if not db_session:
                session.rollback()
            logger.warning("[LocationService] DB store failed")

    def _log_trip_tracking(
        self,
        driver_id: int,
        latitude: float,
        longitude: float,
        speed: float | None,
        heading: float | None,
        accuracy: float | None,
        timestamp: datetime,
        db_session: Session | None = None,
    ) -> bool:
        """Log position dans historique trajets si assignment IN_PROGRESS.

        Args:
            driver_id: ID du chauffeur
            latitude: Latitude
            longitude: Longitude
            speed: Vitesse (m/s)
            heading: Cap (degrés)
            accuracy: Précision GPS (mètres)
            timestamp: Timestamp
            db_session: Session DB (optionnel)

        Returns:
            True si position loggée, False sinon
        """
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
                return False

            # Batch insert: ne logger que toutes les 30s max pour performance
            # (vérifier dernière position loggée)
            last_log = (
                TripTracking.query.filter_by(assignment_id=assignment.id)
                .order_by(TripTracking.timestamp.desc())
                .first()
            )

            if last_log:
                time_since_last = (timestamp - last_log.timestamp).total_seconds()
                if time_since_last < TRIP_TRACKING_MIN_INTERVAL_SEC:
                    return False

            # Logger position
            session = db_session or db.session
            # Créer instance TripTracking avec attributs
            trip_tracking = TripTracking()
            trip_tracking.assignment_id = assignment.id
            trip_tracking.booking_id = assignment.booking_id
            trip_tracking.driver_id = driver_id
            trip_tracking.latitude = latitude
            trip_tracking.longitude = longitude
            trip_tracking.speed = speed
            trip_tracking.heading = heading
            trip_tracking.accuracy = accuracy
            trip_tracking.timestamp = timestamp
            session.add(trip_tracking)
            if not db_session:
                session.commit()

            return True
        except (OperationalError, DBAPIError) as e:
            # Erreurs DB attendues : connexion, timeout
            logger.debug(
                "[LocationService] Trip tracking log failed (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            if not db_session:
                db.session.rollback()
            return False
        except (ValueError, TypeError, AttributeError) as e:
            # Erreurs de validation attendues : données invalides
            logger.debug(
                "[LocationService] Trip tracking log failed (validation error: %s): %s",
                type(e).__name__,
                str(e),
            )
            if not db_session:
                db.session.rollback()
            return False
        except Exception:
            # Erreur inattendue lors du log trip tracking
            logger.debug("[LocationService] Trip tracking log failed")
            if not db_session:
                db.session.rollback()
            return False


# Instance globale (singleton)
_location_service_instance: LocationService | None = None


def get_location_service() -> LocationService:
    """Retourne l'instance singleton du LocationService."""
    global _location_service_instance  # noqa: PLW0603
    if _location_service_instance is None:
        osrm_url = os.getenv("UD_OSRM_BASE_URL", DEFAULT_OSRM_BASE_URL)
        _location_service_instance = LocationService(osrm_base_url=osrm_url)
    return _location_service_instance
