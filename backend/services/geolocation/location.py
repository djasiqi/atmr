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

import contextlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Tuple

import requests
from requests import RequestException, Timeout
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import Session

from application.events.event_bus import publish_event
from domain.assignment_dto import AssignmentDTO
from domain.events.events import DriverLocationUpdatedEvent
from ext import db, redis_client
from models import Assignment, AssignmentStatus, Driver, TripTracking
from repositories.assignment_repository import AssignmentRepository
from repositories.driver_repository import DriverRepository
from services.geolocation.geofencing import get_geofencing_service
from services.geolocation.presence import normalize_location_mode
from services.monitoring.driver_location_metrics import inc_processed

logger = logging.getLogger(__name__)

# Constantes
LAT_THRESHOLD = 90.0
LON_THRESHOLD = 180.0
MIN_POINTS_FOR_MATCHING = 3  # Minimum de points pour map-matching
DEFAULT_OSRM_BASE_URL = os.getenv("UD_OSRM_BASE_URL", "http://osrm:5000")
DEFAULT_DRIVER_LOC_TTL_SEC = int(os.getenv("DRIVER_LOC_TTL_SEC", "1200"))  # 20 min
DEFAULT_MATCH_WINDOW = int(os.getenv("DRIVER_LOC_MATCH_WINDOW", "5"))  # 5 points
OSRM_CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("OSRM_CIRCUIT_BREAKER_THRESHOLD", "5"))
OSRM_CIRCUIT_BREAKER_COOLDOWN_SEC = int(
    os.getenv("OSRM_CIRCUIT_BREAKER_COOLDOWN_SEC", "60")
)

# Missions « en cours » : historique trajet pour tout le cycle actif, pas seulement ONBOARD
# (sinon aucun point pendant EN_ROUTE_* alors que le transport est réel).
TRIP_TRACKING_ASSIGNMENT_STATUSES: frozenset[AssignmentStatus] = frozenset(
    {
        AssignmentStatus.EN_ROUTE_PICKUP,
        AssignmentStatus.ARRIVED_PICKUP,
        AssignmentStatus.ONBOARD,
        AssignmentStatus.EN_ROUTE_DROPOFF,
        AssignmentStatus.ARRIVED_DROPOFF,
    }
)
DEFAULT_GEOFENCE_RADIUS_M = 50.0  # 50m pour détection arrivée pickup/dropoff


def _trip_tracking_min_interval_sec() -> float:
    """Intervalle minimal entre points `trip_tracking` (mission active). Défaut 15 s (alignement mobile)."""
    return float(os.getenv("TRIP_TRACKING_MIN_INTERVAL_SEC", "15"))


# Points légèrement plus vieux que le canon (désordre réseau / batch) : fenêtre élargie
# pour limiter accepted_observability_only inutile. Surchargeable en env.
ARBITRATION_CLOSE_WINDOW_SEC = int(
    os.getenv("DRIVER_LOCATION_ARBITRATION_CLOSE_WINDOW_SEC", "10")
)
MISSION_TOO_OLD_SEC = 120
AVAILABILITY_TOO_OLD_SEC = 600
LOW_ACCURACY_THRESHOLD_M = 1500.0
# v21 : normalisation ``location_mode`` ; sans v21 le backend force ``mission_live`` (arbitrage plus dur).
LOCATION_V21_ENABLED = (
    os.getenv("DRIVER_LOCATION_V21_ENABLED", "true").lower() != "false"
)
TRIP_HISTORY_REPLAY_ENABLED = (
    os.getenv("TRIP_HISTORY_REPLAY_ENABLED", "true").lower() != "false"
)
TRIP_HISTORY_MAX_RECORDED_AGE_SEC = int(
    os.getenv("TRIP_HISTORY_MAX_RECORDED_AGE_SEC", "172800")
)  # 48 h — rejeu hors ligne pour historique analytique


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
    accept_status: str = "accepted_canonical"
    accept_reason: str = ""
    should_fanout: bool = True
    should_persist_db: bool = True
    received_at: str | None = None
    degraded_context: bool = False


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
        self._osrm_failures = 0
        self._osrm_circuit_open_until: datetime | None = None
        self._osrm_degraded_mode = False

    def _is_osrm_circuit_open(self) -> bool:
        if self._osrm_circuit_open_until is None:
            return False
        if datetime.now(UTC) >= self._osrm_circuit_open_until:
            self._osrm_circuit_open_until = None
            self._osrm_degraded_mode = False
            return False
        return True

    def _register_osrm_success(self) -> None:
        self._osrm_failures = 0
        if self._osrm_degraded_mode:
            logger.warning("[LocationService] OSRM recovered, leaving degraded_no_snap")
        self._osrm_degraded_mode = False
        self._osrm_circuit_open_until = None

    def _register_osrm_failure(self, operation: str, error: Exception) -> None:
        self._osrm_failures += 1
        logger.warning(
            "[LocationService] OSRM %s failure (%s): %s (consecutive=%s)",
            operation,
            type(error).__name__,
            str(error),
            self._osrm_failures,
        )
        if self._osrm_failures >= OSRM_CIRCUIT_BREAKER_THRESHOLD:
            self._osrm_degraded_mode = True
            self._osrm_circuit_open_until = datetime.now(UTC).replace(
                microsecond=0
            ) + timedelta(seconds=OSRM_CIRCUIT_BREAKER_COOLDOWN_SEC)
            logger.warning(
                "[LocationService] OSRM degraded_no_snap enabled for %ss after %s failures",
                OSRM_CIRCUIT_BREAKER_COOLDOWN_SEC,
                self._osrm_failures,
            )

    def resolve_normalized_location_mode(
        self, company_id: int | None, location_mode: str
    ) -> str:
        """Mode normalisé identique à ``update_driver_location`` (métriques PR1 / logs)."""
        v21_enabled = self._is_v21_enabled_for_company(company_id)
        return normalize_location_mode(location_mode) if v21_enabled else "mission_live"

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
        location_mode: str = "mission_live",
        recorded_at: datetime | None = None,
        sent_at: datetime | None = None,
        is_background: bool = False,
        mission_id: int | None = None,
        db_session: Session | None = None,
        transport: str = "http",
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
        recorded_dt = recorded_at or timestamp
        sent_dt = sent_at or datetime.now(UTC)
        driver_repo = DriverRepository()
        driver_dto = driver_repo.find_by_id(driver_id)
        company_id = driver_dto.company_id if driver_dto else None
        v21_enabled = self._is_v21_enabled_for_company(company_id)
        normalized_mode = (
            normalize_location_mode(location_mode) if v21_enabled else "mission_live"
        )
        if normalized_mode == "mission_live" and mission_id is None:
            normalized_mode = "availability_presence"
        degraded_context = normalized_mode == "mission_live" and mission_id is None

        # 1. Validation
        if not (-LAT_THRESHOLD <= latitude <= LAT_THRESHOLD):
            raise ValueError(f"Latitude invalide: {latitude}")
        if not (-LON_THRESHOLD <= longitude <= LON_THRESHOLD):
            raise ValueError(f"Longitude invalide: {longitude}")

        snapped_lat, snapped_lon = latitude, longitude
        snap_source = "raw"

        # 2-3. Snap/match uniquement en mission_live
        if normalized_mode == "mission_live":
            try:
                snapped = self._snap_to_road(longitude, latitude)
                if snapped:
                    snapped_lon, snapped_lat = snapped
                    snap_source = "osrm_nearest"
            except (RequestException, Timeout, ConnectionError, OSError) as e:
                logger.debug(
                    "[LocationService] Snap OSRM failed (network error: %s): %s",
                    type(e).__name__,
                    str(e),
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(
                    "[LocationService] Snap OSRM failed (validation error: %s): %s",
                    type(e).__name__,
                    str(e),
                )
            except Exception:
                logger.debug("[LocationService] Snap OSRM failed")

            try:
                matched = self._map_match(driver_id, snapped_lon, snapped_lat)
                if matched:
                    snapped_lon, snapped_lat = matched
                    snap_source = "osrm_match"
            except (RequestException, Timeout, ConnectionError, OSError) as e:
                logger.debug(
                    "[LocationService] Map-matching failed (network error: %s): %s",
                    type(e).__name__,
                    str(e),
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(
                    "[LocationService] Map-matching failed (validation error: %s): %s",
                    type(e).__name__,
                    str(e),
                )
            except Exception:
                logger.debug("[LocationService] Map-matching failed")

        # 4. Stockage Redis + DB
        accept_status, accept_reason, received_at = self._store_location(
            driver_id=driver_id,
            latitude=snapped_lat,
            longitude=snapped_lon,
            speed=speed,
            heading=heading,
            accuracy=accuracy,
            source=snap_source,
            timestamp=timestamp,
            location_mode=normalized_mode,
            recorded_at=recorded_dt,
            sent_at=sent_dt,
            is_background=is_background,
            mission_id=mission_id,
            db_session=db_session,
            degraded_context=degraded_context,
            company_id=company_id,
            transport=transport,
        )
        should_fanout = accept_status == "accepted_canonical"
        should_persist_db = accept_status == "accepted_canonical"

        # 5. Détection geofencing (pickup/dropoff)
        geofencing_service = get_geofencing_service()
        geofence_events: list[str] = []
        if should_persist_db:
            geofence_events = geofencing_service.check_active_assignment_geofencing(
                driver_id=driver_id,
                driver_lat=snapped_lat,
                driver_lon=snapped_lon,
            )

        # 6. Log historique trajet : canon strict OU rejeu analytique (sans toucher au canon Redis)
        trip_logged = False
        if should_persist_db:
            trip_logged = self._log_trip_tracking(
                driver_id=driver_id,
                latitude=snapped_lat,
                longitude=snapped_lon,
                speed=speed,
                heading=heading,
                accuracy=accuracy,
                timestamp=timestamp,
                db_session=db_session,
                history_replay=False,
            )
        elif self._should_append_trip_history(
            location_mode=normalized_mode,
            accept_status=accept_status,
            accept_reason=accept_reason,
            recorded_at=recorded_dt,
        ):
            trip_logged = self._log_trip_tracking(
                driver_id=driver_id,
                latitude=snapped_lat,
                longitude=snapped_lon,
                speed=speed,
                heading=heading,
                accuracy=accuracy,
                timestamp=timestamp,
                db_session=db_session,
                history_replay=True,
            )

        # Event métier (consommable par d'autres services) - sans changer le comportement actuel
        try:
            if should_fanout:
                publish_event(
                    DriverLocationUpdatedEvent(
                        driver_id=driver_id,
                        company_id=company_id,
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

        inc_processed(
            accept_status=accept_status,
            accept_reason=accept_reason,
            location_mode=normalized_mode,
            transport=transport,
        )

        return LocationUpdateResult(
            success=True,
            snapped_lat=snapped_lat,
            snapped_lon=snapped_lon,
            source=snap_source,
            geofence_events=geofence_events,
            trip_logged=trip_logged,
            accept_status=accept_status,
            accept_reason=accept_reason,
            should_fanout=should_fanout,
            should_persist_db=should_persist_db,
            received_at=received_at,
            degraded_context=degraded_context,
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
            if self._is_osrm_circuit_open():
                return None
            url = f"{self.osrm_base_url}/nearest/v1/driving/{longitude},{latitude}"
            r = requests.get(url, params={"number": 1}, timeout=2)
            if r.ok:
                data = r.json()
                waypoints = data.get("waypoints", [])
                if waypoints and waypoints[0].get("location"):
                    loc = waypoints[0]["location"]
                    self._register_osrm_success()
                    return (float(loc[0]), float(loc[1]))  # (lon, lat)
        except (RequestException, Timeout, ConnectionError, OSError) as e:
            self._register_osrm_failure("nearest", e)
        except (ValueError, TypeError, KeyError) as e:
            self._register_osrm_failure("nearest", e)
        except Exception:
            logger.warning("[LocationService] OSRM nearest failed", exc_info=True)
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
            if self._is_osrm_circuit_open():
                return None
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
                        self._register_osrm_success()
                        return (float(loc[0]), float(loc[1]))  # (lon, lat)
        except (RequestException, Timeout, ConnectionError, OSError) as e:
            self._register_osrm_failure("match", e)
        except (ValueError, TypeError, KeyError) as e:
            self._register_osrm_failure("match", e)
        except Exception:
            logger.warning("[LocationService] Map-matching failed", exc_info=True)
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
        location_mode: str,
        recorded_at: datetime,
        sent_at: datetime,
        is_background: bool,
        mission_id: int | None = None,
        db_session: Session | None = None,
        *,
        degraded_context: bool = False,
        company_id: int | None = None,
        transport: str = "http",
    ) -> tuple[str, str, str]:
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
        recorded_iso = recorded_at.isoformat()
        sent_iso = sent_at.isoformat()
        received_dt = datetime.now(UTC)
        received_iso = received_dt.isoformat()
        try:
            from services.monitoring.driver_location_metrics import (
                MAX_CLOCK_SKEW_RECORD_SEC,
                observe_clock_skew_seconds,
            )

            ra = recorded_at if recorded_at.tzinfo else recorded_at.replace(tzinfo=UTC)
            skew_sec = abs((received_dt - ra).total_seconds())
            if skew_sec <= MAX_CLOCK_SKEW_RECORD_SEC:
                observe_clock_skew_seconds(
                    location_mode=location_mode, skew_seconds=skew_sec
                )
        except Exception:
            pass
        accept_status = "accepted_canonical"
        accept_reason = ""
        if not self.redis_client:
            # Sans Redis, aucun arbitrage canonique fiable n'est possible.
            accept_status = "accepted_observability_only"
            accept_reason = "redis_unavailable_no_arbitration"

        # Redis
        if self.redis_client:
            try:
                canonical_key = f"driver:{driver_id}:loc:canonical"
                legacy_key = f"driver:{driver_id}:loc"
                # Compat lecture/écriture pendant migration P2.
                key = canonical_key

                existing_raw = self.redis_client.hgetall(key) or {}
                if not existing_raw:
                    existing_raw = self.redis_client.hgetall(legacy_key) or {}
                existing: dict[str, str] = {}
                for ek, ev in existing_raw.items():
                    try:
                        existing[ek.decode() if isinstance(ek, bytes) else str(ek)] = (
                            ev.decode() if isinstance(ev, bytes) else str(ev)
                        )
                    except Exception:
                        continue

                accept_status, accept_reason = self._arbitrate_update(
                    existing=existing,
                    location_mode=location_mode,
                    recorded_at=recorded_at,
                    accuracy=accuracy,
                )
                try:
                    from services.monitoring.driver_location_metrics import (
                        inc_canonical_overwrite,
                    )

                    had_existing = bool(existing.get("recorded_at") or existing.get("ts"))
                    if accept_status == "accepted_canonical" and had_existing:
                        inc_canonical_overwrite(
                            outcome="accepted", location_mode=location_mode
                        )
                    elif accept_reason == "older_than_canonical":
                        inc_canonical_overwrite(
                            outcome="rejected_older_than_canonical",
                            location_mode=location_mode,
                        )
                except Exception:
                    pass
                existing_company = existing.get("company_id")
                if (
                    existing_company
                    and company_id is not None
                    and existing_company != str(company_id)
                ):
                    accept_status = "rejected_invalid"
                    accept_reason = "cross_tenant_mismatch"
                if (
                    degraded_context
                    and location_mode == "mission_live"
                    and mission_id is None
                    and accept_status == "accepted_canonical"
                    and not accept_reason
                ):
                    accept_reason = "mission_live_missing_mission_id"

                # last_raw toujours mis à jour pour observabilité.
                raw_key = f"driver:{driver_id}:loc:last_raw"
                self.redis_client.hset(
                    raw_key,
                    mapping={
                        "company_id": str(company_id) if company_id else "",
                        "lat": str(latitude),
                        "lon": str(longitude),
                        "speed": str(speed) if speed is not None else "",
                        "heading": str(heading) if heading is not None else "",
                        "accuracy": str(accuracy) if accuracy is not None else "",
                        "ts": ts_iso,
                        "recorded_at": recorded_iso,
                        "sent_at": sent_iso,
                        "received_at": received_iso,
                        "location_mode": location_mode,
                        "is_background": "1" if is_background else "0",
                        "mission_id": str(mission_id) if mission_id is not None else "",
                        "degraded_context": "1" if degraded_context else "0",
                        "source": source,
                        "accept_status": accept_status,
                        "accept_reason": accept_reason,
                    },
                )
                self.redis_client.expire(raw_key, DEFAULT_DRIVER_LOC_TTL_SEC)

                # Écriture canon : seuls les points ``accepted_canonical`` remplissent
                # ``driver:{id}:loc:canonical`` et ``driver:{id}:loc`` (legacy).
                # L'observabilité met à jour ``:last_raw`` mais pas ces hashes → snapshots
                # REST (canon) peuvent diverger du dernier point brut si tout est observabilité.
                if accept_status == "accepted_canonical":
                    canonical_mapping = {
                        "company_id": str(company_id) if company_id else "",
                        "lat": str(latitude),
                        "lon": str(longitude),
                        "speed": str(speed) if speed is not None else "",
                        "heading": str(heading) if heading is not None else "",
                        "accuracy": str(accuracy) if accuracy is not None else "",
                        "ts": ts_iso,
                        "recorded_at": recorded_iso,
                        "sent_at": sent_iso,
                        "received_at": received_iso,
                        "location_mode": location_mode,
                        "is_background": "1" if is_background else "0",
                        "mission_id": str(mission_id) if mission_id is not None else "",
                        "degraded_context": "1" if degraded_context else "0",
                        "source": source,
                    }
                    self.redis_client.hset(canonical_key, mapping=canonical_mapping)
                    self.redis_client.expire(canonical_key, DEFAULT_DRIVER_LOC_TTL_SEC)
                    # Compatibilité transitoire: maintenir l'ancienne clé.
                    self.redis_client.hset(legacy_key, mapping=canonical_mapping)
                    self.redis_client.expire(legacy_key, DEFAULT_DRIVER_LOC_TTL_SEC)
                    try:
                        from services.monitoring.driver_location_metrics import (
                            inc_canonical_redis_write,
                        )

                        inc_canonical_redis_write(
                            location_mode=location_mode, transport=transport
                        )
                    except Exception:
                        pass
                    try:
                        from services.monitoring.driver_location_metrics import (
                            MAX_CLOCK_SKEW_RECORD_SEC,
                            observe_canonical_update_latency_seconds,
                        )

                        sa = sent_at if sent_at.tzinfo else sent_at.replace(tzinfo=UTC)
                        lat_pipe = max(0.0, (received_dt - sa).total_seconds())
                        if lat_pipe <= float(MAX_CLOCK_SKEW_RECORD_SEC):
                            observe_canonical_update_latency_seconds(
                                location_mode=location_mode,
                                transport=transport,
                                seconds=lat_pipe,
                            )
                    except Exception:
                        pass

                # P2: GEO index Redis — index spatial par entreprise pour GEORADIUS/GEOSEARCH
                if company_id is not None and accept_status == "accepted_canonical":
                    try:
                        geo_key = f"driver_locations:geo:{company_id}"
                        self.redis_client.geoadd(
                            geo_key,
                            [longitude, latitude, str(driver_id)],
                        )
                        self.redis_client.expire(geo_key, DEFAULT_DRIVER_LOC_TTL_SEC)
                    except Exception as geo_err:
                        logger.debug(
                            "[LocationService] Redis GEOADD failed (non-blocking): %s",
                            geo_err,
                        )

                # P2: Stream ingestion — XADD pour analytics/replay
                try:
                    stream_key = "driver_location_stream"
                    self.redis_client.xadd(
                        stream_key,
                        {
                            "driver_id": str(driver_id),
                            "company_id": str(company_id) if company_id else "",
                            "lat": str(latitude),
                            "lon": str(longitude),
                            "ts": ts_iso,
                            "recorded_at": recorded_iso,
                            "received_at": received_iso,
                            "location_mode": location_mode,
                            "degraded_context": "1" if degraded_context else "0",
                            "source": source,
                            "accept_status": accept_status,
                            "accept_reason": accept_reason,
                        },
                        maxlen=10000,
                        approximate=True,
                    )
                except Exception as stream_err:
                    logger.debug(
                        "[LocationService] Redis XADD failed (non-blocking): %s",
                        stream_err,
                    )
            except (ConnectionError, OSError, TimeoutError) as e:
                # Erreurs réseau attendues : Redis indisponible, timeout
                logger.warning(
                    "[LocationService] Redis store failed (network error: %s): %s",
                    type(e).__name__,
                    str(e),
                )
                accept_status = "accepted_observability_only"
                accept_reason = "redis_unavailable_no_arbitration"
            except (ValueError, TypeError) as e:
                # Erreurs de validation attendues : données non sérialisables
                logger.warning(
                    "[LocationService] Redis store failed (validation error: %s): %s",
                    type(e).__name__,
                    str(e),
                )
                accept_status = "accepted_observability_only"
                accept_reason = "redis_unavailable_no_arbitration"
            except Exception:
                # Erreur inattendue lors du stockage Redis
                logger.warning("[LocationService] Redis store failed")
                accept_status = "accepted_observability_only"
                accept_reason = "redis_unavailable_no_arbitration"
        if degraded_context and location_mode == "mission_live" and mission_id is None:
            logger.warning(
                "[LocationService] mission_live sans mission_id (driver=%s, company=%s)",
                driver_id,
                company_id,
            )
            if self.redis_client:
                with contextlib.suppress(Exception):
                    self.redis_client.incr(
                        "driver_location:mission_live_missing_mission_id:global"
                    )
                    if company_id:
                        self.redis_client.incr(
                            f"driver_location:mission_live_missing_mission_id:company:{company_id}"
                        )

        # DB
        session = db_session or db.session
        try:
            if accept_status == "accepted_canonical":
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
        return accept_status, accept_reason, received_iso

    def _is_v21_enabled_for_company(self, company_id: int | None) -> bool:
        if not LOCATION_V21_ENABLED:
            return False
        if company_id is None:
            return LOCATION_V21_ENABLED
        raw = os.getenv("DRIVER_LOCATION_V21_TENANT_OVERRIDES", "").strip()
        if not raw:
            return LOCATION_V21_ENABLED
        for token in raw.split(","):
            item = token.strip()
            if not item or ":" not in item:
                continue
            cid_raw, enabled_raw = item.split(":", 1)
            with contextlib.suppress(Exception):
                if int(cid_raw.strip()) != int(company_id):
                    continue
                return enabled_raw.strip().lower() in {"1", "true", "yes", "on"}
        return LOCATION_V21_ENABLED

    def _should_append_trip_history(
        self,
        *,
        location_mode: str,
        accept_status: str,
        accept_reason: str,
        recorded_at: datetime,
    ) -> bool:
        """Historique analytique : points rejetés du canon mais encore utiles en base (replay)."""
        if not TRIP_HISTORY_REPLAY_ENABLED or location_mode != "mission_live":
            return False
        if accept_status != "accepted_observability_only":
            return False
        if accept_reason not in {"older_than_canonical", "too_old_for_mode"}:
            return False
        now = datetime.now(UTC)
        ra = recorded_at if recorded_at.tzinfo else recorded_at.replace(tzinfo=UTC)
        age_sec = max(0.0, (now - ra).total_seconds())
        return age_sec <= float(TRIP_HISTORY_MAX_RECORDED_AGE_SEC)

    def _arbitrate_update(
        self,
        *,
        existing: dict[str, str],
        location_mode: str,
        recorded_at: datetime,
        accuracy: float | None,
    ) -> tuple[str, str]:
        # Payload invalide en pratique filtré avant. On garde la classification ici.
        if location_mode not in {
            "mission_live",
            "availability_presence",
            "passive_last_known",
        }:
            return "rejected_invalid", "invalid_payload"

        now = datetime.now(UTC)
        age_sec = max(0, int((now - recorded_at).total_seconds()))
        max_age = (
            MISSION_TOO_OLD_SEC
            if location_mode == "mission_live"
            else AVAILABILITY_TOO_OLD_SEC
        )
        if age_sec > max_age:
            return "accepted_observability_only", "too_old_for_mode"

        if accuracy is not None and accuracy > LOW_ACCURACY_THRESHOLD_M:
            return "accepted_observability_only", "accuracy_too_low"

        current_recorded = existing.get("recorded_at") or existing.get("ts")
        if not current_recorded:
            return "accepted_canonical", ""
        try:
            current_dt = datetime.fromisoformat(current_recorded.replace("Z", "+00:00"))
        except Exception:
            return "accepted_canonical", ""

        if recorded_at > current_dt:
            return "accepted_canonical", ""

        # Même seconde (quantification GPS / batch) : ne pas laisser la priorité
        # mission_live > availability_presence bloquer le rafraîchissement du canon
        # quand le chauffeur envoie de la disponibilité après une mission.
        if recorded_at == current_dt and location_mode in (
            "mission_live",
            "availability_presence",
        ):
            return "accepted_canonical", ""

        delta = abs((recorded_at - current_dt).total_seconds())
        if delta <= ARBITRATION_CLOSE_WINDOW_SEC:
            priorities = {
                "mission_live": 3,
                "availability_presence": 2,
                "passive_last_known": 1,
            }
            current_mode = normalize_location_mode(existing.get("location_mode"))
            if priorities.get(location_mode, 0) >= priorities.get(current_mode, 0):
                return "accepted_canonical", ""

        return "accepted_observability_only", "older_than_canonical"

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
        *,
        history_replay: bool = False,
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
            history_replay: si True, autorise l'insertion de points plus vieux que le dernier
                log (replay hors ordre) pour l'historique analytique uniquement.

        Returns:
            True si position loggée, False sinon
        """
        try:
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            assignment_repo = AssignmentRepository()
            assignment_dtos = assignment_repo.find_by_driver_id(driver_id)
            active_dtos = [
                dto
                for dto in assignment_dtos
                if dto.status in TRIP_TRACKING_ASSIGNMENT_STATUSES
            ]
            if not active_dtos:
                return False

            # Double mission / plusieurs courses : celle mise à jour la plus récemment
            epoch = datetime(1970, 1, 1, tzinfo=UTC)

            def _trip_tracking_sort_key(dto: AssignmentDTO) -> tuple[datetime, int]:
                return (dto.updated_at or epoch, dto.id)

            active_dtos.sort(key=_trip_tracking_sort_key, reverse=True)
            assignment = Assignment.query.get(active_dtos[0].id)

            if not assignment:
                return False

            min_interval = _trip_tracking_min_interval_sec()
            last_log = (
                TripTracking.query.filter_by(assignment_id=assignment.id)
                .order_by(TripTracking.timestamp.desc())
                .first()
            )

            replay_older = bool(
                history_replay and last_log and timestamp < last_log.timestamp
            )
            if not replay_older and last_log:
                time_since_last = (timestamp - last_log.timestamp).total_seconds()
                if time_since_last < min_interval:
                    return False

            # Logger position
            session = db_session or db.session
            if (
                TripTracking.query.filter_by(
                    assignment_id=assignment.id,
                    timestamp=timestamp,
                ).first()
                is not None
            ):
                return False
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
        except (
            OperationalError,
            DBAPIError,
            ValueError,
            TypeError,
            AttributeError,
        ) as e:
            # DB (connexion, timeout) ou validation
            logger.debug(
                "[LocationService] Trip tracking log failed (%s): %s",
                type(e).__name__,
                str(e),
            )
            if not db_session:
                db.session.rollback()
            return False
        except Exception:
            logger.debug("[LocationService] Trip tracking log failed")
            if not db_session:
                db.session.rollback()
            return False


# Instance globale (singleton)
_location_service_instance: LocationService | None = None


def get_location_service() -> LocationService:
    """Retourne l'instance singleton du LocationService."""
    global _location_service_instance
    if _location_service_instance is None:
        osrm_url = os.getenv("UD_OSRM_BASE_URL", DEFAULT_OSRM_BASE_URL)
        _location_service_instance = LocationService(osrm_base_url=osrm_url)
    return _location_service_instance
