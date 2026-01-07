# backend/tasks/osrm_precompute_tasks.py
"""Tâches Celery pour pré-calculer les matrices OSRM pour zones fréquentes."""

import json
import logging
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Tuple

from celery_app import celery
from ext import db, redis_client
from models import Booking

logger = logging.getLogger(__name__)

# Constantes de configuration
PRECOMPUTE_LOOKBACK_DAYS = 30  # Analyser les 30 derniers jours
PRECOMPUTE_GRID_SIZE = 0.1  # Grille de 0.1° ≈ 11km
PRECOMPUTE_MAX_ZONES = 10  # Maximum 10 zones à pré-calculer
PRECOMPUTE_MIN_BOOKINGS_PER_ZONE = 50  # Minimum 50 bookings pour considérer une zone
PRECOMPUTE_MATRIX_POINTS_PER_ZONE = 20  # Points par zone pour matrice pré-calculée
PRECOMPUTE_CACHE_TTL = 7 * 24 * 3600  # TTL 7 jours pour matrices pré-calculées
PRECOMPUTE_CACHE_PREFIX = "osrm:precomputed:zone:"
MIN_COORDS_FOR_MATRIX = 2  # Minimum 2 coordonnées pour une matrice


def _round_to_grid(coord: Tuple[float, float], grid_size: float) -> Tuple[float, float]:
    """Arrondit une coordonnée à la grille géographique.

    Args:
        coord: (lat, lon)
        grid_size: Taille de la grille en degrés

    Returns:
        Coordonnée arrondie (lat, lon)
    """
    lat, lon = coord
    rounded_lat = round(lat / grid_size) * grid_size
    rounded_lon = round(lon / grid_size) * grid_size
    return (rounded_lat, rounded_lon)


def _identify_frequent_zones(
    company_id: int | None = None, lookback_days: int = PRECOMPUTE_LOOKBACK_DAYS
) -> List[Dict[str, Any]]:
    """Identifie les zones géographiques fréquentes depuis les bookings historiques.

    Args:
        company_id: ID de la company (None = toutes les companies)
        lookback_days: Nombre de jours à analyser en arrière

    Returns:
        Liste de zones avec leurs coordonnées et fréquences
    """
    try:
        # Date de début pour l'analyse
        start_date = datetime.now(UTC) - timedelta(days=lookback_days)

        # Requête pour récupérer les bookings avec coordonnées
        query = db.session.query(Booking).filter(
            Booking.pickup_lat.isnot(None),
            Booking.pickup_lon.isnot(None),
            Booking.dropoff_lat.isnot(None),
            Booking.dropoff_lon.isnot(None),
            Booking.created_at >= start_date,
        )

        if company_id:
            query = query.filter(Booking.company_id == company_id)

        bookings = query.all()

        if not bookings:
            logger.info(
                "[OSRM Precompute] No bookings found for analysis (company_id=%s, days=%d)",
                company_id,
                lookback_days,
            )
            return []

        # Compter les occurrences par zone (grille)
        zone_counter: Counter[Tuple[float, float]] = Counter()
        zone_bookings: Dict[Tuple[float, float], List[Booking]] = defaultdict(list)

        for booking in bookings:
            # Utiliser pickup comme zone principale
            pickup_coord = (
                float(booking.pickup_lat),
                float(booking.pickup_lon),
            )
            zone = _round_to_grid(pickup_coord, PRECOMPUTE_GRID_SIZE)
            zone_counter[zone] += 1
            zone_bookings[zone].append(booking)

        # Sélectionner les zones les plus fréquentes
        top_zones = zone_counter.most_common(PRECOMPUTE_MAX_ZONES)

        frequent_zones = []
        for zone_coord, count in top_zones:
            if count < PRECOMPUTE_MIN_BOOKINGS_PER_ZONE:
                continue

            # Collecter toutes les coordonnées uniques de cette zone
            all_coords = set()
            for booking in zone_bookings[zone_coord]:
                all_coords.add(
                    (
                        float(booking.pickup_lat),  # type: ignore[reportArgumentType]
                        float(booking.pickup_lon),  # type: ignore[reportArgumentType]
                    )
                )
                all_coords.add(
                    (
                        float(booking.dropoff_lat),  # type: ignore[reportArgumentType]
                        float(booking.dropoff_lon),  # type: ignore[reportArgumentType]
                    )
                )

            # Limiter le nombre de points pour la matrice
            coords_list = list(all_coords)[:PRECOMPUTE_MATRIX_POINTS_PER_ZONE]

            frequent_zones.append(
                {
                    "zone_id": f"{zone_coord[0]:.3f},{zone_coord[1]:.3f}",
                    "center": zone_coord,
                    "count": count,
                    "coords": coords_list,
                    "num_points": len(coords_list),
                }
            )

        logger.info(
            "[OSRM Precompute] Identified %d frequent zones (company_id=%s)",
            len(frequent_zones),
            company_id,
        )

        return frequent_zones

    except Exception as e:
        logger.exception("[OSRM Precompute] Error identifying frequent zones: %s", e)
        return []


def _precompute_matrix_for_zone(
    zone: Dict[str, Any], base_url: str, profile: str = "driving"
) -> bool:
    """Pré-calcule une matrice OSRM pour une zone et la stocke dans Redis.

    Args:
        zone: Dictionnaire avec zone_id, center, coords
        base_url: URL de base OSRM
        profile: Profil OSRM (driving, walking, etc.)

    Returns:
        True si succès, False sinon
    """
    try:
        from services.geolocation.osrm import build_distance_matrix_osrm

        coords = zone["coords"]
        zone_id = zone["zone_id"]

        if len(coords) < MIN_COORDS_FOR_MATRIX:
            logger.warning(
                "[OSRM Precompute] Zone %s has < 2 coordinates, skipping", zone_id
            )
            return False

        logger.info(
            "[OSRM Precompute] Computing matrix for zone %s (%d points)",
            zone_id,
            len(coords),
        )

        # Calculer la matrice
        matrix = build_distance_matrix_osrm(
            coords,
            base_url=base_url,
            profile=profile,
            redis_client=redis_client,
        )

        # Stocker dans Redis avec clé spéciale pour pré-calcul
        cache_key = f"{PRECOMPUTE_CACHE_PREFIX}{zone_id}:{profile}"
        matrix_json = json.dumps(matrix)

        if redis_client:
            redis_client.setex(
                cache_key,
                PRECOMPUTE_CACHE_TTL,
                matrix_json.encode("utf-8"),
            )
            logger.info(
                "[OSRM Precompute] ✅ Stored precomputed matrix for zone %s (key=%s)",
                zone_id,
                cache_key,
            )
        else:
            logger.warning(
                "[OSRM Precompute] Redis not available, cannot store precomputed matrix"
            )
            return False

        return True

    except Exception as e:
        logger.exception(
            "[OSRM Precompute] Error precomputing matrix for zone %s: %s",
            zone.get("zone_id", "unknown"),
            e,
        )
        return False


@celery.task(
    name="tasks.osrm_precompute_matrices",
    bind=True,
    acks_late=True,
    task_time_limit=1800,  # 30 minutes max
    task_soft_time_limit=1500,  # 25 minutes soft limit
    max_retries=1,
    autoretry_for=(Exception,),
)
def precompute_osrm_matrices_task(
    self, company_id: int | None = None
) -> Dict[str, Any]:
    """Tâche Celery : Pré-calcule les matrices OSRM pour zones fréquentes.

    Exécutée automatiquement quotidiennement à 3h du matin.

    Args:
        company_id: ID de la company (None = toutes les companies)

    Returns:
        Résultat avec statistiques
    """
    # Utiliser self pour logger le task_id (évite warning linter)
    task_id = getattr(self.request, "id", None) if hasattr(self, "request") else None
    logger.info(
        "[OSRM Precompute] Starting task (task_id=%s, company_id=%s)",
        task_id,
        company_id,
    )

    try:
        import os

        # Récupérer l'URL OSRM depuis les variables d'environnement
        base_url = os.getenv("OSRM_BASE_URL", "http://osrm:5000")
        profile = "driving"

        # Identifier les zones fréquentes
        frequent_zones = _identify_frequent_zones(company_id=company_id)

        if not frequent_zones:
            logger.info(
                "[OSRM Precompute] No frequent zones identified, skipping precompute"
            )
            return {
                "success": True,
                "zones_identified": 0,
                "zones_precomputed": 0,
                "message": "No frequent zones found",
            }

        # Pré-calculer les matrices pour chaque zone
        success_count = 0
        error_count = 0

        for zone in frequent_zones:
            if _precompute_matrix_for_zone(zone, base_url, profile):
                success_count += 1
            else:
                error_count += 1

        logger.info(
            "[OSRM Precompute] ✅ Completed: %d zones precomputed, %d errors",
            success_count,
            error_count,
        )

        return {
            "success": True,
            "zones_identified": len(frequent_zones),
            "zones_precomputed": success_count,
            "zones_errors": error_count,
            "company_id": company_id,
        }

    except Exception as e:
        logger.exception("[OSRM Precompute] Task failed: %s", e)
        raise

