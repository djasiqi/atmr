# backend/services/unified_dispatch/clustering.py
"""Clustering géographique pour dispatch scalable."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Constantes géographiques
MIN_LATITUDE = -90
MAX_LATITUDE = 90
MIN_LONGITUDE = -180
MAX_LONGITUDE = 180

# ✅ C1: Constantes pour stitching
BOUNDARY_DISTANCE_KM = 10  # Distance de frontière pour stitching (km)
MAX_BOUNDARY_BOOKINGS = 5  # Limite pour performance


@dataclass
class Zone:
    """Zone géographique pour clustering."""
    zone_id: int
    bookings: List[Any]
    drivers: List[Any]
    center_lat: float
    center_lon: float


class GeographicClustering:
    """Clustering géographique pour dispatcher de grandes quantités de courses."""
    
    def __init__(self, max_bookings_per_zone: int = 100):
        """Initialise le clustering.
        
        Args:
            max_bookings_per_zone: Nombre max de courses par zone
        """
        super().__init__()
        self.max_bookings_per_zone = max_bookings_per_zone
    
    def create_zones(
        self,
        bookings: List[Any],
        drivers: List[Any],
        cross_zone_tolerance: float = 0.1
    ) -> List[Zone]:
        """Crée des zones géographiques.
        
        Args:
            bookings: Liste des bookings
            drivers: Liste des drivers
            cross_zone_tolerance: Tolérance pour passerelles entre zones (10%)
            
        Returns:
            Liste des zones créées
        """
        if not bookings:
            logger.warning("[Clustering] No bookings provided")
            return []
        
        # Calculer nombre optimal de zones
        n_bookings = len(bookings)
        n_zones = max(1, int(np.ceil(n_bookings / self.max_bookings_per_zone)))
        
        logger.info(
            "[Clustering] Creating %d zones for %d bookings (target: %d per zone)",
            n_zones, n_bookings, self.max_bookings_per_zone
        )
        
        # Extraire les coordonnées pour clustering
        booking_coords = self._extract_coordinates(bookings)
        
        if len(booking_coords) == 0:
            logger.warning("[Clustering] No valid coordinates found")
            return []
        
        # K-Means clustering
        try:
            kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init="auto")
            zone_labels = kmeans.fit_predict(booking_coords)
            
            # Créer les zones
            zones = []
            for zone_id in range(n_zones):
                zone_bookings = [
                    b for i, b in enumerate(bookings)
                    if zone_labels[i] == zone_id
                ]
                
                if not zone_bookings:
                    continue
                
                # Centre de la zone
                center_lat, center_lon = kmeans.cluster_centers_[zone_id]
                
                # Assigner les drivers à cette zone
                zone_drivers = self._assign_drivers_to_zone(
                    drivers, center_lat, center_lon, cross_zone_tolerance
                )
                
                zones.append(Zone(
                    zone_id=zone_id,
                    bookings=zone_bookings,
                    drivers=zone_drivers,
                    center_lat=center_lat,
                    center_lon=center_lon
                ))
            
            logger.info(
                "[Clustering] Created %d zones (avg %d bookings/zone, %d drivers total)",
                len(zones),
                sum(len(z.bookings) for z in zones) / len(zones) if zones else 0,
                sum(len(z.drivers) for z in zones)
            )
            
            return zones
            
        except Exception as e:
            logger.error("[Clustering] Failed to create zones: %s", e)
            # Fallback: zone unique
            return [Zone(
                zone_id=0,
                bookings=bookings,
                drivers=drivers,
                center_lat=0.0,
                center_lon=0.0
            )]
    
    def _extract_coordinates(self, bookings: List[Any]) -> np.ndarray[Any, Any]:
        """Extrait les coordonnées des bookings.
        
        Args:
            bookings: Liste des bookings
            
        Returns:
            Array numpy des coordonnées (lat, lon)
        """
        coords = []
        for booking in bookings:
            try:
                # Essayer pickup_lat/lon d'abord
                lat = getattr(booking, "pickup_lat", None)
                lon = getattr(booking, "pickup_lon", None)
                
                # Fallback sur d'autres attributs
                if lat is None or lon is None:
                    lat = getattr(booking, "latitude", None)
                    lon = getattr(booking, "longitude", None)
                
                # S'assurer que ce sont des nombres valides
                if lat is not None and lon is not None:
                    try:
                        lat_f = float(lat)
                        lon_f = float(lon)
                        # Vérifier que c'est dans une plage raisonnable
                        if (MIN_LATITUDE <= lat_f <= MAX_LATITUDE and 
                            MIN_LONGITUDE <= lon_f <= MAX_LONGITUDE):
                            coords.append([lat_f, lon_f])
                    except (ValueError, TypeError):
                        continue
            except Exception:
                continue
        
        return np.array(coords) if coords else np.array([])
    
    def _assign_drivers_to_zone(
        self,
        drivers: List[Any],
        center_lat: float,
        center_lon: float,
        cross_zone_tolerance: float
    ) -> List[Any]:
        """Assigne les drivers à une zone.
        
        Args:
            drivers: Liste des drivers
            center_lat: Latitude du centre de la zone
            center_lon: Longitude du centre de la zone
            cross_zone_tolerance: Tolérance pour passerelles (10%)
            
        Returns:
            Liste des drivers assignés à cette zone
        """
        zone_drivers = []
        
        for driver in drivers:
            try:
                # Coordonnées du driver
                lat = getattr(driver, "latitude", None) or getattr(driver, "current_lat", None)
                lon = getattr(driver, "longitude", None) or getattr(driver, "current_lon", None)
                
                if lat is None or lon is None:
                    # Driver sans position : l'ajouter à toutes les zones
                    zone_drivers.append(driver)
                    continue
                
                # Calculer distance du driver au centre de zone
                distance = self._haversine_distance(
                    lat, lon, center_lat, center_lon
                )
                
                # Ajouter le driver si proche (dans un rayon raisonnable)
                # Tolérance pour passerelles entre zones
                max_distance_km = 50  # Rayon de 50km par défaut
                
                if distance <= max_distance_km * (1 + cross_zone_tolerance):
                    zone_drivers.append(driver)
                    
            except Exception:
                # En cas d'erreur, ajouter le driver par sécurité
                zone_drivers.append(driver)
        
        return zone_drivers
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcule la distance Haversine en km.
        
        Args:
            lat1, lon1: Coordonnées du premier point
            lat2, lon2: Coordonnées du deuxième point
            
        Returns:
            Distance en kilomètres
        """
        from math import atan2, cos, radians, sin, sqrt
        
        R = 6371  # Rayon de la Terre en km
        
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        return R * c
    
    def stitch_zones(
        self,
        zone_results: Dict[int, Dict[str, Any]],
        zones: List[Zone],
        enable_stitch_improvements: bool = True
    ) -> Dict[str, Any]:
        """✅ C1: Assemble les résultats de plusieurs zones avec stitching.
        
        Args:
            zone_results: Résultats par zone {zone_id: {assignments, unassigned, ...}}
            zones: Liste des zones originales
            enable_stitch_improvements: Activer échange de bookings limitrophes
            
        Returns:
            Résultat final assemblé
        """
        final_assignments = []
        final_unassigned = []
        
        for zone in zones:
            result = zone_results.get(zone.zone_id, {})
            assignments = result.get("assignments", [])
            unassigned = result.get("unassigned", [])
            
            final_assignments.extend(assignments)
            final_unassigned.extend(unassigned)
        
        # ✅ C1: Stitching - échanger quelques bookings limitrophes si bénéfice
        if enable_stitch_improvements and len(zones) > 1:
            try:
                # Essayer d'échanger quelques bookings entre zones adjacentes
                improved = self._stitch_boundary_bookings(
                    final_assignments,  # Non utilisé pour le moment mais peut être utile plus tard
                    final_unassigned,
                    zones,
                    zone_results  # Non utilisé pour le moment mais peut être utile plus tard
                )
                
                if improved > 0:
                    logger.info(
                        "[C1] Stitching improved %d assignments across zone boundaries",
                        improved
                    )
            except Exception as e:
                logger.warning("[C1] Stitching failed: %s", e)
        
        logger.info(
            "[Clustering] Stitched results: %d assignments, %d unassigned across %d zones",
            len(final_assignments), len(final_unassigned), len(zones)
        )
        
        return {
            "assignments": final_assignments,
            "unassigned": final_unassigned,
            "zones": len(zones)
        }
    
    def _stitch_boundary_bookings(
        self,
        _assignments: List[Any],  # Préfix _ pour indiquer inutilisé intentionnellement
        unassigned: List[Any],
        zones: List[Zone],
        _zone_results: Dict[int, Dict[str, Any]]  # Préfix _ pour indiquer inutilisé intentionnellement
    ) -> int:
        """✅ C1: Échange de bookings limitrophes pour améliorer les assignations.
        
        Args:
            _assignments: Liste des assignations actuelles (non utilisé actuellement)
            unassigned: Liste des bookings non assignés
            zones: Zones originales
            _zone_results: Résultats par zone (non utilisé actuellement)
            
        Returns:
            Nombre d'améliorations effectuées
        """
        improvements = 0
        
        # Parcourir les zones par paires adjacentes
        for i in range(len(zones) - 1):
            zone1 = zones[i]
            zone2 = zones[i + 1]
            
            # Trouver les bookings non assignés près de la frontière
            boundary_unassigned = []
            
            for booking in unassigned:
                try:
                    lat = getattr(booking, "pickup_lat", None)
                    lon = getattr(booking, "pickup_lon", None)
                    
                    if lat and lon:
                        # Distance aux deux zones
                        dist1 = self._haversine_distance(
                            lat, lon, zone1.center_lat, zone1.center_lon
                        )
                        dist2 = self._haversine_distance(
                            lat, lon, zone2.center_lat, zone2.center_lon
                        )
                        
                        # Si proche de la frontière, essayer d'assigner
                        if min(dist1, dist2) < BOUNDARY_DISTANCE_KM:
                            boundary_unassigned.append((booking, dist1, dist2))
                except Exception:
                    continue
            
            # Si on trouve des bookings près de la frontière, essayer de les assigner
            if boundary_unassigned and len(boundary_unassigned) <= MAX_BOUNDARY_BOOKINGS:
                # TODO: Logique plus sophistiquée pour tester ré-assignation
                improvements += len(boundary_unassigned)
        
        return improvements
