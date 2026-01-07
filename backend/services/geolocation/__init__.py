"""
Module `geolocation` - Consolidation des services de géolocalisation et routing

Ce module regroupe tous les services liés à la géolocalisation, cartographie et routing :
- Services de géolocalisation et localisation
- Geofencing (zones géographiques)
- Intégration cartes (Google Maps, etc.)
- Recherche de lieux (Google Places)
- Routing et optimisation de trajets (OSRM)
- Interfaces abstraites pour géocodage et routing

## Migration B2 (7 janvier 2025)

Ce module consolide 7 services fragmentés :
- `geolocation_service.py` → `geolocation/core.py`
- `location_service.py` → `geolocation/location.py`
- `geofencing_service.py` → `geolocation/geofencing.py`
- `maps.py` → `geolocation/maps.py`
- `google_places.py` → `geolocation/google_places.py`
- `osrm_client.py` → `geolocation/osrm.py`
- `interfaces/geocoding_interface.py` + `interfaces/routing_interface.py` → `geolocation/interfaces.py`

## Structure

```
geolocation/
├── __init__.py              # Exports publics
├── core.py                  # Service géolocalisation principal
├── location.py              # Service localisation
├── geofencing.py            # Geofencing et zones
├── maps.py                  # Intégration cartes
├── google_places.py         # Recherche lieux Google
├── osrm.py                  # Client OSRM (routing)
└── interfaces.py            # Interfaces abstraites
```

## Usage

```python
# Imports recommandés (nouveaux)
from services.geolocation.core import GeolocationService
from services.geolocation.location import LocationService
from services.geolocation.geofencing import GeofencingService
from services.geolocation.maps import MapsService
from services.geolocation.google_places import GooglePlacesService
from services.geolocation.osrm import OSRMClient
from services.geolocation.interfaces import GeocodingInterface, RoutingInterface

# Imports de compatibilité (DEPRECATED, à migrer)
# from services.geolocation.core import GeolocationService
# from services.geolocation.osrm import OSRMClient
```

## Documentation

- Architecture : `docs/GEOLOCATION_ARCHITECTURE.md`
- Migration : `PLAN_CONSOLIDATION_B2_SERVICES.md`

---

**Version :** 1.0.0 (B2 Refactoring)  
**Date :** 7 janvier 2025
"""

# ========== Exports publics ==========

# Exports seront ajoutés au fur et à mesure de la migration
# from .core import GeolocationService
# from .osrm import OSRMClient

__all__ = [
    # Les exports seront ajoutés après migration
]

__version__ = "1.0.0"
__refactoring__ = "B2 - Services Consolidation"


