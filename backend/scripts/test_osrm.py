#!/usr/bin/env python3
"""Test rapide d'OSRM pour vérifier que le service fonctionne"""

import sys
from pathlib import Path

# Modifier sys.path avant l'import du module local
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from services.osrm_client import route_info  # noqa: E402

try:
    result = route_info(
        (46.2044, 6.1432),
        (46.2050, 6.1450),
        base_url="http://osrm:5000",
        timeout=5
    )
    print("✅ OSRM fonctionne:")
    print(f"   - Distance: {result.get('distance', 0):.0f}m")
    print(f"   - Durée: {result.get('duration', 0):.1f}s")
except Exception as e:
    print(f"❌ Erreur OSRM: {e}")

