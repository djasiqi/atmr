#!/usr/bin/env python3
"""
Script pour réinitialiser manuellement le circuit breaker OSRM
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from services.osrm_client import _osrm_circuit_breaker

def reset_circuit_breaker():
    """Réinitialise le circuit breaker OSRM"""
    with _osrm_circuit_breaker._lock:
        old_state = _osrm_circuit_breaker.state
        old_failures = _osrm_circuit_breaker.failure_count
        
        # Réinitialiser
        _osrm_circuit_breaker.state = "CLOSED"
        _osrm_circuit_breaker.failure_count = 0
        _osrm_circuit_breaker.last_failure_time = None
        
        print(f"✅ Circuit breaker réinitialisé:")
        print(f"   - État: {old_state} → CLOSED")
        print(f"   - Échecs: {old_failures} → 0")
        print(f"   - Last failure: Réinitialisé")

if __name__ == "__main__":
    reset_circuit_breaker()

