#!/usr/bin/env python3
"""Script de test pour vérifier le middleware Prometheus.

Usage:
    # Option 1: Dans Docker
    docker-compose exec api python scripts/test_prometheus_metrics.py
    
    # Option 2: Local (si Python + prometheus-client installés)
    python scripts/test_prometheus_metrics.py
"""

import sys
import time
from pathlib import Path

# Ajouter le backend au path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

import requests # noqa: E402


def test_metrics_endpoint(base_url: str = "http://localhost:5000"):
    """Test l'endpoint de métriques Prometheus."""
    print("🧪 Test middleware Prometheus...")
    print(f"   Base URL: {base_url}")
    
    # 1. Vérifier que l'endpoint existe
    metrics_url = f"{base_url}/prometheus/metrics-http"
    print(f"\n1️⃣ Test endpoint: {metrics_url}")
    
    try:
        resp = requests.get(metrics_url, timeout=5)
        if resp.status_code == 503:
            print("   ❌ ÉCHEC: prometheus_client non installé")
            print("   Solution: pip install prometheus-client")
            return False
        elif resp.status_code != 200:
            print(f"   ❌ ÉCHEC: Status code {resp.status_code}")
            return False
        
        print(f"   ✅ Endpoint accessible (status {resp.status_code})")
        
        # 2. Faire quelques requêtes pour générer des métriques
        print("\n2️⃣ Génération de métriques...")
        for i in range(5):
            try:
                health_resp = requests.get(f"{base_url}/health", timeout=2)
                print(f"   Requête {i+1}/5: /health → {health_resp.status_code}")
                time.sleep(0.1)
            except Exception as e:
                print(f"   ⚠️  Erreur requête {i+1}: {e}")
        
        # 3. Vérifier que les métriques sont présentes
        print("\n3️⃣ Vérification métriques...")
        resp = requests.get(metrics_url, timeout=5)
        content = resp.text
        
        checks = {
            "http_request_duration_seconds": "http_request_duration_seconds" in content,
            "http_requests_total": "http_requests_total" in content,
            "histogram buckets": "le=" in content or "bucket" in content.lower(),
            "labels method/endpoint": 'method="GET"' in content or 'endpoint=' in content,
        }
        
        all_ok = True
        for check_name, check_result in checks.items():
            status = "✅" if check_result else "❌"
            print(f"   {status} {check_name}")
            if not check_result:
                all_ok = False
        
        # 4. Afficher un extrait des métriques
        print("\n4️⃣ Extrait des métriques:")
        lines = content.split("\n")
        relevant_lines = [
            line for line in lines
            if "http_request" in line.lower() or line.startswith("# HELP") or line.startswith("# TYPE")
        ][:10]
        for line in relevant_lines:
            print(f"   {line}")
        
        if all_ok:
            print("\n✅ Tous les tests passent!")
            return True
        else:
            print("\n❌ Certains tests ont échoué")
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ❌ ÉCHEC: Impossible de se connecter à l'API")
        print(f"   Vérifiez que l'API est démarrée sur {base_url}")
        return False
    except Exception as e:
        print(f"   ❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    success = test_metrics_endpoint(base_url)
    sys.exit(0 if success else 1)

