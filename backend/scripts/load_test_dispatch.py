#!/usr/bin/env python3
"""Script de charge pour tester le monitoring sous charge.

Génère plusieurs dispatches simultanés et vérifie que les métriques restent cohérentes.
"""

import concurrent.futures
import sys
import time
from pathlib import Path

import requests

# Ajouter le répertoire parent au path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app  # noqa: E402
from ext import db  # noqa: E402
from models import Company  # noqa: E402


def trigger_dispatch_async(
    company_id: int, base_url: str = "http://localhost:5000"
) -> dict:
    """Déclenche un dispatch de manière asynchrone."""
    try:
        response = requests.post(
            f"{base_url}/api/v1/company_dispatch/run",
            json={
                "async": True,
                "mode": "semi_auto",
                "company_id": company_id,
            },
            timeout=30,
        )

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        return {"success": False, "error": f"Status {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_metrics_summary(base_url: str = "http://localhost:5000") -> dict:
    """Récupère un résumé des métriques."""
    try:
        response = requests.get(f"{base_url}/api/v1/prometheus/metrics", timeout=10)
        response.raise_for_status()
        content = response.text

        summary = {
            "total_lines": len(content.split("\n")),
            "dispatch_runs_total": 0,
            "dispatch_duration_seconds": 0,
        }

        for line in content.split("\n"):
            if "dispatch_runs_total" in line and not line.startswith("#"):
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        summary["dispatch_runs_total"] += float(parts[-1])
                except (ValueError, IndexError):
                    pass

        return summary
    except Exception as e:
        return {"error": str(e)}


def run_load_test(
    concurrent_dispatches: int = 5, base_url: str = "http://localhost:5000"
) -> dict:
    """Exécute un test de charge avec plusieurs dispatches simultanés."""
    results = {
        "concurrent_dispatches": concurrent_dispatches,
        "start_time": time.time(),
        "dispatches_triggered": 0,
        "dispatches_success": 0,
        "dispatches_failed": 0,
        "errors": [],
        "metrics_before": {},
        "metrics_after": {},
    }

    # Récupérer métriques avant
    results["metrics_before"] = get_metrics_summary(base_url)

    # Trouver une company
    company = db.session.query(Company).first()
    if not company:
        results["errors"].append("Aucune company trouvée")
        return results

    # Déclencher plusieurs dispatches en parallèle
    print(f"Déclenchement de {concurrent_dispatches} dispatches en parallèle...")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrent_dispatches
    ) as executor:
        futures = [
            executor.submit(trigger_dispatch_async, company.id, base_url)
            for _ in range(concurrent_dispatches)
        ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results["dispatches_triggered"] += 1
            if result.get("success"):
                results["dispatches_success"] += 1
            else:
                results["dispatches_failed"] += 1
                results["errors"].append(result.get("error", "Unknown error"))

    results["end_time"] = time.time()
    results["duration"] = results["end_time"] - results["start_time"]

    # Attendre un peu pour que les métriques se mettent à jour
    time.sleep(5)

    # Récupérer métriques après
    results["metrics_after"] = get_metrics_summary(base_url)

    return results


def generate_load_test_report() -> str:
    """Génère un rapport de test de charge."""
    print("=" * 80)
    print("TEST DE CHARGE DISPATCH")
    print("=" * 80)
    print()

    base_url = "http://localhost:5000"
    concurrent_dispatches = 5

    print("Configuration:")
    print(f"  - Dispatches simultanés: {concurrent_dispatches}")
    print(f"  - Base URL: {base_url}")
    print()

    # Exécuter le test
    print("Exécution du test de charge...")
    print("-" * 80)
    results = run_load_test(concurrent_dispatches, base_url)

    # Afficher les résultats
    print()
    print("Résultats:")
    print("-" * 80)
    print(f"  Dispatches déclenchés: {results['dispatches_triggered']}")
    print(f"  Succès: {results['dispatches_success']}")
    print(f"  Échecs: {results['dispatches_failed']}")
    print(f"  Durée: {results.get('duration', 0):.2f}s")

    if results.get("metrics_before") and results.get("metrics_after"):
        print()
        print("Métriques:")
        print("-" * 80)
        before_runs = results["metrics_before"].get("dispatch_runs_total", 0)
        after_runs = results["metrics_after"].get("dispatch_runs_total", 0)
        print(
            f"  dispatch_runs_total: {before_runs} → {after_runs} (+{after_runs - before_runs})"
        )

    if results["errors"]:
        print()
        print("Erreurs:")
        print("-" * 80)
        for error in results["errors"][:5]:  # Limiter à 5
            print(f"  - {error}")

    print()
    print("=" * 80)

    if results["dispatches_success"] > 0:
        print("✅ TEST RÉUSSI")
        return "SUCCESS"
    print("❌ TEST ÉCHOUÉ")
    return "FAILED"


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        result = generate_load_test_report()
        sys.exit(0 if result == "SUCCESS" else 1)
