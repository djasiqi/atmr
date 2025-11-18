#!/usr/bin/env python3
"""Script pour tester les métriques en générant des dispatches de test.

Crée des dispatches de test et vérifie que les métriques sont correctement incrémentées.
"""

import sys
import time
from pathlib import Path

import requests

# Ajouter le répertoire parent au path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app  # noqa: E402
from ext import db  # noqa: E402
from models import Booking, Company, Driver  # noqa: E402


def get_metrics_snapshot(base_url: str = "http://localhost:5000") -> dict:
    """Récupère un snapshot des métriques actuelles."""
    try:
        response = requests.get(f"{base_url}/api/v1/prometheus/metrics", timeout=10)
        response.raise_for_status()
        content = response.text

        # Extraire les valeurs des métriques
        metrics = {}
        for line in content.split("\n"):
            if line.strip() and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    metric_name = parts[0].split("{")[0]
                    try:
                        value = float(parts[-1])
                        if metric_name not in metrics:
                            metrics[metric_name] = []
                        metrics[metric_name].append(value)
                    except ValueError:
                        pass

        return metrics
    except Exception as e:
        print(f"Erreur lors de la récupération des métriques: {e}")
        return {}


def trigger_dispatch(company_id: int, base_url: str = "http://localhost:5000", token: str | None = None) -> dict:
    """Déclenche un dispatch via l'API."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.post(
            f"{base_url}/api/v1/company_dispatch/run",
            json={
                "async": True,
                "mode": "semi_auto",
                "company_id": company_id,
            },
            headers=headers,
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()
        return {"error": f"Status {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def validate_metrics_increment(metrics_before: dict, metrics_after: dict, expected_metrics: list[str]) -> dict:
    """Valide que les métriques attendues ont été incrémentées."""
    results = {
        "validated": [],
        "missing": [],
        "errors": [],
    }

    for metric_name in expected_metrics:
        before_values = metrics_before.get(metric_name, [])
        after_values = metrics_after.get(metric_name, [])

        before_sum = sum(before_values) if before_values else 0
        after_sum = sum(after_values) if after_values else 0

        if after_sum > before_sum:
            results["validated"].append(
                {
                    "metric": metric_name,
                    "before": before_sum,
                    "after": after_sum,
                    "increment": after_sum - before_sum,
                }
            )
        elif metric_name not in metrics_after:
            results["missing"].append(metric_name)
        else:
            results["errors"].append(
                {
                    "metric": metric_name,
                    "before": before_sum,
                    "after": after_sum,
                    "message": "Métrique non incrémentée",
                }
            )

    return results


def test_dispatch_metrics() -> str:
    """Teste les métriques en générant des dispatches."""
    print("=" * 80)
    print("TEST DES MÉTRIQUES DISPATCH")
    print("=" * 80)
    print()

    base_url = "http://localhost:5000"

    # Récupérer un snapshot initial
    print("1. Récupération snapshot initial des métriques")
    print("-" * 80)
    metrics_before = get_metrics_snapshot(base_url)
    print(f"   Métriques trouvées: {len(metrics_before)}")

    # Trouver une company de test
    print()
    print("2. Recherche d'une company de test")
    print("-" * 80)
    company = Company.query.first()
    if not company:
        print("❌ Aucune company trouvée")
        return "FAILED"

    print(f"   Company trouvée: {company.id} - {company.name}")

    # Vérifier qu'il y a des bookings et drivers
    bookings_count = Booking.query.filter_by(company_id=company.id).count()
    drivers_count = Driver.query.filter_by(company_id=company.id).count()

    print(f"   Bookings: {bookings_count}")
    print(f"   Drivers: {drivers_count}")

    if bookings_count == 0 or drivers_count == 0:
        print("⚠️ Pas assez de données pour tester")
        return "SKIPPED"

    # Déclencher un dispatch
    print()
    print("3. Déclenchement d'un dispatch de test")
    print("-" * 80)
    dispatch_result = trigger_dispatch(company.id, base_url)

    if "error" in dispatch_result:
        print(f"❌ Erreur lors du dispatch: {dispatch_result['error']}")
        return "FAILED"

    print(f"   Dispatch déclenché: {dispatch_result.get('dispatch_run_id', 'unknown')}")

    # Attendre quelques secondes pour que le dispatch se termine
    print()
    print("4. Attente de la fin du dispatch (10s)...")
    time.sleep(10)

    # Récupérer un snapshot après
    print()
    print("5. Récupération snapshot final des métriques")
    print("-" * 80)
    metrics_after = get_metrics_snapshot(base_url)
    print(f"   Métriques trouvées: {len(metrics_after)}")

    # Valider les incréments
    print()
    print("6. Validation des incréments de métriques")
    print("-" * 80)
    expected_metrics = [
        "dispatch_runs_total",
        "dispatch_duration_seconds",
    ]

    validation = validate_metrics_increment(metrics_before, metrics_after, expected_metrics)

    if validation["validated"]:
        print("✅ Métriques validées:")
        for item in validation["validated"]:
            print(f"   - {item['metric']}: {item['before']} → {item['after']} (+{item['increment']})")

    if validation["missing"]:
        print("⚠️ Métriques manquantes:")
        for metric in validation["missing"]:
            print(f"   - {metric}")

    if validation["errors"]:
        print("❌ Erreurs:")
        for error in validation["errors"]:
            print(f"   - {error['metric']}: {error['message']}")

    print()
    print("=" * 80)

    if len(validation["validated"]) > 0:
        print("✅ TEST RÉUSSI")
        return "SUCCESS"
    print("⚠️ TEST PARTIEL")
    return "PARTIAL"


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        result = test_dispatch_metrics()
        sys.exit(0 if result == "SUCCESS" else 1)
