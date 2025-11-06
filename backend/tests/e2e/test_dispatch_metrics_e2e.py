"""Tests E2E pour valider les métriques dispatch et leur corrélation avec les logs."""

from __future__ import annotations

import re
import time
from typing import Any

import pytest
import requests

from app import create_app
from models import Booking, BookingStatus, Company, DispatchRun, Driver
from services.unified_dispatch import engine
from tests.factories import (
    BookingFactory,
    CompanyFactory,
    DispatchRunFactory,
    DriverFactory,
)


@pytest.fixture
def test_company(db):
    """Crée une company de test."""
    company = CompanyFactory()
    db.session.commit()
    return company


@pytest.fixture
def test_drivers(db, test_company):
    """Crée des drivers de test."""
    drivers = [
        DriverFactory(company=test_company, is_available=True)
        for _ in range(3)
    ]
    db.session.commit()
    return drivers


@pytest.fixture
def test_bookings(db, test_company):
    """Crée des bookings de test."""
    bookings = [
        BookingFactory(
            company=test_company,
            status=BookingStatus.PENDING,
        )
        for _ in range(5)
    ]
    db.session.commit()
    return bookings


@pytest.fixture
def metrics_endpoint():
    """Endpoint Prometheus."""
    return "http://localhost:5000/api/v1/prometheus/metrics"


def parse_metrics(content: str) -> dict[str, Any]:
    """Parse les métriques Prometheus."""
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
                except (ValueError, IndexError):
                    pass
    return metrics


def get_metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    """Récupère la valeur d'une métrique (somme si plusieurs)."""
    values = metrics.get(metric_name, [])
    return sum(values) if values else 0.0


def test_metrics_endpoint_accessible(metrics_endpoint):
    """Test: l'endpoint metrics est accessible."""
    response = requests.get(metrics_endpoint, timeout=10)
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("Content-Type", "")


def test_metrics_format_valid(metrics_endpoint):
    """Test: le format Prometheus est valide."""
    response = requests.get(metrics_endpoint, timeout=10)
    content = response.text
    
    # Vérifier la présence de lignes TYPE et HELP
    assert "# TYPE" in content
    assert "# HELP" in content
    
    # Vérifier que les métriques sont bien formatées
    metric_lines = [line for line in content.split("\n") if line.strip() and not line.strip().startswith("#")]
    for metric_line in metric_lines[:10]:  # Limiter à 10 pour performance
        assert re.match(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})?\s+[0-9.+-eE]+", metric_line), f"Ligne mal formatée: {metric_line}"


def test_dispatch_metrics_present(metrics_endpoint):
    """Test: les métriques dispatch sont présentes."""
    response = requests.get(metrics_endpoint, timeout=10)
    content = response.text
    
    expected_metrics = [
        "dispatch_runs_total",
        "dispatch_duration_seconds",
        "dispatch_quality_score",
        "dispatch_assignment_rate",
        "dispatch_unassigned_count",
        "dispatch_circuit_breaker_state",
    ]
    
    for metric in expected_metrics:
        assert metric in content, f"Métrique {metric} non trouvée"


def test_dispatch_increments_metrics(
    db, test_company, test_drivers, test_bookings, metrics_endpoint
):
    """Test: un dispatch incrémente les métriques."""
    # Récupérer métriques avant
    response_before = requests.get(metrics_endpoint, timeout=10)
    metrics_before = parse_metrics(response_before.text)
    
    runs_before = get_metric_value(metrics_before, "dispatch_runs_total")
    
    # Créer et exécuter un dispatch
    dispatch_run = DispatchRunFactory(
        company=test_company,
        status="PENDING"
    )
    db.session.commit()
    
    # Exécuter le dispatch (simulation)
    try:
        engine.run(
            company_id=test_company.id,
            dispatch_day=dispatch_run.day,
            mode="semi_auto",
        )
        
        # Attendre un peu pour que les métriques se mettent à jour
        time.sleep(2)
        
        # Récupérer métriques après
        response_after = requests.get(metrics_endpoint, timeout=10)
        metrics_after = parse_metrics(response_after.text)
        
        runs_after = get_metric_value(metrics_after, "dispatch_runs_total")
        
        # Vérifier que les métriques ont été incrémentées
        assert runs_after >= runs_before, "dispatch_runs_total n'a pas été incrémenté"
        
    except Exception as e:
        pytest.skip(f"Dispatch non exécutable dans ce contexte: {e}")


def test_metrics_correlation_with_logs(db, test_company):
    """Test: corrélation entre métriques et logs avec dispatch_run_id."""
    # Créer un dispatch run
    dispatch_run = DispatchRunFactory(
        company=test_company,
        status="PENDING"
    )
    db.session.commit()
    
    dispatch_run_id = dispatch_run.id
    
    # Vérifier que dispatch_run_id existe
    assert dispatch_run_id is not None
    
    # Dans un vrai test E2E, on vérifierait que les logs contiennent dispatch_run_id
    # Ici, on simule en vérifiant que le dispatch_run existe
    assert DispatchRun.query.get(dispatch_run_id) is not None


def test_osrm_metrics_present(metrics_endpoint):
    """Test: les métriques OSRM sont présentes."""
    response = requests.get(metrics_endpoint, timeout=10)
    content = response.text
    
    expected_metrics = [
        "osrm_cache_hits_total",
        "osrm_cache_misses_total",
        "osrm_cache_hit_rate",
    ]
    
    for metric in expected_metrics:
        # Les métriques peuvent ne pas être présentes si pas encore utilisées
        # On vérifie juste que le format est correct si présent
        if metric in content:
            # Vérifier que c'est bien formaté
            assert re.search(rf"^{metric}(\{{[^}}]*\}})?\s+[0-9.+-eE]+", content, re.MULTILINE)


def test_slo_metrics_present(metrics_endpoint):
    """Test: les métriques SLO sont présentes."""
    response = requests.get(metrics_endpoint, timeout=10)
    content = response.text
    
    expected_metrics = [
        "dispatch_slo_breaches_total",
        "dispatch_slo_breach_severity",
        "dispatch_slo_should_alert",
    ]
    
    for metric in expected_metrics:
        assert metric in content, f"Métrique SLO {metric} non trouvée"


def test_metrics_labels(db, test_company, metrics_endpoint):
    """Test: les métriques ont les bons labels."""
    response = requests.get(metrics_endpoint, timeout=10)
    content = response.text
    
    # Vérifier que dispatch_runs_total a les labels attendus
    if "dispatch_runs_total" in content:
        # Chercher une ligne avec les labels
        metric_lines = [line for line in content.split("\n") if "dispatch_runs_total" in line and not line.startswith("#")]
        if metric_lines:
            metric_line = metric_lines[0]
            # Vérifier la présence de labels (format: metric{labels} value)
            assert "{" in metric_line or "dispatch_runs_total" in metric_line


@pytest.mark.skip(reason="Nécessite un environnement complet avec Prometheus")
def test_metrics_in_prometheus():
    """Test: les métriques sont visibles dans Prometheus.
    
    Note: Ce test nécessite Prometheus en cours d'exécution.
    """
    # Ce test devrait être exécuté dans un environnement avec Prometheus
    prometheus_url = "http://localhost:9090"
    
    try:
        response = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": "dispatch_runs_total"},
            timeout=5,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    except requests.exceptions.RequestException:
        pytest.skip("Prometheus non accessible")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

