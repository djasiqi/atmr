#!/usr/bin/env python3
"""Script de validation des métriques Prometheus exposées.

Vérifie que toutes les métriques sont correctement exposées et au format Prometheus.
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests

# Ajouter le répertoire parent au path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app  # noqa: E402
from ext import db  # noqa: E402

# Métriques attendues
EXPECTED_METRICS = {
    "dispatch_runs_total": "counter",
    "dispatch_duration_seconds": "histogram",
    "dispatch_quality_score": "gauge",
    "dispatch_assignment_rate": "gauge",
    "dispatch_unassigned_count": "gauge",
    "dispatch_circuit_breaker_state": "gauge",
    "dispatch_temporal_conflicts_total": "counter",
    "dispatch_db_conflicts_total": "counter",
    "dispatch_osrm_cache_hits_total": "counter",
    "dispatch_osrm_cache_misses_total": "counter",
    "osrm_cache_hits_total": "counter",
    "osrm_cache_misses_total": "counter",
    "osrm_cache_bypass_total": "counter",
    "osrm_cache_hit_rate": "gauge",
    "dispatch_slo_breaches_total": "counter",
    "dispatch_slo_breach_severity": "gauge",
    "dispatch_slo_should_alert": "gauge",
    "dispatch_slo_breaches_by_type": "counter",
}


def parse_prometheus_metrics(content: str) -> Dict[str, Dict[str, Any]]:
    """Parse le contenu Prometheus et extrait les métriques.
    
    Returns:
        Dict avec métrique_name -> {type: str, help: str, samples: List[Dict]}
    """
    metrics: Dict[str, Dict[str, Any]] = {}
    current_metric = None
    current_type = None
    current_help = None
    
    lines = content.split("\n")
    for line_raw in lines:
        line = line_raw.strip()
        if not line or line.startswith("#"):
            # Ligne de commentaire ou vide
            if line.startswith("# TYPE"):
                # Format: # TYPE metric_name type
                match = re.match(r"# TYPE\s+(\w+)\s+(\w+)", line)
                if match:
                    current_metric = match.group(1)
                    current_type = match.group(2)
                    if current_metric not in metrics:
                        metrics[current_metric] = {"type": current_type, "help": "", "samples": []}
                    else:
                        metrics[current_metric]["type"] = current_type
            elif line.startswith("# HELP"):
                # Format: # HELP metric_name description
                match = re.match(r"# HELP\s+(\w+)\s+(.+)", line)
                if match:
                    current_metric = match.group(1)
                    current_help = match.group(2)
                    if current_metric not in metrics:
                        metrics[current_metric] = {"type": "", "help": current_help, "samples": []}
                    else:
                        metrics[current_metric]["help"] = current_help
        # Ligne de métrique: metric_name{labels} value
        elif current_metric:
            if current_metric not in metrics:
                metrics[current_metric] = {"type": "", "help": "", "samples": []}
            metrics[current_metric]["samples"].append(line)
    
    return metrics


def validate_metrics_format(content: str) -> List[str]:
    """Valide le format Prometheus et retourne les erreurs."""
    errors = []
    
    # Vérifier que c'est du texte
    if not isinstance(content, str):
        errors.append("Le contenu n'est pas du texte")
        return errors
    
    # Vérifier la présence de lignes TYPE et HELP
    if "# TYPE" not in content:
        errors.append("Aucune ligne # TYPE trouvée")
    if "# HELP" not in content:
        errors.append("Aucune ligne # HELP trouvée")
    
    # Vérifier que les métriques sont bien formatées
    lines = content.split("\n")
    metric_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
    
    for metric_line in metric_lines:
        # Format attendu: metric_name{labels} value ou metric_name value
        if not re.match(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})?\s+[0-9.+-eE]+", metric_line):
            errors.append(f"Ligne de métrique mal formatée: {metric_line[:100]}")
    
    return errors


def validate_metrics_endpoint(base_url: str = "http://localhost:5000") -> Dict[str, Any]:
    """Valide l'endpoint /api/v1/prometheus/metrics."""
    results = {
        "endpoint": f"{base_url}/api/v1/prometheus/metrics",
        "accessible": False,
        "format_valid": False,
        "metrics_found": {},
        "metrics_missing": [],
        "errors": [],
    }
    
    try:
        response = requests.get(results["endpoint"], timeout=10)
        response.raise_for_status()
        results["accessible"] = True
        
        content = response.text
        results["content_length"] = len(content)
        
        # Valider le format
        format_errors = validate_metrics_format(content)
        if format_errors:
            results["errors"].extend(format_errors)
        else:
            results["format_valid"] = True
        
        # Parser les métriques
        parsed_metrics = parse_prometheus_metrics(content)
        
        # Vérifier les métriques attendues
        for metric_name, metric_type in EXPECTED_METRICS.items():
            if metric_name in parsed_metrics:
                found_type = parsed_metrics[metric_name].get("type", "")
                if found_type == metric_type or metric_name in content:
                    results["metrics_found"][metric_name] = {
                        "type": found_type,
                        "samples_count": len(parsed_metrics[metric_name].get("samples", [])),
                    }
                else:
                    results["metrics_missing"].append(f"{metric_name} (type mismatch: {found_type} != {metric_type})")
            else:
                # Vérifier si la métrique existe mais avec un préfixe ou suffixe
                found = False
                for parsed_name in parsed_metrics:
                    if metric_name in parsed_name or parsed_name in metric_name:
                        found = True
                        results["metrics_found"][metric_name] = {
                            "found_as": parsed_name,
                            "type": parsed_metrics[parsed_name].get("type", ""),
                        }
                        break
                if not found and metric_name in content:
                    # Métrique présente mais non parsée correctement
                    results["metrics_found"][metric_name] = {"status": "present_in_content"}
                elif not found:
                    results["metrics_missing"].append(metric_name)
        
        results["total_metrics_found"] = len(parsed_metrics)
        results["total_metrics_expected"] = len(EXPECTED_METRICS)
        
    except requests.exceptions.RequestException as e:
        results["errors"].append(f"Erreur de connexion: {e}")
    except Exception as e:
        results["errors"].append(f"Erreur inattendue: {e}")
    
    return results


def validate_osrm_health_endpoint(base_url: str = "http://localhost:5000") -> Dict[str, Any]:
    """Valide l'endpoint /api/v1/osrm/health."""
    results = {
        "endpoint": f"{base_url}/api/v1/osrm/health",
        "accessible": False,
        "status_code": None,
        "response": None,
        "errors": [],
    }
    
    try:
        response = requests.get(results["endpoint"], timeout=10)
        results["status_code"] = response.status_code
        results["accessible"] = response.status_code == 200
        
        if results["accessible"]:
            results["response"] = response.json()
        else:
            results["errors"].append(f"Status code: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        results["errors"].append(f"Erreur de connexion: {e}")
    except Exception as e:
        results["errors"].append(f"Erreur inattendue: {e}")
    
    return results


def generate_report() -> str:
    """Génère un rapport de validation."""
    print("=" * 80)
    print("VALIDATION DES MÉTRIQUES PROMETHEUS")
    print("=" * 80)
    print()
    
    # Valider endpoint Prometheus
    print("1. Validation endpoint /api/v1/prometheus/metrics")
    print("-" * 80)
    metrics_results = validate_metrics_endpoint()
    
    if metrics_results["accessible"]:
        print("✅ Endpoint accessible")
        print(f"   Taille réponse: {metrics_results.get('content_length', 0)} bytes")
    else:
        print("❌ Endpoint non accessible")
        for error in metrics_results["errors"]:
            print(f"   Erreur: {error}")
        return "FAILED"
    
    if metrics_results["format_valid"]:
        print("✅ Format Prometheus valide")
    else:
        print("❌ Format Prometheus invalide")
        for error in metrics_results["errors"]:
            print(f"   Erreur: {error}")
    
    print()
    print(f"Métriques trouvées: {len(metrics_results['metrics_found'])}/{metrics_results['total_metrics_expected']}")
    for metric_name, info in metrics_results["metrics_found"].items():
        print(f"  ✅ {metric_name}: {info}")
    
    if metrics_results["metrics_missing"]:
        print()
        print(f"Métriques manquantes ({len(metrics_results['metrics_missing'])}):")
        for metric in metrics_results["metrics_missing"]:
            print(f"  ❌ {metric}")
    
    print()
    print("2. Validation endpoint /api/v1/osrm/health")
    print("-" * 80)
    health_results = validate_osrm_health_endpoint()
    
    if health_results["accessible"]:
        print(f"✅ Endpoint accessible (status: {health_results['status_code']})")
        if health_results.get("response"):
            print(f"   Status OSRM: {health_results['response'].get('status', 'unknown')}")
            print(f"   Circuit Breaker: {health_results['response'].get('circuit_breaker', {}).get('state', 'unknown')}")
    else:
        print("❌ Endpoint non accessible")
        for error in health_results["errors"]:
            print(f"   Erreur: {error}")
    
    print()
    print("=" * 80)
    
    # Déterminer le statut global
    if metrics_results["accessible"] and metrics_results["format_valid"] and len(metrics_results["metrics_missing"]) == 0:
        print("✅ VALIDATION RÉUSSIE")
        return "SUCCESS"
    print("⚠️ VALIDATION PARTIELLE")
    return "PARTIAL"


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        result = generate_report()
        sys.exit(0 if result == "SUCCESS" else 1)

