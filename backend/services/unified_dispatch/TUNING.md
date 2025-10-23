# Guide de Calibrage des Paramètres

## Vue d'ensemble

Ce document fournit des recommandations pour calibrer les paramètres du système de dispatch en fonction des besoins spécifiques de chaque entreprise.

## Paramètres de base

### 1. Temps de service (`ServiceTimesSettings`)

```python
@dataclass
class ServiceTimesSettings:
    pickup_service_min: int = 5      # Temps de pickup (minutes)
    dropoff_service_min: int = 10    # Temps de dropoff (minutes)
    min_transition_margin_min: int = 15  # Marge entre courses (minutes)
```

**Recommandations par type d'entreprise**:

#### Transport médical

```python
ServiceTimesSettings(
    pickup_service_min=10,    # Installation patient
    dropoff_service_min=15,   # Désinstallation patient
    min_transition_margin_min=20  # Marge de sécurité
)
```

#### Transport scolaire

```python
ServiceTimesSettings(
    pickup_service_min=3,     # Montée rapide
    dropoff_service_min=3,    # Descente rapide
    min_transition_margin_min=10  # Marge réduite
)
```

#### Transport de personnes âgées

```python
ServiceTimesSettings(
    pickup_service_min=8,     # Temps d'aide
    dropoff_service_min=8,    # Temps d'aide
    min_transition_margin_min=15  # Marge standard
)
```

### 2. Regroupement de courses (`PoolingSettings`)

```python
@dataclass
class PoolingSettings:
    enabled: bool = True                    # Activer le regroupement
    time_tolerance_min: int = 10            # Tolérance temporelle (±min)
    pickup_distance_m: int = 500            # Distance max entre pickups (m)
    max_detour_min: int = 15                # Détour max acceptable (min)
```

**Recommandations par contexte**:

#### Zone urbaine dense

```python
PoolingSettings(
    enabled=True,
    time_tolerance_min=5,      # Tolérance réduite
    pickup_distance_m=300,     # Distance réduite
    max_detour_min=10          # Détour réduit
)
```

#### Zone rurale

```python
PoolingSettings(
    enabled=True,
    time_tolerance_min=15,     # Tolérance étendue
    pickup_distance_m=1000,    # Distance étendue
    max_detour_min=25          # Détour étendu
)
```

#### Transport médical (regroupement limité)

```python
PoolingSettings(
    enabled=False,             # Désactiver le regroupement
    time_tolerance_min=5,
    pickup_distance_m=200,
    max_detour_min=5
)
```

### 3. Poids heuristiques (`HeuristicWeights`)

```python
@dataclass
class HeuristicWeights:
    proximity: float = 0.4      # Proximité géographique
    fairness: float = 0.3       # Équité entre chauffeurs
    capacity: float = 0.2       # Utilisation de la capacité
    urgency: float = 0.1        # Priorité aux urgences
```

**Profils d'optimisation**:

#### Optimisation coût (minimiser la distance)

```python
HeuristicWeights(
    proximity=0.6,    # Priorité à la proximité
    fairness=0.2,     # Équité réduite
    capacity=0.1,     # Capacité réduite
    urgency=0.1       # Urgences standard
)
```

#### Optimisation équité (répartir équitablement)

```python
HeuristicWeights(
    proximity=0.2,    # Proximité réduite
    fairness=0.5,     # Priorité à l'équité
    capacity=0.2,     # Capacité standard
    urgency=0.1       # Urgences standard
)
```

#### Optimisation urgence (priorité aux urgences)

```python
HeuristicWeights(
    proximity=0.3,    # Proximité standard
    fairness=0.2,     # Équité réduite
    capacity=0.2,     # Capacité standard
    urgency=0.3       # Priorité aux urgences
)
```

## Paramètres avancés

### 1. Configuration temporelle (`TimeSettings`)

```python
@dataclass
class TimeSettings:
    pickup_buffer_min: int = 5      # Marge avant pickup
    dropoff_buffer_min: int = 5     # Marge avant dropoff
    pickup_window_min: int = 10     # Fenêtre de pickup
    dropoff_window_min: int = 10    # Fenêtre de dropoff
    horizon_min: int = 240          # Horizon de planification (4h)
    horizon_max: int = 1440         # Horizon max (24h)
```

**Ajustements recommandés**:

#### Service ponctuel

```python
TimeSettings(
    pickup_buffer_min=2,     # Marge réduite
    dropoff_buffer_min=2,    # Marge réduite
    pickup_window_min=5,     # Fenêtre réduite
    dropoff_window_min=5,    # Fenêtre réduite
    horizon_min=120,         # Horizon 2h
    horizon_max=480          # Horizon max 8h
)
```

#### Service continu

```python
TimeSettings(
    pickup_buffer_min=10,    # Marge étendue
    dropoff_buffer_min=10,   # Marge étendue
    pickup_window_min=20,    # Fenêtre étendue
    dropoff_window_min=20,   # Fenêtre étendue
    horizon_min=480,         # Horizon 8h
    horizon_max=2880         # Horizon max 48h
)
```

### 2. Configuration d'urgence (`EmergencyPolicy`)

```python
@dataclass
class EmergencyPolicy:
    enabled: bool = True                    # Activer les urgences
    max_emergency_drivers: int = 3          # Max chauffeurs d'urgence
    emergency_penalty_multiplier: float = 2.0  # Pénalité pour non-urgence
    emergency_timeout_min: int = 30         # Timeout urgence (min)
```

**Profils d'urgence**:

#### Transport médical (urgences fréquentes)

```python
EmergencyPolicy(
    enabled=True,
    max_emergency_drivers=5,        # Plus de chauffeurs d'urgence
    emergency_penalty_multiplier=3.0,  # Pénalité élevée
    emergency_timeout_min=15        # Timeout réduit
)
```

#### Transport scolaire (urgences rares)

```python
EmergencyPolicy(
    enabled=True,
    max_emergency_drivers=1,        # Moins de chauffeurs d'urgence
    emergency_penalty_multiplier=1.5,  # Pénalité modérée
    emergency_timeout_min=60        # Timeout étendu
)
```

## Processus de calibrage

### 1. Analyse des données historiques

```python
# Analyser les performances passées
def analyze_historical_performance(company_id, days=30):
    # Récupérer les métriques des 30 derniers jours
    metrics = get_dispatch_metrics(company_id, days)

    # Analyser les tendances
    assignment_rate = metrics['avg_assignment_rate']
    avg_run_time = metrics['avg_run_time_sec']
    unassigned_reasons = metrics['unassigned_reasons']

    return {
        'assignment_rate': assignment_rate,
        'run_time': avg_run_time,
        'bottlenecks': unassigned_reasons
    }
```

### 2. Tests A/B

```python
# Configuration de test A
settings_a = Settings(
    service_times=ServiceTimesSettings(
        pickup_service_min=5,
        dropoff_service_min=10,
        min_transition_margin_min=15
    ),
    pooling=PoolingSettings(
        time_tolerance_min=10,
        pickup_distance_m=500
    )
)

# Configuration de test B
settings_b = Settings(
    service_times=ServiceTimesSettings(
        pickup_service_min=8,
        dropoff_service_min=12,
        min_transition_margin_min=20
    ),
    pooling=PoolingSettings(
        time_tolerance_min=15,
        pickup_distance_m=750
    )
)

# Comparer les performances
def compare_configurations(settings_a, settings_b, test_duration_days=7):
    # Exécuter les deux configurations en parallèle
    # Comparer les métriques de performance
    pass
```

### 3. Optimisation automatique

```python
# Algorithme d'optimisation des paramètres
def optimize_parameters(company_id, target_metrics):
    """
    Optimise automatiquement les paramètres pour atteindre les métriques cibles.
    """
    current_settings = get_company_settings(company_id)

    # Définir les plages de recherche
    parameter_ranges = {
        'pickup_service_min': (3, 15),
        'dropoff_service_min': (5, 20),
        'time_tolerance_min': (5, 30),
        'pickup_distance_m': (200, 1000)
    }

    # Algorithme de recherche (ex: grid search, random search)
    best_settings = None
    best_score = 0

    for settings in generate_settings_combinations(parameter_ranges):
        score = evaluate_settings(settings, target_metrics)
        if score > best_score:
            best_score = score
            best_settings = settings

    return best_settings
```

## Recommandations par secteur

### Transport médical

**Priorités**: Sécurité, ponctualité, confort patient

```python
Settings(
    service_times=ServiceTimesSettings(
        pickup_service_min=10,
        dropoff_service_min=15,
        min_transition_margin_min=20
    ),
    pooling=PoolingSettings(
        enabled=False,  # Pas de regroupement
        time_tolerance_min=5,
        pickup_distance_m=200,
        max_detour_min=5
    ),
    emergency=EmergencyPolicy(
        enabled=True,
        max_emergency_drivers=5,
        emergency_penalty_multiplier=3.0,
        emergency_timeout_min=15
    ),
    heuristic=HeuristicWeights(
        proximity=0.3,
        fairness=0.2,
        capacity=0.2,
        urgency=0.3  # Priorité aux urgences
    )
)
```

### Transport scolaire

**Priorités**: Ponctualité, sécurité, efficacité

```python
Settings(
    service_times=ServiceTimesSettings(
        pickup_service_min=3,
        dropoff_service_min=3,
        min_transition_margin_min=10
    ),
    pooling=PoolingSettings(
        enabled=True,
        time_tolerance_min=5,
        pickup_distance_m=300,
        max_detour_min=10
    ),
    emergency=EmergencyPolicy(
        enabled=True,
        max_emergency_drivers=2,
        emergency_penalty_multiplier=2.0,
        emergency_timeout_min=30
    ),
    heuristic=HeuristicWeights(
        proximity=0.5,  # Priorité à la proximité
        fairness=0.3,
        capacity=0.1,
        urgency=0.1
    )
)
```

### Transport de personnes âgées

**Priorités**: Confort, sécurité, flexibilité

```python
Settings(
    service_times=ServiceTimesSettings(
        pickup_service_min=8,
        dropoff_service_min=8,
        min_transition_margin_min=15
    ),
    pooling=PoolingSettings(
        enabled=True,
        time_tolerance_min=15,
        pickup_distance_m=500,
        max_detour_min=20
    ),
    emergency=EmergencyPolicy(
        enabled=True,
        max_emergency_drivers=3,
        emergency_penalty_multiplier=2.5,
        emergency_timeout_min=45
    ),
    heuristic=HeuristicWeights(
        proximity=0.4,
        fairness=0.4,  # Priorité à l'équité
        capacity=0.1,
        urgency=0.1
    )
)
```

## Monitoring et ajustement

### Métriques à surveiller

1. **Taux d'assignation**: > 85%
2. **Temps d'exécution**: < 5 secondes
3. **Satisfaction client**: > 90%
4. **Utilisation des chauffeurs**: 70-90%

### Signaux d'alerte

- Taux d'assignation < 70%
- Temps d'exécution > 10 secondes
- Nombreuses courses non assignées
- Surcharge des chauffeurs

### Actions correctives

1. **Taux d'assignation faible**:

   - Augmenter `time_tolerance_min`
   - Augmenter `pickup_distance_m`
   - Réduire `min_transition_margin_min`

2. **Temps d'exécution élevé**:

   - Réduire `horizon_max`
   - Optimiser les poids heuristiques
   - Améliorer le cache OSRM

3. **Surcharge des chauffeurs**:
   - Augmenter `min_transition_margin_min`
   - Réduire `pickup_service_min`
   - Ajuster les fenêtres de travail

## Outils de diagnostic

### Script d'analyse des performances

```python
#!/usr/bin/env python3
"""
Script d'analyse des performances du dispatch
"""

import requests
import json
from datetime import datetime, timedelta

def analyze_dispatch_performance(company_id, days=7):
    """Analyse les performances du dispatch sur une période donnée"""

    # Récupérer les métriques de santé
    response = requests.get(
        f"http://localhost:5000/api/company_dispatch_health/health/trends?days={days}",
        headers={"Authorization": f"Bearer {get_token()}"}
    )

    if response.status_code == 200:
        data = response.json()

        # Analyser les tendances
        assignment_rates = data['assignment_rates']
        run_times = data['run_times']

        avg_assignment_rate = sum(assignment_rates) / len(assignment_rates)
        avg_run_time = sum(run_times) / len(run_times)

        print(f"Taux d'assignation moyen: {avg_assignment_rate:.2%}")
        print(f"Temps d'exécution moyen: {avg_run_time:.2f}s")

        # Recommandations
        if avg_assignment_rate < 0.85:
            print("⚠️  Taux d'assignation faible - Considérer l'ajustement des paramètres")

        if avg_run_time > 5.0:
            print("⚠️  Temps d'exécution élevé - Considérer l'optimisation")

        return {
            'assignment_rate': avg_assignment_rate,
            'run_time': avg_run_time,
            'recommendations': generate_recommendations(avg_assignment_rate, avg_run_time)
        }

    return None

def generate_recommendations(assignment_rate, run_time):
    """Génère des recommandations basées sur les métriques"""
    recommendations = []

    if assignment_rate < 0.85:
        recommendations.append("Augmenter time_tolerance_min")
        recommendations.append("Augmenter pickup_distance_m")
        recommendations.append("Réduire min_transition_margin_min")

    if run_time > 5.0:
        recommendations.append("Réduire horizon_max")
        recommendations.append("Optimiser les poids heuristiques")
        recommendations.append("Améliorer le cache OSRM")

    return recommendations

if __name__ == "__main__":
    company_id = 1
    analysis = analyze_dispatch_performance(company_id)

    if analysis:
        print("\nRecommandations:")
        for rec in analysis['recommendations']:
            print(f"- {rec}")
```
