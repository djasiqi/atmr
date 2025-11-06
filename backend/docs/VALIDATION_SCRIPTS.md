# Documentation des Scripts de Validation

Guide d'utilisation des scripts de validation pour le monitoring et les métriques.

## Scripts disponibles

1. `validate_metrics.py`: Valide que les métriques sont exposées
2. `validate_logs_correlation.py`: Valide la corrélation des logs
3. `test_dispatch_metrics.py`: Teste les métriques en générant des dispatches
4. `load_test_dispatch.py`: Test de charge pour valider le monitoring sous charge

## validate_metrics.py

### Usage

```bash
cd backend
python scripts/validate_metrics.py
```

### Fonctionnalités

- Vérifie que l'endpoint `/api/v1/prometheus/metrics` est accessible
- Valide le format Prometheus
- Vérifie que toutes les métriques attendues sont présentes
- Teste l'endpoint `/api/v1/osrm/health`

### Résultats

Le script génère un rapport avec:

- ✅ Métriques validées
- ❌ Métriques manquantes
- ⚠️ Erreurs de format

### Exemple de sortie

```
VALIDATION DES MÉTRIQUES PROMETHEUS
================================================================================

1. Validation endpoint /api/v1/prometheus/metrics
--------------------------------------------------------------------------------
✅ Endpoint accessible
   Taille réponse: 15234 bytes
✅ Format Prometheus valide

Métriques trouvées: 17/17
  ✅ dispatch_runs_total: {'type': 'counter', 'samples_count': 5}
  ✅ dispatch_duration_seconds: {'type': 'histogram', 'samples_count': 3}
  ...

✅ VALIDATION RÉUSSIE
```

## validate_logs_correlation.py

### Usage

```bash
cd backend
python scripts/validate_logs_correlation.py
```

### Fonctionnalités

- Valide que les logs contiennent `dispatch_run_id`
- Vérifie la présence de `trace_id` et `span_id`
- Teste la corrélation entre logs et métriques
- Analyse les dispatches récents (24h)

### Résultats

Le script génère un rapport avec:

- Nombre de dispatches analysés
- Taux de corrélation
- Logs manquants

### Exemple de sortie

```
VALIDATION CORRÉLATION LOGS
================================================================================

Validation des dispatches récents (24h)
--------------------------------------------------------------------------------
Dispatches trouvés: 10
Dispatches validés: 10

Détails par dispatch:
  DispatchRun 123:
    - Logs analysés: 15
    - Avec dispatch_run_id: 15
    - Avec trace_id: 15
    - Taux corrélation: 100.0%

✅ VALIDATION RÉUSSIE
```

## test_dispatch_metrics.py

### Usage

```bash
cd backend
python scripts/test_dispatch_metrics.py
```

### Fonctionnalités

- Génère un dispatch de test
- Vérifie que les métriques sont incrémentées
- Compare les métriques avant/après
- Teste les différents modes (auto, semi_auto, manual)

### Prérequis

- Une company avec des bookings et drivers
- API backend accessible

### Résultats

Le script génère un rapport avec:

- Métriques avant/après
- Incréments validés
- Erreurs éventuelles

### Exemple de sortie

```
TEST DES MÉTRIQUES DISPATCH
================================================================================

1. Récupération snapshot initial des métriques
--------------------------------------------------------------------------------
   Métriques trouvées: 15

2. Recherche d'une company de test
--------------------------------------------------------------------------------
   Company trouvée: 1 - Test Company
   Bookings: 10
   Drivers: 5

3. Déclenchement d'un dispatch de test
--------------------------------------------------------------------------------
   Dispatch déclenché: 123

6. Validation des incréments de métriques
--------------------------------------------------------------------------------
✅ Métriques validées:
   - dispatch_runs_total: 10 → 11 (+1)
   - dispatch_duration_seconds: 5.2 → 6.1 (+0.9)

✅ TEST RÉUSSI
```

## load_test_dispatch.py

### Usage

```bash
cd backend
python scripts/load_test_dispatch.py
```

### Fonctionnalités

- Génère plusieurs dispatches simultanés (par défaut: 5)
- Vérifie que les métriques restent cohérentes
- Teste les limites du système
- Mesure la performance sous charge

### Configuration

Modifier le nombre de dispatches simultanés dans le script:

```python
concurrent_dispatches = 5  # Changer selon besoins
```

### Résultats

Le script génère un rapport avec:

- Nombre de dispatches réussis/échoués
- Durée totale
- Métriques avant/après
- Erreurs éventuelles

### Exemple de sortie

```
TEST DE CHARGE DISPATCH
================================================================================

Configuration:
  - Dispatches simultanés: 5
  - Base URL: http://localhost:5000

Exécution du test de charge...
--------------------------------------------------------------------------------

Résultats:
--------------------------------------------------------------------------------
  Dispatches déclenchés: 5
  Succès: 5
  Échecs: 0
  Durée: 12.34s

Métriques:
--------------------------------------------------------------------------------
  dispatch_runs_total: 50 → 55 (+5)

✅ TEST RÉUSSI
```

## Interprétation des résultats

### SUCCESS

Tous les tests sont passés. Le système est opérationnel.

### PARTIAL

Certains tests sont passés mais pas tous. Vérifier les warnings.

### FAILED

Les tests ont échoué. Vérifier les erreurs et corriger.

### SKIPPED

Les tests ont été ignorés (pas assez de données, etc.).

## Procédures de validation pré-déploiement

### 1. Validation complète

```bash
# Valider les métriques
python scripts/validate_metrics.py

# Valider la corrélation
python scripts/validate_logs_correlation.py

# Tester les métriques
python scripts/test_dispatch_metrics.py
```

### 2. Test de charge (optionnel)

```bash
python scripts/load_test_dispatch.py
```

### 3. Vérifier les résultats

- Tous les scripts doivent retourner `SUCCESS`
- Vérifier qu'aucune métrique n'est manquante
- Vérifier que la corrélation est > 80%

## Intégration CI/CD

Les scripts peuvent être intégrés dans le pipeline CI/CD:

```yaml
# Exemple GitHub Actions
- name: Validate metrics
  run: |
    cd backend
    python scripts/validate_metrics.py
    python scripts/validate_logs_correlation.py
```

## Troubleshooting

### Scripts ne trouvent pas les métriques

1. Vérifier que l'API backend est démarrée
2. Vérifier que l'endpoint `/api/v1/prometheus/metrics` est accessible
3. Vérifier que `prometheus_client` est installé

### Corrélation échoue

1. Vérifier que les logs contiennent `dispatch_run_id`
2. Vérifier que OpenTelemetry est configuré
3. Vérifier que le contexte de logging est correct

### Tests de métriques échouent

1. Vérifier qu'il y a des bookings et drivers
2. Vérifier que les dispatches peuvent être déclenchés
3. Vérifier que les métriques sont bien enregistrées
