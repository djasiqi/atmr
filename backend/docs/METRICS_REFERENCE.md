# Référence des Métriques Prometheus

Documentation complète de toutes les métriques exposées par l'application ATMR.

## Métriques Dispatch

### dispatch_runs_total

**Type**: Counter

**Labels**:

- `status`: Status du run (`running`, `completed`, `failed`)
- `mode`: Mode du dispatch (`auto`, `semi_auto`, `manual`)
- `company_id`: ID de l'entreprise

**Description**: Nombre total de runs dispatch exécutés.

**Exemple PromQL**:

```promql
# Taux de runs par seconde
rate(dispatch_runs_total[5m])

# Runs complétés par mode
sum(dispatch_runs_total{status="completed"}) by (mode)
```

### dispatch_duration_seconds

**Type**: Histogram

**Labels**:

- `mode`: Mode du dispatch
- `company_id`: ID de l'entreprise

**Buckets**: 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0

**Description**: Durée d'exécution du dispatch en secondes (histogramme).

**Exemple PromQL**:

```promql
# P95 de la durée
histogram_quantile(0.95, rate(dispatch_duration_seconds_bucket[5m]))

# P50 de la durée
histogram_quantile(0.50, rate(dispatch_duration_seconds_bucket[5m]))

# Durée moyenne
rate(dispatch_duration_seconds_sum[5m]) / rate(dispatch_duration_seconds_count[5m])
```

### dispatch_quality_score

**Type**: Gauge

**Labels**:

- `dispatch_run_id`: ID du dispatch run
- `company_id`: ID de l'entreprise

**Description**: Score de qualité du dispatch (0-100).

**Exemple PromQL**:

```promql
# Score moyen par entreprise
avg(dispatch_quality_score) by (company_id)

# Score le plus récent
topk(1, dispatch_quality_score)
```

### dispatch_assignment_rate

**Type**: Gauge

**Labels**:

- `dispatch_run_id`: ID du dispatch run
- `company_id`: ID de l'entreprise

**Description**: Taux d'assignation des bookings (0-100%).

**Exemple PromQL**:

```promql
# Taux d'assignation moyen
avg(dispatch_assignment_rate) by (company_id)

# Taux d'assignation le plus récent
topk(1, dispatch_assignment_rate)
```

### dispatch_unassigned_count

**Type**: Gauge

**Labels**:

- `dispatch_run_id`: ID du dispatch run
- `company_id`: ID de l'entreprise

**Description**: Nombre de bookings non assignés après le dispatch.

**Exemple PromQL**:

```promql
# Nombre de bookings non assignés par entreprise
sum(dispatch_unassigned_count) by (company_id)
```

### dispatch_circuit_breaker_state

**Type**: Gauge

**Labels**:

- `company_id`: ID de l'entreprise

**Description**: État du circuit breaker OSRM:

- `0`: CLOSED (normal)
- `1`: OPEN (échecs répétés)
- `2`: HALF_OPEN (test de récupération)

**Exemple PromQL**:

```promql
# Circuit breaker ouvert
dispatch_circuit_breaker_state == 1

# Circuit breaker en état normal
dispatch_circuit_breaker_state == 0
```

### dispatch_temporal_conflicts_total

**Type**: Counter

**Labels**:

- `dispatch_run_id`: ID du dispatch run
- `company_id`: ID de l'entreprise

**Description**: Nombre total de conflits temporels détectés (validation temporelle stricte).

**Exemple PromQL**:

```promql
# Taux de conflits temporels par seconde
rate(dispatch_temporal_conflicts_total[5m])
```

### dispatch_db_conflicts_total

**Type**: Counter

**Labels**:

- `dispatch_run_id`: ID du dispatch run
- `company_id`: ID de l'entreprise

**Description**: Nombre total de conflits DB (violations de contraintes uniques).

**Exemple PromQL**:

```promql
# Taux de conflits DB par seconde
rate(dispatch_db_conflicts_total[5m])
```

### dispatch_osrm_cache_hits_total

**Type**: Counter

**Labels**:

- `dispatch_run_id`: ID du dispatch run
- `company_id`: ID de l'entreprise

**Description**: Nombre total de hits dans le cache OSRM.

### dispatch_osrm_cache_misses_total

**Type**: Counter

**Labels**:

- `dispatch_run_id`: ID du dispatch run
- `company_id`: ID de l'entreprise

**Description**: Nombre total de misses dans le cache OSRM.

## Métriques OSRM

### osrm_cache_hits_total

**Type**: Counter

**Labels**:

- `cache_type`: Type de cache (`route`, `table`, `matrix`)

**Description**: Nombre total de hits dans le cache Redis OSRM.

### osrm_cache_misses_total

**Type**: Counter

**Labels**:

- `cache_type`: Type de cache

**Description**: Nombre total de misses dans le cache Redis OSRM.

### osrm_cache_bypass_total

**Type**: Counter

**Description**: Nombre total de bypass cache (Redis non disponible).

### osrm_cache_hit_rate

**Type**: Gauge

**Description**: Taux de réussite du cache OSRM (0-1).

**Exemple PromQL**:

```promql
# Hit rate moyen
avg(osrm_cache_hit_rate)

# Hit rate < 70% (alerte)
osrm_cache_hit_rate < 0.70
```

## Métriques SLO

### dispatch_slo_breaches_total

**Type**: Counter

**Labels**:

- `window_minutes`: Fenêtre de temps en minutes

**Description**: Nombre total de breaches SLO détectées.

**Exemple PromQL**:

```promql
# Taux de breaches par seconde
rate(dispatch_slo_breaches_total[5m])
```

### dispatch_slo_breach_severity

**Type**: Gauge

**Labels**:

- `severity`: Sévérité (`info`, `warning`, `critical`)

**Description**: Sévérité des breaches SLO:

- `0`: info
- `1`: warning
- `2`: critical

### dispatch_slo_should_alert

**Type**: Gauge

**Description**: Devrait alerter (0=no, 1=yes).

**Exemple PromQL**:

```promql
# Alertes actives
dispatch_slo_should_alert == 1
```

### dispatch_slo_breaches_by_type

**Type**: Counter

**Labels**:

- `type`: Type de breach

**Description**: Breaches SLO par type.

**Exemple PromQL**:

```promql
# Breaches par type
sum(rate(dispatch_slo_breaches_by_type[5m])) by (type)
```

## Requêtes PromQL utiles

### Taux de succès des dispatches

```promql
sum(rate(dispatch_runs_total{status="completed"}[5m])) /
sum(rate(dispatch_runs_total[5m]))
```

### Durée moyenne des dispatches

```promql
rate(dispatch_duration_seconds_sum[5m]) /
rate(dispatch_duration_seconds_count[5m])
```

### Taux d'assignation moyen

```promql
avg(dispatch_assignment_rate) by (company_id)
```

### Hit rate cache OSRM

```promql
osrm_cache_hit_rate
```

### Breaches SLO critiques

```promql
dispatch_slo_breach_severity{severity="critical"} > 0
```

## Alertes recommandées

### Dispatch échoué

```promql
rate(dispatch_runs_total{status="failed"}[5m]) > 0.1
```

### Circuit breaker ouvert

```promql
dispatch_circuit_breaker_state == 1
```

### Cache hit rate bas

```promql
osrm_cache_hit_rate < 0.70
```

### Breaches SLO critiques

```promql
dispatch_slo_should_alert == 1
```

### Conflits temporels élevés

```promql
rate(dispatch_temporal_conflicts_total[5m]) > 1
```
