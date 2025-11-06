# Guide de Configuration Monitoring Prometheus/Grafana

Ce guide explique comment configurer et démarrer l'environnement de monitoring Prometheus et Grafana pour ATMR.

## Prérequis

- Docker et Docker Compose installés
- Accès au réseau Docker
- Ports disponibles: 9090 (Prometheus), 3000 (Grafana)

## Démarrage

### 1. Démarrer les services

```bash
docker-compose up -d prometheus grafana
```

Vérifier que les services sont démarrés:

```bash
docker-compose ps prometheus grafana
```

### 2. Accéder aux interfaces

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
  - Utilisateur par défaut: `admin`
  - Mot de passe par défaut: `admin` (changé au premier login)

### 3. Vérifier que Prometheus scrape les métriques

Dans Prometheus, aller dans Status → Targets et vérifier que `atmr-backend` est UP.

L'endpoint scrapé est: `http://api:5000/api/v1/prometheus/metrics`

## Configuration

### Prometheus

La configuration Prometheus se trouve dans `prometheus/prometheus.yml`:

- **Scrape interval**: 15s
- **Evaluation interval**: 15s
- **Retention**: 30 jours

### Grafana

Grafana est configuré via provisioning:

- **Datasource**: Configuré automatiquement (`grafana/provisioning/datasources/prometheus.yml`)
- **Dashboards**: Auto-provisionnés depuis `grafana/dashboards/`

Les dashboards disponibles:

1. **ATMR Dispatch Overview**: Vue d'ensemble des métriques dispatch
2. **ATMR Dispatch SLO Monitoring**: Monitoring des SLO breaches

## Métriques disponibles

### Métriques Dispatch

- `dispatch_runs_total`: Nombre total de runs dispatch
- `dispatch_duration_seconds`: Durée d'exécution (histogramme)
- `dispatch_quality_score`: Score de qualité (0-100)
- `dispatch_assignment_rate`: Taux d'assignation (%)
- `dispatch_unassigned_count`: Nombre de bookings non assignés
- `dispatch_circuit_breaker_state`: État du circuit breaker OSRM
- `dispatch_temporal_conflicts_total`: Conflits temporels
- `dispatch_db_conflicts_total`: Conflits DB

### Métriques OSRM

- `osrm_cache_hits_total`: Hits cache OSRM
- `osrm_cache_misses_total`: Misses cache OSRM
- `osrm_cache_bypass_total`: Bypass cache (Redis HS)
- `osrm_cache_hit_rate`: Taux de réussite cache (0-1)

### Métriques SLO

- `dispatch_slo_breaches_total`: Total breaches détectées
- `dispatch_slo_breach_severity`: Sévérité des breaches (0=info, 1=warning, 2=critical)
- `dispatch_slo_should_alert`: Devrait alerter (0=no, 1=yes)
- `dispatch_slo_breaches_by_type`: Breaches par type

## Utilisation des dashboards

### Dashboard Dispatch Overview

Affiche:

- Graphique dispatch_runs_total (par status, mode)
- Graphique dispatch_duration_seconds (histogramme p50/p95)
- Graphique dispatch_quality_score
- Graphique dispatch_assignment_rate
- Graphique dispatch_unassigned_count
- Graphique circuit breaker state
- Graphique conflits temporels/DB
- Graphique cache hit rate OSRM

### Dashboard SLO Monitoring

Affiche:

- Graphique SLO breaches par type
- Graphique should_alert
- Graphique breach severity
- Tableau des dernières breaches

## Troubleshooting

### Prometheus ne scrape pas les métriques

1. Vérifier que l'API backend est accessible:

   ```bash
   curl http://localhost:5000/api/v1/prometheus/metrics
   ```

2. Vérifier les logs Prometheus:

   ```bash
   docker-compose logs prometheus
   ```

3. Vérifier la configuration dans Prometheus UI: Status → Configuration

### Grafana ne peut pas se connecter à Prometheus

1. Vérifier que Prometheus est accessible depuis Grafana:

   ```bash
   docker-compose exec grafana wget -O- http://prometheus:9090/api/v1/query?query=up
   ```

2. Vérifier la configuration du datasource dans Grafana: Configuration → Data Sources

### Les dashboards ne s'affichent pas

1. Vérifier que les fichiers JSON sont dans `grafana/dashboards/`
2. Vérifier les logs Grafana:

   ```bash
   docker-compose logs grafana
   ```

3. Importer manuellement dans Grafana: Dashboards → Import → Upload JSON

## Variables d'environnement

Vous pouvez configurer Grafana via variables d'environnement dans `docker-compose.yml`:

- `GRAFANA_ADMIN_USER`: Utilisateur admin (défaut: `admin`)
- `GRAFANA_ADMIN_PASSWORD`: Mot de passe admin (défaut: `admin`)

## Alertes

Les alertes Prometheus sont configurées dans:

- `prometheus/alerts-dispatch.yml`: Alertes dispatch
- `prometheus/alerts-slo.yml`: Alertes SLO

Pour activer Alertmanager, décommenter la configuration dans `prometheus/prometheus.yml`.

## Sauvegarde

Les données Prometheus sont persistées dans le volume Docker `prometheus-data`.

Pour sauvegarder:

```bash
docker-compose exec prometheus promtool tsdb dump /prometheus > backup.tar.gz
```

Pour restaurer, utiliser `promtool tsdb create` (voir documentation Prometheus).
