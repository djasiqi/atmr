# Guide de Corrélation des Logs

Ce guide explique comment utiliser les identifiants de corrélation (`dispatch_run_id`, `trace_id`) pour tracer un dispatch complet à travers les logs et métriques.

## Identifiants de corrélation

### dispatch_run_id

Identifiant unique d'un dispatch run. Présent dans tous les logs liés à un dispatch.

**Format dans les logs**: `[dispatch_run_id=123]`

### trace_id

Identifiant OpenTelemetry pour tracer une requête à travers plusieurs services.

**Format dans les logs**: `trace_id=abc123`

### span_id

Identifiant OpenTelemetry pour un span spécifique dans une trace.

**Format dans les logs**: `span_id=def456`

## Utilisation dans les logs

Tous les logs de dispatch incluent automatiquement `dispatch_run_id` dans le contexte de logging.

### Exemple de log

```
2025-01-15 10:30:45 [INFO] [dispatch_run_id=123] [company_id=1] Starting dispatch run
2025-01-15 10:30:46 [INFO] [dispatch_run_id=123] [trace_id=abc123] Collected 10 bookings
2025-01-15 10:30:47 [INFO] [dispatch_run_id=123] [trace_id=abc123] Running heuristics
2025-01-15 10:30:48 [INFO] [dispatch_run_id=123] [trace_id=abc123] Applied 8 assignments
```

## Corrélation avec les métriques

Les métriques Prometheus incluent `dispatch_run_id` dans les labels pour permettre la corrélation.

### Exemple de métrique

```prometheus
dispatch_quality_score{dispatch_run_id="123",company_id="1"} 85.5
dispatch_assignment_rate{dispatch_run_id="123",company_id="1"} 80.0
```

## Recherche dans les logs

### Avec grep

```bash
# Rechercher tous les logs d'un dispatch
grep "dispatch_run_id=123" /var/log/atmr/backend.log

# Rechercher avec trace_id
grep "trace_id=abc123" /var/log/atmr/backend.log
```

### Avec journald (si configuré)

```bash
journalctl -u atmr-backend | grep "dispatch_run_id=123"
```

### Avec ELK/Loki

```promql
{dispatch_run_id="123"}
```

## Corrélation logs ↔ métriques

### Étape 1: Identifier le dispatch_run_id

Depuis les logs ou l'interface utilisateur, notez le `dispatch_run_id`.

### Étape 2: Rechercher les métriques

Dans Prometheus, rechercher:

```promql
dispatch_quality_score{dispatch_run_id="123"}
dispatch_assignment_rate{dispatch_run_id="123"}
```

### Étape 3: Corréler avec les logs

Utiliser le `trace_id` pour trouver tous les logs liés:

```bash
grep "trace_id=abc123" /var/log/atmr/backend.log
```

## Exemples de requêtes

### Tracer un dispatch complet

1. **Identifier le dispatch_run_id** depuis l'interface ou les logs
2. **Rechercher les logs**:
   ```bash
   grep "dispatch_run_id=123" /var/log/atmr/backend.log | tail -20
   ```
3. **Vérifier les métriques**:
   ```promql
   dispatch_quality_score{dispatch_run_id="123"}
   dispatch_assignment_rate{dispatch_run_id="123"}
   dispatch_unassigned_count{dispatch_run_id="123"}
   ```

### Analyser les erreurs

1. **Identifier les erreurs**:
   ```bash
   grep "dispatch_run_id=123" /var/log/atmr/backend.log | grep ERROR
   ```
2. **Vérifier les conflits**:
   ```promql
   dispatch_temporal_conflicts_total{dispatch_run_id="123"}
   dispatch_db_conflicts_total{dispatch_run_id="123"}
   ```

### Corréler avec OpenTelemetry

Si OpenTelemetry est configuré, utiliser le `trace_id` pour voir la trace complète dans Jaeger/Zipkin.

## Format des logs

Les logs sont structurés avec les champs suivants:

```
timestamp [LEVEL] [dispatch_run_id=X] [company_id=Y] [trace_id=Z] message
```

### Exemple complet

```
2025-01-15 10:30:45 [INFO] [dispatch_run_id=123] [company_id=1] [trace_id=abc123] [span_id=def456] Starting dispatch run for company 1
```

## Scripts de validation

Utiliser les scripts de validation pour vérifier la corrélation:

```bash
# Valider la corrélation des logs
python backend/scripts/validate_logs_correlation.py

# Tester les métriques
python backend/scripts/validate_metrics.py
```

## Best Practices

1. **Toujours inclure dispatch_run_id** dans les logs de dispatch
2. **Utiliser trace_id** pour corréler avec OpenTelemetry
3. **Documenter les corrélations** dans les tickets d'incident
4. **Utiliser les métriques** pour identifier les problèmes de performance
5. **Utiliser les logs** pour comprendre le contexte détaillé

## Troubleshooting

### dispatch_run_id manquant

Si `dispatch_run_id` n'apparaît pas dans les logs, vérifier que le contexte de logging est correctement configuré.

### trace_id manquant

Si `trace_id` n'apparaît pas, vérifier que OpenTelemetry est configuré et activé.

### Métriques non corrélées

Si les métriques ne contiennent pas `dispatch_run_id`, vérifier que les fonctions d'enregistrement des métriques incluent ce label.
