# Runbook - Procédures d'Incident

## Vue d'ensemble

Ce document décrit les procédures à suivre en cas d'incident avec le système de dispatch unifié.

## Incidents courants et solutions

### 1. OSRM en panne

**Symptômes**:

- Erreurs 500 sur les endpoints de dispatch
- Logs: "OSRM service unavailable"
- Taux d'assignation très faible

**Diagnostic**:

```bash
# Vérifier la santé d'OSRM
curl -s http://localhost:5000/route/v1/driving/6.6323,46.5197;6.6330,46.5200

# Vérifier les logs
tail -f /var/log/atmr/dispatch.log | grep OSRM
```

**Solution**:

1. **Court terme**: Le circuit breaker active automatiquement le fallback euclidien
2. **Moyen terme**: Redémarrer le service OSRM
   ```bash
   sudo systemctl restart osrm
   ```
3. **Long terme**: Vérifier la configuration réseau et les ressources

**Vérification**:

```bash
# Tester un dispatch simple
curl -X POST http://localhost:5000/api/company_dispatch/run \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"for_date": "2024-01-15", "mode": "heuristic_only"}'
```

### 2. Dispatch bloqué (clé Redis expirée)

**Symptômes**:

- Message: "Another dispatch is already running"
- Dispatch qui ne se termine jamais
- Verrous Redis orphelins

**Diagnostic**:

```bash
# Vérifier les clés Redis
redis-cli keys "dispatch:lock:*"
redis-cli keys "dispatch:enqueued:*"

# Vérifier les processus Celery
celery -A celery_app inspect active
```

**Solution**:

1. **Identifier la clé problématique**:

   ```bash
   redis-cli keys "dispatch:lock:*"
   ```

2. **Supprimer la clé expirée**:

   ```bash
   redis-cli del "dispatch:lock:1:2024-01-15"
   ```

3. **Vérifier les tâches Celery bloquées**:
   ```bash
   celery -A celery_app inspect active
   celery -A celery_app control revoke <task_id>
   ```

**Prévention**:

- Surveiller les TTL des clés Redis
- Alerter sur les dispatches qui durent > 10 minutes

### 3. Queue Celery saturée

**Symptômes**:

- Dispatches qui ne se lancent pas
- Logs: "Celery queue full"
- Temps de réponse élevés

**Diagnostic**:

```bash
# Vérifier la queue Celery
celery -A celery_app inspect active
celery -A celery_app inspect scheduled
celery -A celery_app inspect reserved

# Vérifier les workers
celery -A celery_app inspect stats
```

**Solution**:

1. **Augmenter le nombre de workers**:

   ```bash
   celery -A celery_app worker --concurrency=4 --loglevel=info
   ```

2. **Nettoyer la queue**:

   ```bash
   celery -A celery_app purge
   ```

3. **Redémarrer les workers**:
   ```bash
   sudo systemctl restart celery-worker
   ```

### 4. Taux d'assignation faible

**Symptômes**:

- Taux d'assignation < 70%
- Nombreuses réservations non assignées
- Alertes de performance

**Diagnostic**:

```bash
# Vérifier les métriques de santé
curl -s http://localhost:5000/api/company_dispatch_health/health \
  -H "Authorization: Bearer $TOKEN"

# Analyser les raisons de non-assignation
curl -s http://localhost:5000/api/company_dispatch_health/health \
  -H "Authorization: Bearer $TOKEN" | jq '.unassigned_reasons'
```

**Solution**:

1. **Analyser les raisons**:

   - `no_driver_available`: Augmenter le nombre de chauffeurs
   - `capacity_exceeded`: Ajuster les capacités des véhicules
   - `time_window_infeasible`: Étendre les fenêtres de travail
   - `no_emergency_drivers`: Former plus de chauffeurs d'urgence

2. **Ajuster les paramètres**:
   ```python
   # Dans settings.py
   settings.pooling.time_tolerance_min = 15  # Augmenter la tolérance
   settings.service_times.min_transition_margin_min = 10  # Réduire la marge
   ```

### 5. Erreurs de validation des paramètres

**Symptômes**:

- Erreurs 400 sur les endpoints
- Message: "Paramètres invalides"
- Logs Marshmallow

**Diagnostic**:

```bash
# Vérifier les logs de validation
tail -f /var/log/atmr/dispatch.log | grep "Paramètres invalides"
```

**Solution**:

1. **Vérifier le format des dates**: YYYY-MM-DD
2. **Vérifier les modes**: auto, heuristic_only, solver_only
3. **Vérifier la structure des overrides**

### 6. Problèmes de cache Redis

**Symptômes**:

- Performances dégradées
- Appels OSRM répétés
- Erreurs de cache

**Diagnostic**:

```bash
# Vérifier l'état de Redis
redis-cli ping
redis-cli info memory

# Vérifier les clés de cache
redis-cli keys "osrm:*"
redis-cli keys "dispatch:*"
```

**Solution**:

1. **Nettoyer le cache**:

   ```bash
   redis-cli flushdb
   ```

2. **Redémarrer Redis**:

   ```bash
   sudo systemctl restart redis
   ```

3. **Vérifier la configuration**:
   ```bash
   redis-cli config get maxmemory
   redis-cli config get maxmemory-policy
   ```

## Commandes de diagnostic

### Vérification de l'état général

```bash
# Santé du système
curl -s http://localhost:5000/api/company_dispatch_health/health \
  -H "Authorization: Bearer $TOKEN"

# Tendances de performance
curl -s http://localhost:5000/api/company_dispatch_health/health/trends?days=7 \
  -H "Authorization: Bearer $TOKEN"
```

### Vérification des services

```bash
# OSRM
curl -s http://localhost:5000/route/v1/driving/6.6323,46.5197;6.6330,46.5200

# Redis
redis-cli ping

# Celery
celery -A celery_app inspect ping
```

### Logs à consulter

```bash
# Logs de dispatch
tail -f /var/log/atmr/dispatch.log

# Logs d'OSRM
tail -f /var/log/osrm/osrm.log

# Logs de Redis
tail -f /var/log/redis/redis.log

# Logs de Celery
tail -f /var/log/celery/worker.log
```

## Procédures de récupération

### Récupération complète du système

1. **Arrêter tous les services**:

   ```bash
   sudo systemctl stop celery-worker
   sudo systemctl stop osrm
   sudo systemctl stop redis
   ```

2. **Nettoyer les ressources**:

   ```bash
   redis-cli flushall
   rm -f /tmp/celerybeat-schedule*
   ```

3. **Redémarrer dans l'ordre**:

   ```bash
   sudo systemctl start redis
   sudo systemctl start osrm
   sudo systemctl start celery-worker
   ```

4. **Vérifier la santé**:
   ```bash
   curl -s http://localhost:5000/api/company_dispatch_health/health \
     -H "Authorization: Bearer $TOKEN"
   ```

### Récupération partielle

1. **Redémarrer un service spécifique**:

   ```bash
   sudo systemctl restart <service-name>
   ```

2. **Vérifier l'impact**:
   ```bash
   # Tester un dispatch simple
   curl -X POST http://localhost:5000/api/company_dispatch/run \
     -H "Authorization: Bearer $TOKEN" \
     -d '{"for_date": "2024-01-15", "mode": "heuristic_only"}'
   ```

## Alertes et monitoring

### Métriques critiques

- Taux d'assignation < 70%
- Temps d'exécution > 10 secondes
- Disponibilité OSRM < 95%
- Erreurs de validation > 5%

### Actions automatiques

- Circuit breaker OSRM (5 échecs consécutifs)
- TTL automatique des verrous (5 minutes)
- Anti-duplication des runs (5 minutes)

## Contacts d'urgence

- **Équipe DevOps**: devops@atmr.com
- **Équipe Backend**: backend@atmr.com
- **On-call**: +41 21 XXX XX XX

## Escalade

1. **Niveau 1**: Diagnostic et solution standard
2. **Niveau 2**: Intervention manuelle et configuration
3. **Niveau 3**: Escalade vers l'équipe de développement
4. **Niveau 4**: Escalade vers l'équipe d'architecture
