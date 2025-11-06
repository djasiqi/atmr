# ✅ 2.11: Configuration PagerDuty pour Alertes SLO

Ce document explique comment configurer PagerDuty pour recevoir les alertes SLO breach depuis Prometheus/Alertmanager.

## 1. Créer une intégration PagerDuty

1. **Connecter au dashboard PagerDuty**

   - URL: https://app.pagerduty.com
   - Créer un nouveau service ou utiliser un service existant

2. **Ajouter une intégration Prometheus**

   - Dans le service PagerDuty: **Integrations** → **Add integration**
   - Sélectionner **Prometheus**
   - Copier la **Integration Key** générée

3. **Configurer les alertes**
   - Définir les règles d'escalade (on-call, etc.)
   - Configurer les notifications (SMS, email, push)

## 2. Configurer Alertmanager

1. **Copier l'exemple de configuration**

   ```bash
   cp prometheus/alertmanager.example.yml prometheus/alertmanager.yml
   ```

2. **Remplacer la clé PagerDuty**

   ```yaml
   # Dans prometheus/alertmanager.yml
   pagerduty_configs:
     - service_key: "VOTRE_INTEGRATION_KEY_ICI"
   ```

3. **Déployer Alertmanager**

   **Option A: Docker Compose**

   ```yaml
   alertmanager:
     image: prom/alertmanager:latest
     volumes:
       - ./prometheus/alertmanager.yml:/etc/alertmanager/alertmanager.yml
     ports:
       - "9093:9093"
     command:
       - "--config.file=/etc/alertmanager/alertmanager.yml"
   ```

   **Option B: Kubernetes**

   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: alertmanager
   spec:
     template:
       spec:
         containers:
           - name: alertmanager
             image: prom/alertmanager:latest
             volumeMounts:
               - name: config
                 mountPath: /etc/alertmanager
         volumes:
           - name: config
             configMap:
               name: alertmanager-config
   ```

4. **Mettre à jour prometheus.yml**
   ```yaml
   alerting:
     alertmanagers:
       - static_configs:
           - targets:
               - alertmanager:9093 # Décommenter cette ligne
   ```

## 3. Tester les alertes

1. **Vérifier que Prometheus peut joindre Alertmanager**

   ```bash
   curl http://prometheus:9090/api/v1/alertmanagers
   ```

2. **Vérifier que Alertmanager peut joindre PagerDuty**

   ```bash
   curl http://alertmanager:9093/api/v2/status
   ```

3. **Déclencher une alerte de test**

   - Via l'interface Prometheus: http://prometheus:9090/alerts
   - Ou via une alerte manuelle en modifiant les métriques

4. **Vérifier dans PagerDuty**
   - L'alerte doit apparaître dans PagerDuty
   - Les détails doivent inclure: summary, description, runbook_url

## 4. Types d'alertes configurées

### Alertes Critiques (Page Oncall)

- `APISLOLatencyBreachCritical`: Latence > 1 violation/seconde
- `APISLOErrorRateBreach`: Taux d'erreurs > 5% sur 5min
- `APISLOAvailabilityBreach`: Disponibilité < seuil sur 15min
- `DispatchSLOBreachRepeatedPagerDuty`: Breaches répétés dispatch
- `DispatchSLOSeverityCritical`: Severity critique dispatch
- `CriticalHealthCheckFailing`: Health checks critiques échouent

### Alertes Warnings (Notification Slack)

- `APISLOLatencyBreachRepeated`: Violations latence répétées
- `APILatencyP95High`: Latence p95 > 2s
- `DispatchSLOBreachCountHigh`: > 10 breaches en 1h
- `GlobalSLOBreachesElevated`: Taux global élevé

## 5. Runbooks

Les runbooks sont référencés dans les annotations des alertes via `runbook_url`.

Exemples de runbooks à créer:

- `/runbooks/api-slo-latency.md`: Diagnostic et résolution problèmes latence API
- `/runbooks/api-slo-error-rate.md`: Analyse taux d'erreurs API
- `/runbooks/api-slo-availability.md`: Troubleshooting disponibilité
- `/runbooks/dispatch-slo-breach.md`: Résolution breaches dispatch
- `/runbooks/dispatch-slo-critical.md`: Actions immédiates pour breaches critiques
- `/runbooks/health-check-failure.md`: Diagnostic health checks
- `/runbooks/global-slo-summary.md`: Vue d'ensemble violations système

## 6. Métriques surveillées

### API SLO Metrics

- `api_slo_latency_breach_total`: Compteur violations latence
- `api_slo_error_breach_total`: Compteur violations taux d'erreurs
- `api_slo_availability_breach_total`: Compteur violations disponibilité
- `api_slo_request_duration_seconds`: Histogramme latences (pour p95)

### Dispatch SLO Metrics

- `dispatch_slo_breaches_total`: Compteur breaches dispatch
- `dispatch_slo_breach_severity`: Severity actuelle (gauge)
- `dispatch_slo_should_alert`: Flag alert (0=no, 1=yes)

### Global Metrics

- `http_requests_total`: Total requêtes HTTP (pour health checks)

## 7. Variables d'environnement

```bash
# PagerDuty Integration Key (stockée dans alertmanager.yml, pas env var)
PAGERDUTY_INTEGRATION_KEY=<key>

# Slack Webhook (optionnel, pour fallback)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

## 8. Dépannage

### Alertes ne partent pas vers PagerDuty

1. Vérifier logs Alertmanager: `docker logs alertmanager`
2. Vérifier que Prometheus peut joindre Alertmanager
3. Vérifier que la clé PagerDuty est correcte
4. Vérifier les routes dans alertmanager.yml

### Alertes trop nombreuses

- Ajuster `group_wait` et `group_interval` dans alertmanager.yml
- Augmenter `for` dans les règles d'alerte (alerts-slo.yml)
- Utiliser `inhibit_rules` pour supprimer alertes redondantes

### Alertes manquantes

- Vérifier que les métriques SLO sont bien exposées: `curl http://api:5000/prometheus/metrics-http`
- Vérifier que Prometheus scrape bien les métriques
- Vérifier les labels dans les alertes (matchent avec les métriques)
