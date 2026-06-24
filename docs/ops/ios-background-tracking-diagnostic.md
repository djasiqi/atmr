# Diagnostic iOS — tracking GPS stale en arrière-plan (A3)

**Statut** : investigation cause racine — Lot 1 (mesure) déployé, Lot 2 (correctifs) en attente d'analyse 24–48 h.

## Constat production (A0)

- Taux stale Apple ~84 % (`driver_device_stale_fix_total{manufacturer="Apple"}`)
- Les heartbeats iOS arrivent avec `last_fix_age_seconds` / `native_last_fix_age_seconds` **déjà élevés côté device** (1h40–5h43) → le problème est **mobile/device**, pas le pipeline Kafka serveur.

## Signaux collectés (Lot 1 — A1 + A3.0)

Champs remontés dans `device_health` et persistés en `driver_device_health_events` :

| Champ | Source mobile | Interprétation |
|-------|---------------|----------------|
| `app_version` | `expo-application` | Ventilation stale par version app |
| `os_version` | `expo-device` | Ventilation stale par major iOS |
| `native_last_fix_age_seconds` | task native GPS | Âge fix côté task background (vs JS) |
| `native_task_running` | `trackingRuntime` | Task location native active ? |
| `ios_accuracy_authorization` | inféré / API | `full` vs `reduced` |
| `ios_low_power_mode` | `expo-battery` | Mode économie d'énergie |
| `ios_background_refresh_status` | `expo-background-fetch` | Background App Refresh |

## Audit permissions iOS (checklist chauffeur)

1. **Localisation** : `Always` + **Precise Location** activée
2. **Background App Refresh** : activé pour Lirie
3. **Low Power Mode** : désactivé en service
4. **UIBackgroundModes** : `location` présent dans `app.json` / build natif
5. **Batterie** : pas d'optimisation agressive (Settings → Lirie → Background)

## Hypothèses priorisées

| # | Hypothèse | Signal discriminant |
|---|-----------|---------------------|
| H1 | iOS suspend la task background location | `native_task_running=false` + fix age élevé |
| H2 | Permission réduite (Precise off) | `ios_accuracy_authorization=reduced` |
| H3 | Low Power Mode / Background Refresh denied | `ios_low_power_mode=true` ou `background_refresh=denied` |
| H4 | App tuée par OS (memory pressure) | `app_state=background` + task stopped + fix très vieux |

## Requêtes d'analyse (post Lot 1, 24–48 h)

```bash
# Audit stale par chauffeur (A0)
docker compose exec api python -m scripts.report_driver_tracking_coverage \
  --stale-audit --stale-hours 48 --output /tmp/stale-audit.csv
```

```sql
-- iOS stale : comparer fix JS vs native
SELECT driver_id, app_version, os_version,
       last_fix_age_seconds, native_last_fix_age_seconds,
       ios_low_power_mode, ios_background_refresh_status,
       ios_accuracy_authorization, native_task_running, recorded_at
FROM driver_device_health_events
WHERE platform = 'ios'
  AND last_fix_age_seconds > 300
ORDER BY recorded_at DESC
LIMIT 50;
```

## Correctifs Lot 2 (planifiés — A3.1/A3.2/A3.3)

| Lot | Action | Fichiers cibles |
|-----|--------|-----------------|
| A3.1 | Relance tracking sur `AppState` foreground/background | `mobile/unified-app/src/features/driver/services/` |
| A3.2 | Watchdog si fix GPS > X minutes | task manager + heartbeat |
| A3.3 | UX chauffeur si iOS bloque background | écran readiness / bannière |
| A3.4 | Significant location changes (fallback) | config native iOS |

## Tests device

- iPhone 12 (iOS 16+)
- iPhone XR (iOS 15/16)
- Scénarios : 30 min background, Low Power Mode ON/OFF, toggle permissions

## Métriques / alertes

- `TrackingAppleStaleHigh` — stale > 60 % ventilé par `app_version` / `os_version`
- `driver_device_ios_health_total` — signaux background iOS
- Dashboard : `driver-tracking-health` (Grafana)
