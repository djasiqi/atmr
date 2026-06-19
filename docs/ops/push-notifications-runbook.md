# Runbook — Couverture push chauffeur

## Quand le token push devient actif

Le token n'est **pas** activé à la création du compte. Conditions cumulatives :

1. App mobile installée, session chauffeur `ready`
2. Feature flag `driver_push_enabled` actif
3. Disclosure notifications acceptée (`DriverNotificationDisclosureHost`)
4. Permission OS notifications accordée (unique point : `registerPushToken.ts`)
5. POST réussi `POST /api/v1/driver/save-push-token` → `device_tokens.is_active=true`

Ordre sécurisé du flush pending :

```text
disclosure acceptée → permission OS → flush pending → save-push-token
```

## Garde disclosure sur flush pending

`flushPendingPushTokenRegistrations` ne doit **jamais** POSTer si la disclosure n'est pas acceptée ou si la permission OS est absente. Test unitaire : `pendingPushTokenRegistration.test.ts`.

## Diagnostic ops

```bash
# Rapport complet (optionnel driver_id)
./scripts/push-notifications-diagnostic.sh
./scripts/push-notifications-diagnostic.sh 6858

# Audit tokens (via Docker)
docker compose exec api python scripts/audit_device_tokens.py --list-drivers-without-token
docker compose exec api python scripts/audit_device_tokens.py --report

# Couverture admin
curl -H "Authorization: Bearer $ADMIN_JWT" \
  "https://<host>/api/v1/admin/push-coverage/drivers?operational_only=true&without_token_only=true"
```

## Interprétation endpoint admin

| Champ | Signification |
|-------|----------------|
| `last_driver_activity_at` | Dernière activité device (pas un login strict) |
| `token_created_at` | Création du token actif le plus récent |
| `token_updated_at` | Dernière ré-enregistrement |
| `push_status` | `operational`, `no_token`, `stale_token`, `token_invalid`, `expo_fallback_unreliable` (Android sans token FCM natif) |
| `app_version` | Version app depuis Redis `driver:{id}:device_health` |

Exemple actionnable : `token_invalid` + dernier succès il y a 42j + Android + v1.0.3 → demander réouverture app / réinstallation.

## Cas production connus

| Driver | Problème | Action |
|--------|----------|--------|
| 6858 | Token FCM invalidé (`token_unregistered`), jamais ré-enregistré | Ouvrir app, accepter notifications, test push |
| 7755 | Aucun token jamais enregistré | Première session complète + disclosure + permission |

## Lifecycle tokens

- Désactivation immédiate : `token_unregistered`, `DeviceNotRegistered`, `SenderIdMismatchError`
- Stale : ≥5 échecs consécutifs **et** >30j sans succès (`PUSH_DEVICE_TOKEN_STALE_DAYS`)
- Cron Celery : `notifications.deactivate_stale_device_tokens` (quotidien)

## Métriques Prometheus / Grafana

Dashboard Grafana : `push-notifications-atmr`

| Métrique | Usage |
|----------|--------|
| `push_operational_drivers_*` | Couverture chauffeurs `is_active && is_available` |
| `push_token_registration_success_total` | Détecter régressions release **avant** baisse couverture |
| `push_token_registration_failure_total` | Pic d'échecs post-release |

**Limite :** `push_operational_drivers_total` dépend de la qualité du statut `is_available`. Des chauffeurs laissés `is_available=true` sans travailler gonflent le dénominateur et abaissent artificiellement le ratio.

## Alertes (rollout)

| Alerte | Statut |
|--------|--------|
| `PushDriverCoverageLowBootstrap` (25%) | **Active** |
| `PushTokenRegistrationFailureSpike` | **Active** (seuil volumétrique) |
| `PushDriverCoverageWarning` (95%) | Commentée — activer après nettoyage flotte |
| `PushDriverCoverageCritical` (90%) | Commentée — idem |

Calibrer `PushTokenRegistrationFailureSpike` sur volumétrie réelle (<20 chauffeurs : `increase > 10` / 30min).

## Test push manuel

```bash
# Côté chauffeur authentifié
POST /api/v1/driver/me/test-push
```

Bouton profil app mobile : « Déclencher test push ».

## STOP GATE OPS (clôture projet)

Validation obligatoire sur chauffeurs **6858** et **7755** :

1. Ouvrir app, disclosure + permission
2. Token actif en base (`device_tokens.is_active=true`)
3. Test push reçu
4. `GET /admin/push-coverage/drivers?driver_id=XXXX` → `push_status=operational`
5. Grafana : `push_operational_drivers_with_active_token_total` augmente

```sql
SELECT driver_id, is_active, provider, platform, created_at, last_push_success_at, last_push_error_code
FROM device_tokens WHERE driver_id IN (6858, 7755) ORDER BY updated_at DESC;
```

## STOP GATE métier (phases futures)

| Phase | Comportement | Statut |
|-------|--------------|--------|
| 1 | Warning admin + badge « push non opérationnel » | Implémenté (endpoint admin) |
| 2 | Exclusion dispatch auto sans push | Décision métier |
| 3 | Blocage disponibilité chauffeur | Validation métier requise |

**Aucun** blocage automatique de `driver.is_available` dans ce lot.
