# Runbook — Couverture push chauffeur

> **iOS — aucune notification** : voir le diagnostic Phase A
> [`docs/ops/push-ios-no-notifications-audit.md`](push-ios-no-notifications-audit.md)
> (statuts canoniques, test-push forcé `provider=fcm|expo`, gate de sortie).

## Gate iOS / contexte chauffeur

Mêmes prérequis que Android : contexte **chauffeur** `ready`, disclosure acceptée,
permission OS accordée. Sur iOS, vérifier aussi `aps-environment`, Bundle ID et clé APNs Firebase.

`provider_accepted` (FCM `message_id` ou ticket Expo) ≠ livraison device.
`mobile_received` / `mobile_opened` / `business_acknowledged` sont des étapes distinctes.

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

## ⚠️ RUNBOOK PUSH ANDROID — contexte chauffeur obligatoire

**Les tests FCM chauffeur sont invalides** si le contexte actif est `company`, `institution` ou `client`.

`DriverNotificationsBridge` (seul composant qui enregistre `provider=fcm` côté chauffeur) ne se monte **que** si :

```text
activeContext.context_type === "driver"
status === "ready"
driverId != null
driver_push_enabled === true
```

En contexte **entreprise**, c’est `CompanyNotificationsBridge` qui peut enregistrer un token **Expo** — ce qui produit exactement le symptôme :

```text
device_tokens.provider = expo   (mis à jour)
device_tokens.provider = fcm    (absent)
```

→ **Ce n’est pas une preuve que FCM Android est cassé** ; c’est souvent une preuve que le flux chauffeur n’a jamais tourné.

### Avant tout test FCM / push chauffeur

- [ ] Contexte actif = **chauffeur** (pas entreprise)
- [ ] Écran chauffeur visible (Disponible, missions…) — pas dispatch company
- [ ] `driverId` connu (ex. 7514)
- [ ] Disclosure notifications acceptée (modale **Continuer** si bandeau)
- [ ] Permission notifications Android accordée
- [ ] Attendre **20–30 s** après connexion avant la requête SQL

**Qualification incident recommandée** tant que le gate ci-dessous n’est pas PASS :

> Couverture FCM chauffeur non démontrée — tests exécutés majoritairement hors contexte chauffeur.

**Ne pas qualifier** en P1/P2 « FCM Android cassé » sans gate PASS/FAIL en contexte chauffeur avéré.

Chaîne technique (rappel) :

```text
DriverNotificationsBridge monté
  → useRegisterPushTokenEffect
  → getDriverFcmToken()
  → registerFcm()
  → device_tokens.provider = fcm
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

# Vérification FCM natif Android (SHA-1 / expo_fallback)
./scripts/verify-fcm-token-coverage.sh --driver-id 7514
./scripts/verify-fcm-token-coverage.sh --android-expo-only
docker exec atmr-backend-1 python scripts/verify_fcm_token_coverage.py --driver-id 7514 --expect-fcm
```

Procédure complète SHA-1 Firebase : [firebase-fcm-sha1-procedure.md](./firebase-fcm-sha1-procedure.md)

```bash
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
| 7514 | Couverture FCM non démontrée ; token Expo présent ; tests initiaux souvent en contexte `company:1` | Exécuter **STOP GATE FCM-7514-A** (contexte chauffeur avéré) — voir ci-dessous |

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

## STOP GATE FCM-7514-A (couverture FCM chauffeur — Android)

Gate formel pour démontrer (ou infirmer) l’enregistrement FCM natif du driver **7514** sur un appareil (ex. S23).

### Prérequis app (conditions cumulatives)

| # | Condition |
|---|-----------|
| 1 | Contexte actif = **chauffeur** (`activeContext.context_type = driver`) |
| 2 | `driverId = 7514` |
| 3 | Disclosure notifications acceptée |
| 4 | Permission notifications OS accordée |
| 5 | Attente **20–30 s** après connexion chauffeur |

### Exécution

1. Déconnexion complète de l’app
2. Connexion **chauffeur 7514** — vérifier visuellement l’UI chauffeur (pas entreprise)
3. Noter l’**heure exacte** de connexion (`T0`)
4. Attendre 20–30 s
5. Requête SQL prod immédiate :

```sql
SELECT
    provider,
    platform,
    is_active,
    updated_at
FROM device_tokens
WHERE driver_id = 7514
ORDER BY updated_at DESC;
```

Script ops :

```bash
docker exec atmr-backend-1 python scripts/verify_fcm_token_coverage.py --driver-id 7514 --expect-fcm
docker exec atmr-backend-1 python scripts/verify_fcm_token_coverage.py --driver-id 7514 --gate-json
```

Sortie gate compacte (`--gate-json`) :

```json
{
  "driver_id": 7514,
  "fcm_present": false,
  "active_provider": "expo",
  "status": "FAIL"
}
```

### Critères

| Résultat | Verdict |
|----------|---------|
| `provider=fcm`, `platform=android`, `is_active=true`, `updated_at >= T0` | **PASS** — couverture FCM démontrée |
| Token Expo inchangé, pas de FCM | **FAIL** — flux FCM non achevé ; cause racine **non identifiée** sans télémétrie |
| Aucune ligne `updated_at >= T0` | **FAIL** — enregistrement push jamais atteint |

**Qualification incident (juin 2026, driver 7514 / S23)** :

```text
P1 MOBILE PUSH — Android Driver FCM registration path not completing
État : REPRODUCED
Impact : token Expo uniquement, aucun FCM natif en base
Cause racine : NON IDENTIFIÉE (gate FAIL confirmé)
```

### Observabilité prod (P0 — après déploiement backend + OTA mobile)

**Logs backend** — corréler le point de rupture sans logcat JS :

```bash
# A) save-push-token jamais appelé vs rejeté
docker logs atmr-backend-1 --since 30m 2>&1 | grep save_push_token

# B) Télémétrie mobile structurée (5 événements)
docker logs atmr-backend-1 --since 30m 2>&1 | grep driver_push_telemetry
```

| Événement `driver_push_telemetry` | Interprétation |
|-----------------------------------|----------------|
| `driver_push.bridge_mounted` + `enabled=false` | Cas **A** — bridge non actif (contexte, flags, driverId) |
| `driver_push.disclosure_blocked` | Cas **B** — disclosure non acceptée |
| `driver_push.permission_blocked` | Cas **C** — permission OS refusée |
| `driver_push.get_token_failed` | Cas **D** — `messaging().getToken()` échoue |
| `driver_push.token_acquired` sans `register_success` | `getToken()` OK, `registerFcm()` / POST en échec (zone grise D→E) |
| `driver_push.register_success` sans `save_push_token received` | Cas **E** — échec réseau / API côté client |
| `save_push_token received` + status≠200 | Cas **F** — rejet backend |

Endpoint mobile : `POST /api/v1/driver/me/telemetry/push`  
Endpoint token : `POST /api/v1/driver/save-push-token` (logs `save_push_token received/outcome`).

> ⚠️ L’ingest local (`localhost:7242/ingest`) est **__DEV__ uniquement** — invisible en prod.  
> La télémétrie push prod passe par `/driver/me/telemetry/push` + logs backend.

### Vérifier l’OTA chargée (Sentry)

Tags Sentry mobile (MonitoringProvider) :

- `expo_update_id` — attendu `74c41b05-…` pour l’OTA diagnostic FCM-GATE
- `runtime_version` — ex. `1.0.5`
- `is_embedded_launch` — `false` si bundle OTA appliqué

Sans OTA diagnostic appliquée, les logs `[FCM-GATE]` console.info restent invisibles en release.

### Si FAIL — logcat natif (complément)

```powershell
adb logcat -v time | Select-String 'FirebaseMessaging|RNFirebase|FIS_AUTH|SERVICE_NOT_AVAILABLE'
```

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

## OTA auto-reload (prod)

⚠️ **Désactivé en prod** (`EXPO_PUBLIC_OTA_AUTO_RELOAD_ENABLED=0`) après incident juin 2026. Rollback OTA group `a9012280-…` ; rechargement manuel (double relance app) requis.

Comportement actuel :

1. `expo-updates` télécharge l’OTA au boot (`checkAutomatically: ON_LOAD`)
2. L’utilisateur doit **fermer et rouvrir** l’app pour appliquer le bundle
3. Reprise auto-reload : uniquement après stabilisation (`OtaAutoReloadProvider` non monté)

Validation terrain (S23 / driver sans FCM) — **uniquement en contexte chauffeur** :

```powershell
adb logcat -v time | Select-String 'FirebaseMessaging|RNFirebase|dev.expo.updates'
```

> En APK prod, utiliser **logs backend** (`save_push_token`, `driver_push_telemetry`) et SQL / `verify_fcm_token_coverage.py --gate-json`.  
> Les `console.info` JS et l’ingest local `localhost:7242` sont **invisibles** en release.

Attendu après swipe-kill + relance ×2 **en session chauffeur** :

- `dev.expo.updates` télécharge l’OTA si applicable
- SQL : ligne `provider=fcm` pour le driver testé (gate FCM-7514-A)

Voir [firebase-fcm-sha1-procedure.md](./firebase-fcm-sha1-procedure.md) pour SHA-1 et checklist complète.
