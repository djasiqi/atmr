# Procédure — Firebase SHA-1 + vérification token FCM (Android)

## Contexte

Sur Android production (`ch.liri.operations`, `@react-native-firebase/messaging`), un token **FCM natif** (`provider=fcm`) est requis pour les notifications **app tuée**. Un token **Expo seul** (`ExponentPushToken[...]`) peut être enregistré avec succès mais la livraison en arrière-plan reste **non fiable** (`push_status=expo_fallback_unreliable`).

Cause fréquente : **SHA-1 Play App Signing** absent ou incorrect dans Firebase Console.

| Référence | Valeur |
|-----------|--------|
| Package Android | `ch.liri.operations` |
| Projet Firebase | `driver-app-c0260` |
| Fichier mobile | `mobile/unified-app/google-services.json` |
| Flag mobile | `driver_fcm_native_enabled` / `EXPO_PUBLIC_ENABLE_DRIVER_FCM_NATIVE=1` |

---

## Prérequis — contexte chauffeur (obligatoire)

Avant toute vérification SHA-1 ou logcat FCM, exécuter le **STOP GATE FCM-7514-A** décrit dans [push-notifications-runbook.md](./push-notifications-runbook.md).

Sans contexte `driver`, `DriverNotificationsBridge` n’est pas monté → aucun token `provider=fcm` possible, même si Firebase est parfaitement configuré.

---

## Étape 1 — Récupérer le SHA-1 (Play Console)

1. [Google Play Console](https://play.google.com/console) → application **Lirie Operations**
2. **Release** → **Setup** → **App integrity**
3. Section **App signing key certificate** → copier **SHA-1 certificate fingerprint**

> **Important :** utiliser le certificat **App signing** (Google re-signe l’AAB), pas seulement la clé d’upload locale.

### SHA-1 upload (APK internes / EAS preview uniquement)

Si vous testez un **APK signé upload key** (pas Play Store) :

```bash
# Keystore EAS (credentials locales) ou keystore upload
keytool -list -v -keystore <chemin-keystore> -alias <alias>
```

Ajouter **aussi** ce SHA-1 dans Firebase si l’APK n’est pas passé par Play App Signing.

---

## Étape 2 — Enregistrer le SHA-1 dans Firebase

1. [Firebase Console](https://console.firebase.google.com/) → projet **`driver-app-c0260`**
2. ⚙ **Project settings** → onglet **Your apps**
3. App Android **`ch.liri.operations`**
4. **Add fingerprint** → coller le SHA-1 Play App Signing (format `AA:BB:CC:...`)
5. Enregistrer

Optionnel : ajouter le SHA-256 (même écran) — utile pour App Links, pas obligatoire pour FCM.

**Propagation :** 5–15 minutes en général. Pas besoin de republier l’AAB Play Store ; un **OTA** ou simple redémarrage app suffit côté client.

---

## Étape 3 — Vérifier côté appareil (logcat)

Sur le téléphone du chauffeur (USB / `adb`) :

```powershell
adb logcat -c
adb logcat -v time | Select-String 'driver.push.fcm|save-push-token|dev.expo.updates'
```

**Séquence attendue** (session chauffeur ouverte, notifications autorisées) :

| Événement | Signification |
|-----------|----------------|
| `driver.push.fcm.get_token_start` | Tentative FCM native |
| `driver.push.fcm.token` + `token_present=true` | Token FCM obtenu |
| POST `save-push-token` `provider=fcm` | Enregistrement backend OK |

**Échec typique SHA-1 / config Firebase :**

| Événement | Action |
|-----------|--------|
| `driver.push.fcm.unavailable` | Lire `reason` / `error_code` dans le log |
| Pas de `get_token_start` | Flag `driver_fcm_native_enabled` ou session non chauffeur |
| Token Expo seul en base | FCM non obtenu — reprendre étapes 1–2 |

Relance app : fermeture complète → rouvrir 2× si OTA récent.

---

## Étape 4 — Vérifier en base (script ops)

### Rapport chauffeur cible

```bash
# Dev local (Docker compose)
./scripts/verify-fcm-token-coverage.sh --driver-id 7514

# Prod (SSH)
docker exec atmr-backend-1 python scripts/verify_fcm_token_coverage.py --driver-id 7514
```

**Sortie attendue si OK :**

```json
"fcm_coverage": "fcm_native_ok",
"active_fcm_android_count": 1
```

**Sortie problème SHA-1 / FCM non obtenu :**

```json
"fcm_coverage": "android_expo_only",
"recommendations": [
  "Vérifier SHA-1 Play App Signing dans Firebase Console..."
]
```

### Gate CI / ops (exit code)

```bash
docker exec atmr-backend-1 python scripts/verify_fcm_token_coverage.py --driver-id 7514 --expect-fcm
# exit 0 = FCM OK, exit 1 = pas de token FCM Android actif
```

### Rapport flotte

```bash
./scripts/verify-fcm-token-coverage.sh --report
./scripts/verify-fcm-token-coverage.sh --android-expo-only
```

### SQL complémentaire

```sql
SELECT id, driver_id, provider, platform, is_active,
       LEFT(token, 12) AS token_prefix,
       created_at, updated_at, last_push_success_at, last_push_error_code
FROM device_tokens
WHERE driver_id = 7514
ORDER BY updated_at DESC;
```

Attendu : au moins une ligne `provider=fcm`, `platform=android`, `is_active=true`, préfixe token ≠ `ExponentPush`.

---

## Étape 5 — Test push bout en bout

1. App chauffeur : profil → **Déclencher test push**  
   ou `POST /api/v1/driver/me/test-push` (JWT chauffeur)
2. Vérifier réception **app tuée** (swipe kill → envoi test depuis admin / autre device)
3. Admin : `GET /api/v1/admin/push-coverage/drivers?driver_id=7514` → `push_status=operational`

---

## Checklist STOP GATE (chauffeur pilote)

- [ ] **Contexte chauffeur actif** (gate FCM-7514-A — pas entreprise)
- [ ] SHA-1 Play App Signing présent dans Firebase (`ch.liri.operations`)
- [ ] Logcat : `driver.push.fcm.token` avec `token_present=true`
- [ ] Base : `device_tokens.provider=fcm` actif pour Android
- [ ] `verify_fcm_token_coverage.py --driver-id XXX --expect-fcm` → exit 0
- [ ] Test push reçu app tuée
- [ ] `push_status=operational` (pas `expo_fallback_unreliable`)

---

## Dépannage rapide

| Symptôme | Cause probable | Correctif |
|----------|----------------|-----------|
| Expo token en base, pas FCM | SHA-1 manquant / mauvais certificat | Étape 1–2 |
| FCM en base, push échoue `SenderIdMismatchError` | `google-services.json` ≠ projet Firebase backend | Aligner projet + regénérer JSON |
| FCM en base, `token_unregistered` | Token révoqué / réinstall | Réouvrir app, ré-enregistrer |
| Bandeau notifications persistant | Disclosure locale perdue (`pm clear`) | OTA sync permissions (juin 2026) ou Continuer dans modale |

---

## Fichiers liés

- Runbook push : [push-notifications-runbook.md](./push-notifications-runbook.md)
- Script vérification : `backend/scripts/verify_fcm_token_coverage.py`
- Wrapper shell : `scripts/verify-fcm-token-coverage.sh`
- Mobile FCM : `mobile/unified-app/src/features/driver/firebaseMessaging.ts`
- Enregistrement token : `mobile/unified-app/src/core/notifications/registerPushToken.ts`
