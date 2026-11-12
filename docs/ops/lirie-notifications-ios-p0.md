# LIRIE — NOTIF-IOS-P0

Fiabilité des notifications push iOS — diagnostic vs audit sept. 2026, correctifs ciblés, tests locaux.

**Date :** 2026-11-12  
**Branche :** `candidate/lirie-notif-ios-p0-20261112`  
**Base :** `origin/candidate/lirie-fin-caisse-20261112` @ `1f12a9db`  
**Worktree :** `D:\lirie-backups\_candidate_lirie_notif_ios_p0_20261112`  
**WIP `c:\Users\jasiq\atmr` :** non modifié  
**Merge / deploy / flags ops :** non effectués  

---

## Verdicts

| Gate | Résultat |
|------|----------|
| PUSH BACKEND | **PASS** (code + tests locaux) |
| PUSH MOBILE | **PARTIAL** (contrat enregistrement/logout OK ; pas de Jest device ; FG iOS FCM non prouvé) |
| IOS DEVICE DELIVERY | **NOT VERIFIED** |
| SECURITY | **PASS** (isolation cibles + privacy + dédup destinataires ; pas d’envoi à des comptes réels) |
| READY FOR REMOTE CI | **YES** (commits locaux prêts ; aucun push effectué) |

---

## A. État de l'anomalie historique

Audit [`push-ios-no-notifications-audit.md`](push-ios-no-notifications-audit.md) (2026-09-28) :

1. **Crash Celery Prometheus** (`eta_accuracy_rate` / Incorrect label names) après claim Redis → perte d’envoi.
2. **Doublon Expo + FCM** même `device_id` iOS à l’envoi chauffeur.
3. **Validation appareil** en pause.
4. Routage FCM iOS (`android` → `ios`) déjà **PASS** en prod ce jour-là.

Confrontation nov. 2026 (base FIN-CAISSE `1f12a9db`) :

| Incident sept. | État code nov. | Preuve |
|----------------|----------------|--------|
| Prometheus fatale | **Corrigé avant NOTIF-IOS-P0** | `818355ba`, `test_push_metrics_non_fatal.py` |
| Lease / claim Redis | **Corrigé avant** | `fe07b2ac`, `test_push_delivery_lease.py` |
| Dédup iOS driver | **Corrigé avant** | `5f720740`, `prepare_driver_push_targets` |
| Routage platform iOS | **Corrigé avant** | `d6f05831`, `infer_fcm_platform` |
| Dédup iOS **company** + **Kafka consumer** | **Manquait** → corrigé ici | `company_push.py`, `kafka_consumer.py` |
| Logout body `device_installation_id` | **Trous** → corrigé ici | `auth.py` + `client.ts` |
| Réception iPhone | **Toujours ouverte** | aucun journal terrain nov. |

L’audit sept. n’est **pas** une preuve de panne nov. pour les deux P0 code driver déjà hotfixes ; il reste une preuve d’absence de validation appareil.

---

## B. Causes encore présentes (avant correctifs NOTIF) / risques résiduels

Confirmées et traitées dans ce lot :

1. `send_push_to_company_sync` envoyait **tous** les `DeviceToken` actifs (Expo+FCM iOS) sans `prepare_push_targets`.
2. Consumer Kafka chauffeur même écart (contournait la sélection Celery).
3. Logout mobile envoyait `device_installation_id` mais le backend n’acceptait que `device_id` / `deviceId` / header `X-Device-ID` pour l’invalidation push.

Risques **non corrigés** (hors scope rewrite / flags) :

- Phase C `IOS_DISABLE_EXPO_ON_FCM_UPSERT` **off** : lignes Expo+FCM restent actives en DB (dédup à la sélection seulement).
- Lifecycle tokens `PUSH_DEVICE_TOKEN_LIFECYCLE_ENABLED` **off** par défaut.
- Affichage foreground FCM iOS via fallback Expo `scheduleNotificationAsync` (Android Notifee) — non prouvé sur device.
- Pas d’endpoint `save-push-token` **client** dédié (notifications client plutôt portail/inbox).
- Docs ops / evidence journal encore partiellement figés à sept. (bannière audit alignée ; journal terrain vide).

---

## C. Corrections réalisées

1. **`prepare_push_targets`** (générique owner) + alias `prepare_driver_push_targets` — `backend/services/notifications/push_device_selection.py`
2. **Company push** utilise `prepare_push_targets` — `backend/services/notifications/company_push.py`
3. **Kafka consumer** utilise `prepare_push_targets` — `backend/services/notifications/kafka_consumer.py`
4. **Logout** accepte `device_installation_id` pour invalidation push — `backend/routes/auth.py`
5. **Mobile logout** envoie aussi `device_id` (= installation id) — `mobile/unified-app/src/core/api/client.ts`
6. Bannière audit ops mise à jour — `docs/ops/push-ios-no-notifications-audit.md`

Pas de réécriture du système push. Pas d’activation de flags Phase B/C.

---

## D. Tests exécutés

Exécution Docker (image snapshot `atmr-api-pytest-snap:notif-ios-p0`, réseau `atmr-stack`, mount worktree `/app`) :

**Batch 1 — 23 passed**

- `test_push_device_selection.py` (dont nouveau `test_prepare_push_targets_company_ios_dedup`)
- `test_company_push_ios_dedup.py` (nouveau)
- `test_push_metrics_non_fatal.py`
- `test_upsert_device_token.py`

**Batch 2 — 52 passed, 1 failed puis corrigé**

- `test_push_delivery_lease.py` ………
- `test_push_privacy_access_control.py` ….
- `test_notification_targets.py` (matrice A/B/C métier cibles)
- `test_ios_push_flags.py` / `test_ios_fcm_platform_and_expo_tickets.py`
- `test_logout_push_device_installation_id.py` → **PASS** après fix monkeypatch

**Total prouvé localement :** ≥ 75 tests verts sur le périmètre push/logout/isolation cibles.

Aucun envoi push à des chauffeurs/clients réels. Aucun `test-push` FCM/Expo terrain.

---

## E. Résultats des scénarios métier

| ID | Scénario | Résultat |
|----|----------|----------|
| A | Nouvelle mission → Chauffeur | **BACKEND PASS** (matrice `assigned_*` + pipeline driver) — device **NOT VERIFIED** |
| B | Modification mission | **BACKEND PASS** (`updated_*` targets) — device **NOT VERIFIED** |
| C | Annulation transport | **BACKEND PASS** (`cancelled_*`) — device **NOT VERIFIED** |
| D | Notification Entreprise | **BACKEND PASS** (company push + dédup iOS ajoutée) — device **NOT VERIFIED** |
| E | Notification Client | **PARTIAL** (inbox/portail ; pas de chaîne push mobile client équivalente) |
| F | Connecté puis déconnecté | **BACKEND+MOBILE CONTRACT PASS** (invalidation via `device_id` / `device_installation_id` / header) — device **NOT VERIFIED** |
| G | Plusieurs appareils / compte | **BACKEND PASS** (1 cible / `device_id` ; multi-device = multi-cibles) |
| H | Réinstall / renew token | **CONTRACT PASS** (upsert + listeners renew code) — device **NOT VERIFIED** |
| I | Événement rejoué / idempotence | **BACKEND PASS** (lease SENT skip + tests DELIVERY) |
| J | Isolation inter-entreprises | **BACKEND PASS** (privacy access control + targets ; self-notify company guard) |

---

## F. État des builds iOS

- Bundle : `ch.liri.operations` ; `app.json` buildNumber **72** / version **1.0.14**
- EAS : profils `development` / canary avec `EXPO_PUBLIC_ENABLE_DRIVER_PUSH=1`, `EXPO_PUBLIC_ENABLE_DRIVER_FCM_NATIVE=1`, `EXPO_PUBLIC_ENABLE_COMPANY_PUSH=1`
- `UIBackgroundModes` inclut `remote-notification`
- **Aucune nouvelle build iOS produite** dans ce lot
- Credentials APNs / Firebase : non exportés, non inspectés en clair

---

## G. Validation appareil

```text
IOS DEVICE DELIVERY = NOT VERIFIED
DELIVERY DEVICE NOT VERIFIED
```

Aucun iPhone de test + build alignée n’a été utilisé ici.  
`provider_accepted` ≠ preuve de réception.  
Ne pas annoncer **PUSH IOS PASS END-TO-END**.

Gate terrain recommandé (synthé uniquement) :

1. `POST /driver/me/test-push` `fcm` puis `expo` sur compte de test
2. Tracer `correlation_id` → tray iOS → open → page mission
3. Répéter company save-push-token + test équivalent si endpoint dispo

---

## H. Compatibilité MFA (#104)

Diff MFA tip (`626de1c3`) vs base FIN-CAISSE sur chemins push services / `registerPushToken` / bridges : **vide**.

MFA touche surtout `backend/routes/auth.py`, `sessionProvider.tsx`, `client.ts` (login 202 + sessions métier).

Impacts à anticiper au stacking (sans merge effectué) :

- Enregistrement push **après** `issue_business_access_token` (pas pendant interrupt MFA).
- Reprise du correctif logout `device_installation_id` / `body.device_id` lors du rebase MFA : le tip MFA **n’a pas encore** l’alias `device_installation_id` sur l’invalidation push (mêmes lignes historiques `device_id`/`deviceId`/header seulement).

Vérification : **statique PASS** (pas de conflit code push) ; **intégration runtime MFA×push : non exécutée** dans ce lot.

---

## I. Commits

Locaux uniquement (aucun `git push`) :

| SHA | Message |
|-----|---------|
| `bcd1b842` | `fix(push-ios): dedup company/kafka et logout device_installation_id` |
| `67747005` | `docs(ops): rapport NOTIF-IOS-P0 et alignement audit iOS` |

Branche ahead de `origin/candidate/lirie-fin-caisse-20261112` de **2** commits.

---

## J. Points restant à traiter

1. Validation réception **appareil iOS réel** (FCM puis Expo) + journal evidence.
2. Porter les alias logout vers la branche MFA au moment du stack.
3. Ne pas activer Phase B/C / lifecycle tokens avant preuve FCM positive.
4. Optionnel après preuve : ne plus poster Expo iOS si FCM OK (réduit bruit DB).
5. Client mobile push dédié si produit l’exige.
6. CI distante : ouvrir Draft PR **après** validation humaine (non fait ici).

---

## Restrictions respectées

- Aucun merge des Drafts #101–#104  
- Aucun déploiement / migration prod / flag ops ON  
- Aucun travail PAY-SAFERPAY  
- WIP principal préservé  
- Pas d’audit général hors notifications iOS  
