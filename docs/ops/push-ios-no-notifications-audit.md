# Audit — iOS : aucune notification (Sujet 1)

**Statut officiel (2026-09-28, 22:35 Genève) :**

```text
PUSH iOS ROUTING          = PASS PROD
PUSH DELIVERY RELIABILITY = P0 FAIL / HOTFIX REQUIRED
DEVICE PROVIDER DEDUP     = FAIL
REAL-DEVICE VALIDATION    = PAUSED
```

Le routage FCM iOS (`d6f05831`) est en production depuis 21:33. La course 40500 à 22:20 a prouvé le routage (FCM iOS accepté) et le doublon Expo+FCM du même `device_id`. À 22:29–22:30, les notifications d'assignation et de réattribution ont été perdues : la tâche Celery meurt sur Prometheus (`eta_accuracy_rate`, `Incorrect label names`) après que le claim Redis est déjà consommé. Aucun token n'a été modifié. La validation appareil est en pause tant que ce hotfix n'est pas en production.

DELIVERY-06, assumé `at-least-once` : si le provider accepte puis que le processus meurt avant l'écriture de `SENT`, le retry peut renvoyer. Le même `notification_id` trace les deux acceptations. Ce n'est pas une perte de notification.

Le crash carte `AIRGoogleMap` / `LIRIE-MOBILE-1K` est un sujet séparé : `MAP CRASH = CODE FIXED / NEW iOS BUILD + DEVICE VALIDATION PENDING` (`5f7a8d87`). Le déploiement backend ne le valide pas.

Historique 2026-09-23 : `P0 — DEVICE TEST REQUIRED` (Phase A, pas de changement FCM/Expo/APNs avant `test-push`). Fiche : [`p0-mobile-session-push-evidence.md`](p0-mobile-session-push-evidence.md). `provider_accepted` n’est pas `mobile_received`.

## Gate — PUSH iOS, validation production

Aucune autre correction avant ce test. Ne pas modifier la ligne FCM à la main. Si le passage `android → ios` n’a pas lieu à l’étape 1, arrêter : le défaut est encore dans l’enregistrement du token, avant l’envoi.

```text
PUSH iOS — PRODUCTION VALIDATION GATE

Précondition
[PASS] backend déployé après 21:33 (Genève, 2026-09-28)
[PASS] ligne FCM iPhone encore platform=android (id 774, last_seen 21:22, avant la fin du déploiement)
[PASS] ligne non modifiée manuellement

Étape 1 — Ré-enregistrement
[ ] ouvrir Lirie sur l’iPhone après 21:33
[ ] vérifier nouveau last_seen_at
[ ] FCM : platform android → ios automatiquement
[ ] fournisseur reste fcm
[ ] token reste active

Étape 2 — Assignation
[ ] assigner une course à Emmenez Moi
[ ] S23 reçoit
[ ] iPhone reçoit

Étape 3 — Modification / statut
[ ] modifier la course ou son statut
[ ] S23 reçoit
[ ] iPhone reçoit

Étape 4 — États iOS
[ ] foreground
[ ] background
[ ] application fermée

Étape 5 — Preuves serveur
[ ] routage FCM = ios
[ ] payload comporte une alerte APNs visible
[ ] ticket Expo enregistré
[ ] receipt Expo récupéré
[ ] processed > 0
[ ] aucune erreur DeviceNotRegistered sur le token actuel
```

## Gate — crash carte iOS (séparé)

Le déploiement backend ne couvre pas ce sujet.

```text
MAP CRASH — IOS BUILD GATE

[ ] nouvelle build contenant 5f7a8d87
[ ] installée sur iPhone 16 Pro Max
[ ] même parcours carte que lors du crash de 14:38
[ ] répétition navigation / ouverture / fermeture carte
[ ] aucun nouveau LIRIE-MOBILE-1K
[ ] aucun NSInvalidArgumentException AIRGoogleMap
```

## Symptôme

Android reçoit correctement les notifications ; iOS n’en reçoit aucune.

## Principe

- Phase A = **diagnostic + observabilité**, sans modification de routage FCM/Expo.
- Ne pas confondre : file Celery ≠ acceptation provider ≠ réception mobile ≠ ouverture ≠ ACK métier.
- `business_acknowledged` est la seule preuve fiable de prise en charge d’une assignation urgente.
- Un `message_id` FCM ou un ticket Expo `ok` = acceptation technique uniquement.

## Causes les plus probables

1. Token iOS incorrect ou appartenant à un autre projet Firebase
2. Clé APNs absente, expirée ou mal associée dans Firebase
3. Bundle ID incorrect / non aligné
4. `aps-environment` development vs production incompatible
5. Build TestFlight avec config Firebase différente du debug
6. Token enregistré sous le mauvais provider
7. Notification envoyée avant contexte chauffeur `ready`
8. Payload iOS invalide ou silencieux involontaire

Le double chemin FCM+Expo peut provoquer doubles notifs / observabilité confuse, mais n’explique pas naturellement l’**absence totale**.

## Checklist audit bout-en-bout

- [ ] Génération / enregistrement token APNs via FCM (`provider=fcm`, `platform=ios`)
- [ ] Transmission `POST /driver/save-push-token` (contexte chauffeur `ready`)
- [ ] Association token → chauffeur / entreprise / `device_id`
- [ ] Environnement APNs (`aps-environment`) identifié pour le build
- [ ] Clé APNs (`.p8`) vérifiée dans Firebase Console
- [ ] Capabilities : Push Notifications + Background Modes → Remote notifications
- [ ] Format payload iOS (alert vs background / `content-available`)
- [ ] Permission notification accordée sur l’iPhone
- [ ] Traitement FG / BG / app tuée
- [ ] Nettoyage tokens invalides (`invalid_token` confirmé uniquement)

## Fiche preuve (par build testé)

Renseigner pour **development** et **TestFlight/production** :

```text
app_version:
build_number:
bundle_id:
firebase_project_id:
aps_environment:
installation_id (= device_id):
token_created_at:
token_last_seen_at:
last_successful_provider_acceptance_at:
last_mobile_received_at:
```

## Procédure test-push forcé

```bash
# FCM uniquement (chauffeur authentifié)
POST /api/v1/driver/me/test-push
{ "provider": "fcm" }

# Expo uniquement
POST /api/v1/driver/me/test-push
{ "provider": "expo" }
```

Suivre le `correlation_id` retourné dans les logs :

```text
création test → sélection device → tentative provider → réponse provider
→ receipt Expo (si applicable) → mobile_received → mobile_opened
```

## Statuts canoniques (`delivery_status`)

```text
queued | provider_accepted | provider_rejected | configuration_error
| invalid_token | retry_pending | failed | mobile_received | mobile_opened
| business_acknowledged
```

Champ séparé Expo : `provider_receipt_status` = `pending | ok | error | not_applicable`.

Aliases dashboards uniquement (jamais en nouveau stockage) : `sent`→`provider_accepted`, `rejected`→`provider_rejected`.

## Rétention

| Canal | Rétention |
|-------|-----------|
| Logs détaillés par tentative | 30 jours |
| Preuves test-push | 90 jours |
| Agrégats métriques | durée monitoring standard |

Prometheus — labels autorisés uniquement : `platform`, `provider`, `delivery_status`, `notification_type`, `error_category`.

## Gate de sortie Phase A

| Élément | Résultat exigé |
|---------|----------------|
| Token FCM iOS actif | prouvé |
| Token Expo iOS actif | prouvé si utilisé |
| `device_id` stable | prouvé |
| Projet Firebase | identifié |
| Bundle ID | aligné |
| `aps-environment` | identifié |
| Clé APNs Firebase | vérifiée manuellement |
| Test FCM forcé | résultat documenté |
| Test Expo forcé | résultat documenté |
| Logs corrélés | disponibles |
| Réception / ouverture | observée ou absence expliquée |
| Cause racine | identifiée, ou hypothèses ouvertes listées |
| Correctif | appliqué et retesté si cause identifiée |

## Phases suivantes (conditionnelles)

- **Phase B** : `IOS_NATIVE_FCM_PREFERRED` — fallback Expo uniquement sur `failure_before_send`.
- **Phase C** : `IOS_DISABLE_EXPO_ON_FCM_UPSERT` — convergence scoped `device_id`.
- **Phase D** : NO-GO suppression Expo tant que versions actives non mesurées.

## Journal de preuve terrain

_À remplir lors des tests appareil réel (dev + TestFlight/prod)._
_Statut 2026-09-23 : journal toujours vide. Gate Phase A non close. Utiliser la fiche `=== INCIDENT PUSH IOS ===` de [`p0-mobile-session-push-evidence.md`](p0-mobile-session-push-evidence.md)._

### Build development

- Date :
- Résultat FCM forcé :
- Résultat Expo forcé :
- Cause racine / hypothèses ouvertes :

### Build TestFlight / production

- Date :
- Résultat FCM forcé :
- Résultat Expo forcé :
- Cause racine / hypothèses ouvertes :

## ✅ Implémenté

- Document d’audit + lien depuis le runbook push.
- Module [`backend/services/notifications/push_delivery_status.py`](../../backend/services/notifications/push_delivery_status.py) : statuts canoniques, classification, sanitization, `deduplication_key`.
- Envoi push enrichi ([`push.py`](../../backend/services/notifications/push.py)) : logs `[push_attempt]`, champs provider normalisés, politique retry Phase A (`PUSH_RETRY_DELAYS_SEC`, `failed`/`retry_exhausted`).
- `SENDER_ID_MISMATCH` → `configuration_error` sans désactivation token ([`firebase_push.py`](../../backend/services/notifications/firebase_push.py), [`device_token_lifecycle.py`](../../backend/services/notifications/device_token_lifecycle.py)).
- Receipts Expo : stockage ticket Redis + tâche Celery `notifications.fetch_expo_push_receipts` ([`expo_receipts.py`](../../backend/services/notifications/expo_receipts.py)).
- `POST /driver/me/test-push` : `provider=fcm|expo`, `correlation_id`, ACL `device_token_id` scoped chauffeur.
- ACK mobile : `mobile_received` / `mobile_opened` via `ack_kind`.
- Flags Phase B/C off par défaut ([`ios_push_flags.py`](../../backend/services/notifications/ios_push_flags.py)) ; sélection / upsert conditionnels.
- Mobile : `deduplication_key` reconnu dans `buildStableDedupeKey`.
- Métrique Prometheus `push_attempt_status_total` (cardinalité contrôlée).
- Tests unitaires backend + mobile.

### Reste à faire (gate de sortie Phase A — ops / terrain)

- ✅ **Statut 2026-09-28** : routage push iOS `DEPLOYED / REAL-DEVICE VALIDATION PENDING`. La fiche de collecte reste [`p0-mobile-session-push-evidence.md`](p0-mobile-session-push-evidence.md). Le gate Phase A de 2026-09-23 (clé APNs, `test-push` forcé) n’est pas clos par ce déploiement.
- Exécuter sur un iPhone réel `POST /api/v1/driver/me/test-push` avec `{"provider":"fcm"}` puis `{"provider":"expo"}`, et distinguer `provider_accepted` de `mobile_received`.
- Vérifier manuellement la clé APNs Firebase, Bundle ID, `aps-environment` du binaire TestFlight.
- Activer Phase B/C uniquement après preuve FCM positive sur l’appareil.
