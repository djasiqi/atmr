# Stop Gate S23 — Push Android chauffeur

Runbook de validation pour le correctif doublon notifications Android chauffeur.

## Prérequis bloquants (GO 1 / GO 2)

### GO 1 — Backend prod à jour

Commits requis sur l'image déployée :

| Commit | Contenu |
|--------|---------|
| `d5c61f25` | FCM Android data-only, sélection token unique |
| `0297c8eb` | Dedup Expo/FCM |

```bash
git merge-base --is-ancestor d5c61f25 HEAD && echo "GO 1 OK"
git merge-base --is-ancestor 0297c8eb HEAD && echo "GO 1 OK"
```

| Résultat | Preuve | Exécuté le | Exécuté par |
|----------|--------|------------|-------------|
| PASS / FAIL | SHA image prod ou sortie commande | | |

### GO 2 — Token chauffeur test (ex. driver 7514)

```sql
SELECT id, provider, platform, device_id, is_active, updated_at
FROM device_tokens
WHERE driver_id = 7514 AND is_active = true;
-- Attendu : provider=fcm, platform=android, is_active=true
```

```bash
docker exec atmr-backend-1 python3 /app/scripts/verify_fcm_token_coverage.py --driver-id 7514 --gate-json
```

| Résultat | Preuve | Exécuté le | Exécuté par |
|----------|--------|------------|-------------|
| PASS / FAIL | Sortie SQL + gate-json | | |

## Audit Notifee (post-refactor)

```bash
rg "notifee\.displayNotification" mobile/unified-app/src
```

Attendu : **1 seul hit** dans `pushLocalDisplay.ts` (scope chauffeur).

| Résultat | Preuve | Exécuté le | Exécuté par |
|----------|--------|------------|-------------|
| PASS / FAIL | Sortie `rg` | | |

## Preuves minimales par gate

Pour chaque gate, joindre :

```text
push_dispatch_id=...
push_sent_summary push_sent_count=...
push_display_local source_channel=fcm source=... dedupe_key=...
```

## Tableau de clôture incident

| Gate | Scénario | Résultat | Preuve (logs / capture) | Exécuté le | Exécuté par |
|------|----------|----------|-------------------------|------------|-------------|
| #1 | Foreground | PASS / FAIL | `push_dispatch_id`, `push_sent_summary`, `push_display_local` | | |
| #2 | Background verrouillé | PASS / FAIL | idem | | |
| #3 | Force-stop / swipe Samsung | PASS / FAIL | idem + `source=headless` ou `background` | | |
| #4 | Re-assign même mission < 5 min | PASS / FAIL | idem + `push_duplicate_skipped` attendu | | |
| #5 | FCM + Expo simultanés | PASS / FAIL | idem + `selected_devices=1`, `provider=fcm` | | |
| #6 | Missions A + B (~30s) | PASS / FAIL | 2× `push_dispatch_id`, 2× `push_display_local` | | |
| #7 | Multi-appareils S23 + S25 | PASS / FAIL | `push_sent_count=2`, 1 preuve par appareil | | |
| #8 | Reboot complet *(post-release)* | PASS / FAIL / N/A | idem + GO 2 post-reboot | | |

**Critère de clôture doublon :** gates **#1–#7 = PASS**. Gate **#8** recommandée post-release (N/A acceptable à la clôture initiale).

## Gate #3 — Samsung force-stop vs swipe away

| Méthode | Procédure S23 |
|---------|---------------|
| Swipe away | Récents → glisser l'app hors de la liste |
| Force-stop | Paramètres → Apps → Lirie Operations → Forcer l'arrêt |

Exécuter **les deux** pour le gate #3. Si force-stop ne délivre pas la notif : documenter comme limitation OEM (distinct du bug doublon).

## Gate #6 — Missions différentes

Assigner mission A, attendre ~30s, assigner mission B. Attendu : **2 notifications**, 2 `push_dispatch_id` distincts.

## Gate #7 — Multi-appareils FCM

Chauffeur avec S23 + S25 (2 tokens FCM actifs, `device_id` différents). Attendu : `selected_devices=2`, `push_sent_count=2`, 1 carte par appareil, 0 doublon par appareil.

## Gate #8 — Reboot (optionnel post-release)

S23 → reboot complet → connexion chauffeur → vérifier GO 2 → assigner mission. Attendu : `provider=fcm` actif, `push_sent_count=1`, 1 notification.

## Gate #9 — Réassignation (ancien chauffeur)

Réassigner une mission du chauffeur test vers un autre chauffeur. Attendu :

- 1× `push_display_local` sur l'**ancien** chauffeur avec `type=booking_reassigned` et `dedupe_key=booking:{id}:event:reassigned`
- Titre « Course réassignée » avec nom client (mode detailed)
- 0 notification sur le nouveau chauffeur en doublon avec l'assignation

| Résultat | Preuve | Exécuté le | Exécuté par |
|----------|--------|------------|-------------|
| PASS / FAIL | logs backend + capture tray ancien chauffeur | | |

## Correctif suivi incident (post-OTA `019eeb26`)

✅ **Implémenté** : correctif complémentaire doublon assignation + notification réassignation manquante.

- **Mobile** : handler background FCM n'affiche plus via le callback provider (`display: false`) — un seul `displayLocalDriverPush` ; clé stable `booking:{id}:event:assigned` prioritaire sur `event_id` différent ; verrou `inFlightDisplayKeys` anti-race.
  - Fichiers : `pushLocalDisplay.ts`, `notificationDedupStore.ts`, `firebaseMessaging.ts`, `NotificationsProvider.tsx`
- **Backend** : `fanout_driver_booking_reassigned` utilise `build_push_message(EVENT_REASSIGNED)` avec `dedupe_key`, `mission_id`, `event_id` ; handler charge le booking pour le corps métier.
  - Fichiers : `fanout.py`, `push_message_builder.py`, `driver_handlers.py`

**Reste à faire (ops)** : déployer backend + publier OTA mobile, puis exécuter gates #1–#9 sur S23.

## Livrable archivé

Runbook complété + extraits Celery/logcat + capture tray S23 sans doublon visible.
