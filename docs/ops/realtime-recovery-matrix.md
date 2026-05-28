# Realtime Recovery Matrix — gate D3 avant PR C ≥5 % et PR D

**Statut** : audit cartographie initial — 2026-05-28
**Objectif** : garantir qu'après perte WS, reconnect, foreground iOS, Android Doze, polling cassé ou relay drop, le mobile peut reconstruire un état cohérent via REST.

---

## 1. Synthèse exécutive


| Aspect                                                    | État                                                                                                                                                                         |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bridges Socket.IO                                         | 2 (driver + company) — aucun bridge passenger/admin/client/institution                                                                                                       |
| Reconnect WS                                              | Driver = manuel `scheduleReconnect()`. Company = auto socket.io (20 tentatives, 300 ms→8 s)                                                                                  |
| AppState foreground                                       | Driver = `runtimeResume` (token + resync + flush). Company = reconnect socket si idle/failed (pas de resync métier)                                                          |
| NetInfo → reconnect WS                                    | ❌ Aucun listener NetInfo ne force un reconnect socket                                                                                                                        |
| Watchdog staleness                                        | Company freshness 60 s/5 min (dispatch event). Fleet map silence 120 s → refetch GPS                                                                                         |
| `connection.authority` event                              | ❌ Émis par ws-service, **jamais consommé** côté mobile                                                                                                                       |
| `presence_heartbeat` event                                | ❌ **Jamais émis** côté mobile                                                                                                                                                |
| Ack `event_ack_batch`                                     | ✅ Driver + company, événements `CRITICAL_ACK_EVENTS`                                                                                                                         |
| Dispatch events (`dispatch_assignment`, `dispatch_run_*`) | ❌ **Listés dans CRITICAL_ACK_EVENTS et CRITICAL_EVENT_TYPES mais ABSENTS de `CONTEXT_REALTIME_CHANNELS.company`** → bridge ne s'y abonne pas → events silencieusement perdus |
| `company_data_stale_resync`                               | ⚠️ Dispatché par bridge mais **consommé nulle part** (sauf GPS via autre event)                                                                                              |
| `company_socket_reconnected`                              | ✅ Refetch GPS snapshot. ❌ Pas de refetch dispatch/missions/chat                                                                                                              |
| Endpoints REST snapshot                                   | ✅ Existent pour GPS, dispatch list/detail, chat, missions, inbox                                                                                                             |


**Verdict** : 4 gaps bloquants identifiés (G1–G4) avant PR C 20–50 % et PR D.

---

## 2. Architecture par couche

### Couche 1 — Bridges Socket.IO


| Bridge  | Fichier                                                          | Auth                                    | Transport                                                                                             | Reconnect                                          | Headers canary                    |
| ------- | ---------------------------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------- | --------------------------------- |
| Driver  | `src/core/realtime/realtimeManager.ts:238-805`                   | JWT via auth handshake                  | `["websocket","polling"]`                                                                             | Manuel `scheduleReconnect()`, `reconnection:false` | ✅ via `getWsCanaryExtraHeaders()` |
| Company | `src/features/company/realtime/companyRealtimeBridge.ts:470-722` | JWT auth + `Authorization` header natif | Web `["websocket","polling"]`, natif `["websocket"]` puis fallback `["polling"]` avec `upgrade:false` | Auto socket.io 20 tentatives 300 ms→8 s            | ✅ via `getWsCanaryExtraHeaders()` |


### Couche 2 — AppState / Foreground


| Surface | Hook                                                                       | Comportement foreground                                                                                                                             |
| ------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Driver  | `src/features/driver/runtimeResume.ts:29-105`                              | Token refresh → `realtimeManager.connect()` si flag → `reconcileDriverMissions` → flush offline queue → flush tracking → invalidation mission scope |
| Driver  | `src/features/driver/services/syncEngine.ts:213-245`                       | `scheduleDriverMissionSync(..., "foreground")` avec guard anti-burst                                                                                |
| Company | `app/(app)/(company)/_layout.tsx:58-75`                                    | Si bridge `idle` → `connect()`. Si `failed`/`reconnecting` → `reconnect()`. **Pas de resync métier**.                                               |
| Company | `src/features/company/dashboard/useCompanyDashboardScreenModel.ts:511-524` | Refetch dashboard **uniquement si stale OU socket pas healthy/fresh**                                                                               |


### Couche 3 — Watchdog / Staleness


| Source                                    | Seuil                       | Action                                                       |
| ----------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| `companyRealtimeState.ts:15-18`           | fresh < 60 s, stale ≥ 5 min | Patch snapshot `dataFreshness`                               |
| `companyRealtimeBridge.ts:407-443`        | Tick 30 s                   | Si nouvelle staleness → dispatch `company_data_stale_resync` |
| `useCompanyDriverLiveTracking.ts:394-403` | GPS silence 120 s           | Refetch `/companies/me/drivers/locations`                    |
| `driver/hooks.ts:297-319`                 | Heartbeat 60 s              | **Local telemetry uniquement, pas d'emit WS**                |


### Couche 4 — TanStack Query invalidation


| WS event                                             | Action invalidation                                                                                    | Source                                   |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| `booking_updated` (missionId)                        | invalide `rideDetails(missionId)` + `dashboard`                                                        | `company/hooks.ts:366-397`               |
| `booking_updated` (sans missionId)                   | invalide `dashboard` + `missions` + `dispatch-delays` + `optimizer`                                    | `company/hooks.ts:400-444`               |
| `booking_created`, `urgent_alert`                    | invalide `dashboard` + `missions` + `inbox`                                                            | `company/hooks.ts:361-363`               |
| `team_chat_message`                                  | patch local hub si flag, sinon invalide unread/threads/messages                                        | `company/hooks.ts` + `messages/hooks.ts` |
| `driver_location_update`, `driver_live_state_update` | invalidation `driversLocations` batch 500 ms + patch UI cache map                                      | `useCompanyDriverLiveTracking.ts`        |
| `dispatch_assignment`                                | ❌ **Aucune** — event jamais reçu (cf. G1)                                                              |                                          |
| `dispatch_run_started`                               | ❌ **Aucune** — event jamais reçu (cf. G1)                                                              |                                          |
| `dispatch_run_completed`                             | ❌ **Aucune** — event jamais reçu (cf. G1)                                                              |                                          |
| `dispatch_run_failed`                                | ❌ **Aucune** — event jamais reçu (cf. G1)                                                              |                                          |
| `company_data_stale_resync`                          | ❌ **Aucun consumer hors GPS** (cf. G2)                                                                 |                                          |
| `company_socket_reconnected`                         | ✅ Refetch GPS snapshot. ❌ Pas de refetch dispatch/missions/chat (cf. G3)                               |                                          |
| `mission_updated` driver                             | Patch cache `driverQueryKeys.missions(contextId)`. Sur ordering douteux → invalidation detail + resync | `driver/realtime.ts:130-213`             |


### Couche 5 — Endpoints REST snapshot disponibles


| Feature                | Canonique                                             | Fallback                 | Fichier                    |
| ---------------------- | ----------------------------------------------------- | ------------------------ | -------------------------- |
| GPS company            | `GET /api/v1/companies/me/drivers/locations`          | `/drivers/live`          | `companyApi.ts:454-606`    |
| Dispatch list company  | `GET /api/v1/company_mobile/dispatch/v1/rides`        | `/dispatch/v1/rides`     | `companyApi.ts`            |
| Dispatch detail        | `/company_mobile/dispatch/v1/rides/:id`               | `/dispatch/v1/rides/:id` | `companyApi.ts`            |
| Dashboard realtime     | `/company_mobile/dispatch/v1/dashboard/realtime`      | `/dispatch/v1/...`       | `companyApi.ts`            |
| Dispatch delays        | `/dispatch/v1/delays`                                 | `companyApi.ts:646-678`  |                            |
| Chat inbox             | `/conversations/inbox`                                | —                        | `companyInboxApi.ts:25-55` |
| Chat thread messages   | `/messages/:companyId/hub/threads/:threadId/messages` | —                        | `messages/api.ts`          |
| Dispatch chat messages | `/company_mobile/dispatch/v1/chat/messages`           | —                        | `companyApi.ts:1361-1429`  |
| Notifications          | `/companies/notifications`                            | —                        | `companyInboxApi.ts`       |
| Driver missions        | `/driver/me/bookings` + `/driver/me/bookings/since`   | —                        | `driver/api.ts:159-180`    |
| Driver mission detail  | `/driver/me/bookings/:id`                             | —                        | `driver/api.ts`            |


### Couche 6 — Monitoring frontend


| Métrique                                          | État                                                            |
| ------------------------------------------------- | --------------------------------------------------------------- |
| Sentry init                                       | ✅ via `MonitoringProvider.tsx` si DSN défini                    |
| Socket connect/disconnect/reconnect_failed driver | ✅ `runtimeStabilityMetrics.ts`                                  |
| Reconnect attempts company                        | ✅ `realtimeMetrics.ts`                                          |
| Map setState/sec, GPS batch flush/coalesce        | ✅ `perfKpi.ts` + `realtimeMetrics.ts`                           |
| `stale_duration` (s) frontend                     | ❌ Manquant                                                      |
| `resync_count` par feature                        | ❌ Manquant                                                      |
| Reporting Sentry sur perte connexion              | ⚠️ Via telemetry ingest seulement, pas `captureException` dédié |


### Couche 7 — Feature flags Phase 2


| Flag                                         | Source                              | Comportement                  |
| -------------------------------------------- | ----------------------------------- | ----------------------------- |
| `ws_service_canary`                          | env `EXPO_PUBLIC_WS_CANARY_ENABLED` | Active header `X-WS-Canary`   |
| `realtime_socket_enabled`                    | env/runtime                         | On par défaut en dev          |
| `company_realtime_enabled`                   | env/runtime                         | Auto si dispatch actif        |
| `realtime_auth_reconnect_enabled`            | runtime                             | Reconnect après refresh token |
| `realtime_reconnect_circuit_breaker_enabled` | runtime                             | Circuit breaker reconnect     |
| `realtime_resync_transition_gate_enabled`    | runtime                             | Gate transition resync        |
| Manquant : `connection_authority_telemetry`  | —                                   | Pas de flag, pas de consumer  |


---

## 3. Matrice par feature


| Feature                    | WS event reçu                                                                      | Stale detect                                 | Reconnect resync                       | Foreground resync         | REST snapshot                    | Verdict           |
| -------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------- | -------------------------------------- | ------------------------- | -------------------------------- | ----------------- |
| **GPS drivers**            | `driver_location_update`, `driver_live_state_update`, `company_socket_reconnected` | ✅ 120 s map silence                          | ✅ Refetch sur reconnect                | ⚠️ Refresh stale-only     | `/drivers/locations`             | **OK**            |
| **Bookings list/detail**   | `booking_updated`, `booking_created`, `booking_cancelled`                          | ⚠️ Indirect via company freshness            | ❌ Pas de refetch missions              | ⚠️ Stale-only             | `/dispatch/v1/rides` + detail    | **PARTIEL**       |
| **Dispatch assignment**    | ❌ `dispatch_assignment` non écouté                                                 | ❌                                            | ❌                                      | ❌                         | `/dispatch/v1/rides`             | **❌ BLOQUANT G1** |
| **Dispatch run lifecycle** | ❌ `dispatch_run_started/completed/failed` non écoutés                              | ❌                                            | ❌                                      | ❌                         | `/dispatch/v1/status`            | **❌ BLOQUANT G1** |
| **Team chat**              | `team_chat_message`, `conversation_message`                                        | ⚠️ Polling unread 15 s si socket pas healthy | ❌ Pas de refetch threads sur reconnect | ✅ Refetch on focus        | `/conversations/inbox` + threads | **PARTIEL**       |
| **Urgent alert**           | `urgent_alert`                                                                     | ❌                                            | ❌                                      | ⚠️ Stale-only             | inbox + missions                 | **PARTIEL**       |
| **Mission driver**         | `mission_updated`, `mission_assigned`                                              | ⚠️ Ordering/gap detect                       | ✅ `scheduleDriverMissionSync`          | ✅ `runtimeResume` complet | `/driver/me/bookings/since`      | **OK (driver)**   |
| **Notifications**          | Inbox indirect via invalidation                                                    | ❌                                            | ❌                                      | ❌                         | `/companies/notifications`       | **PARTIEL**       |
| **Delay invalidation**     | `delay_invalidated`                                                                | ❌                                            | ❌                                      | ❌                         | `/dispatch/v1/delays`            | **PARTIEL**       |


---

## 4. Gaps identifiés (priorité bloquant PR D)

### G1 — Dispatch events jamais écoutés côté mobile **[CRITIQUE — bloquant PR C 20% et PR D]**

**Faits** :

- Backend `event_contract.py` : `dispatch_assignment`, `dispatch_run_started/completed/failed` sont `CRITICAL_EVENT_TYPES`
- Mobile `wsCanary.ts` : ces events sont dans `CRITICAL_ACK_EVENTS` → mobile s'attend à les recevoir et à les acker
- Mobile `contextRegistry.ts:10-24` : `CONTEXT_REALTIME_CHANNELS.company` ne les contient pas
- Mobile `companyRealtimeBridge.ts:445-460` : `SOCKET_EVENTS.forEach((eventName) => socket.on(...))` → ne s'abonne qu'aux events de `CONTEXT_REALTIME_CHANNELS.company`

**Conséquences** :

- Dispatcher émet `dispatch_assignment` → mobile ne le reçoit jamais → cache UI désynchronisé
- Pas d'ack envoyé → ws-service incrémente `confirmed_critical_miss` → **faux positifs métriques qui peuvent déclencher rollback canary** alors que tout est OK
- Le pipeline E2E latency monitoring est **structurellement biaisé**

**Fix minimal** :

```typescript
// src/core/realtime/contextRegistry.ts
company: [
  "company_dispatch_update",
  "new_booking",
  "booking_updated",
  "booking_cancelled",
  "booking_message",
  "booking_message_sent",
  "team_chat_message",
  "conversation_message",
  "urgent_alert",
  "driver_location_update",
  "driver_live_state_update",
  "optimizer_status_changed",
  "delay_invalidated",
  // Phase 2 PR B/C — gate D3
  "dispatch_assignment",
  "dispatch_run_started",
  "dispatch_run_completed",
  "dispatch_run_failed",
],
```

Et ajout du mapping `invalidateCompanyQueriesForEvent` pour ces events (probablement vers dashboard + missions, et detail si `missionId` présent).

**Effort** : 1 PR ≈ 2 h (registry + invalidation + test).

---

### G2 — `company_data_stale_resync` jamais consommé **[BLOQUANT PR C ≥5 %]**

**Faits** :

- `companyRealtimeBridge.ts:407-443` : si données stale (5 min sans event), dispatch `company_data_stale_resync` via `contextRealtimeRouter`
- Recherche `company_data_stale_resync` consumer → 0 résultat hors le dispatcher

**Conséquences** :

- Si WS reste silencieux 5 min (background long, polling cassé, relay down) → mobile ne sait pas qu'il faut refetch dispatch/missions/chat/notifications
- L'utilisateur voit des données obsolètes sans signal visuel
- Le watchdog backend Phase 2 (dedup, GPS) ne suffit pas à compenser

**Fix minimal** :

- Hook `useCompanyStaleResyncListener(contextId)` qui :
  - écoute `company_data_stale_resync` via `contextRealtimeRouter`
  - invalide dashboard + missions + inbox + dispatch-delays
  - log `realtimeMetrics.staleResyncTriggered` (nouveau compteur)

**Effort** : 1 PR ≈ 1.5 h.

---

### G3 — Reconnect ne déclenche que GPS resync (pas dispatch/chat/missions) **[BLOQUANT PR D]**

**Faits** :

- `useCompanyDriverLiveTracking.ts:358-360` consomme `company_socket_reconnected` → refetch GPS uniquement
- Aucun consumer ne refetch dispatch/missions/chat/inbox sur reconnect

**Conséquences** :

- Background iOS → app relance → socket reconnect → GPS revient mais dispatch peut être obsolète si event raté pendant background
- Le bridge company émet `connect` → mobile reçoit `company_socket_reconnected` event SEULEMENT pour GPS → **autres caches non rafraîchis**

**Fix minimal** :

- Étendre le hook G2 (ou ajouter en parallèle) : `useCompanyReconnectResyncListener` qui invalide dashboard + missions + inbox + threads chat sur `company_socket_reconnected`
- Throttling : pas plus d'une invalidation toutes les 30 s pour éviter storm

**Effort** : 1 PR ≈ 1.5 h.

---

### G4 — `connection.authority` jamais consommé **[OBSERVABILITÉ — pas bloquant mais à faire avant 20 %]**

**Faits** :

- ws-service émet `connection.authority` à chaque connect avec `{authority: "ws-service", canary: true, version}`
- Aucun consumer mobile

**Conséquences** :

- Impossible de savoir côté mobile sur quel chemin il est (backend vs ws-service)
- Impossible de tagger Sentry / telemetry avec `authority` pour faire des comparaisons cross-population
- Le pipeline `confirmed_critical_miss` ne peut pas être segmenté par chemin

**Fix minimal** :

- Consumer `connection.authority` dans les 2 bridges
- Set tag Sentry `realtime.authority`
- Émettre metric `realtime.authority_observed_total{authority=...}`

**Effort** : 1 PR ≈ 1 h.

---

## 5. Plan d'implémentation B (proposition pour validation)

Découpage en 3 PRs ciblées, indépendantes, rollback rapide :

### PR mobile-D3.1 — Fix G1 (dispatch events) **[BLOQUANT PR C ≥5 %]**

- `contextRegistry.ts` : ajouter les 4 events dispatch_*
- `company/hooks.ts` : `invalidateCompanyQueriesForEvent` pour dispatch_assignment + dispatch_run_*
  - `dispatch_assignment` → dashboard + missions + ride detail si missionId
  - `dispatch_run`_* → dashboard + missions
- Test : `contextRegistry.test.ts` vérifie présence des 4 events
- Test : invalidation déclenchée pour chaque event
- Acceptation : ws-service `confirmed_critical_miss` doit tomber à ~0 quand mobile est canary

### PR mobile-D3.2 — Fix G2 + G3 (resync sur stale + reconnect) **[BLOQUANT PR C ≥20 %]**

- Nouveau hook `useCompanyRecoveryListener(contextId)` monté au layout company
- Consomme `company_data_stale_resync` ET `company_socket_reconnected`
- Throttle 30 s + log metric `realtime.recovery_resync_total{trigger=stale|reconnect}`
- Invalidation cohérente : dashboard + missions + inbox + threads chat
- Test : mock router dispatch + vérifier invalidations + throttle

### PR mobile-D3.3 — Fix G4 (authority telemetry) **[POST-CANARY 5 %]**

- Listener `connection.authority` dans driver + company bridges
- Tag Sentry `realtime.authority` + metric ingest
- Pas de logique métier dépendante (read-only observability)

---

## 6. Critères GO/no-go PR C ≥5 %


| Critère                                            | Statut                                     |
| -------------------------------------------------- | ------------------------------------------ |
| G1 fixé (dispatch events écoutés et invalident)    | ❌ À faire (PR D3.1)                        |
| G2 fixé (stale resync consommé)                    | ❌ À faire (PR D3.2)                        |
| G3 fixé (reconnect resync pour dispatch/chat)      | ❌ À faire (PR D3.2)                        |
| G4 fixé (authority observée)                       | ⚠️ Souhaitable mais pas bloquant (PR D3.3) |
| Tests mobile passent                               | À vérifier après chaque PR                 |
| Drill rollback staging réel                        | Bloquant — à planifier ops                 |
| `confirmed_critical_miss` ≈ 0 en staging mixed pop | Mesure à valider après PR D3.1             |


---

## 7. Critères GO/no-go PR D 100 %

Tout ce qui précède **PLUS** :

- Trigger automatique mode dégradé Kafka (D1) — implémentation lag check
- Soak staging 72 h avec mobile mixed pop sans incident
- Métrique `staleResyncTriggered` < seuil défini
- Pas de spike `confirmed_critical_miss` sur cohorte canary 50 %

---

## 8. Risques résiduels post-fixes


| Risque                                | Mitigation                                                                  |
| ------------------------------------- | --------------------------------------------------------------------------- |
| iOS background long (Doze équivalent) | App refresh sur foreground (déjà OK) + stale resync (PR D3.2)               |
| Android Doze                          | Idem + recovery via push FCM si critique                                    |
| Polling fallback cassé Traefik        | Tester en staging explicitement avec `forceNew: true` polling-only          |
| NetInfo lag (4G→WiFi)                 | Reconnect auto socket.io OU forcer reconnect via listener NetInfo (post-D3) |
| Cross-pod ws-service après scaling    | Documenté `ws-service-multi-instance.md`                                    |
