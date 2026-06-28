# Passation — Préparation production : chaîne GPS chauffeur (tracking temps réel + Kafka + carte)

> Document de passation pour la mise en production des correctifs réalisés sur la chaîne
> de tracking GPS chauffeur (mobile → backend → Redis/Postgres → frontend), la voie durable
> Kafka (Option A), et le rendu des marqueurs sur la carte entreprise.
>
> ⚠️ **Le tracking en arrière-plan Android 16** : correctif natif implémenté, **validation BG en cours** — voir §6.

---

## 0. TL;DR — à faire avant prod

1. ~~**Retirer toute l'instrumentation de debug** (logs `AGENTBG` / `AGENTMK`) — §5.~~ ✅ **Implémenté** (session debug 2026-06-26).
2. **Trancher le rendu des marqueurs en prod** : le workaround dev « marqueurs classiques forcés » dans `DriverLiveMap.jsx` doit être revu (§4) — en prod avec un **vrai Map ID Cloud**, les AdvancedMarkers devraient fonctionner.
3. **Configurer l'env backend** : flags Kafka (4+1) + (optionnel) `TRACKING_SOCKET_KAFKA_MIRROR_ENABLED`, et le rate limiter WS — §2.
4. **Configurer l'env frontend** : Maps cloud + **Map ID réel** — §4.
5. **Vérifier les flags mobile EAS** (déjà majoritairement en place dans `eas.json` profil `production`) — §3.
6. **Connaître les limites** : arbitrage de présence socket = best-effort mono-worker (§2.4) ; background non résolu (§6).

---

## 1. Correctifs livrés cette itération (résumé)

| # | Zone | Fichier | Mécanisme | Flag / config | Statut prod |
|---|------|---------|-----------|---------------|-------------|
| 1 | Socket multi-contexte | `backend/sockets/chat.py` (connect) | Authentifie en **driver** un compte `role=company` possédant un profil driver, via `surface=driver` / `context_id=driver:{id}` | aucun (logique) | ✅ ship tel quel |
| 2 | Anti-storm ACK | `backend/sockets/chat.py` (branche rate-limit) | ACK des batches rate-limités (`driver_location_batch_ack`) pour que la file mobile draine | aucun (logique) | ✅ ship |
| 3 | Backoff mobile | `mobile/.../driverRealtimeBridge.ts` | Respecte `retry_after` sur `rate_limit_exceeded` (un seul flush différé) | aucun (logique) | ✅ ship (build mobile) |
| 4 | Arbitrage de présence | `backend/sockets/chat.py` (connect) | Déconnecte les anciens sockets du même driver (single-socket) | aucun (logique) | ⚠️ ship + **limite multi-worker** (§2.4) |
| 5 | Miroir Kafka (Option A) | `chat.py` + `ingest_producer.py` | Publie le point socket dans `driver.location.raw` en **fire-and-forget** (non bloquant) | **`TRACKING_SOCKET_KAFKA_MIRROR_ENABLED`** (défaut `false`) | ⚙️ ship **dormant**, activer si voulu (§2.3) |
| 6 | Anti double-traitement Kafka | `ingest_consumer.py`, `processed_fanout_consumer.py` | `skip persist` + `skip fanout` pour `source="socket_batch"` | aucun (logique) | ✅ ship (n'agit que si #5 ON) |
| 7 | Rendu marqueurs | `frontend/.../DriverLiveMap.jsx` | Forçage **marqueurs classiques** (workaround dev) | ⚠️ **hardcodé** `const GOOGLE_MAPS_USE_JS_STYLES = true` | 🛑 **à revoir avant prod** (§4) |
| 8 | Reprise session tracking | `backend/sockets/chat.py` (batch) | Takeover si `trk_sess_{ts}_*` entrante **≥** session Redis active ; ACK explicite si batch stale | aucun (logique) | ✅ **Implémenté** : évite famine canonical après reconnect mobile |
| 9 | Rollback DB par batch | `backend/sockets/chat.py` (batch) | `db.session.rollback()` en entrée handler | aucun (logique) | ✅ **Implémenté** : isole session SQLAlchemy des erreurs push-token |
| 10 | Récupération FGS Android | `backgroundLocationTask.ts` | Si contexte mission à jour mais FGS OS arrêté → relance `startLocationUpdatesAsync` | aucun (logique) | ✅ **Implémenté** : telemetry `tracking.background.fgs_recover` |
| 11 | ACK `session_conflict` mobile | `driverRealtimeBridge.ts`, `driverTrackingQueue.ts` | Draine batch stale + `reconcileAfterSessionConflict()` + flush immédiat | aucun (logique) | ✅ **Implémenté** : évite famine après reprise session backend |
| 12 | Drain backlog `socket_emitted` | `driverTrackingQueue.ts` | Si socket mort ou file ≥30 → libère `socket_emitted` + force HTTP | `EXPO_PUBLIC_DRIVER_TRACKING_BACKLOG_FORCE_HTTP` (déf. 30) | ✅ **Implémenté** : débloque files 500+ |

Détails des causes racines : voir `docs/ops/passation-debug-background-tracking.md` (s'il existe) ou l'historique de debug.

---

## 2. Backend — configuration d'environnement production

### 2.1 Kafka (déjà documenté) — `env.kafka.production.example`
Les **4 flags clients** doivent être cohérents (tous ON) + le **5e** (persistance) sous *stop gate* :

```env
KAFKA_ENABLED=true
TRACKING_INGEST_ASYNC_ENABLED=true
TRACKING_PROCESSED_FANOUT_ENABLED=true
WS_KAFKA_CONSUMER_ENABLED=true
TRACKING_INGEST_PERSIST_ENABLED=true        # uniquement après P0-1/P0-3/P1-1a (cf. gps-tracking-pipeline.md)
```
Topics prod = **`.v2`** (`driver.location.raw.v2`, etc.). Bootstrap 3 brokers, RF=2. Voir `docs/ops/kafka-production.md`.

### 2.2 Rate limiter WebSocket (`backend/services/monitoring/websocket_rate_limiter.py`)
Défauts **stabilisés** : `driver_location_batch` = **2 / 10 s** par chauffeur (cadence FG mobile ~8 s + marge retry). Côté mobile, pacing client (`socketBatchPacing.ts`, min 5 s entre émissions socket) + drain intelligent de la queue + ACK `rate_limited` non destructif.

```env
WS_DRIVER_LOCATION_BATCH_LIMIT=2
WS_DRIVER_LOCATION_BATCH_WINDOW_SEC=10
WS_DRIVER_LOCATION_LIMIT=1
WS_DRIVER_LOCATION_WINDOW_SEC=1
# Mobile (Expo) — aligner sur la fenêtre serveur
EXPO_PUBLIC_DRIVER_SOCKET_BATCH_MIN_INTERVAL_MS=5000
```

### 2.3 Miroir socket → Kafka (Option A) — **nouveau flag**
```env
# OFF par défaut. Activer SEULEMENT si on veut la voie durable Kafka alimentée aussi par le socket.
TRACKING_SOCKET_KAFKA_MIRROR_ENABLED=false
```
Si activé en prod :
- Le consumer doit tourner avec `TRACKING_INGEST_PERSIST_ENABLED=true` (la voie socket reste source de vérité ; le consumer **skip** la persistance + le fanout pour `source="socket_batch"` → pas de double-écriture/fanout).
- Bénéfice : durabilité/replay/analytics + multi-instance. Coût : latence nulle ajoutée (fire-and-forget) mais charge Kafka accrue.
- ⚠️ Le producteur tourne sous worker **gevent** : `enqueue_fire_and_forget` ne fait **pas** de `future.get()` (non bloquant) et borne `KAFKA_MAX_BLOCK_MS` (défaut 1000). À valider sous charge avant activation prod.

### 2.4 ⚠️ Limite connue — arbitrage de présence (#4)
`chat.py` scanne `_SID_INDEX` (cache **par worker**) et appelle `socketio.server.disconnect(old_sid)`. En prod **multi-workers** (gunicorn > 1 + `message_queue` Redis), un socket dupliqué sur un **autre** worker n'est **pas** vu par le scan local → l'arbitrage est **best-effort mono-worker**.
- Acceptable si 1 worker (dev/petit prod).
- Pour multi-worker : prévoir un **registre driver→sid partagé (Redis)** + déconnexion cross-worker via le manager Socket.IO. À planifier (non bloquant si le churn de sockets est maîtrisé).

### 2.5 Fichiers compose
- `docker-compose.yml` : l'API expose désormais les vars Kafka (`KAFKA_ENABLED`, `TRACKING_INGEST_ASYNC_ENABLED`, `TRACKING_SOCKET_KAFKA_MIRROR_ENABLED`, `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_MAX_BLOCK_MS`) avec **défauts `false`/sûrs** → configurables par l'env serveur.
- `docker-compose.kafka.dev.yml` : **dev uniquement** (broker unique, flags ON, `TRACKING_SOCKET_KAFKA_MIRROR_ENABLED=true`, `TRACKING_INGEST_PERSIST_ENABLED=true`). **Ne pas** utiliser en prod — prod = `docker-compose.kafka.yml` + `env.kafka.production.example`.

---

## 3. Mobile — flags EAS (`mobile/unified-app/eas.json`, profil `production`)

Les flags tracking pertinents sont **déjà** activés dans le profil `production` :
`EXPO_PUBLIC_ENABLE_DRIVER_SOCKET`, `ENABLE_BG_LOCATION`, `ENABLE_TRACKING_PERSISTENT_QUEUE`, `ENABLE_TRACKING_HTTP_FALLBACK`, `ENABLE_TRACKING_REAL_ACK_SEMANTICS`, `ENABLE_TRACKING_SAFE_STALE_FALLBACK`, `ENABLE_TRACKING_ADAPTIVE_CADENCE`, `ENABLE_TRACKING_PRESENCE_MODE`, `ENABLE_TRACKING_SELF_HEAL_WATCH`, etc.

URLs prod (déjà bonnes) :
```env
EXPO_PUBLIC_API_BASE_URL=https://api.lirie.ch/api/v1
EXPO_PUBLIC_DRIVER_SOCKET_URL=https://api.lirie.ch
```

À noter :
- `ENABLE_TRACKING_RECOVERY_CASCADE=0` et `ENABLE_TRACKING_STATE_MACHINE=0` en prod ; profils `production-gps-phase2/3` pour les activer progressivement (canary background).
- Le **backoff rate-limit** (#3) est dans le code → embarqué dans le build mobile, **pas de flag**.
- ⚠️ **Ne jamais builder prod avec le wrapper `easUpdateProd`/env dev** ; les valeurs `EXPO_PUBLIC_*` sont gelées au bundle.
- ✅ **Implémenté** (2026-06-27) : alignement locales fr-CH production — `src/i18n/productionLocale.ts`, permissions `app.json` en français, textes FGS dans `eas.json` (`EXPO_PUBLIC_DRIVER_BG_*`), canaux Android FR, preflight `check-build-ready.js` (flags GPS + patch Android 16).

---

## 4. Frontend — Maps & rendu marqueurs (⚠️ POINT CRITIQUE)

### 4.1 Cause racine du « marqueur invisible »
En **dev**, le Map ID résolu était `DEMO_MAP_ID` → les **AdvancedMarkerElement** restaient *attachés mais non peints* (sous-arbre `content-visibility`), prouvé par l'invariant `attached=5 / painted=0`. Workaround dev : forcer les marqueurs **classiques** (canvas) dans `DriverLiveMap.jsx`.

### 4.2 Ce qu'il faut faire en prod
Config Maps prod (cf. `frontend/maps.env.example`) :
```env
REACT_APP_GOOGLE_MAPS_API_KEY=<clé web restreinte HTTP referrer>
REACT_APP_GOOGLE_MAPS_LIBRARIES=marker
REACT_APP_GOOGLE_MAPS_LIRIE_STYLE=cloud
REACT_APP_GOOGLE_MAPS_MAP_ID=<VRAI Map ID Cloud — Maps → Map Management, PAS l'ID de style>
```

🛑 **Décision à prendre** sur le workaround `DriverLiveMap.jsx` (`const GOOGLE_MAPS_USE_JS_STYLES = true`) :
- **Option A (recommandée)** : **revenir** à l'import de `GOOGLE_MAPS_USE_JS_STYLES` depuis `mapUtils` (retirer le `const … = true` hardcodé). Avec un **vrai Map ID** en prod, les AdvancedMarkers se peignent correctement (le bug venait de `DEMO_MAP_ID`). À **valider en preview avec le vrai Map ID** avant de retirer le workaround.
- **Option B (fallback fiable)** : conserver les **marqueurs classiques** partout (rendu canvas robuste, insensible au content-visibility) — au prix d'un avertissement de dépréciation Google. Acceptable si on ne veut pas dépendre du Map ID/AdvancedMarkers.

→ **Tant que le vrai Map ID prod n'est pas validé, ne pas livrer le workaround tel quel en prod sans avoir tranché.**

---

## 5. 🧹 Nettoyage instrumentation debug

✅ **Implémenté** (2026-06-26) : retrait `AGENTBG` / `AGENTMK` / `_agent_debug_log` dans `chat.py`.

Vérifier l'absence de résidus :
```bash
rg -n "AGENTBG|AGENTMK|agent log|127.0.0.1:7539" mobile frontend backend
```
(Note : `backend/sockets/chat.py` — l'instrumentation `AGENTDBG_CONNECT` a déjà été retirée ; re-vérifier.)

Fichiers temporaires de debug à supprimer du repo s'ils traînent : `bg_capture.txt`, `fgs_poll.txt`, `dev_screen.png`, `debug-*.log`.

---

## 6. Bug BG Android 16 — correctif natif (en validation)

**Cause racine confirmée (2026-06-26)** : `startLocationUpdatesAsync` démarre le FGS mais la voie `PendingIntent` FLP d'expo-location n'établit pas de requête OS livrable sur **Android 16 / targetSDK 36** → `task_invoked` jamais en BG.

✅ **Implémenté** — patch natif `expo-location` (`patches/expo-location+19.0.8.patch`) :
- **`LocationTaskConsumer.kt`** : pour les tâches avec `foregroundService`, bascule sur **`LocationCallback`** (même voie que le watch FG qui fonctionne) au lieu du `PendingIntent` FLP ;
- **Ordre corrigé** : démarrage FGS → `onServiceConnected` → `requestLocationUpdates` (requête OS après FGS actif) ;
- **Exécution directe** : quand FGS + callback actifs, la tâche JS est invoquée sans passer par `JobScheduler` ;
- **`buildFromSource`** : `expo-location` ajouté dans `package.json` → `expo.autolinking.buildFromSource` ;
- Source de référence versionnée : `mobile/unified-app/native-patches/expo-location/LocationTaskConsumer.kt`.

**Test décisif** (mock en mouvement, app en BG) :
```bash
adb logcat -s LocationTaskConsumer TaskService
adb shell dumpsys location | grep -E '10889|ch.liri|operations'
# Attendu : log "Started location updates via LocationCallback (FGS path)"
# + requête OS visible + telemetry tracking.background.task_invoked
```

⚠️ **Statut** : patch compilé et APK debug installé sur device ; **validation BG complète requise** avant prod. Le foreground reste validé indépendamment.

---

## 7. Checklist de mise en production

- [ ] Retirer l'instrumentation `AGENTBG` / `AGENTMK` (§5) + fichiers temporaires. ✅ fait (§5)
- [ ] Activer Kafka dev : `docker compose -f docker-compose.yml -f docker-compose.kafka.dev.yml up -d atmr_api tracking-kafka-consumer tracking-processed-fanout` — ✅ **validé** 2026-06-26 (`KAFKA_ENABLED=true`, `TRACKING_SOCKET_KAFKA_MIRROR_ENABLED=true`, consumers healthy)
- [ ] Trancher le rendu marqueurs (§4) : vrai Map ID prod validé en preview, ou conserver classiques.
- [ ] Env backend : Kafka (4+1 flags, topics `.v2`) ; `TRACKING_SOCKET_KAFKA_MIRROR_ENABLED` décidé (OFF par défaut) ; rate limiter vérifié.
- [ ] Env frontend : `REACT_APP_GOOGLE_MAPS_*` (cloud + Map ID réel).
- [ ] Build mobile EAS profil `production` (flags tracking déjà OK ; URLs prod ; pas d'env dev).
- [ ] Vérifier le mono/multi-worker API vs arbitrage de présence (§2.4).
- [ ] Lint + tests verts (`docker exec atmr-atmr_api python -m pytest …` ; `npx jest …`).
- [ ] Plan de rollback Kafka : `TRACKING_INGEST_PERSIST_ENABLED=false` + `TRACKING_INGEST_ALLOW_REPUBLISH_ONLY=true`.
- [ ] Background Android 16 : valider patch natif expo-location (§6) sur device réel avec mock en mouvement.

---

## 8. Références
- `env.kafka.production.example` — env Kafka prod.
- `frontend/maps.env.example` — env Maps prod.
- `docs/ops/kafka-production.md`, `docs/ops/gps-tracking-pipeline.md` — runbooks détaillés.
- `mobile/unified-app/eas.json` — profils de build & flags mobile.
