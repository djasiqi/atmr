# Entreprise — reload Metro 2026-09-06

```text
SMOKE PERF             = PRÊT
INVARIANTS CODE        = VALIDÉS
VALIDATION DEVICE      = SEUL GATE RESTANT
PERF-01                = RESTE OPEN
AUTH-GATE-01           = DISTINCT
DEPLOY                 = BLOQUÉ
```

`react_commit ~2 s` = coût dans le **rendu/commit React** du dashboard. Menu dev disparu et captures ADB en échec = conséquences de cette saturation, pas un défaut ADB ni un crash Android natif.

L’`exit 134` Metro/V8 est un incident **dev distinct**. Sans stderr / dump V8, la cause n’est pas démontrée. **Ne pas le comptabiliser comme crash de `ch.liri.operations`.**

Ne pas rejouer le boot : les preuves sont archivées.

Ce n’est **pas** un cold start natif. `COMPANY-COLD-01` reste non testé.

Journal de référence : `docs/ops/_smoke_company_reload_01_2026-09-06/metro-boot1-pre-reload.txt`
(S23, Lirie Dev, Metro 8081, 2026-09-06 19:52:56Z).

## COMPANY-AUTH-GATE-01 = FAIL / DISTINCT

Défaut **distinct** de la saturation dashboard. L’intercepteur coupe le HTTP (`0` × 401), mais le critère strict n’est pas tenu : des callers partent encore avant `SESSION_READY`.

✅ **Implémenté** (barrière partielle) : alignée sur `SESSION_READY`.

- `mobile/unified-app/src/core/network/companySessionNetworkGate.ts`
- intercepteur `ERR_COMPANY_SESSION_NOT_READY` dans `src/core/api/client.ts`
- `enabled` React Query : `contextId` **et** session réseau prête

### Mesure boot JS 19:52Z

| Critère | Attendu | Observé |
|---|---|---|
| HTTP 401 avant `SESSION_READY` | 0 | **0** |
| GET protégés qui atteignent le serveur | 0 | **0** (status `undefined`, 0–4 ms) |
| `/auth/*` avant `SESSION_READY` | autorisé | csrf 200, refresh 200 (7,1 s), bootstrap 200 |
| Tentatives locales bloquées | 0 | **14** (leak callers) |
| Premier GET protégé HTTP après `SESSION_READY` | 200 | notifications, delays/live, realtime, drivers/live, status, rides, request-offers — tous **200** |

Les six `401` du journal initial ont disparu. Le refresh n’est plus déclenché par une rafale protégée : il reste le chemin normal `/auth/refresh-token` (~7,1 s), puis `SESSION_READY` à **13,958 ms**.

Le leak restant : des callers (prefetch J±1 `prefetchAdjacentDispatchMissions`, `requestWithFallback`) invoquent encore l’API avant `SESSION_READY`. L’intercepteur les coupe — `contract_failure` avec `status: null`, **pas** de 401.

`CACHE_READY` n’a **pas** été émis : `peekCompanyColdStartSnapshot` a manqué (`cache_hit: false` sur missions / dashboard).

`/auth/*` reste autorisé. Aucune purge. Aucun changement GPS / queue chauffeur.

## COMPANY-DASHBOARD-PERF-01 = FAIL / SATURATION PROUVÉE

Toujours **bloqueur**. `react_commit ~2 s` = rendu/commit React. Menu dev / captures ADB en échec = conséquences, pas crash natif.

Jalons boot JS (depuis cold-start origin) :

```text
APP_JS_READY              1,277 ms
SESSION_RESTORED          1,335 ms
DASHBOARD_MOUNTED         1,679 ms
DASHBOARD_DATA_READY      2,025 ms   (meaning: cache_or_first_data, interactive: false)
MAP_READY                 3,516 ms
SESSION_READY            13,958 ms
OVERLAY_HIDDEN           16,014 ms
DASHBOARD_INTERACTIVE    16,014 ms   (= overlay retiré)
SOCKET_HEALTHY           18,639 ms
DRIVER_MARKERS_MOUNTED   22,711 ms
```

`DASHBOARD_DATA_READY` ≠ interactif. Les jalons sont maintenant séparés.

### Phases `perf.company.dashboard.phase`

| Phase | n | somme | max | Verdict |
|---|---|---|---|---|
| `snapshot` | 39 | 145 ms | 14 ms | sain (y compris rematerialize 120 puis 287 drivers) |
| `markers` | 8 | 2 ms | 1 ms | sain |
| `overlays` | 4 | 3 ms | 2 ms | sain |
| `realtime_fusion` | 0 | — | — | pas de batch socket mesuré sur cette fenêtre |
| `react_commit` | 109 | **218,6 s** | **4,05 s** | **bloqueur** |

`perf.js_long_task` : 53 tâches, ~**112 s** cumulées, max **4,07 s**.

Après le boot, un rematerialize (~287 sources, 6–13 ms) toutes les ~3 s est suivi d’un `react_commit` de **2,0–2,5 s** (dashboard_model + fleet_map). La dérivation est cheap ; le commit React de l’arbre carte/dashboard ne l’est pas.

`perf.company.screen.usable` dashboard : premier commit modèle après overlay, puis jusqu’à **61 s** (`trigger: model_commit`) — l’écran se repeint en continu, donc « usable » est ré-émis trop tard.

Aucune optimisation dashboard dans ce passage : mesures seulement.

## COMPANY-LAZY-BOOT-01

Avant toute navigation utilisateur, Metro a encore bundlé :

- `messages/_layout.tsx` + `messages/index.tsx` (~3,2–3,4 s)
- `settings.tsx` (~5,2 s)
- `clients-facturation.tsx` (~5,0 s)

Inchangé. Amplifié en dev ; n’explique pas les `react_commit` de 2 s **après** ces bundles.

## Points sains (reconfirmés)

- Un seul socket entreprise, `SOCKET_HEALTHY` sans reconnexion.
- Aucun socket chauffeur. Tracking GPS inactif côté entreprise.
- Aucun `409` / `429` / `5xx` / HTTP `401`.
- Aucun lien avec le P0 FGS chauffeur.

## Anomalies secondaires (inchangées)

- `ownerKey: "anonymous"` au register push alors que le contexte est `company:1`.
- SecureStore > 2048 octets.

## Metro V8 ≠ crash app

Une tentative de menu dev / `/reload` a coincé UIAutomator : conséquence de la saturation JS (`react_commit` ~2 s), pas d’un défaut ADB. Metro a ensuite exit **134** (abort V8). **Cause non prouvée** (pas de stderr / dump). **`COMPANY-APP-CRASH` = non observé** — ne pas le comptabiliser comme crash de `ch.liri.operations`.

Le boot instrumenté reste **exploitable / archivé**. Ne pas le rejouer pour cette isolation. Metro `8081` a été relancé.

## Isolation du commit (code, 2026-09-06 22:04)

Attribution à partir du code + du journal, **sans rejouer le boot**.

| Source | Verdict | Preuve |
|---|---|---|
| Transformations snapshot / rematerialize | **pas le coût** | 6–14 ms ; dérivation cheap |
| Dérivation marqueurs / overlays | **pas le coût** | 0–2 ms |
| Mises à jour live **socket** | **pas le cadenceur** | `realtime_fusion` = 0 sur la fenêtre |
| Invalidations React Query | **secondaire** | refetch `drivers/live` ~30 s, pas toutes les 5 s |
| Tick fraîcheur 5 s → **churn d’identités** | **déclencheur** | `freshnessTick` force `refreshAgeForUnchangedSources` ; `applyLocalLocationFreshness` alloue un nouvel objet dès que `last_seen_seconds` change → liste nouvelle → memo carte cassé |
| Rendu/commit carte + shell | **coût** | `react_commit` 2–4 s ; `dashboard_model` et `fleet_map` mesurent le **même** commit (ne pas sommer) ; `DRIVER_MARKERS_MOUNTED` = 1 au boot — à confirmer `spatial_count` / `marker_count` sur les pics suivants |
| Sections hors écran / lazy | **amplificateur dev** | messages / settings / clients-facturation bundlés ; pas la cause des 2 s après bundle |

L’île couverture (`CockpitLiveCoverageIsland`) calcule l’âge avec son tick local (`coverageNowMs` + `recorded_at`).

✅ **Implémenté** (correctif, non validé device) : le tick 5 s ne propage plus vers la carte.

- `last_seen_at` (`recorded_at`) immuable ; plus d’injection de `last_seen_seconds` dans la collection carte
- 287 objets + liste réutilisés tant qu’il n’y a ni franchissement `live → recent → stale` ni mouvement spatial
- refetch live à même ancre (coords + last_seen_at) : `replaced = 0`
- `freshnessTick` retiré du state : l’intervalle n’appelle `setDrivers` que si la liste change
- extras : `reused`, `replaced`, `list_identity_changed`, `spatial_count`, `marker_count`

`dashboard_model` et `fleet_map` mesurent le même commit : ne pas additionner.

## Gate `COMPANY-DASHBOARD-PERF-01`

PASS uniquement si, sur l’ensemble de la fenêtre device :

```text
p95 react_commit < 200 ms
max react_commit  < 1 s
js_long_task ≥ 1 s = 0
dashboard manipulable pendant tick/refetch
```

`dashboard_model` et `fleet_map` = le même commit : ne pas additionner.

## Smoke device — 4 cas (complet, aucun cas de plus)

Invariants code validés :

- cas 1 et 2 : liste entièrement stable, aucun `setDrivers` ;
- cas 3 et 4 : nouvelle liste justifiée, **1 objet remplacé sur 287** ;
- les 286 autres références et les clés des marqueurs restent stables.

Le device doit dire si React Native réconcilie uniquement l’élément concerné, ou si le changement de liste suffit encore à un commit global de `MapView`.

```text
1. Trois ticks immobiles
   replaced = 0
   setDrivers = 0
   aucun commit carte corrélé

2. Refetch avec même ancre
   replaced = 0
   spatial_count = 0
   référence de liste inchangée

3. Un changement spatial
   replaced = 1 attendu
   spatial_count = 1
   commit borné

4. Un changement de fraîcheur (live → recent ou recent → stale)
   replaced = 1 attendu
   spatial_count = 0
   marker_count stable
   commit borné
```

Si le cas 4 échoue malgré `replaced = 1`, le prochain niveau est une **projection spatiale dédiée à la carte**. Aucun autre changement n’est justifié avant cette mesure.

Un pic hors gate doit citer `surface` + `replaced` + `spatial_count` + `marker_count`.

## Suite

1. Validation device — seul gate restant pour PERF-01.
2. AUTH-GATE-01 reste distinct.
3. `COMPANY-COLD-01` après le gate device.
