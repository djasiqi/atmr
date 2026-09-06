# LIRIE — Optimisation d’exécution

**Statut** : chantier figé. Smoke S23 exécuté — **PASS**. Aucune nouvelle micro-optimisation tant qu’un problème mesurable en utilisation réelle ne justifie de rouvrir.  
**Périmètre** : Cockpit carte + Courses, mode **manuel uniquement**.  
**Règle** : `NO UX REDESIGN` · `NO GPS LOGIC CHANGE` · `NO BUSINESS RULE CHANGE` · `NO SEMI-AUTO`.

```text
APP RESPONSIVENESS + OPTIMIZATION = PASS / CLOSED

OPT-01…07      CLOSED
OPT-08 CODE    PASS
OPT-04D        HOLD — NO ACTION

DEVICE FINAL SMOKE
S23            PASS (2026-09-06)

CODE STATUS    CLOSED
PRODUCT GATE   PASS

NAV-01         PASS / CLOSED
FIX-DEV-CHUNK  PASS
FIX-CLIENTSELECTOR-LOOP  PASS
CREATE RIDE UX ACTIF — P0 only (pas NAV-01)

Aucun retour à lazy:false
Aucune réouverture optimisation

COCKPIT MAP FULL-BLEED   correctif UI distinct — pas NAV-01
```

```text
OBJECTIF PRINCIPAL
LIRIE MOBILE OPTIMIZATION

moins de CPU / rerenders / réseau / travail hors écran
moins de mémoire / payloads
meilleur démarrage, carte/liste, autonomie

MOUNTED ≠ ACTIVE
lazy: false conservé (nav 1–7 ms)
Cockpit hors focus = état + viewport + sélection
sans travail visuel coûteux
```

On n’optimise **pas** le comportement produit. Le réseau lent (ex-RUN 02) devient **OPT-08** (validation finale), pas le prochain chantier.

## Objectifs ressentis

| Étape | Cible |
|---|---|
| Touch → feedback visuel | < 100 ms |
| Touch → ouverture locale | < 200 ms |
| Navigation écran | immédiate |
| Données réseau | ensuite |
| Réseau lent | UI utilisable |
| Réseau absent | dernier état connu |

Ce ne sont pas les temps backend : ce sont les temps ressentis.

## Contrat verrouillé (hors scope)

```text
UX GPS COVERAGE       = PASS / LOCKED
GPS DATA SEMANTICS    = LOCKED
DEVICE / BACKGROUND   = chantier séparé
BATTERY CONCLUSION    = interdite avant terrain
```

Si `N/T` est faux : source de données / présence / timestamps / transport — pas de redesign cockpit.

## Ordre

| ID | Sujet | Statut |
|---|---|---|
| P0 | Isolation mode manuel (semi-auto / optimiseur hors arbre) | ✅ **Implémenté** |
| PERF-01 | Instrumentation tap / nav / API / cache / rerender | ✅ **PASS** |
| PERF-02 | Navigation + touch responsiveness Cockpit + Courses | ✅ **PASS** (RUN 01) |
| PERF-03 | Cache-first / stale-while-revalidate | ✅ **PASS** |
| PERF-04 | Déplier / replier courses 100 % local | ✅ **PASS** |
| PERF-05 | Détails / Éditer ouverts immédiatement | ✅ **PASS** |
| PERF-05B | Détails snapshot > 300 ms | **HOLD** — métrique 13 s non interprétable |
| PERF-06 | Isoler updates GPS des rerenders généraux | **superseded** par OPT-02 |
| PERF-07 | Recherche + date : debounce / cancel / prefetch | ✅ **PASS** |
| **OPT-01** | Background work : Cockpit monté hors focus → suspendre le visuel | ✅ **CLOSED** (smoke S23) |
| **OPT-02** | Isolation render Cockpit : N/T ≠ carte ; marker X ≠ les autres | ✅ **CLOSED** (smoke S23) |
| **OPT-03** | Courses : RideCard memo / FlatList / virtualisation | ✅ **CLOSED** (smoke S23) |
| **OPT-04A** | Inventaire HTTP Cockpit + Courses | ✅ **Implémenté** (inventaire, pas encore de code) |
| **OPT-04B** | Pagination journée / ne pas confondre page et journée | ✅ **Implémenté** |
| OPT-04C | Prefetch J±1 : 1re page seulement | conservé (pas de changement 04B) |
| OPT-04D | Payload DTO liste (garder snapshot PERF-05) | **HOLD** |
| OPT-04E | Invalidations / refetch ciblés | ✅ **Implémenté** |
| OPT-05 | React Query : `staleTime` / `gcTime` par nature de donnée | ✅ **Implémenté** |
| OPT-06 | Mémoire : rétention bornée (pas un gcTime plus court) | ✅ **Implémenté** |
| OPT-07 | Cold start : critique vs différé | ✅ **Implémenté** |
| OPT-08 | Réseau lent / offline / reconnect (ex-RUN 02 / PERF-08) | ✅ **PASS** (code + FINAL S23 SMOKE) |
| **NAV-01** | Transitions barre : lazy au boot, preload CODE après idle | ✅ **CLOSED** (ne pas rouvrir) |
| FIX-DEV-CHUNK | `unknown module` Fast Refresh — plus d’`import()` idle RideCreateModal | ✅ **PASS** |
| FIX-CLIENTSELECTOR-LOOP | `Maximum update depth` — callbacks stables + setState no-op | ✅ **PASS** |
| GPS | Couverture / sémantique / carte | **OUT OF SCOPE** |

## Baseline code (constat, pas encore de fix)

- Onglets : `lazy: false`, `freezeOnBlur: false`, `detachInactiveScreens: false` — les écrans restent montés. **OPT-01** : hors focus, le travail visuel cockpit est suspendu (`MOUNTED ≠ ACTIVE`).
- Cockpit : `endPageLoad` attend `dashboardQuery.isSuccess && !isFetching` — le « ready » est encore lié au réseau.
- Courses : `useCompanyDispatchMissionsQuery({ date })` — **une clé / un GET par jour** ; search et status sont filtrés localement.
- Courses : `handleToggleMissionExpand` est déjà local (`setState` seul) — PERF-04 = confirmer qu’aucune API n’est branchée sur le dépliage.
- HTTP : compteur d’URL existant ; **durée** round-trip ajoutée (PERF-01).
- Pas de `persistQueryClient` : au cold start, pas de snapshot flotte / courses hors mémoire React Query.
- `AppButton` a déjà un état `pressed`. Les `Pressable` custom des cartes courses sont à auditer (PERF-02).

## Instrumentation PERF-01

✅ **Implémenté** : mesures uniquement, aucun changement d’UX / GPS / métier.

Événements / buckets (tier `dev`) :

- `perf.tap` — phases `feedback` / `local` / `navigation` (durée depuis le `onPressIn`)
- `perf.query.cache` — hit / miss (`dispatch.missions:*`, `dispatch.dashboard:*`)
- `perf.api.roundtrip` — durée HTTP (succès et erreur)
- `screen_render` — compteur Cockpit + Courses
- existant : `fleet_map.driver_marker_render`, `fleet_map.enrich_fleet_drivers_ms`

Branchement :

| Surface | Fichier | Mesure |
|---|---|---|
| Onglets + FAB | `CompanyFloatingTabBar.tsx` | tap → feedback → navigation |
| Déplier / Détails / Éditer / date | `rides.tsx` | tap local ou navigation |
| LIVE / date / notif / stats | `CompanyFleetCockpit.tsx` | tap local + `recordScreenRender` |
| Liste courses | `useCompanyDispatchMissionsQuery` | cache hit/miss à chaque clé |
| Dashboard | `useCompanyDashboardQuery` | cache hit/miss à chaque date |
| HTTP | `core/api/client.ts` | `_perfStartedAt` → `recordApiRoundtrip` |

Dump : `__dumpPerfReport__()` / `__getPerfReport__()` en `__DEV__`.

## Cartographie gestes ↔ réseau (PERF-02 scan)

Les gestes ci-dessous **n’attendent pas le réseau** aujourd’hui (OK) :

| Geste | Comportement actuel |
|---|---|
| Cockpit → Courses → Cockpit | Onglets déjà `lazy: false` / `detachInactiveScreens: false` — écrans restent montés |
| Déplier / replier une course | `setExpandedMissionId` uniquement |
| Éditer | Modal ouverte tout de suite avec préremplissage liste (`rideEditInitial`) |
| Réassigner / Transférer | Modal ouverte tout de suite ; spinner **dans** la modal pendant le GET chauffeurs / partenaires |
| LIVE / calendrier / stats / notifications cockpit | `setState` local |
| Barre du bas | `navigation.navigate` immédiat |
| Chips filtres courses | `setStatus` local sur la journée déjà chargée |

Les gestes / flux qui **bloquent encore le ressenti** :

| Geste | Blocage | Ticket |
|---|---|---|
| Recherche « Client, adresse ou chauffeur… » | ~~1 GET / caractère~~ → filtre local | ✅ PERF-07 |
| Changement de date Courses | Cache hit instantané ; miss = skeleton du jour, **jamais les courses de J sous J+1** | ✅ PERF-03 |
| Détails | Snapshot Courses immédiat ; GET en background ; spinner plein écran **seulement** deep link sans cache | ✅ PERF-05 |
| Annuler | **Pas de modal** : le tap lance directement `cancelCompanyRide` (spinner sur le chip). Hors scope métier pour l’instant | PERF-02 (confirm UI = sujet séparé) |
| Cold start Cockpit / Courses | Pas de `persistQueryClient` — pas de snapshot disque, écran vide jusqu’au premier GET | PERF-03 / 08 |
| `endPageLoad` cockpit | Milestone « ready » attend `dashboardQuery.isSuccess && !isFetching` | mesure seulement |

## PERF-07 + PERF-03 Courses

✅ **Implémenté** (2026-09-05). Règle : **rapidité oui, ambiguïté de date jamais**.

```text
PÉRIMÈTRE
- Courses uniquement
- aucune modification GPS / carte
- aucune règle métier
- Annuler inchangé
```

### Recherche locale (PERF-07)

- GET = journée (`date` seulement). `q` / `search` ne partent plus.
- Filtre local : patient, adresses, chauffeur, institution / propriétaire, `search_index`.
- Casse et accents neutralisés.
- Effacer la recherche restaure la journée.

### Date + cache (PERF-03)

- Clé React Query = `(contexte, date)`. Pas de `placeholderData` inter-jours.
- **Cache hit J±1** : liste du jour cible tout de suite + refetch si stale (`staleTime` = `companyDetail`).
- **Cache miss** : skeleton léger « Chargement de cette journée… » — header / recherche / filtres restent utilisables. **Aucune carte de J n’est affichée sous J+1.**
- Prefetch J-1 / J+1 **uniquement après succès de J**, debounce 220 ms (évite la tempête si 5→6→7).
- Dedup = clé React Query + `staleTime` : retour sur une date déjà visitée / préchargée sans 4 GET complets.

Fichiers : `filterMissionsByLocalSearch.ts`, `prefetchAdjacentDispatchMissions.ts`, `hooks.ts` (`useCompanyDispatchMissionsQuery`), `rides.tsx`, `CompanyRidesMissionFlatList.tsx`, `companyQueryKeys.ts`.

### Gates

```text
PERF-07 SEARCH
[x] 0 HTTP par caractère
[x] résultat local immédiat
[x] effacer la recherche restaure toute la journée
[x] aucun résultat hors date

PERF-03 DATE
[x] cache hit J±1 immédiat
[x] aucun full-screen spinner
[x] jamais de données J présentées comme J+1
[x] prefetch J-1/J+1
[x] requêtes dédupliquées
[x] retour sur date déjà visitée quasi instantané

REGRESSION
[x] expand local inchangé
[x] Edit immédiat inchangé
[x] Réassigner/Transférer inchangés
[x] Annuler inchangé
[x] GPS/carte inchangés
```

## PERF-05 Détails

✅ **Implémenté** (2026-09-05).

```text
tap Détails → navigation → snapshot local → GET background → serveur autoritaire
```

- Snapshot mémorisé depuis la liste (`rememberRideDetailSnapshot`) ; repli cache journées.
- Clé stable `ride-details / missionId`. Fresh (`staleTime` 30 s) = pas de GET. Stale = affichage + refresh.
- Champs inconnus du DTO liste (téléphone, naissance, réf., facture, destination, historique) : skeleton local, jamais `Téléphone : —`.
- GET en échec : snapshot conservé + « Impossible d’actualiser » + Réessayer.
- Chauffeur A (liste) puis B (GET) → l’écran devient B.
- Deep link sans snapshot ni cache : spinner initial légitime.

Instrumentation : `perf.mission_details` (`tap`, `navigation`, `snapshot_render`, `cache_hit`, `http_complete`, `server_reconciled`).  
Métrique clé : **TAP → SNAPSHOT VISIBLE** (`snapshot_render`).

```text
PERF-05 DETAILS
[x] navigation sans attendre le GET
[x] snapshot Courses visible immédiatement
[x] aucun full-screen spinner si snapshot/cache disponible
[x] skeleton uniquement pour les données inconnues
[x] cache détail réutilisé
[x] fresh cache = pas de GET inutile
[x] stale cache = refresh background
[x] réponse serveur remplace les valeurs obsolètes
[x] erreur GET ne détruit pas le snapshot
[x] deep link sans snapshot toujours supporté
[x] retour puis réouverture quasi instantané
[x] aucune mutation métier modifiée
```

## P0 — MANUAL MODE ISOLATION

✅ **Implémenté** (2026-09-05) : isolation architecturale **avant** tout RUN PERF.

Le mode manuel est le seul actif. Semi-auto et optimiseur restent dans le dépôt mais **hors arbre React** et **refusés côté service**. Ce n’est pas un masquage de boutons : la branche n’est jamais montée, donc aucun hook / GET / timer / mutation moteur au montage de Courses.

```text
DISPATCH MODE — LOCK
MANUAL = ON
SEMI_AUTO DISPATCH = OFF
OPTIMIZER          = OFF
FULLY_AUTO         = OFF
DEV = OFF · PROD = OFF
```

Double barrière :

| Couche | Comportement si flag `false` |
|---|---|
| UI / arbre React | `MaybeCompanyRidesEngineActions` retourne `null` **avant** `require` du module moteur — aucun hook, aucun effet |
| Service / action | `runCompanyDispatch`, `runCompanyOptimizer`, `getOptimizerStatus`, switch hors `manual` → `DispatchFeatureDisabledError` (0 HTTP) |

Fichiers :

- `mobile/unified-app/src/features/company/dispatch/dispatchModeLock.ts`
- `mobile/unified-app/src/features/company/components/rides/MaybeCompanyRidesEngineActions.tsx` (gate)
- `mobile/unified-app/src/features/company/components/rides/CompanyRidesEngineActions.tsx` (code dormant)
- `mobile/unified-app/app/(app)/(company)/rides.tsx` — plus de GET mode, plus de CTA
- garde dans `companyApi.ts`, query optimizer `enabled: false`, prefetch optimizer coupé, invalidation optimizer ignorée

```text
P0 — MANUAL MODE ISOLATION
[x] bouton "Lancer le dispatch" absent
[x] bouton "Lancer l’optimiseur" absent
[x] aucun hook semi-auto monté
[x] aucun appel HTTP semi-auto
[x] aucun timer/polling semi-auto
[x] aucune mutation automatique
[x] aucun calcul d’optimisation au render
[x] aucune modification d’une mission
[x] parcours manuel identique avant/après
[x] PERF instrumentation toujours active

STATUS:
RUN 01 TOUCH = PASS
OPT-01 + OPT-02 + OPT-03 = CLOSED (smoke S23)
OPT-04A = FAIT
OPT-04B = IMPLÉMENTÉ
OPT-04E = IMPLÉMENTÉ
OPT-05 = IMPLÉMENTÉ
OPT-06 = IMPLÉMENTÉ
OPT-07 = CLOSED
OPT-08 CODE = PASS
OPT-04D HOLD
CODE STATUS = CLOSED
PRODUCT GATE = PASS (FINAL S23 SMOKE)
APP RESPONSIVENESS + OPTIMIZATION = PASS / CLOSED
```

## RUN 01 — 2026-09-06 (S23 Lirie Dev, API Docker locale)

✅ **Joué** : scénario Cockpit + Courses (PID JS `26289`, ~00:19–00:31).  
`__getPerfReport__(50)` **non obtenu** (CDP timeout puis app envoyée au launcher par KEYCODE_BACK). Reconstruction depuis `logcat` `[perf-kpi]`.

| Étape | Résultat |
|---|---|
| Idle Cockpit 20 s | Fait |
| Cockpit → Courses | Fait (`tab.rides` nav **1–7 ms**) |
| Déplier / replier | Fait (`rides.expand` local **1–2 ms**) |
| Recherche « Sonia » puis effacer | Fait (filtre local, 0 KPI tap — TextInput non branché) |
| J → J+1 → J → J+1 | Fait (`rides.date` local **1 ms**) — Aujourd’hui = dim. 6 sept. |
| Détails → retour → Détails | 1re ouverture **confirmée écran** (#45711, snapshot + skeletons) ; 2e = KPI nav **14 ms** |
| Retour Cockpit | Fait (`tab.dashboard` nav **2 ms**) |
| LIVE ouvrir / fermer | **Incomplet** (pas de KPI LIVE ; BACK a mis l’app au launcher) |
| Idle final 20 s | Fait **sur le launcher**, pas sur le Cockpit |

Taps (chemin critique) :

| Action | feedback | local / nav |
|---|---:|---:|
| `tab.rides` | 0 ms | 1–7 ms |
| `rides.expand` | 0 ms | 1–2 ms |
| `rides.date` | 0 ms | 1 ms |
| `rides.details` | 0 ms | **14–19 ms** |
| `tab.dashboard` | 0 ms | 2 ms |

Hors chemin critique (infra locale **non représentative** de la prod) : `api.roundtrip` 13–83 s (`wslrelay` / `127.0.0.1:15100` instable). `js_long_task` jusqu’à 70 s — ne pas traiter comme un jank UI réel.

`mission_details.snapshot_render` = **13241 ms** (TAP → premier `detailView.data`). À **revérifier** : le 1er tap Détails a émis la nav à 00:28:08 mais l’écran n’était pas encore visible ; la 2e ouverture n’a pas réémis `snapshot_render` (même `rideId`). Visuellement, quand Détails s’est affiché : snapshot Courses + skeletons + « Actualisation… » (contrat PERF-05).

Pendant l’idle Cockpit puis sur Courses (`lazy: false`) : `fleet_map.enrich_fleet_drivers` continue (22→25) p95 **7 ms** ; `screen_render` `company.dashboard` 18→21. Signal PERF-06 **faible** (enrich cheap), pas une preuve de jank 5 s.

### Arbre de décision (après RUN 01)

```text
TAP → feedback / nav / expand / date   = PASS (< 20 ms)
HTTP sur le chemin critique            = non (les taps n’attendent pas)
Cockpit rerender 5 s = cause #1        = NON prouvé (enrich 7 ms)
Nav Cockpit↔Courses lente + HTTP=0     = NON (nav 1–7 ms)
snapshot_render > 300 ms               = métrique 13 s douteuse — ne pas lancer PERF-05B
cold start                             = hors RUN 01 (session/wslrelay)
UI rapide, API lentes                  = oui en local, pas un verdict backend
```

### Verdict RUN 01 — sensation de lourdeur au toucher = **FERMÉ**

Ne pas refaire RUN 01 uniquement pour le JSON. Le logcat `[perf-kpi]` répond à la question. Le `snapshot_render` à 13 s n’est **pas** une preuve PERF-05B tant que le point de départ de la métrique n’est pas fiable.

```text
APP RESPONSIVENESS — RUN 01

Touch feedback          PASS
Tab navigation          PASS
Expand / collapse       PASS
Date interaction        PASS
Details navigation      PASS
Network on touch path   NO

PERF-06 map isolation   superseded OPT-02
PERF-05B details        HOLD
COLD START              OPT-07 IMPLÉMENTÉ
GPS                     OUT OF SCOPE

RUN 01 TOUCH            PASS
```

```text
PERF-01 instrumentation       PASS
PERF-07 local search          PASS
PERF-03 date/cache/prefetch   PASS
PERF-05 details snapshot      PASS
RUN-01 touch responsiveness   PASS

NEXT
OPT-04D HOLD (mesure JSON)
OPT-05 = implémenté (staleTime / gcTime par famille)

RUN 02 / PERF-08
différé → OPT-08 (validation finale, pas le prochain chantier)
```

## OPT-01 + OPT-02 — travail hors écran + isolation render Cockpit

✅ **Implémenté** (2026-09-06). `lazy: false` inchangé. Aucune sémantique GPS / métier / UX.

```text
MOUNTED ≠ ACTIVE
Cockpit hors focus
→ état / viewport / sélection conservés
→ pas d’horloge N/T 5 s
→ pas de freshnessTick 5 s
→ pas de decay visuel 90 ms
→ pas d’enrich / cluster / recentrage auto
→ pas de recordScreenRender
→ pulse LIVE arrêté
```

Au retour focus : un tick `Date.now()` + un enrich des dernières positions cache. `N/T` et le vieillissement rattrapent sans changer les seuils 30 s / 120 s.

OPT-02 :

- Horloge N/T isolée dans `CockpitLiveCoverageIsland` — un tick LIVE ne rerender plus `FleetMap` / Upcoming / tab bar.
- `CockpitMapBlock` mémoïsé ; hors focus, `drivers` / `missions` ignorés.
- `rematerializeLiveDrivers` : un GPS sur SS ne recrée que SS.
- `DriverMarker` : `last_seen_seconds` brut hors comparateur (bucket `location_status` / stale visuel).

Fichiers :

- `mobile/unified-app/src/features/company/dashboard/cockpitVisualWork.ts`
- `mobile/unified-app/src/features/company/utils/liveDriverListMaterialize.ts`
- `mobile/unified-app/src/features/company/components/dashboard/CockpitLiveCoverageIsland.tsx`
- `mobile/unified-app/src/features/company/components/dashboard/CompanyFleetCockpit.tsx`
- `mobile/unified-app/src/features/company/realtime/useCompanyDriverLiveTracking.ts`
- `mobile/unified-app/src/features/company/components/maps/useOperationalFleetMap.ts`
- `mobile/unified-app/src/features/company/components/maps/OperationalFleetMap.tsx`
- `mobile/unified-app/src/features/company/components/maps/DriverMarker.tsx`

```text
OPT-01 BACKGROUND WORK
[x] Cockpit + Courses restent montés
[x] travail visuel suspendu hors focus
[x] cache / sockets continuent
[x] rattrapage N/T + âge au refocus
[x] aucune règle GPS changée

OPT-02 RENDER ISOLATION
[x] tick N/T ≠ rerender carte
[x] position SS ≠ reconstruction des autres markers
[x] pas de redesign overlay
```

## OPT-03 — Courses list rendering

✅ **Implémenté** (2026-09-06). UX / recherche locale / cache J±1 / Détails / métier / GPS / semi-auto inchangés.

```text
expand Sonia
→ RideCard Sonia rerender
→ autres RideCard = 0 (isExpanded boolean + memo)

mission 45711 change
→ nouvelle réf. 45711
→ autres missions = mêmes réfs
```

- `CompanyRidesMissionRow` mémoïsé avec `areCompanyRidesMissionRowPropsEqual` : plus de `expandedMissionId` passé à toutes les cartes.
- `renderItem` ne spread plus tout `props`. Callbacks stables, `keyExtractor` = `mission_id`.
- Contenu expanded (itinéraires + chips Détails / Réassigner / …) **non monté** si `expanded === false`. Modals toujours au niveau écran.
- `reconcileDispatchMissionList` + `structuralSharing` React Query : un refetch ne recrée pas toute la journée.
- Recherche : `getMissionNormalizedSearchIndex` (WeakMap) — « S » puis « So » = `includes` sur un index déjà normalisé.
- `FlatList` existante conservée (`Screen scroll={false}`). Pas de `getItemLayout` (hauteur variable). Fenêtre inchangée (12 / 8 / 8) + `removeClippedSubviews`.
- Tests de charge locale : 10 / 30 / 60 / 100 missions (refs + filtre).

Fichiers :

- `mobile/unified-app/src/features/company/components/rides/CompanyRidesMissionFlatList.tsx`
- `mobile/unified-app/src/features/company/components/rides/companyRidesMissionRowProps.ts`
- `mobile/unified-app/src/features/company/utils/dispatchMissionListReconcile.ts`
- `mobile/unified-app/src/features/company/utils/filterMissionsByLocalSearch.ts`
- `mobile/unified-app/src/features/company/hooks.ts`
- `mobile/unified-app/src/features/company/components/DispatchRideListCard.tsx`
- `mobile/unified-app/app/(app)/(company)/rides.tsx`

```text
OPT-03 COURSES
[x] RideCard mémoïsé utilement
[x] refs missions inchangées conservées
[x] expand Sonia ne rerender pas toute la liste
[x] une mise à jour mission ne recrée pas toutes les missions
[x] renderItem / callbacks stables
[x] keyExtractor stable
[x] virtualisation non cassée
[x] contenu expanded non monté quand fermé
[x] recherche utilise un index normalisé stable
[x] aucune mutation/règle métier modifiée
```

**OPT-03 CLOSED** après smoke S23 combiné (2026-09-06). **OPT-04B implémenté** : le client conserve `total` / `page` / `page_size` et complète J en fond.

## SMOKE S23 — OPT-01 + OPT-02 + OPT-03 (2026-09-06)

✅ **PASS** — Lirie Dev Galaxy S23, Metro `--dev-client` :8081, API Docker locale. Une journée = 1 course (Sonia DUPONT `#45711`, dim. 6 sept.). Pas de nouvelle campagne de benches.

```text
1. Cold reload DEV                         OK (session company:1)
2. Cockpit 10–15 s                         OK (Sam. 5 Sept., LIVE 0/287, 1 à assigner / 1 en retard)
3. Courses                                 OK (Dim. 6 Sept., 1 carte Sonia NON ASSIGNÉ)
4. Scroll rapide haut → bas → haut         OK (1 carte : pas de disparition / saut)
5. Expand Sonia                            OK (itinéraires + chips ; rides.expand local 2–3 ms)
6. Collapse Sonia                          OK
7. Expand une autre mission                N/A (une seule mission ce jour)
8. Recherche "Sonia" → effacer             OK (filtre local, carte conservée)
9. Détails → retour                        OK (nav 25 ms, cache_hit snapshot PERF-05)
10. Courses → Cockpit                      OK
11. viewport + sélection Cockpit           OK (date Sam. 5, LIVE 0/287, feuille Prochaines courses, course 10:00)
12. Cockpit → Courses                      OK (Dim. 6, Sonia toujours là, expand conservé)
```

Régressions cherchées : **aucune visible**.

- Pas de flash semi-auto / bouton dispatch.
- Expand fluide, contenu expanded seulement à l’ouverture.
- Recherche locale correcte.
- Retour Cockpit : état visuel conservé (OPT-01 `MOUNTED ≠ ACTIVE`).
- `tab.rides` / `tab.dashboard` nav **2–7 ms** quand le thread JS n’est pas saturé.

Hors scope smoke (bruit infra DEV, pas un verdict OPT-01/02/03) :

- Round-trips API 3–22 s (`wslrelay` / hangs `:15100`) → `js_long_task` 3–15 s collés aux GET ; un tap Courses peut peindre avec 4–8 s de retard.
- `recovery_resync_stale_missions` a invalidé toute la famille missions (**19 s** observé) — OPT-04E.
- Retour Détails : écran blanc ~2 s puis Cockpit (long task 8 s), puis Courses via onglet. Pas un redesign ; le stack `ride-details` + thread JS saturé.
- Filtre horloge + caractère `"c"` résiduels = taps adb hors cible, pas un bug produit.

## OPT-04 — réseau / payload / pagination

Règle : **ne pas faire** les appels, calculs et transferts inutiles. Pas `page_size = 500` (le serveur plafonne déjà à `MAX_PAGE_SIZE = 100`).

```text
OPT-04A  inventaire HTTP     FAIT
OPT-04B  pagination journée  IMPLÉMENTÉ
OPT-04C  prefetch J±1        conservé (page 1 seulement)
OPT-04D  payload DTO         HOLD
OPT-04E  invalidations / refetch   IMPLÉMENTÉ
OPT-05   query cache policy        IMPLÉMENTÉ
OPT-06   memory retention          IMPLÉMENTÉ
```

### OPT-04A — inventaire HTTP Cockpit + Courses

Sources : hooks montés (`lazy: false` → les deux écrans fetch au boot) + logs S23 + contrat API.

Défauts React Query (`instrumentedQueryClient`) : `staleTime` 30 s, `refetchOnMount: "ifStale"`. **`refetchOnWindowFocus` / `refetchOnReconnect` = défaut TanStack = true** (non overridés).

| Endpoint | Fréquence / déclencheur | Taille / objets (S23) | staleTime | Focus / reconnect | Invalidations | Prefetch | Champs vraiment utilisés | Classe |
|---|---|---|---|---|---|---|---|---|
| `GET /company_mobile/dispatch/v1/rides?date=&page_size=50` | Mount Cockpit **et** Courses ; change de date ; recovery ; `booking_*` large | 1 item aujourd’hui ; 6× au boot (dont 1 `status: undefined`) ; 6–22 s | 30 s (`companyDetail`) | oui (défaut) | `missions` `exact: false` (J + J±1) ; recovery resync | J±1 via `prefetchAdjacentDispatchMissions` (même GET page 1 / 50) | Liste + snapshot Détails : patient, heure, statut, adresses, chauffeur, institution, ids | **CRITICAL** + **OVERFETCH contrat** (page ≠ journée) |
| `GET /company_mobile/dispatch/v1/dashboard/realtime` | Mount Cockpit ; date ; recovery | Agrégats 1/0/1/0 | 20 s (`companyList`) | oui | dashboard `exact: true` | non | Compteurs + prochaines courses | **CRITICAL** cockpit |
| `GET /company_mobile/dispatch/v1/status` | Mount Cockpit (mode) | petit | 30 s | oui | — | non | Mode manuel (déjà locked) | **BACKGROUND** (mode déjà connu) |
| `GET /companies/me/drivers/live` | Tracking + recovery | **287 locations**, 9× boot, 3–10 s | 30 s | oui | recovery `drivers_locations` | non | LIVE N/T + carte | **CRITICAL** sémantique GPS — **OVERFETCH** volume roster |
| `GET /company_dispatch/delays/live` + `/delays` | Mount Courses ; focus | petits, 200 rapides | 20 s | oui | `dispatch-delays` large | non | Retard pickup carte | **BACKGROUND** Courses |
| `GET /companies/notifications` | Inbox dashboard + refetch focus Courses | 30 items | 20 s | oui | inbox | non | Badge cloche | **BACKGROUND** |
| `GET /company_mobile/dispatch/v1/mode` | Boot | 16 s observé | — | — | — | non | Confirmation manuel | **REDUNDANT** si status déjà là |
| `GET /company-settings/billing`, `/billing/parties`, `/companies/me/clients` | Layout / autres onglets montés | 3–16 s | list/slow | oui | — | non | Pas Cockpit/Courses | **BACKGROUND** boot |
| `GET /companies/me`, `/companies/me/invoices` ×2 | Boot | 16 s | — | oui | — | non | Hors écran actif | **REDUNDANT** au boot dispatch |
| `GET /messages/…/hub/threads`, `unread-count` | Badge messages | 16 s | 20–45 s | **focus true** (hooks chat) | recovery chat | non | Badge tab | **BACKGROUND** |
| `GET /company/request-offers` | Badge menu | jusqu’à 32 s | — | oui | recovery offers | non | Badge 99+ | **BACKGROUND** |
| `GET /company_mobile/dispatch/v1/rides/:id` | Détails / Éditer | 1 mission | 30 s | oui | ride-details ciblé si `missionId` | snapshot liste | Écran Détails | **CRITICAL** à l’ouverture ; cache_hit S23 |

Contrat pagination **côté API** (inchangé) + **client OPT-04B** :

```text
API GET /rides
  DEFAULT_PAGE_SIZE = 20
  MAX_PAGE_SIZE     = 100
  client envoie     page_size = 50
  réponse           { page, page_size, total, items }

client (OPT-04B)
  → { missions, total, page_size, loaded, is_complete, next_page, refreshed_at }
  → page 1 immédiate
  → J ouvert : pages 2..N séquentielles en fond
  → J±1 : page 1 seulement
  → « aucun résultat » seulement si is_complete
```

`prefetchAdjacentDispatchMissions` : après succès de J, GET J-1 et J+1 avec le **même** `page_size: 50` (pas la journée complète non plus — même trou de complétude si on ouvre J+1 avec >50 courses, mais le cache « première page » est déjà le bon plafond pour J±1).

Invalidations trop larges (OPT-04E) :

- `booking_updated` **sans** `missionId` → dashboard + **toutes** les clés missions + delays.
- `booking_created` / `urgent_alert` → dashboard + missions `exact: false` + inbox.
- `performCompanyRecoveryResync` → dashboard + inbox + delays + chat + offers + live drivers. **Missions : plus de famille** (OPT-04E). Observé S23 avant 04E : `recovery_resync_stale_missions` **19 s**.
- Cockpit appelle **aussi** `useCompanyDispatchMissionsQuery` → le GET journée est payé même sans ouvrir Courses (`lazy: false`).

DTO liste (`_build_ride_summary`) : résumé booking + assignment + coords (géocode serveur si manquant) + durée/distance. Pas d’historique facturation complet dans ce DTO — OPT-04D = vérifier le JSON réel vs champs liste + snapshot PERF-05, pas tout retirer.

```text
OPT-04A
[x] inventaire endpoints Cockpit + Courses
[x] classification CRITICAL / BACKGROUND / REDUNDANT / OVERFETCH
[x] page_size 50 ≠ journée complète documenté
[x] API total/page déjà là — client les conserve (OPT-04B)
[x] prefetch J±1 = même GET page 1
[x] invalidations recovery / booking trop larges
```

### OPT-04B — pagination journée (2026-09-06)

✅ **Implémenté**. Pas de changement backend, pas de `page_size=500`. `page_size` client reste **50**.

```text
GET J page 1 → affichage immédiat
si total > 50 → page 2 en background (séquentiel)
merge structural-sharing (refs page 1 conservées)
J±1 prefetch = page 1 seulement
promotion J+1 → J : cache page 1 + pages 2..N
```

Contrat client (`CompanyDispatchMissionListResponse`) :

- `missions` / `total` / `page_size` / `loaded` / `is_complete` / `next_page` / `pagination_error` / `date`
- `is_complete = loaded >= total` (et page non vide si encore attendue)

Recherche : hits immédiats sur le chargé ; **pas** de « Aucun résultat » tant que J n’est pas complète → « Recherche dans les N courses restantes… ». Effacer la recherche réaffiche les missions déjà reçues.

Erreur page N : pages 1..N-1 conservées, `pagination_error`, retry reprend `next_page` (pas un GET journée entier). Pied de liste : « Chargement de N courses supplémentaires… » ou bouton Réessayer. Pas de spinner plein écran.

Fichiers :

- `mobile/unified-app/src/features/company/utils/dispatchDayPagination.ts`
- `mobile/unified-app/src/features/company/utils/useCompleteOpenDispatchDay.ts`
- `mobile/unified-app/src/features/company/hooks.ts` (`completeDay` uniquement sur Courses)
- `mobile/unified-app/src/features/company/api/companyApi.ts` / `contracts.ts`
- `mobile/unified-app/app/(app)/(company)/rides.tsx`
- `mobile/unified-app/src/features/company/components/rides/CompanyRidesMissionFlatList.tsx`

```text
OPT-04B GATES
[x] page 1 visible immédiatement
[x] total/page/page_size conservés par le client
[x] J se complète en background jusqu’à total
[x] pages chargées séquentiellement
[x] aucune duplication mission_id
[x] ordre final identique au contrat serveur
[x] références des missions existantes conservées
[x] recherche immédiate sur items déjà chargés
[x] « aucun résultat » seulement si J complète
[x] erreur page N ne détruit pas pages 1..N-1
[x] retry reprend la pagination
[x] J±1 reste page 1 seulement
[x] promotion J+1 → J réutilise page 1 en cache
[x] aucune règle métier / GPS / UX principale modifiée
```

**NEXT (avant 04E)** : OPT-04E — ✅ fermé ci-dessous.

### OPT-04E — invalidations / resync ciblés (2026-09-06)

✅ **Implémenté**. Règle : **CACHE PATCH FIRST**, invalidate seulement si nécessaire, refetch le plus petit scope autoritaire.

```text
mission change
→ GET ride-details / payload autoritaire
→ patch #mission dans la journée qui la contient
→ ride-details si déjà en cache
→ J-1 / J+1 intacts
→ dashboard invalidé (agrégat non patchable)
```

`recovery_resync_*` ne fait plus `invalidate missions exact:false`. Sans `mission_id` : refetch **J observé** uniquement. Date connue : refetch **exactement** cette journée.

Mutations (assign / edit / create / cancel / …) : même chemin, raison `mutation`. Une journée `loaded=87 is_complete=true` reste complète après patch. Une pagination partielle (`pagination_error`) n’est pas jetée.

Focus / reconnect :

- `refetchOnReconnect: false` sur la query rides (le recovery listener est la source reconnect)
- `refetchOnWindowFocus` seulement si `completeDay` (J ouvert) et pas de sync autoritaire récente (8 s)
- J±1 prefetch : 0 observer → pas de rafale
- pull-to-refresh Courses : `manual` ; plus de 2e GET rides sur `useFocusEffect`

Compteur DEV : `rides_fetch_reason` = `initial | pagination | date_change | focus | reconnect | recovery | mutation | manual`.

Fichiers :

- `mobile/unified-app/src/features/company/utils/dispatchMissionCachePatch.ts`
- `mobile/unified-app/src/features/company/utils/ridesFetchReason.ts`
- `mobile/unified-app/src/features/company/realtime/useCompanyRecoveryListener.ts`
- `mobile/unified-app/src/features/company/hooks.ts`
- `mobile/unified-app/src/features/company/useRideForms.ts`

```text
OPT-04E GATES
[x] recovery ne fait plus d'invalidation famille rides
[x] J±1 survivent à une recovery de J
[x] mission récupérée patchée par mission_id
[x] références des missions non modifiées conservées
[x] ride-details correspondant réconcilié
[x] journée complète reste complète après patch
[x] pagination déjà chargée n'est pas jetée
[x] mutation ciblée ≠ refetch journée complète
[x] agrégats invalidés uniquement si réellement dépendants
[x] focus ne refetch pas inutilement J±1
[x] reconnect ne produit pas de rafale de rides
[x] aucun double refresh mutation/recovery/focus
[x] serveur reste autoritaire
[x] aucune règle métier / GPS / UX modifiée
```

**NEXT (avant 05)** : OPT-05 — ✅ fermé ci-dessous.

### OPT-05 — politique de cache React Query (2026-09-06)

✅ **Implémenté**. Pas de backend, pas de DTO, pas de GPS, pas d’UX. OPT-04D reste HOLD.

Inventaire avant : presque tout à `companyList` 20 s / `companyDetail` 30 s / défaut 30 s. `gcTime` = défaut TanStack 5 min → une journée de 87 courses pouvait être jetée. Après OPT-04E, les patchs/événements sont la source de fraîcheur : un staleTime court ne faisait que relancer des GET sans événement métier.

| Famille | staleTime | gcTime | Focus | Exemples |
|---|---|---|---|---|
| `realtime` | 10 s | 5 min | oui si stale | `drivers/live` (sémantique GPS inchangée) |
| `operational` | 2 min | 30 min | oui si stale (J ouvert) | rides J, dashboard, delays, inbox |
| `adjacent` | 10 min | 30 min | non | prefetch J±1 |
| `historical` | 15 min | 30 min | non | journées hors voisinage |
| `detail` | 30 s | 15 min | non | ride-details |
| `referential` | 10 min | 60 min | non | clients, factures, billing, mode |

`staleTime` ≠ `gcTime` : quitter Courses ne reconstruit plus la journée. Retour J / Détails = cache immédiat, refresh seulement si la famille le permet.

Fichiers : `mobile/unified-app/src/core/queryCachePolicy.ts`, `instrumentedQueryClient.ts` (`gcTime` défaut 15 min), `hooks.ts`, `useRideForms.ts`, `prefetchAdjacentDispatchMissions.ts`.

```text
OPT-05 GATES
[x] inventaire staleTime / gcTime de toutes les queries critiques
[x] aucune valeur globale appliquée aveuglément
[x] données temps réel restent réellement fraîches
[x] J actif conserve la cohérence métier
[x] J±1 ne refetch pas inutilement
[x] retour Courses utilise son cache
[x] retour Détails utilise son cache
[x] référentiels stables ne sont pas rechargés à répétition
[x] cache suffisamment long pour navigation rapide
[x] événements/mutations continuent de patcher/invalider correctement
[x] aucune donnée métier obsolète masquée par un staleTime excessif
```

**NEXT (avant 06)** : OPT-06 — ✅ fermé ci-dessous.

### OPT-06 — mémoire / rétention (2026-09-06)

✅ **Implémenté**. `gcTime` OPT-05 **inchangé** (30 min). On évacue les journées **hors fenêtre utile**, pas le cache qui sert la nav.

Rétention fonctionnelle :

```text
toujours : J actif (Courses + Cockpit) + J-1 + J+1
+ 2 dates LRU récemment vues
query observée = jamais retirée
```

Scénario 6→7→8→9→10 : garde 9/10/11 + extras 8/7 ; **6 évacué**. Retour 6 = refetch (plus J±1). 6→7→6 = tout piné, instantané.

Ride-details : pas de « max 5 ». DTO ≈ résumé liste (quelques Ko). Filet seulement si **> 40 inactifs**. Snapshot secondaire (PERF-05) borné à **2**.

Caches secondaires bornés : dedup invalidation, reconcile 04E. Instrumentation DEV : compteurs (jours / details), **pas** de snapshots missions/drivers.

Flotte : OPT-01 freeze carte hors focus + structural sharing live drivers déjà en place. Pas de fusion des copies API/normalized (GPS locked).

Fichiers : `dispatchQueryRetention.ts`, `rideDetailSnapshotStore.ts`, `rides.tsx`, `useCompanyDashboardScreenModel.ts`.

```text
OPT-06 GATES
[x] pas de croissance continue à chaque changement de date
[x] pas de croissance continue à chaque ouverture Détails
[x] anciens objets mission libérables
[x] aucun listener/timer dupliqué après navigation
[x] aucun cache secondaire non borné
[x] cache React Query utile conservé
[x] journées anciennes évacuables
[x] carte ne garde pas d'anciennes matérialisations (freeze OPT-01)
[x] instrumentation DEV bornée
[x] aucune régression de la navigation instantanée J / J±1
[x] aucun changement GPS / métier / UX
```

**NEXT (avant 07)** : OPT-07 — ✅ fermé ci-dessous.

### OPT-07 — cold start (2026-09-06)

✅ **Implémenté**. Pas un bench JS de 200 ms : on a **sorti des requêtes, initialisations et modules entiers** du chemin critique.

Principe verrouillé :

```text
AVANT PREMIER ÉCRAN UTILE → uniquement l’indispensable
APRÈS PREMIER ÉCRAN      → secondaire / prefetch / modules non visibles
```

#### OPT-07A — graphe de boot

Hors chemin critique (plus au mount Cockpit / barre / prefetch switch) :

- `GET /companies/me` (réglages seulement)
- invoices / clients (`clients-facturation` + onglet Factures)
- GET `/mode` (le `GET status` suffit pour le header)
- inbox, unread chat, offres institution
- delays, pages 2..N Courses, prefetch J±1
- optimizer / engine (LOCK OFF, déjà gated)

`prefetchContextTarget` ne précharge plus que **dashboard + missions J page 1 + drivers/live**.

#### OPT-07B — startup en couches

```text
session locale valide → shell → Cockpit (snapshot / structure)
→ CRITICAL : rides J p1, dashboard, drivers/live, dispatch status
→ BACKGROUND : inbox, unread, offers, delays, prefetch
```

Le Cockpit ne fait plus `Promise.all` de 10 GET avant d’afficher. `loading` ne bloque plus si un snapshot / cache existe. `endPageLoad` ne attend plus `!isFetching`.

Auth : snapshot offline authentifié + contexte → `status=ready` **avant** `fetchBootstrap`. Refresh serveur en fond. Révoquée → logout inchangé. Échec réseau après entrée locale → on ne repasse pas en `error`.

#### OPT-07C — snapshot disque borné

Persisté uniquement : J page 1 (≤ 50), résumé dashboard, roster (≤ 120, `recorded_at` réel, **sans** `last_seen_seconds`), dispatch status. `cached_at` / `server_refreshed_at` par tranche. Au lancement : hydrate RQ puis refetch stale. Une position disque avec `location_status: live` et `recorded_at` vieux de 10 min **n’est pas LIVE** (règles 30 s / 120 s).

#### OPT-07D — lazy modules

`lazy: false` **conservé** pour Cockpit + Courses. Tous les autres onglets (`chat`, `clients-facturation`, `invoices`, `settings`, détails, offres, carte) sont `lazy: true`. Modals Courses (création / édition / assign / transfert / horaire) : `React.lazy` au premier tap.

Fichiers : `companyColdStartGraph.ts`, `companyColdStartPhase.ts`, `companyColdStartSnapshot.ts`, `prefetchContextTarget.ts`, `sessionProvider.tsx`, `hooks.ts`, `(company)/_layout.tsx`, `rides.tsx`, `useCompanyDashboardScreenModel.ts`.

```text
OPT-07 GATES
[x] aucun engine semi-auto/optimizer au boot
[x] aucun billing/invoices au boot sans nécessité visible
[x] aucun GET mode redondant
[x] premier écran ne dépend pas des données secondaires
[x] requête background lente ne bloque pas l'app
[x] Cockpit/Courses restent lazy:false
[x] snapshot disque strictement borné
[x] snapshot périmé ne devient jamais donnée "LIVE"
[x] serveur réconcilie silencieusement le snapshot
[x] modules secondaires chargés tardivement
[x] aucune règle métier/GPS/UX modifiée
[x] aucun travail supprimé s'il est réellement nécessaire au premier écran
```

**NEXT (avant 08)** : OPT-08 — ✅ fermé ci-dessous (corrections ciblées, pas de nouveau chantier).

### OPT-08 — robustesse lent / offline / reconnect (2026-09-06)

✅ **Implémenté** comme **validation + corrections ciblées**, pas un nouveau chantier d’architecture.

Défauts observés dans le code (pas un bench device) :

| Défaut | Correction |
|---|---|
| Mutation offline : retry 15 s × 2 + message générique | `assertCompanyOnlineForMutation` immédiat, `retry: false` réseau, message verrouillé |
| Assign optimistic : rollback seulement 409/422 | rollback sur **toute** erreur (pas de succès fictif) |
| Journée absente du cache offline → skeleton puis « Aucune course » | état `offline_unavailable` explicite, jamais le J précédent |
| Recovery reconnect invalidait inbox / chat / offres | resync = J observé + dashboard + delays + live uniquement |

Lent 800/2000 ms : déjà couvert par OPT-07 (snapshot/cache d’abord, recherche locale, Détails snapshot, pas d’overlay global). GPS : vieillissement local 30 s / 120 s sans réseau (OPT-07C). Cold start offline : session locale → ready sans bootstrap (OPT-07).

```text
OPT-08 GATES (code)
[x] navigation / cache indépendants du réseau (OPT-03…07)
[x] aucun overlay global pour un GET lent
[x] recherche locale
[x] Détails snapshot avant serveur
[x] pagination n’empêche pas d’utiliser la page déjà chargée
[x] journée absente offline = message clair
[x] mauvaise date jamais affichée
[x] mutations : pas de succès fictif, refus immédiat offline
[x] pas de retries agressifs (query/mutation réseau)
[x] reconnect : pas de rafale J±1 / billing / chat / offres
[x] snapshot GPS ancien ≠ LIVE
[x] échec réseau ≠ session révoquée
```

Fichiers : `companyOfflinePolicy.ts`, `hooks.ts`, `useRideForms.ts`, `companyApi.ts`, `useCompanyRecoveryListener.ts`, `CompanyRidesMissionFlatList.tsx`, `rides.tsx`.

Tests Jest (2026-09-06) : `companyOfflinePolicy`, `useCompanyRecoveryListener`, `companyColdStartSnapshot`, `canEnterFromLocalSession` — **PASS**.

Smoke device : ✅ **PASS** — FINAL S23 SMOKE ci-dessous.

```text
CODE STATUS    CLOSED
PRODUCT GATE   PASS
OPT-04D        HOLD — NO ACTION
```

Chantier figé. Aucune nouvelle micro-optimisation tant qu’un problème mesurable en utilisation réelle ne justifie de le rouvrir.

## FINAL S23 SMOKE — OPT-08 / OPT-07 device (2026-09-06)

Smoke fonctionnel 5–10 min. Pas de bench, pas de JSON perf, pas de millisecondes.

```text
FINAL S23 SMOKE

1. Relancer proprement Expo / Metro
2. Cold start ONLINE
   → shell/Cockpit sans attendre les données secondaires

3. 800 ms
   → Cockpit → Courses → recherche → Détails

4. 2000 ms
   → même parcours
   → aucun blocage global

5. OFFLINE
   → caches consultables
   → date non cachée = message offline correct
   → mutation = refus immédiat
   → aucun succès optimiste fictif

6. RECONNECT
   → J observé + dashboard + delays + live
   → pas J±1 / billing / clients / chat / offres en rafale

7. Kill app → OFFLINE → relaunch
   → session locale
   → shell
   → snapshot
   → GPS périmé reste hors LIVE
```

✅ **PASS** — Lirie Dev Galaxy S23, 2026-09-06 ~03:01–03:09. Metro relancé proprement après le crash `@expo/cli ./utils/env` (install pendant Metro). Latence injectée via proxy local `15102` (`800` / `2000` / `OFFLINE`) ; OFFLINE = `adb reverse` API retiré (Wi‑Fi ADB conservé).

| Étape | Verdict | Constat |
|---|---|---|
| 1. Metro propre | PASS | Expo `--dev-client` :8081 ; `env.js` présent ; crash précédent = env DEV |
| 2. Cold start ONLINE | PASS | BootSplash ready ~1,6 s ; `DASHBOARD_MOUNTED` avant GET rides / unread / delays |
| 3. 800 ms | PASS | Courses + recherche « Sonia » locale + Détails snapshot ; pas d’overlay global |
| 4. 2000 ms | PASS | Cockpit / cache utilisables ; pas de blocage global (un overlay Inspect Expo ouvert par taps adb, refermé) |
| 5. OFFLINE | PASS | Cache J consultable ; Mar. 8 sept. (hors cache) = *Cette journée n’est pas disponible hors connexion* (jamais le J précédent) |
| 5b. Mutation | PASS après fix ciblé | 1er essai : « Network Error » + *Aucun chauffeur disponible*. Fix : `getDispatchApiErrorMessage` + masquer l’empty roster si erreur. Retest : *Connexion indisponible. Impossible d’effectuer cette action pour le moment.* Confirmer désactivé. |
| 6. RECONNECT | PASS | Viewport Courses / Dim. 6 / Sonia conservé. Recovery = dashboard + delays + live. Pas de rafale J±1 / billing / clients / chat / offres |
| 7. Kill → OFFLINE → relaunch | PASS | `SESSION_READY` puis Cockpit snapshot (1 à assigner, 10:00) ; statut **RECONNEXION…** — pas LIVE ; session non révoquée |

Corrections ciblées observées sur device (pas un nouveau chantier) : `rides.tsx` (assign + bandeau jour), `AssignDriverModal.tsx`.

```text
APP RESPONSIVENESS + OPTIMIZATION = PASS / CLOSED
OPT-04D = HOLD — NO ACTION
```

Les 13–83 s `wslrelay` / `127.0.0.1:15100` restent un **bruit d’infra de dev**, pas un profil.

Pour **chaque** profil, rejouer exactement le même parcours :

```text
1. Cockpit
2. Courses
3. Recherche "Sonia" → effacer
4. J → J+1
5. Détails → retour
6. Cockpit
```

### Gates

```text
800 ms
[ ] aucun écran bloqué
[ ] taps toujours immédiats
[ ] cache utilisé
[ ] pas de spinner plein écran inutile

2000 ms
[ ] même comportement
[ ] Détails affiche le snapshot
[ ] date en cache immédiate
[ ] date hors cache = skeleton de la bonne journée
[ ] aucune donnée ancienne présentée comme actuelle

OFFLINE
[ ] dernière donnée connue reste visible
[ ] navigation locale fonctionne
[ ] recherche locale fonctionne
[ ] message réseau non bloquant
[ ] aucune boucle/retry agressive

RECONNECT
[ ] reprise automatique
[ ] pas de tempête HTTP
[ ] pas de doublons
[ ] pas de reset écran
[ ] pas de perte de sélection/navigation
[ ] données serveur réconciliées proprement
```

### Règle de décision (stricte)

```text
SI 2000 ms = PASS
ET OFFLINE/RECONNECT = propre
→ APP RESPONSIVENESS + SLOW NETWORK = CLOSED

SINON
→ corriger uniquement le défaut observé
→ pas de nouveau chantier générique
```

Objectif du verdict : **application résistante à 2 s de latence** — pas « relay local dysfonctionnel pendant 50 s ».

Login **local uniquement** (pas prod) — `info@emmenez-moi.ch` / `LirieLocal2026!`.

### NAV-01 — transitions barre (2026-09-06)

```text
NAV-01 = CLOSED

BOOT
Cockpit + Courses uniquement

POST-BOOT IDLE
preload code :
- Chat
- Menu

INTERDIT AU PRELOAD
- montage écran
- GET
- clients
- factures
- settings
- données Chat
- RideCreateModal (voir FIX-DEV-CHUNK)
```

✅ **Implémenté** + **smoke S23**. **Ne se rouvre pas.** Ne pas y rattacher les correctifs d’usage ci-dessous, ni le layout map full-bleed. `lazy: false` uniquement Cockpit + Courses. **Aucun retour à `lazy:false` global. Aucune réouverture optimisation.**

```text
COLD START → Cockpit + Courses
APRÈS PREMIER ÉCRAN STABLE → import() Chat / Menu
                             sans monter les écrans
                             sans GET Chat / Clients / Invoices / Settings
AU TAP → module déjà en mémoire → shell immédiat → données ensuite
+ → React.lazy d’origine au tap (pas un preload idle)
```

Fichiers : `companyTabModulePreload.ts`, `(company)/_layout.tsx`, `rides.tsx`, `companyColdStartGraph.ts`.

```text
NAV-01 — BOTTOM TAB TRANSITIONS

[x] Cockpit → Courses immédiat (lazy:false inchangé)
[x] Courses → Cockpit immédiat
[x] premier Chat = code déjà évalué (GET threads seulement au mount)
[x] premier Menu = settings / clients-facturation déjà évalués
[x] bouton + = React.lazy au tap (pas d’import() idle — évite `unknown module` Metro)
[x] préchargement CODE uniquement après lane background
[x] aucun GET déclenché par le preload
[x] aucun écran secondaire monté au boot
[x] cold start OPT-07 conservé
[x] aucune règle métier/GPS modifiée
```

#### Smoke S23 NAV-01 (2026-09-06 ~12:33–12:38)

Lancement propre (pas Fast Refresh), idle ≥ 2 s, puis `perf.tap` navigation :

| Transition | 1er tap | 2e tap | Lecture |
|---|---|---|---|
| Cockpit → Courses | 6–10 ms | 2–4 ms | immédiat |
| Courses → Cockpit | 2 ms | 1–2 ms | immédiat |
| Chat | **2 ms** | **2 ms** | 1er ≈ 2e, rapide |
| `+` Create | 164 ms | — | pas un lazy-chunk (pas 1er lent / 2e rapide) |

Chat : shell `Messages` + cache local tout de suite ; `GET /messages/.../hub/threads` ensuite (`page_load` inbox 915 ms). Ce n’est pas le coût du module.

Verdict discriminant : **1er tap ≈ 2e tap et rapides** → NAV-01 **CLOSED**.  
Les `js_long_task` ~4 s dashboard (API GPS) restent hors scope — pas une campagne perf, pas un retour à `lazy:false`.

### Correctifs d’usage (hors NAV-01)

Deux défauts indépendants de la navigation. **Ne rouvrent pas NAV-01.**

```text
NAV-01 = CLOSED

FIX-DEV-CHUNK = PASS
FIX-CLIENTSELECTOR-LOOP = PASS

Aucun retour à lazy:false
Aucune réouverture optimisation
```

**FIX-DEV-CHUNK** — Fast Refresh / Metro : un `import()` idle séparé de `RideCreateModal` (`rideCreateModalLoad.ts`) créait un second chunk invalide → `unknown module`. Fichier supprimé. Le `+` conserve uniquement le `React.lazy` d’origine dans `rides.tsx`. Preload idle = Chat + Menu uniquement.

**FIX-CLIENTSELECTOR-LOOP** — `ClientSelector` / `AddressSelector` : callback instable → `setState` parent même sans changement → `Maximum update depth exceeded`. Callbacks `useCallback` + `nextSuggestionFields()` no-op + effet via ref.

Smoke pratique S23 (2026-09-06 ~13:01–13:04) : **redémarrage complet Lirie Dev** (force-stop, pas Fast Refresh).

```text
NAV-01 — ouverture / navigation du module
+ → Créer une réservation          = PASS
Pas de unknown module
Pas de crash lazy
Modal accessible
Champs principaux présents
```

`tab.create` 17 ms, bundle `RideCreateModal` 282 ms. Metro après ce boot : **0** `unknown module`, **0** `Maximum update depth exceeded`.

Le comportement interne du formulaire (focus client, overlays, flux) n’appartient **pas** à NAV-01. Voir [create-ride-ux-2026-09-06.md](create-ride-ux-2026-09-06.md) — chantier **CREATE RIDE UX**, ouvert ensuite.

### Cockpit map full-bleed (2026-09-06) — layout only

Correctif visuel : la bande `E.BG` entre la carte et la barre du bas venait du **Tabs navigator** qui retirait la hauteur de la nav à la scène. La map était déjà `absoluteFill` dans le cockpit.

✅ **Implémenté** : nav entreprise en overlay ; « Prochaines courses » au-dessus de la nav ; clearance Courses / Clients pour ne pas passer sous la pilule. **Aucun** changement GPS / métier / OPT / NAV-01.

```text
COCKPIT MAP FULL-BLEED

[x] Map = hauteur écran complète
[x] Bottom nav = overlay absolu
[x] Upcoming rides = overlay absolu
[x] aucun paddingBottom réservé à la nav dans MapView
[x] safe-area absorbée par la nav, pas par la map
[x] aucune bande entre carte et bottom bar
[x] aucune règle GPS/métier
```

## Protocoles de test (PERF-09)

A. Wi-Fi / 5G normal — **RUN 01 PASS** (touch)  
B. Réseau lent simulé — **RUN 02 A/B** (`800 ms`, `2000 ms`)  
C. Perte réseau / reconnexion — **RUN 02 C/D** (`OFFLINE` → `ONLINE`)

Mesurer : TAP → feedback, TAP → nav, TAP → screen rendered, API start → response, cache hit/miss, rerender écran, rerender marqueurs. Sur RUN 02 : surtout **UI encore là** + **0 HTTP sur search / date cache / nav**.
