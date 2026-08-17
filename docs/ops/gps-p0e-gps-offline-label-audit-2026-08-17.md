# P0-E — Audit bascule « GPS hors ligne » (fraîcheur ≠ transmission)

## Accord produit

```text
mission active + tracking + coords connues
→ marqueur visible en permanence ✅
immobile → mêmes coords OK, pas « disparition »
« position ancienne » ≠ « GPS hors ligne »
GPS hors ligne → uniquement perte réelle de remontée (critère explicite)
```

## Condition exacte aujourd’hui (chaîne)

### 1) Backend REST — `location_status`

Fichier : `backend/services/company_driver_locations.py`  
Âge : `recorded_at` (puis `received_at` / `ts`) via  
`resolve_location_freshness_timestamp` → `last_seen_seconds`.

Seuils mission (`presence.py`, `MISSION_LIVE_THRESHOLDS`) :

| Âge (`recorded_at`) | `location_status` |
|---------------------|-------------------|
| ≤ 20 s | `live` |
| ≤ 90 s | `recent` |
| ≤ 300 s | `stale` |
| **> 300 s** | **`offline`** |

`offline_reason` = `no_signal` dès que `location_status == offline`.

**Point critique :** l’horloge de fraîcheur est le **`recorded_at` de capture GPS**, pas l’heure du dernier PUT HTTP réussi.  
Un retry idempotent (même `event_id`, même payload immuable, HTTP 202) **ne rajeunit pas** le statut.

### 2) Frontend carte — libellé « GPS hors ligne »

`mapUtils.getDriverFreshnessLabel` :

- `last_known` → « GPS hors ligne — … »
- `offline` ou `position_source=db_fallback` → « GPS hors ligne »

`companyDriverProjections.resolveDriverMapProjection` :

- `gpsFreshness` ∈ {`offline`, `offline_unknown`, `stale`, `last_known`}  
  ou `db_fallback` → `visualTreatment` = `gps_offline` / `gps_stale`, **`visualStatus = offline`**

`fleetDriverLocationPresence` : si backend envoie `location_status=offline|last_known` → présence locale `last_known` ; **`countedAsLocated = false`** (seul live|recent compte pour `N/8 en direct`).

`DriverLiveMap` : busy/assigned + offline/stale → chip **« GPS hors ligne »** ; KPI `locatedCount` baisse → bannière type « Aucun GPS récent » si 0/N.

Le marqueur n’est pas forcément détruit (`showMarker` true pour `last_known`), mais il est **traité comme hors-ligne / hors « en direct »**.

## Lien avec canary 135 #2

```text
PUT 202 continue     = vrai (retries items existants)
DLE/canonical figés  = vrai (pas de nouvel event_id / recorded_at)
REST offline         = âge(recorded_at) > 300 s ★
FE « GPS hors ligne」 = conséquence de location_status=offline ★
```

Donc : **« l’app transmet encore » ≠ « le backend reçoit des positions fraîches au sens fraîcheur carte ».**  
Les 202 observés ne prouvaient pas de nouvelles captures ; ils prouvaient un flush/retry. Le label offline n’est pas un bug d’affichage isolé dans ce run — il suit le clock `recorded_at` figé.

## Ce qui est trompeur (contrat vs code)

Le code **mélange** :

1. absence de **nouveau fix GPS** (recorded_at ne bouge pas),
2. **âge de la dernière position connue**,
3. libellé **« GPS hors ligne »** / `no_signal`,
4. et le KPI **« en direct »** (éviction live).

Alors que le produit veut :

1. pipeline encore alimenté / session active → pas « hors ligne »,
2. position inchangée (immobile) → OK, reste visible,
3. âge élevé → « position ancienne », pas immédiatement « GPS hors ligne ».

## Discriminants à ne pas confondre

| Signal | Signifie |
|--------|----------|
| PUT 202 | HTTP accepté (peut être retry) |
| nouvel `event_id` + `recorded_at` récent | vraie nouvelle capture |
| `location_status=offline` | âge(recorded_at) > 300 s (mission) |
| FE « GPS hors ligne » | projection de ce statut / last_known |
| conflict=0 | immutabilité 135 OK (orthogonal) |

## Cadre figé (confirmé 2026-08-17 ~20:05)

```text
P-TECH  = priorité immédiate (nouveaux event_id / recorded_at en FG)
P-UX    = ensuite, train séparé
IMMUTABILITY 135 ✅
HOME #3 HOLD ⛔
PATCH UX HOLD ⛔
NEXT = « GPS prêt » / « retry FG » uniquement
     = pré-gate diagnostique (même mission/session, app FG)
     = PASS = nouvelles captures réelles (≥3 eid), PAS PUT 202 seuls
```
