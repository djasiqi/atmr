# P0-F UI — Présence GPS flotte

**Statut** : implémenté côté UI (mobile + web).  
**Merge** : uniquement après canary P0-F téléphone (Android → analyse → iOS → puis UI).  
**PR** : séparée du canary GPS / Kafka / P0-E.

## Architecture (trois axes)

- **Métier** (`resolveFleetOperationalStatus`) — busy / assigned / available…  
  `last_known` n’est **plus** un override métier.
- **Présence GPS** (`resolveDriverLocationPresence`) — `live` | `recent` | `stale` | `last_known` | `offline_unknown`.
- **Device** (`tracking_display_status`, ex. `degraded_constrained`) — diagnostic, hors machine de fraîcheur.

Invariant amont figé (mission active) : **immobile ≠ stale** — la fraîcheur se calcule sur l’âge du fix, pas sur le changement de coords. Voir [gps-presence-vs-position-model.md](gps-presence-vs-position-model.md).

## Modules

| Surface | Fichier |
|--------|---------|
| Mobile resolver + labels | `mobile/unified-app/src/features/company/components/maps/driverLocationPresence.ts` |
| Mobile fraîcheur locale | `mobile/unified-app/src/features/company/utils/localDriverLocationFreshness.ts` |
| Web resolver | `frontend/src/utils/fleetDriverLocationPresence.js` |
| Web projections | `frontend/src/utils/companyDriverProjections.js` (`isNonLiveGpsPosition`) |

Seuils **uniques** : `LOCAL_LIVE_MAX_SECONDS = 30`, `LOCAL_RECENT_MAX_SECONDS = 120`.

## Compteur N/T

- `T` = roster flotte (`enriched` / snapshot `/me/drivers/live` fusionné, y compris sans coords).
- `N` = présence `live` | `recent` uniquement.
- Badge gated : `rosterResolved` → sinon « Localisation… » (jamais `0/0` ni T partiel).
- Filtre carte : n’altère jamais N/T ; hint séparé « N affiché(s) ».

## Géométrie

`spatialDrivers = filter(showMarker)` pour markers, clustering, fit bounds, recenter.  
Interdit : placeholder `0,0` pour `offline_unknown`.

## Contrat mobile `position_source`

Propagé via `CompanyDriverLiveLocation`, `normalizeLocation`, merges realtime / freshnessTick.

## Hors scope

- Mutation globale de `getFreshnessStatus` (seuils 20/90/300).
- Backend skip coords / Kafka / P0-E.
- 6ᵉ état `offline` dans le type présence.
