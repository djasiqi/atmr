# P0-F UI — Présence GPS flotte

**Statut** : implémenté côté UI (mobile + web).  
**Cockpit LIRIE N/T** : **PASS / LOCKED** (2026-09-05) — ne plus redessiner pendant le debug GPS.  
**Merge** : uniquement après canary P0-F téléphone (Android → analyse → iOS → puis UI).  
**PR** : séparée du canary GPS / Kafka / P0-E.

## Contrat verrouillé — cockpit GPS coverage

```text
UX GPS COVERAGE       = PASS / LOCKED
GPS DATA SEMANTICS    = LOCKED (N/T n’est pas décoratif)
DEVICE / BACKGROUND   = à valider séparément
BATTERY CONCLUSION    = interdite avant validation terrain
```

- `T` = chauffeurs **actifs** uniquement (`is_active !== false`).
- `N` = chauffeurs actifs dont la présence est réellement `live` | `recent` après vieillissement.
- Aging = toutes les 5 s. `LIVE ≤ 30 s`, `RECENT ≤ 120 s`, `> 120 s` = hors ligne.
- Header : `Date | ● LIVE | N/T en direct | Notifications`.
- Barre ops : `à assigner | en cours | en retard | disponibles` — jamais de couverture GPS.
- Tap chauffeur (feuille) : fermer → sélectionner le marqueur → recentrer si position connue.
- `fleet-map` : badge carte conservé, pas de capsule LIVE dupliquée.

Si `N/T` est faux (ex. `1/7`), traiter **source de données / présence / timestamps / transport**. Ne pas rouvrir l’UX cockpit.

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

- `T` = flotte **active** uniquement (`is_active !== false`) — snapshot `/me/drivers/live` avec `active_only=True`, pas les comptes historiques.
- `N` = présence `live` | `recent` uniquement, vieillie localement (timeout 30 s / 120 s). Un `location_status=live` périmé ne reste pas dans N.
- `0/T` et `T/T` : même capsule, ratio explicite, pas d’état visuel supplémentaire.
- Badge gated : `rosterResolved` → sinon « Localisation… » (jamais `0/0` ni T partiel).
- Filtre carte : n’altère jamais N/T ; hint séparé « N affiché(s) ».
- ✅ **Implémenté** : cockpit LIRIE — le compteur n’est plus une pastille flottante en haut à gauche. Il est intégré à la capsule LIVE du bandeau (`● LIVE | N/T en direct`). Clic → feuille « Suivi en direct » (hors-ligne : `Dernière position : il y a X min`). Toucher un chauffeur ferme la feuille, sélectionne le marqueur et recentre la carte. La barre « à assigner / en cours / en retard / disponibles » reste exclusivement opérationnelle.
- Carte standalone / inline (`fleet-map`, `EnterpriseLiveMap`) : badge carte conservé.

## Géométrie

`spatialDrivers = filter(showMarker)` pour markers, clustering, fit bounds, recenter.  
Interdit : placeholder `0,0` pour `offline_unknown`.

### Cockpit map full-bleed (layout uniquement)

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

✅ **Implémenté** : la carte cockpit (`absoluteFill`) occupe toute la scène. La barre d’onglets entreprise est en overlay (`tabBarStyle.position = absolute` + wrapper `CompanyFloatingTabBar`) : elle n’ôte plus sa hauteur à la map. « Prochaines courses » utilise `resolveUpcomingRidesBottomOffset` (offset nav complet, plus de plafond 14 px). Safe-area inférieure = `paddingBottom` de la pilule, pas un `paddingBottom` / `tabBarHeight` sur `MapView`. Aucun changement de sémantique GPS.

Fichiers : `floatingTabScreenOptions.ts`, `(company)/_layout.tsx`, `CompanyFloatingTabBar.tsx`, `companyFleetCockpitLayout.ts`, `DriverBottomSheet.tsx`, `rides.tsx` (clearance liste après overlay).

## Contrat mobile `position_source`

Propagé via `CompanyDriverLiveLocation`, `normalizeLocation`, merges realtime / freshnessTick.

## Hors scope

- Mutation globale de `getFreshnessStatus` (seuils 20/90/300).
- Backend skip coords / Kafka / P0-E.
- 6ᵉ état `offline` dans le type présence.
