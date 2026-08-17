# P0-D D5 — chaîne mobile post-cut (read-only) — VERDICT

```text
PROD                   = 286737a2 ✅
BACKEND                = GELÉ
D4-B                   = hors scope runtime
HOT-PATCH              = ROLLBACK ✅
ANDROID 126            = NOT READY
GENERAL DISTRIBUTION   = NO-GO
PATCH                  = NO-GO
CAPTURE                = 2026-08-16T16:03:30Z → 16:08:53Z (device local +2)
DEVICE                 = SM-S911B USB RFCW20QC53W / driver 20135 / build 126
```

Artefacts : `docs/ops/_release_exec_p0d_2026-08-16/d5_task_chain/`  
Script : `run_d5_task_chain.ps1` (FG 90s → HOME 180s).

## Question

> Après le cut, la `background-location-task` Expo est-elle encore réellement exécutée avec un payload Location, et jusqu’où le JS progresse-t-il avant l’absence de PUT ?

## Verdict (étage 1)

```text
PREMIER ÉTAGE QUI CESSE = FLP → LocationTaskConsumer (payload Location)
                         PAS encore TaskService / JS / enqueue / flush / HTTP
```

Discriminant retenu : **D5-B** (delivery native → JS sans locations).

| Étape | Observation D5 | Statut |
|-------|----------------|--------|
| FGS `LocationTaskService` | vivant, `isForeground=true`, `startRequested=true` | OK |
| Cut HOME | `MainActivity onTop=false` @ 18:05:22 ; launcher TOP ; `curProcState=4` (FGS) | OK |
| FLP | `GmsPassiveListener_FLP` reçoit encore des fixes (speed≈0) | OK (système) |
| Delivery app | `FusedLocation … blocked - too close` / `too fast` (centaines) | **COUPÉ** |
| `LocationTaskConsumer` | `Location unavailable for foreground-service task delivery` ~toutes les **10s**, **FG + HOME** | **COUPÉ** |
| `TaskService Finished` / `background-location-task` | **0** sur toute la fenêtre | **jamais atteint** |
| Télémétrie JS task / enqueue | 0 | N/A |
| HTTP PUT / LOC PG (15 min) | **0** | N/A |

Chaîne observée :

```text
FLP fix (système)
  → delivery vers request Expo BLOQUÉE (too close / too fast)
  → LocationTaskConsumer : Location unavailable (cadence ~10s)
  → PAS de locationBundles → PAS d’executeTask
  → PAS de TaskService Finished 'background-location-task'
  → JS task / enqueue / flush / PUT jamais entrés
```

## Contrôle validité du run

- HOME **effectif** (pas un faux FG) : activity top = launcher ; process Liri `curProcState=4` (FOREGROUND_SERVICE), pas TOP.
- Le dumpsys FGS `getFgsAllowStart=PROC_STATE_TOP` est un **raison d’allow historique**, pas l’état process courant.
- Silence PUT/LOC **déjà en FG** (90s) : le run est **stationnaire** ; ce n’est pas un discriminant « HOME only », c’est un discriminant **« aucun nouveau fix livré à Expo »**.

## Alignement canary 15:23:19Z / D3-C

| Source | Task Finished | unavailable | PUT/LOC |
|--------|---------------|-------------|---------|
| Canary P0-D post-cut | logcat JS insuffisant | — | 0 PUT après 15:23:19Z (backlog drain puis silence) |
| D3-C (plus tôt) | Finished **continue** ~20s | oui ~10s | LOC stop ~HOME+46s (puis D4-B explique une partie) |
| **D5 (maintenant)** | Finished **0** | oui ~10s **FG+HOME** | 0 PUT / 0 LOC |

Lecture unifiée : quand FLP **ne livre plus** de positions au consumer Expo, la task JS **n’entre pas**.  
D3-C montrait un étage plus bas (Finished sans LOC) sous un autre régime (fixes encore livrés / backlog).  
D5 isole l’étage **amont** dominant en stationnaire / post-drain : **pas de payload Location → pas de task**.

## Hypothèses fermées / ouvertes

Fermées pour cette fenêtre :

- D5-A « TaskManager mort / task plus invoquée » au sens FGS down → **non** (FGS up ; consumer ticke).
- D5-C/D/E (gate JS, queue, transport) → **non atteignables** (pas d’entrée JS).

Ouvertes (pas de patch) :

- Pourquoi la request Expo reçoit `unavailable` + `too close/too fast` alors que FLP passive voit des fixes (filtre distance/intervalle request vs provider).
- Rejouer D5 **en mouvement** pour voir si Finished réapparaît (confirmer le filtre FLP comme levier unique).

## Statut figé

```text
NOM = RELEASE-ONLY EXPO LOCATION DELIVERY FAILURE
D5-B = CONFIRMED DIFFERENTIAL ✅ (A/B stationnaire Prod vs Dev)
  stationnaire comme cause = RULED OUT
  premier étage = delivery FLP → LocationTaskConsumer (release)
PATCH = NO-GO
BACKEND = GELÉ
```

A/B + diff 4 familles :
- [`../d5_ab_stationary/D5_AB_STATIONARY.md`](../d5_ab_stationary/D5_AB_STATIONARY.md)
- [`../d5_release_only/D5_RELEASE_ONLY_DIFF.md`](../d5_release_only/D5_RELEASE_ONLY_DIFF.md)

## Lien modèle produit

Le modèle figé **POSITION ≠ PRÉSENCE** ([gps-presence-vs-position-model.md](../../gps-presence-vs-position-model.md)) exige un heartbeat de présence même à coords identiques.  
Sur Prod126 le heartbeat n’arrive pas faute de **delivery Expo** ; l’A/B montre que ce n’est **pas** « stationnaire = pas de fix » (Dev livre à l’arrêt).
