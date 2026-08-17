# P0-D D3-C — chaine Fused → Consumer → Task (Prod 126, HOME 180s)

```text
DEVICE   = SM-S911B USB RFCW20QC53W
BINARY   = Prod 126 (non-DEBUGGABLE)
CYCLE    = TOP healthy → HOME 180s, sample ~10s
DRIVER   = 20135 / mission #38224
PATCH    = NO-GO
```

Artefacts : `docs/ops/_release_exec_p0d_2026-08-16/d3c_delivery/`  
Script : `run_d3c_delivery_chain.ps1`

## Verdict matrice

| Etage | Critere | Resultat |
|-------|---------|----------|
| D3-C1 | FGS true + registration Fused **disparue** | **RULED OUT** |
| D3-C2 | registration presente, **aucun** event consumer | **NOT CONFIRMED** (events presents) |
| D3-C3 | consumer OK, **aucune** invocation task | **NOT CONFIRMED** (TaskService Finished continue) |

```text
FGS shell                 = true toute la fenetre (180s)
getFgsAllowStart          = PROC_STATE_TOP (pas de DENIED)
startRequested            = true
Registration Fused live   = PRESENTE (WorkSource{10905}, Request HIGH_ACCURACY @+8s, hash F318B210)
GmsPassiveListener_FLP    = fixes ~10s toute la fenetre
Location unavailable      = oui des HOME+3s (cadence ~10s)
TaskService Finished      = oui ~toutes les 20s jusqu'a HOME+180s
LOC backend               = OK jusqu'a ~HOME+46s puis STOP
```

### Lecture causale

```text
D3-C1  RULED OUT
  → la subscription Fused (WorkSource package) ne disparait pas avant les LOC

D3-C2  non comme "callback mort"
  → LocationCallback recoit encore onLocationAvailability
  → MAIS isLocationAvailable=false alors que FLP systeme a des fixes
  → rupture qualitative provider→callback Expo (pas une deregistration)

D3-C3  non comme "task jamais invoquee"
  → TaskService Finished 'background-location-task' continue apres l'arret LOC
  → les 3 premiers Finished (~15:57:09 / :29 / :50) alignent les 3 derniers LOC
  → ensuite : task tourne encore, backend n'enregistre plus
  → rupture **task JS → upload/effet** (ou payload vide filtre), pas consumer→TaskManager
```

Nouvelle precision (sous-etage utile) :

```text
D3-C3b — task invoquee (Finished) mais LOC backend = 0
         apres ~HOME+46s
```

Ce n'est **pas** encore "Play Services/Samsung only" : registration + FLP passive OK ; le trou est entre callback Expo / task effect et le backend.

## Timeline (locale device UTC+2)

| T | FGS | Reg 10905 | unavailable | Task Finished | LOC prod (UTC) |
|---|-----|-----------|-------------|---------------|----------------|
| T0 15:57:05 | true | oui | 0 | — | ...13:56:51 |
| HOME+10..40 | true | oui | oui | oui | 13:57:09, :30, :51 |
| HOME+50..180 | true | oui | oui | **oui** | **0** |

Dernier LOC : `2026-08-16T13:57:51Z` ≈ HOME+46s.  
Aucun `stopSelf` / AM Stopping LocationTaskService.

## Alignement avec le modele

- Warning commun Dev/Prod **non causal seul** : confirme (ici + FGS/reg/task encore vivants).
- `DENIED` **pas** observe sur ce cycle clean 180s → reste consequence de paths empoisonnes / recovery BG.
- `debuggable=false` **pas** cause racine de ce cycle (FGS allow reste TOP).

## Statut fige

```text
D1       essentially excluded
D2       RULED OUT
D3-A     contributif / non discriminant
D3-B     non observe
D3-C     LEADING
  D3-C1  RULED OUT (ce cycle)
  D3-C2  NOT as dead callback; availability=false vs FLP OK = signal
  D3-C3  NOT as no-invoke; D3-C3b LEADING (Finished sans LOC)
D3-D     consequence release / environnement

PATCH               NO-GO
GENERAL DISTRIBUTION NO-GO
```

## Suite read-only

1. Instrumenter / correlater **payload** de chaque `Finished task` (taille locations / HTTP) — sans patch runtime prod : logcat OkHttp / telemetry si present, ou build diag.
2. Verifier `deferredUpdatesDistance/Interval` + `shouldReportDeferredLocations` sous `mIsHostPaused` vs mouvements reels.
3. Ne pas whitelister batterie ; ne pas patcher P0-D.
