# Canary A+B + FULL C3 — 2026-08-14

```text
CANARY                 = A+B combiné (P0-A lifecycle + P0-B presence hydrate)
DEVICE                 = S23 SM-S911B
DRIVER                 = 19 (surveillance d20_hits=0)
MISSION                = 26 EN_ROUTE / EN_ROUTE_PICKUP
CAPTURE                = docs/ops/_c3_ab_2026-08-14/
FENÊTRE PRINCIPALE     = ~16:57–17:22 Genève
REPRISE                = ~18:12–18:18 (pré-check + T12 rejeu)
ADB                    = 100.81.106.54:40175 (reprise ; run principal :39179)
```

## Freeze officiel (figé)

```text
P0-A                 CLOSED / PASS ✅
P0-B                 CLOSED / PASS ✅
CANARY A+B            PASS ✅
FULL C3               PASS ✅
C3 GLOBAL             PASS ✅

P0-C                  OPEN (scindé)
P0-C-NATIVE           CONFIRMED — gps-p0-c-native.md
P0-C-LEDGER           CONFIRMED — gps-p0-c-ledger.md
C-SEQUENCING          CONTRIBUTING / PARTIAL
C4 replay 17:20       EXCLUDED
PATCH P0-C*           NO-GO
```

Preuves **16:57–17:22** suffisantes pour A/B/C3. Suite incident : [P0-C parent](gps-p0-c-loc-stale-after-pause.md).

## Pré-check (run principal)

| Check | Résultat |
|-------|----------|
| Metro + reverse 8081/15100 | OK |
| Session driver 19 | OK |
| Mission 26 | OK |
| `auth_not_usable` | 0 |
| `concurrent_both` / `err_fg` / `start_fail` | 0 |
| `d20_hits` | 0 |

→ **PRE-CHECK PASS**

## Matrice C3 (A+B) — run principal

Tous les `SIG_*` du timeline : `auth_not_usable=0`, `concurrent_both=0`, `err_fg=0`, `start_fail=0`, `d20_hits=0`.

| Test | Scénario | Verdict A+B | Notes |
|------|----------|-------------|-------|
| P0 | Pré-check FG | **PASS** | ensure/SESSION OK |
| T3 | Shade | **PASS** | err_fg=0 |
| T2 | HOME | **PASS** | pid churn OK |
| T_home_app | HOME↔app | **PASS** | |
| T8/T9 | Recents / osc | **PASS** | vs baseline FAIL (ERR_FOREGROUND) |
| T6 | Lock 30/60 s | **PASS** | vs baseline FAIL |
| T7 | Shade→lock | **PASS** | |
| T5 | BG 15/30/60 s | **PASS** (nuance) | pid logcat parfois vide ; DB/FGS = vérité terrain |
| T10 | Anti-zombie 3 min | **PASS** | recover + LOC |
| T11 | Net OFF→ON | **PASS** | vs baseline FAIL |
| T_cold | Force-stop / cold | **PASS** | FGS/nat OK ; LOC m26 OK |
| T12 | Stabilisation 5 min | **PASS** | FGS true ; LOC m26 jusqu’à 17:20:10 ; nfix≈20 s ; auth=0 |

Timeline : `T12_STABILIZE_5MIN_START` 17:15:02 → `SIG_T12_stabilize` 17:20:06 → `DONE_FULL_C3_AB` 17:22:49.

### Preuve T12 (principale)

```text
SIG_T12_stabilize auth_not_usable=0 concurrent_both=0 err_fg=0 start_fail=0 d20_hits=0
FGS_19 / NATIVE_19 = true (health récent)
LOC19 … 17:20:10 mission=26
BOOK26 (26, EN_ROUTE, 19)
```

Fichiers : `summary_T12_stabilize.txt`, `snap_T12_stabilize.txt`, `logcat_T12_stabilize.txt`.

## Pause / reprise

`PAUSE.txt` notait un arrêt pendant T12 (~17:19). En fait le runner a **terminé** T12 + `DONE_FULL_C3_AB` (captures présentes).

Reprise demandée 18:12 (ADB `:40175` + Metro) :

| Étape | Résultat |
|-------|----------|
| Pré-check reprise | Fragile : `auth_not_usable=1` (1 hit), FGS dip, LOC fenêtre 15 m = 0 |
| T12 rejeu 5 min | Orchestration A/B **OK** (`auth_not_usable=0`, err_fg=0, overlap=0, FGS 17/20) |
| Cadence LOC reprise | **FAIL** : `MAX_LOC19` toujours 17:20:10 ; health `fix_stale` (nfix 55→358 s) ; acks `duplicate` / `ingested_non_persisted` ; queue_depth≈35 |

→ La reprise **ne réfute pas** le PASS de la matrice principale. Symptôme promu en **[P0-C](gps-p0-c-loc-stale-after-pause.md)** (runtime sain, LOC stagnant, acks `duplicate` / `ingested_non_persisted`). Aucun patch en cours.

Captures reprise : `summary_resume_*.txt`, `snap_resume_*.txt`, `run_resume_t12.ps1`.

## Critères vs baseline C3 (pré-patch)

| Critère baseline FAIL | A+B run principal |
|----------------------|-------------------|
| START∧STOP overlap (A1) | 0 |
| `ERR_FOREGROUND…` orchestration | 0 |
| Lock / osc / anti-zombie dead | PASS |
| `auth_not_usable` headless | 0 sur toute la matrice |
| T12 FGS + LOC + nfix sain | PASS (17:15–17:20) |
| Cross-driver 20 | d20_hits=0 |

## Verdict

```text
P0-A                   = CLOSED / PASS ✅
P0-B                   = CLOSED / PASS ✅
CANARY A+B             = PASS ✅
FULL C3                = PASS ✅
C3 GLOBAL              = PASS ✅
P0-C-LEDGER-CLIENT     = CONFIRMED
P0-C-LEDGER-SERVER     = CONFIRMED
P0-C-NATIVE            = CLOSED / REQUALIFIED (N4)
PATCH P0-C*            = NO-GO
```

## Implémentation

✅ **Implémenté** : matrice C3 A+B ; freeze A/B/C3 PASS ; diagnostic causal P0-C clos (ledger HOL, NATIVE requalifié) — [gps-p0-c-loc-stale-after-pause.md](gps-p0-c-loc-stale-after-pause.md).
