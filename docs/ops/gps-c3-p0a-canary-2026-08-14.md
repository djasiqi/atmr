# Canary P0-A — rejeu scénarios A (2026-08-14)

```text
CANARY                 = P0-A seul
GIT TAG                = gps-canary-p0a-2026-08-14
COMMIT                 = 479cd60d
DRIVER                 = 19
MISSION                = 26
P0-B                   = NON hydraté (auth_not_usable toujours possible en headless)
CAPTURE                = docs/ops/_c3_p0a_2026-08-14/
FENÊTRE                = ~14:55–15:11 Genève
```

## Critère principal P0-A

```text
start_in_flight=1 ∧ stop_in_flight=1   → JAMAIS observé (Metro nlo_*)
ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED
  provoqué par notre orchestration     → 0 occurrence
NATIVE_ERR_N (health)                  → 0 sur tous les snaps
```

## Pré-check (avant stress)

| Check | Résultat |
|-------|----------|
| Cold / session driver 19 | OK (`SESSION_READY`, reconcile `driver:19`) |
| Mission 26 | OK |
| FGS notification | OK (`FOREGROUND_SERVICE`, « Mission en cours ») |
| `fgs_running` / `native_task_running` | true |
| PUT /location | OK (LOC mission 26 réguliers) |
| concurrent in_flight | 0 |
| spam nlo_start | 0 |

→ **PRE-CHECK PASS**

## Matrice scénarios A

| Test | Scénario | concurrent ∧ | err_fg | fails | FGS fin | LOC | Verdict A |
|------|----------|--------------|--------|-------|---------|-----|-----------|
| P0 | Pré-check | 0 | 0 | 0 | 9/17 | 9 | **PASS** |
| T2 | HOME ×15 | 0 | 0 | 0 | dip 0/25 pendant stress* | 9 (staleness courte) | **PASS A1** (pas de race) |
| T3 | Shade ×25 | 0 | 0 | 0 | récupère 2→true | +3 LOC | **PASS** (était FAIL/A1 baseline) |
| HOME↔app | ×12 | 0 | 0 | 0 | 4/25 puis true | OK | **PASS** |
| T6 | Lock/unlock ×5 | 0 | 0 | 0 | 5/25 | 14 | **PASS A1** (baseline FAIL + ERR_FOREGROUND) |
| T9 | Oscillation ×12 | 0 | 0 | 0 | 7/25 | 14 | **PASS A1** |
| T10 | Anti-zombie 3 min BG | 0 | 0 | 0 | **15/25** fin true | **19** | **PASS** (baseline : détecte mais runtime mort) |
| T12 | Stabilisation 5 min FG | 0 | 0 | 0 | **24/25** | **20** | **PASS A** (baseline FGS=0 LOC=0 nfix→1308) |

\* Pendant HOME agressif, health montre un **dip FGS** (`fgs_not_running`) sans `native_start_error` ni chevauchement START/STOP. Reprise ensuite (shade / home↔app / lock).

## Preuve Metro (T3 shade — seul snap avec nlo_*)

```text
starts=10 stops=8 concurrent_both=0 err_fg=0 fails=0
```

Fichier : `metro_T3_shade.txt` (15 ko). Aucune ligne avec `start_in_flight=1` et `stop_in_flight=1` simultanés.

## Nuances / hors critère A strict

- `auth_not_usable` / headless skip : toujours possible (**P0-B**, volontairement non patché).
- T12 : `constraint_reason=fix_stale` avec `fix`/`nfix` élevés alors que **FGS=true** et **LOC continuent** → GNSS / métrique age, pas mort runtime C3-T12.
- Dip FGS sous HOME×15 : à surveiller, mais **pas** la race A1 ni `ERR_FOREGROUND` d’orchestration.

## Verdict

```text
ROOT CAUSE A       = CONFIRMED
PATCH A            = IMPLEMENTED
CANARY A           = PASS ✅

P0-B               = depuis : IMPLEMENTED (unitaires) / CANARY B PENDING
C3 GLOBAL          = PASS ✅ (canary A+B — gps-c3-ab-canary-2026-08-14.md)
SUITE              = P0-C (gps-p0-c-loc-stale-after-pause.md) — ne pas rouvrir A
```

```text
P0-A CANARY (critère concurrence + ERR_FOREGROUND orchestration) = PASS
C3 GLOBAL historique (baseline pré-A)                            = FAIL (obsolète pour A seul)
```

**Prochaine étape** : canary B ciblé (sans C3 complet) — [gps-p0-b-headless-auth-hydration.md](gps-p0-b-headless-auth-hydration.md).

## Implémentation

✅ **Implémenté** : rejeu automatisé ADB des scénarios A (HOME, shade, HOME↔app, lock, osc, anti-zombie, stabilize 5 min) + captures PG/Metro dans `_c3_p0a_2026-08-14/`. Critère principal P0-A **PASS**.
