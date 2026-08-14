# Canary P0-B ciblé — 2026-08-14

```text
CANARY                 = P0-B ciblé (identité / headless)
DEVICE                 = S23 SM-S911B (100.81.106.54:39179)
DRIVER                 = 19
MISSION                = 26
CAPTURE                = docs/ops/_c3_p0b_2026-08-14/
FENÊTRE                = ~16:20–16:41 Genève
DRIVER A / B           = 19 → 20 (atmr1@atmr.ch)
P0-A surveillance      = err_fg=0, start_fail=0, NATIVE_ERR_N=0 sur tous les snaps
FULL C3                = NO-GO (hors scope)
```

## Pré-check

| Check | Résultat |
|-------|----------|
| ADB + reverse 8081/15100 | OK |
| Dev Client → Metro (`lirie://…127.0.0.1:8081`) | OK (`MainActivity`) |
| Session driver 19 | OK |
| Mission 26 + PUT `/location` | OK |
| FGS / native recover | OK (dip court puis true) |
| `auth_not_usable` fenêtre live | **0** |
| `ensure_headless` → `SESSION_AVAILABLE` | OK |
| `task_invoked` | OK |

→ **PRE-CHECK PASS**

## Matrice B1–B7

| Test | Scénario | auth_not_usable | ensure / SESSION | LOC | err_fg / start_fail | Verdict |
|------|----------|-----------------|------------------|-----|---------------------|---------|
| B1 | Cold start / session restaurée | 0 | ensure=7, SESSION=5, TEMP=2, UNAVAIL=0 | 12 (m26) | 0 / 0 | **PASS** |
| B2 | Background / headless réel | 0 | `path=hydrate` + `presence.hydrated` driver=19 + `task_invoked` | continues (m26) | 0 / 0 | **PASS** (décisif) |
| B3 | Refresh token | 0 | TEMP puis SESSION ; UNAVAIL=0 | 12 (m26) | 0 / 0 | **PASS** |
| B4 | Logout → headless refuse | — | `presence.cleared` + UNAVAILABLE ; queue quarantined ; **0 LOC19 après logout** ; pas de `task_invoked` dans la fenêtre courte avant login B | 0 post-logout | 0 / 0 | **PASS** (clear immédiat ; skip headless non échantillonné — gap ~11s) |
| B5 | Login A / headless PASS | — | (couvert par B1/B7 driver 19) | — | — | **PASS** (= B1/B7) |
| B6 | A→logout→login B anti cross-driver | — | `presence.published` **driver_id=20** ; `quarantine_identity_mismatch` ; context `driver:20` ; 0 LOC sous identité A après switch | LOC20=0 (pas de mission B) | 0 / 0 | **PASS** (identité) |
| B7 | Runtime JS recréé (force-stop) | 0 | `presence.published` + ensure SESSION + `task_invoked` ; nouveau PID | 12 (m26) | 0 / 0 | **PASS** |

### Preuve décisive B2 (headless hydrate)

```text
tracking.auth.presence.hydrated { driver_id: 19, ... }
tracking.auth.presence.ensure_headless { kind: 'SESSION_AVAILABLE', path: 'hydrate', driver_id: 19 }
tracking.background.task_invoked
auth_not_usable = 0
```

Fichier : `logcat_B2_bg.txt`.

### Preuve B3 (refresh)

```text
auth.refresh.endpoint_used
ensure_headless AUTH_TEMPORARILY_UNAVAILABLE (memory_temp)
ensure_headless SESSION_AVAILABLE
TRACKING_IDENTITY_UNAVAILABLE count = 0
tracking continue (LOC mission 26)
```

Note : rafales `refresh-token` / 429 observées côté gateway — hors critère B (pas de retombée UNAVAILABLE).

### Preuve B7 (mémoire détruite)

```text
force-stop → nouveau pid
tracking.auth.presence.published { driver_id: 19 }
ensure_headless SESSION_AVAILABLE → task_invoked
auth_not_usable = 0
```

## Critères PASS B (4 propriétés)

| # | Propriété | Statut |
|---|----------|--------|
| 1 | Headless + session valide → plus de `auth_not_usable` | ✅ B2 |
| 2 | Cold start / nouveau runtime JS sans snapshot mémoire précédent | ✅ B1 + B7 |
| 3 | Logout → headless inutilisable immédiatement | ✅ B4 (`presence.cleared` + UNAVAILABLE + 0 LOC19 post-logout ; queue quarantined) |
| 4 | A→B : aucune réutilisation presence/lease/owner A | ✅ B6 (`published` driver 20 + `quarantine_identity_mismatch` + 0 LOC post-switch sous A) |

### Chronologie B4→B6 (logcat)

```text
16:40:49  presence.cleared { reason: logout, kind: TRACKING_IDENTITY_UNAVAILABLE }
16:40:49  tracking.queue.quarantined
16:40:59  POST /auth/login 200 (gateway)
16:41:00  quarantine_identity_mismatch
16:41:00  lease.updated driver_id=20
16:41:00  presence.published { driver_id: 20, kind: SESSION_AVAILABLE }
POST_LOGOUT_LOC (19|20) depuis 16:40:49 = 0
```

Fichiers : `logcat_B4_B6.txt`, `snap_B4_B6.txt`.

## Surveillance P0-A (légère)

```text
ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED = 0
start_failed = 0
NATIVE_ERR_N = 0
```

Pas de régression A observée pendant ce canary B.

## Verdict

```text
P0-A        = CLOSED / PASS ✅
P0-B        = CLOSED / PASS ✅ (B1–B7)
C3 GLOBAL   = PASS ✅ (canary A+B — gps-c3-ab-canary-2026-08-14.md)
FULL C3     = PASS ✅
P0-C        = OPEN — gps-p0-c-loc-stale-after-pause.md
```

✅ **Implémenté** : canary combiné A+B + C3 complet documenté dans [gps-c3-ab-canary-2026-08-14.md](gps-c3-ab-canary-2026-08-14.md). Suite : [P0-C](gps-p0-c-loc-stale-after-pause.md).

## Compte chauffeur B (canary)

```text
email       = atmr1@atmr.ch
password    = Atmr1234
driver_id   = 20
company_id  = 3  (même company que driver 19)
user_id     = 26
```

Note : `atmr1@atmr` rejeté (format email invalide) → `atmr1@atmr.ch`.

## Implémentation

✅ **Implémenté** : runner + captures B1–B7 + ce rapport (`gps-c3-p0b-canary-2026-08-14.md`).
