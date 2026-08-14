# C3 execution — 2026-08-14 (S23 / mission 26)

```text
DIAGNOSTIC                 = CLOSED
ROOT CAUSE A               = CONFIRMED
ROOT CAUSE B               = CONFIRMED
PATCH A / B runtime        = LIVRÉ + canary A+B
DESIGN P0-A                = READY (gps-p0-a-lifecycle-design.md)
C3 GLOBAL (baseline ce doc)= FAIL / P0  ← historique pré-patch
C3 GLOBAL (post A+B)       = PASS ✅ — gps-c3-ab-canary-2026-08-14.md
P0-C                       = OPEN — gps-p0-c-loc-stale-after-pause.md
DRIVER                     = 19
MISSION                    = 26 EN_ROUTE / EN_ROUTE_PICKUP
INSTRUMENTATION            = P0-A nlo_* (session baseline)
CAPTURE                    = docs/ops/_c3_data_2026-08-14/
CAPTURE A+B                = docs/ops/_c3_ab_2026-08-14/
```

Documents liés : [RCA mission 26](gps-mission-26-rca-2026-08-14.md) · [P0-A ticket](gps-p0-a-native-restart-race.md) · [**P0-A design**](gps-p0-a-lifecycle-design.md) · [P0-B](gps-p0-b-headless-auth-hydration.md) · [**Canary A+B / C3 PASS**](gps-c3-ab-canary-2026-08-14.md) · [**P0-C**](gps-p0-c-loc-stale-after-pause.md)

✅ **Implémenté** (suite) : rejeu C3 combiné A+B PASS — voir [gps-c3-ab-canary-2026-08-14.md](gps-c3-ab-canary-2026-08-14.md). Le corps ci-dessous reste la **baseline FAIL** pré-patch. Symptôme post-pause → [P0-C](gps-p0-c-loc-stale-after-pause.md).

---

## État de départ

| Check | Résultat |
|-------|----------|
| ADB wireless | OK (mDNS) |
| reverse 8081 / 15100 | OK |
| Metro | running |
| App | MainActivity + JS bundle |
| booking 26 | EN_ROUTE |
| assignment | EN_ROUTE_PICKUP |
| fgs_running | true (avant stress) |

---

## Matrice — résultats

| Test | Scénario | Verdict | Preuves clés |
|------|----------|---------|--------------|
| 1 | Foreground 2 min | **PASS** | PUT `/location` réguliers 200 ; FGS≈13/14 ; LOC_N=11 ; 0 nlo fail |
| 2 | HOME ×15 | **PASS** (réf.) | FGS 6/6 ; pas de start_failed ; A1 non reproduit (comme prévu) |
| 3 | Shade agressif | **FAIL / A1** | Voir preuve A1 ci-dessous |
| 4 | Shade+HOME ×10 | **PASS partiel** | FGS 4/4 pendant fenêtre courte ; pas de nouveau fail immédiat |
| 5 | BG 15/30/60 s | **PASS partiel** | FGS majoritairement true ; LOC faible mais non nul |
| 6 | Lock / unlock | **FAIL** | FGS tombe ; `ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED` ; nfix→400–500 s ; LOC→0 |
| 7 | Shade→POWER ×5 | **FAIL** | Même erreur Expo ; FGS_TRUE_RATIO 0/3 |
| 8 | Recents | **FAIL** | Cascade nlo + ERR_FOREGROUND ; flips AppState élevés |
| 9 | Oscillation max | **FAIL** | 11 cycles ; rafale `ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED` + `NativeTaskInactive` |
| 10 | Anti-zombie 3 min | **FAIL** | Détection OK (`fgs_not_running` / `native_start_failure`) ; **restart natif n’obtient pas FGS** ; nfix→310 s |
| 11 | Réseau OFF→ON | **FAIL** (USB) | ADB USB survivé pendant OFF ; Network Error / health_send_failed ; **0 PUT** après ON ; queue_depth=0 (pas de buffer local utile) ; HTTP 429=0 ; anti_zombie + ERR_FOREGROUND persistent |
| 12 | Stabilisation 5 min | **FAIL** | 5 min FG USB : FGS_TRUE=0/5 ; nfix **712→1308 s** ; LOC=0 ; PUT=0 ; `auth_not_usable`×5 ; runtime **reste dégradé** |

---

## Preuve A1 (démontrée) — TEST 3 ~10:23:34

Critère satisfait : **START et STOP concurrents** puis `start_failed`.

```text
10:23:34.179  start_requested
              start_attempt_id = nlo_start_cap_mssok1co_ossd3pn2ih
              start_in_flight  = 1
              stop_in_flight   = 0

10:23:34.235  stop_requested
              stop_attempt_id  = nlo_stop_cap_mssok1f6_e666w0x0cf
              stop_in_flight   = 1
              start_in_flight  = 1          ← chevauchement

10:23:34.250  start_failed
              start_attempt_id = nlo_start_cap_mssok1co_ossd3pn2ih
              error_name       = NativeTaskInactive
              (puis nouvel start_requested nlo_start_cap_mssok1t0_…)
```

Fichier : `_c3_data_2026-08-14/` + logcat process ; health post-fail dans `snap_A1_*.txt`.

### Code Expo discriminant (après lock / osc)

```text
error_name = Error
error_code = ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED
error_message = Call to function 'ExpoLocation.startLocationUpdatesAsync' has been rejected.
```

Exemple health :

```text
[nlo_start_…] ensure_manager_state:fgs_recover:
Error/ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED:
Call to function 'ExpoLocation.startLocationUpdatesAsync' has been rejected
```

→ L’inconnu A (« pourquoi rejected ? ») est maintenant **partiellement levé** : Android refuse le démarrage FGS hors fenêtre foreground autorisée, souvent pendant / juste après transitions AppState (shade, lock, recover en background).

---

## P0-B (inchangé)

Toujours observé pendant la session :

```text
tracking.background.task.skipped / resume.rejected_stale_owner
reason = auth_not_usable
```

Bug confirmé ; **non corrigé** (NO-GO).

---

## Anti-zombie (T10)

| Étape | Résultat |
|-------|----------|
| 1. Détecte stale / fgs_not_running | ✅ |
| 2. Demande restart (`fgs_recover`, `native_start_failure`) | ✅ |
| 3. Obtient runtime natif actif | ❌ (`ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED` répété ; nfix croît) |

---

## C3 — acceptation (matrice RCA)

| # | Scénario | Verdict run |
|---|----------|-------------|
| 1 | Foreground continu | PASS |
| 2 | FG→BG | FAIL (après stress / lock) |
| 3 | BG→FG | FAIL (double START / recover fail) |
| 4 | Écran lock | FAIL |
| 5 | Unlock | FAIL (pas de restauration durable) |
| 6 | Réseau OFF→ON | FAIL (0 PUT reprise ; runtime déjà mort) |
| 7 | Oscillation AppState | FAIL (A1 + ERR_FOREGROUND) |
| 8 | Anti-zombie | FAIL (détecte, ne restaure pas) |
| 9 | Headless | FAIL (`auth_not_usable`) |
| 10 | Native runtime cohérent | FAIL |
| 11 | Cadence ≤30 s | FAIL (nfix → 1308 s en T12) |
| 12 | Observabilité | **PASS** (nlo_* + `ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED`) |

```text
C3 GLOBAL = FAIL / P0
```

---

## GO / NO-GO

| Action | Statut |
|--------|--------|
| Patch fonctionnel P0-A | ❌ NO-GO (attendre GO explicite) |
| Patch fonctionnel P0-B | ❌ NO-GO |
| Rejouer T11/T12 après reconnect ADB | recommandé |
| Concevoir correctifs A et B **séparés** | OK conceptuellement |

### Orientation future P0-A (non implémentée)

- Ne jamais appeler `startLocationUpdatesAsync` si `AppState !== active` (déjà partiellement deferred) **et** si une opération STOP est in-flight.
- Sérialiser strictement STOP→START ; backoff sur `ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED`.
- Lock/unlock : différer recover jusqu’à foreground réel stable.

### Orientation future P0-B (non implémentée)

- Hydrater `SESSION_AVAILABLE` au login (ticket dédié).

---

## Implémentation

### T11 / T12 (USB) — suite

```text
ADB USB                    = RFCW20QC53W
T11 OFF                    = Network Error (health) ; ADB reste device
T11 ON                     = Wi-Fi Holding-BA.AL ; PUT=0 ; FGS toujours false
T11B lock+net              = nfix 712→984 ; anti_zombie_fix_stale
T12 5 min                  = nfix → 1308 ; FGS=0 ; PUT=0 ; auth_not_usable×5
```

✅ **Implémenté** : matrice C3 T1–T12 exécutée ; preuve A1 ; code Expo `ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED` ; captures `_c3_data_2026-08-14/` ; **aucun patch A/B**.

**Reste à faire** : GO explicite pour concevoir correctifs P0-A et P0-B **séparés** (pas avant).
