# Certification états chauffeur — FG / BG (device canary unique)

Référence produit : [`docs/contracts/gps-driver-product-contract.md`](../contracts/gps-driver-product-contract.md).  
Device GPS (B2/B3/POC-1A) : [`_release_exec_p0d_2026-08-16/b2_canary_78a1c73c/B3_STATUS.md`](./_release_exec_p0d_2026-08-16/b2_canary_78a1c73c/B3_STATUS.md).  
Flotte multi-devices : [`gps-fleet-e2e-certification.md`](./gps-fleet-e2e-certification.md) = **HOLD**.

```text
FLEET 3/10/20 DEVICES              = HOLD
E2E FLEET CERTIFICATION            = HOLD

DEVICE GPS CERTIFICATION           = CLOSED ✅
GPS DEVICE DEV                     = FREEZE ✅

DRIVER STATE CERT                  = OPEN ★
  ordre                            = C01→C11 strict
  C01–C09                          = PASS ✅
  C10 IN_PROGRESS / BG             = HOLD ★ NEXT
  ARRIVED-SOT-1/1B/2A-D            = PASS ✅
  NEXT ★                           = C10 IN_PROGRESS / BG (mission 54)
  C07                              = _driver_state_cert_2026-08-21/C07_FG_PASS.md
  C08                              = _driver_state_cert_2026-08-21/C08_BG_PASS.md
  C09                              = _driver_state_cert_2026-08-21/C09_FG_PASS.md
  SOT2D                            = _driver_state_cert_2026-08-21/ARRIVED_SOT2D_PASS.md

PROD                               = NO-GO
```

**Un device canary** (Samsung). Pas de nouveau développement GPS. Ne pas rouvrir B2/B3/POC-1A sauf régression factuelle.

---

## Définition FG / BG

```text
FOREGROUND
= app chauffeur visible à l'écran

BACKGROUND
= app envoyée au HOME
+ non force-stoppée
+ chauffeur toujours EN SERVICE
+ FGS attendu vivant
```

Deep Doze / écran verrouillé = déjà certifié B3. Ici : **chaque état métier continue correctement quand l’app quitte le premier plan**.

---

## Mapping états métier → statut tracking

| État certif | Statut produit (mobile) | Mode attendu | Cadence P9 |
|-------------|-------------------------|--------------|------------|
| S0 Sans mission | `mission_id=null` | `availability_presence` | ≈60 s |
| S1 Mission assignée | `ASSIGNED` (dans fenêtre T−lead/T+grace) | `mission_live` | ≈20 s |
| S2 En route | `EN_ROUTE` | `mission_live` | ≈20 s |
| S3 Arrivé | `ARRIVED` | `mission_live` | ≈20 s |
| S4 À bord | `IN_PROGRESS` | `mission_live` | ≈20 s |

`ARRIVED` **≠ fin de tracking** — LIVE continue.  
Retour terminal : mission terminale → LIVE 20 s → PRESENCE 60 s si toujours EN SERVICE (soft, sans stop/restart).

---

## Matrice C01→C10

| ID | État | App | Mode | Cadence | Verdict |
|----|------|-----|------|--------:|---------|
| C01 | Sans mission | FG | PRESENCE | ~60 s | ✅ PASS |
| C02 | Sans mission | BG | PRESENCE | ~60 s | ✅ PASS |
| C03 | ASSIGNED éligible | FG | LIVE | ~20 s | ✅ PASS |
| C04 | ASSIGNED éligible | BG | LIVE | ~20 s | ⛔ FAIL |
| C05 | EN_ROUTE | FG | LIVE | ~20 s | ✅ |
| C06 | EN_ROUTE | BG | LIVE | ~20 s | ✅ |
| C07 | ARRIVED | FG | LIVE | ~20 s | ✅ PASS |
| C08 | ARRIVED | BG | LIVE | ~20 s | ✅ PASS |
| C09 | IN_PROGRESS (à bord) | FG | LIVE | ~20 s | ✅ PASS |
| C10 | IN_PROGRESS (à bord) | BG | LIVE | ~20 s | ✅ PASS |
| C11 | TERMINALE → PRESENCE | FG | PRESENCE | ~60 s | ✅ PASS |

Puis transitions retour (même certif) :

```text
C11  MISSION TERMINALE → LIVE 20 → PRESENCE 60 (EN SERVICE)
     soft 20000→60000 · Unregister=0 · FLP_REMOVE=0 · FGS restart=0
```

---

## Grille de preuve commune (chaque C0x)

```text
STATE / app_state
driver_id / mission_id / mission_status
tracking_mode / task_mode
FGS / owner_gen / tracking_session_id
P9_count / P9_median_delta
last_event_id / capture_id / recorded_at / lat/lng
PUT / backend_ingest / backend_persist / driver_projection / carte
Unregister / FLP_REMOVE / FGS_restart
VERDICT
```

### Corrélation d’au moins un événement par état

```text
P9.event_id / capture_id / recorded_at
  =
PUT.event_id / capture_id / recorded_at
  =
POSTGRES.event_id / capture_id / recorded_at
  =
PROJECTION / CARTE (driver_id, recorded_at, lat/lng)
```

### Invariants natifs (tous les C0x sauf S0→S1 soft attendu)

```text
Unregister        = 0
FLP_REMOVE        = 0
FGS restart       = 0
owner rotation    = 0   (sauf switch mission explicite hors scope)
session rotation  = 0 rotations *inattendues*
                  TTL `ttl_or_missing` (≥1800 s) + chaîne continue = autorisée ✅
```

S0→S1 : soft-update **60000→20000** autorisé (POC-1A déjà PASS).  
Retour terminal : soft **20000→60000** autorisé.

**C03 mode consistency** : ✅ **CLOSED** (patch `missionScheduling` sur `app_resume` + anti-wipe). Pendant ASSIGNED éligible, `location_mode` payload = `mission_live` continu ; `app_resume` interval = 20000.

---

## Détail par état

### S0 — Sans mission (`is_available=true`, `mission_id=null`)

**C01 FG** : `availability_presence` · `presence_window` · FGS ON · P9≈60 s · chaîne P9→carte.

**C02 BG** : HOME → PRESENCE continue · FGS vivant · owner/session stables · P9≈60 s · 0 Unregister / remove / restart · backend + carte continuent.

### S1 — Mission assignée (tracking-eligible, fenêtre T−lead/T+grace)

**C03 FG** : mission visible · eligibility LIVE · soft 60→20 · P9≈20 s · 0 Unregister/remove/restart/owner/session rotate.

**C04 BG** : HOME → `mission_live` · P9≈20 s · FGS · transport + projection + carte.

### S2 — EN_ROUTE

**C05/C06** : status mobile=backend=EN_ROUTE · LIVE ≈20 s · **changement de statut métier ≠ STOP** · 0 interruption native · chaîne complète FG et BG.

### S3 — ARRIVED

**C07/C08** : **ARRIVED → LIVE continue** (pas de retour PRESENCE, pas de 60 s, pas de stop). BG : FGS ON · P9≈20 s · backend/carte progressent (attente devant établissement).

### S4 — À bord (`IN_PROGRESS`)

**C09 FG** : LIVE ≈20 s · lat/lng / recorded_at / event_id progressent (déplacement réel).

**C10 BG** : **gate critique** — À BORD + HOME + mouvement → LIVE obligatoire · FGS · P9≈20 s · PUT/ingest/projection/carte · 0 Unregister/remove/restart.

---

## Gate global

```text
C01…C11 = PASS
= DRIVER STATE CERTIFICATION PASS ✅

GO PROD fleet E2E = HOLD (certification séparée)

9/11 ou C11 FAIL
= NO-GO ⛔
```

Run sheet opérationnel : [`_driver_state_cert_2026-08-21/C01_C10_RUN.md`](./_driver_state_cert_2026-08-21/C01_C10_RUN.md).
Preuve C11 : [`_driver_state_cert_2026-08-21/C11_TERM_PASS.md`](./_driver_state_cert_2026-08-21/C11_TERM_PASS.md).

---

## Ordre d’exécution recommandé (canary)

```text
1. C01 S0-FG → C02 S0-BG
2. Créer mission ASSIGNED éligible → C03 FG → C04 BG
3. Transition EN_ROUTE → C05 FG → C06 BG
4. Transition ARRIVED → C07 FG → C08 BG
5. Transition IN_PROGRESS → C09 FG (mouvement) → C10 BG (mouvement)
6. Terminer mission → C11 LIVE→PRESENCE soft
```
