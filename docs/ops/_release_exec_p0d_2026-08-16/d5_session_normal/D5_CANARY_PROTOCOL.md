# D5 — Protocole canary interne (étroit)

**Statut** : `C1+C2+C3+C4 = PASS ✅` — `CANARY INTERNE = VALIDATED ✅` — `DISTRIBUTION = NO-GO ⛔`

```text
RCA STRUCTUREL   = CLOSED / SUFFISANT ✅
PATCH DESIGN     = DONE ✅
IMPLEMENTATION   = DONE ✅
TESTS            = 34 PASS ✅ (T9–T12 owner_version_mismatch + recheck)
CANARY INTERNE   = VALIDATED ✅
C1 PREFLIGHT     = PASS ✅ versionCode 127
C1 RUN           = PASS ✅
C2 RUN           = PASS ✅
C3 RUN           = PASS ✅
  → d5_canary/C3_summary.txt
C4 RUN (129)     = FAIL ❌ (historique)
C4_130 ATTR      = DONE ✅ W1 = owner_version_mismatch CLOSED
FIX W1           = DONE ✅
BUILD 131        = INSTALLED ✅
C4 RE-RUN (131)  = PASS ✅
  → d5_canary/C4_summary_131.txt
  Unregister=0 / NATIVE_STOP_ENTRY=0 / L1 non-destructif
CANARY VALIDATED = YES (interne C1–C4) ✅
RELEASE REVIEW   = FAIL ❌
  → d5_canary/RELEASE_REVIEW_131.md
  B1 SHA OPEN / B2 QA OPEN / B3 FIXED ✅
  NEXT = commit SHA FINAL S puis AAB production
DISTRIBUTION     = NO-GO ⛔
BACKEND          = READ-ONLY / observationnel uniquement
```

Référence patch : `D5_PATCH_DESIGN.md`  
Baseline D5 fail : session normale Prod126 (`T_FAIL` 21:18:49.975)

---

## 0. Prérequis binaire

| Item | Exigence |
|------|----------|
| Build | Release interne contenant le patch ownership (pas Play) |
| Package | `ch.liri.operations` (ou canal interne dédié) |
| Device | Même famille que D5 si possible (Samsung) |
| Mission | `IN_PROGRESS` réelle |
| Backend | **Observation only** — aucun changement serveur |
| Force-stop | Hors protocole (ne pas reproduire cold-start force) |

---

## 1. Gate principale (4 protections)

Sur toute la fenêtre canary, avec mission `IN_PROGRESS` :

```text
unexpected Unregister          = 0
Register↔Unregister storm       = 0
STOP gen stale atteint natif    = 0
B2 stop direct                  = 0
self_heal_restart destructif    = 0
transient mission null → STOP   = 0   (sauf confirmation stable + mission absente)
```

Tracking attendu :

```text
MISSION IN_PROGRESS
  → HOME / background normal
  → Task Finished continue
  → PUT / LOC continue
```

---

## 2. Séquences ciblées

### C1 — Baseline

```text
FG → HOME → background prolongé (≥ 5–10 min)
→ Task Finished continue
→ PUT/LOC continue
→ FGS stable
```

### C2 — Churn React / query

```text
refetch missions / changements scheduling / FG↔BG
→ mission reste active
→ aucun cleanup destructif
→ aucun Unregister inattendu
```

### C3 — Transient loss

```text
reproduire si possible un trou local court (mission React null bref)
→ pending ~2,5 s (confirm)
→ mission revient
→ native STOP = 0
→ telemetry : transient_loss.pending puis annulation (pas confirmed+stop)
```

### C4 — Recovery / freshness UNKNOWN

```text
freshness partiellement UNKNOWN (pas de lastSent / lastFix observables)
→ self-heal L1 seulement (ensure / wake)
→ pas d'Unregister
→ pas de reason self_heal_restart destructif
```

### Contrôle B2 / generation (smoking gun ownership)

```text
START N+1 pendant STOP N en vol
→ shouldAbortNativeStop = true
→ Location.stopLocationUpdatesAsync NON appelé
→ telemetry tracking.lifecycle.stop.abandoned
→ task reste enregistrée
→ cadence Location continue
```

---

## 3. Smoking gun

### Ancien D5 (FAIL)

```text
Unregister → Register → FGS → Unregister → …
```

### Nouveau binaire (PASS ownership)

```text
STOP requested
→ state/generation re-check
→ stale STOP aborted
→ task reste enregistrée
→ cadence Location continue
```

C’est la preuve la plus importante que l’ownership fonctionne en release.

---

## 4. Signaux logcat / télémétrie à capturer

### Natif (TaskService / Expo)

| Pattern | Interprétation |
|---------|----------------|
| `Unregistering 'background-location-task'` | STOP natif — compter ; classer légitime vs inattendu |
| `Finished task 'background-location-task'` | Cadence saine |
| `Could not find a location task` | FAIL si hors STOP légitime |
| `Registering` + Unregister en rafale (~300 ms) | Storm D5 → FAIL |

### JS / telemetry (si sink actif en release interne)

| Event | Attendu |
|-------|---------|
| `tracking.lifecycle.stop.requested` | OK |
| `tracking.lifecycle.stop.abandoned` | OK sur race START N+1 |
| `tracking.lifecycle.stop.executed` | Seulement STOP légitimes |
| `tracking.lifecycle.transient_loss.pending` | OK en C3 |
| `tracking.lifecycle.transient_loss.confirmed` | Seulement si mission absente durable |
| `tracking.watch.restarted` + `recovery_level=L1` | OK C4 |
| `recovery_level=L2` / `self_heal_restart` destructif | FAIL sauf preuve native L2 |

### Backend (lecture seule)

| Signal | Attendu |
|--------|---------|
| `PUT /api/v1/driver/me/location` | Cadence continue |
| LOC rows mission | Continue, même `mission_id` |

---

## 5. STOP légitimes (ne pas compter comme D5)

```text
- mission terminale confirmée (COMPLETED / CANCELLED métier)
- logout / leave driver context
- hardStop context_left_driver
```

Tout autre Unregister pendant `IN_PROGRESS` = **suspect D5**.

---

## 6. Critères PASS / FAIL

### PASS (fenêtre complète C1–C4)

```text
TaskService unexpected Unregister = 0
"Could not find a location task"  = 0
Register/Unregister storm         = 0

Finished background task          = cadence continue
PUT                               = continue
LOC                               = continue

self-heal L1                      = non destructif
generation stale STOP             = aborted (si race observée)
FGS                               = stable

crash / ANR                       = 0
```

### FAIL

```text
CANARY FAIL
→ capturer FIRST divergence (timestamp + pattern)
→ pas de rollback conceptuel automatique
→ attribuer laquelle des 4 protections a cédé :
   P1 ownership / gen guard
   P2 B2 bypass
   P3 self-heal L1
   P4 transient React
→ DISTRIBUTION reste NO-GO
```

### Après PASS

```text
CANARY PASS
→ D5 PATCH = CANARY VALIDATED ✅
→ ensuite seulement revue GO/NO-GO distribution
```

---

## 7. Artefacts à déposer

Sous `docs/ops/_release_exec_p0d_2026-08-16/d5_canary/` (créer au run) :

```text
timeline.txt
logcat_continuous.txt
samples.csv          (optionnel, poll svc)
C1_summary.txt … C4_summary.txt
analyze_verdict.txt  (PASS/FAIL + compteurs)
apk_or_build_id.txt  (fingerprint binaire)
```

Script d’analyse post-hoc : `analyze_d5_canary.ps1` (compteurs Unregister / storm / Finished).

---

## 8. Hors scope canary

- Distribution Play / OTA prod
- Modification backend
- Force-stop cold-start
- Attribution T1 exacte
- Recette GPS complète (précision, batterie longue durée)
