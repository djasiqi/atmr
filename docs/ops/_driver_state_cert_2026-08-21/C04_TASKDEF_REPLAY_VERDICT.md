# C04 — Replay TASKDEF_PROBE (2026-08-21 ~19:45–19:52 UTC+2)

```text
MISSION     = 50 ASSIGNED (CANARY-C04-TASKDEF)
DEVICE      = SM-S911B  adb-RFCW20QC53W-…
PATCH FONC. = AUCUN
C05         = HOLD
```

## Verdict mécanique

```text
A) local=true  registry=false  registered=true  started=true
   → SHADOW FLAG
   → OBSERVÉ : 0 fois ★
   → patch shadow flag = NON AUTORISÉ

B) local=false registry=false  registered=true  started=true
   → NEVER-DEFINED / definition lost
   → OBSERVÉ : OUI (frontière Metro HMR, run 1) ★
   → puis heal immédiat via exit_reason=defineTask_called

C) local=true registry=true + P8>0 + J1=0
   → OBSERVÉ : NON

D) local=true registry=true + P8=J1
   → OBSERVÉ : OUI (cold start clean, run 2) ★
   → P8=6 J1=6 J7=6
```

**Synthèse :** le discriminant A (shadow flag) n’est **pas** confirmé.  
Le pattern B apparaît au **reload Metro** (local+registry à false, registration native encore true). Sur cold start propre, la définition est saine (D).

```text
PATCH shadow-flag (if taskDefined return → registry authority)
= HOLD — A non prouvé

NEXT RCA
= pourquoi après certains HMR/long-lived process,
  defineTaskIfNeeded n’est pas rejoué (B non soigné)
  → alignement scope-module / re-init au wake
  ≠ patch skip local seul
```

---

## Run 1 — HMR (logcat_C04_TASKDEF_20260821_194611.txt)

Séquence utile :

```text
19:47:37.393 T1/T2
  local=false registry=false registered=true started=true
  js_runtime_id=js_mt38sdc0_dukk4k
  → BRANCHE B ★

19:47:37.393 T3 exit_reason=defineTask_called
  local=true registry=true registered=true started=true
  → heal définition

19:47:37.399 T2/T3 exit_reason=local_taskDefined_true
  local=true registry=true  (skip sain, registry OK)

19:47:39 HOME
  T4 : puis registered/started flippent false (FGS stop lié HMR — contamination)
```

`local=true registry=false` : **0 hits**.

---

## Run 2 — cold start clean (logcat_C04_TASKDEF_CLEAN_20260821_195032.txt)

```text
T1/T2 : local=false registry=false registered=false started=false
T3    : defineTask_called → local=true registry=true
T3    : local_taskDefined_true + registry=true (skip OK)
T4 HOME : local=true registry=true registered=true started=true ★

P8=6 J1=6 J7=6  → branche D (définition saine sur ce process)
```

---

## Preuve causale A (non obtenue)

Séquence demandée **absente** :

```text
T2 local=true registry=false
T3 exit_reason=local_taskDefined_true
→ HOME → P8 → J1 absent
```

À la place, le skip `local_taskDefined_true` n’apparaît qu’avec **registry=true**.

---

## Artifacts

```text
logcat_C04_TASKDEF_20260821_194611.txt       (HMR + B transitoire)
logcat_C04_TASKDEF_CLEAN_20260821_195032.txt (cold D)
mission_id=50
```
