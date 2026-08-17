# D5 — Conception patch (single ownership)

**Statut** : `PATCH DESIGN = GO` — `IMPLEMENTATION = DONE ✅` — `CANARY INTERNE = NEXT` — `DISTRIBUTION = NO-GO ⛔`

Tip de référence PROD126 : `286737a2`.  
RCA structurel : `D5_RCA_SORTIE_OBLIGATOIRE.md` + `D5_LOCAL_DATA_HOLE_AND_LASTSENT_FINAL.md`.

**Gate distribution** : canary interne requis — protocole `D5_CANARY_PROTOCOL.md` — avant toute décision Play/OTA.
**Implémenté** : 4 protections §1–5 ; tests T1/T2/T4/T5/T6/T8 ; ordre §9 étapes 1–5.

### ✅ Implémenté

1. **Lifecycle owner** — `requestTrackingStop` dans `driverTrackingBridge.ts` + abort guard pré-natif dans `backgroundLocationTask.ts` (`shouldAbortNativeStop` juste avant `Location.stopLocationUpdatesAsync`). Types : `tracking/trackingLifecycleOwner.ts`.
2. **B2** — `ineligible_tracking_state` et `presence_fg_outside_window` passent par `requestTrackingStop` (plus d’appel direct non gardé).
3. **Self-heal L1** — `forceRestartTrackingWatch` n’appelle plus `stopBackground` par défaut ; L2 via `allowDestructiveRestart`. `shouldTriggerAntiZombie` ignore le fallback `startedAge` (UNKNOWN).
4. **Hook transient** — `useDriverTracking` : null React → confirmation `TRANSIENT_MISSION_LOSS_CONFIRM_MS` avant STOP ; mission active → START sans cleanup destructif.
5. **Tests** — `driverTrackingBridge.test.ts` (D5), `trackingSelfHeal.test.ts` (T4/L1), `trackingLifecycleOwner.arch.test.ts` (T8).

### Canary (NEXT)

Protocole étroit : `D5_CANARY_PROTOCOL.md` (C1–C4 + gate B2).  
Analyse post-hoc : `analyze_d5_canary.ps1`.  
Backend observationnel uniquement. Distribution gelée jusqu’à `CANARY VALIDATED`.

---

## 0. Pourquoi on patche sans T1 attribuable

```text
D5 RCA STRUCTURAL        = SUFFICIENT FOR PATCH DESIGN ✅
EXACT T1 SOURCE          = UNATTRIBUTED / ARTEFACT-LIMITED ★
SELF-HEAL FIRST STOP     = LEADING CONDITIONAL ★

B2 BYPASS                = CONFIRMED DEFECT ✅
DUAL AUTHORITY           = CONFIRMED DEFECT ✅
SELF-HEAL FALLBACK       = UNSAFE DESIGN CONDITION ✅
```

Deux défauts suffisent à rendre D5 possible, indépendamment du micro-trigger :

```text
DÉFAUT 1 — B2
ensureManagerState / présence FG
→ stopBackgroundLocationTask(...)  SANS lifecycleGeneration
→ Unregister natif possible après clear / en parallèle d''un START

DÉFAUT 2 — self-heal
lastSentAt=null ∧ lastFix=null ∧ startedAge>60s
→ stopBackground("self_heal_restart")
→ Unregister sans ownership unique / sans preuve panne native
```

Le patch ne cherche **pas** à « fixer T1 ». Il rend ces chemins incapables de produire un Unregister destructif stale.

---

## 1. Architecture cible

```text
React Hook ─────────────┐
Manager (eligibility) ──┤
Health / self-heal ─────┤
Recovery / anti-zombie ─┤
                        ▼
              TRACKING LIFECYCLE OWNER
              desiredState ∈ {RUNNING, STOPPED, RECOVERING}
              lifecycleGeneration
              missionId (bridge intention)
              recoveryLevel ∈ {NONE, L1_NON_DESTRUCTIVE, L2_DESTRUCTIVE}
                        │
                  unique START / STOP
                  (dernier check gen IMMÉDIATEMENT
                   avant Location.stopLocationUpdatesAsync)
                        │
                        ▼
                 Expo Location / FGS
```

**Interdit** après patch :

```text
Hook / Manager / Self-heal / Recovery
  → appel direct stopBackgroundLocationTask
  → appel direct Location.stopLocationUpdatesAsync
```

Seul le lifecycle owner matérialise le STOP natif.

---

## 2. Protection P0 — Ownership unique du STOP

### API unique

```ts
type StopRequest = {
  reason: string;
  expectedGeneration: number;
  expectedMissionId?: number | null;
  /** EXPLICIT vs TRANSIENT — voir §5 */
  authority: "explicit" | "transient_loss" | "recovery_l2";
};

requestTrackingStop(req: StopRequest): Promise<"stopped" | "abandoned" | "deferred">;
requestTrackingStart(...): Promise<...>; // inchangé conceptuellement ; bump gen
```

### Invariance

```text
START génération N+1
arrive à n''importe quel moment
avant Location.stopLocationUpdatesAsync

→ tout STOP génération N devient stale
→ native Unregister = 0 ✅
```

### Check final (obligatoire)

Juste **avant** `Location.stopLocationUpdatesAsync` (dans le owner, sous lock) :

```text
1. expectedGeneration === lifecycleGeneration ?
2. desiredState === STOPPED ?
3. missionId encore compatible avec ce STOP ?
4. aucun START plus récent (même gen) ?

NON à l''un → abandon ; log telemetry abandoned_stale_stop
```

Les checks **avant** les `await` (flush/sync) restent utiles mais **insuffisants** seuls — D5 exige le check immédiatement pré-natif.

### Callers actuels à migrer (tip `286737a2`)

| Site | Fichier | Action |
|------|---------|--------|
| `ensureManagerState` ineligible | `driverTrackingBridge.ts` ~1395 | `requestTrackingStop({ reason: "ineligible_tracking_state", ... })` |
| présence FG hors fenêtre | idem ~1448 | idem `presence_fg_outside_window` |
| `stopTrackingRuntime` | idem ~1324 | via owner |
| `stopBackground` self-heal | bridge → `stopBackgroundLocationTask` | L1/L2 via owner |
| lease / owner natif | `backgroundLocationTask.ts` ~1713+ | via owner (ou owner interne unique) |
| hook cleanup STOP | `useDriverTracking` → stop bridge | via owner + §5 transient |

`stopBackgroundLocationTask` devient **privé au module owner** (ou re-export test-only).  
Test architectural : grep/ESLint fail si nouvel import hors owner.

---

## 3. Protection P0 — Supprimer le bypass B2

Aujourd''hui (CONFIRMED) :

```ts
if (!eligibility.trackingEligible) {
  void stopBackgroundLocationTask("ineligible_tracking_state"); // ← B2
  ...
}
```

Cible :

```ts
if (!eligibility.trackingEligible) {
  trackingManager.setDesired("STOPPED"); // conceptuel
  void requestTrackingStop({
    reason: "ineligible_tracking_state",
    expectedGeneration: lifecycleGeneration,
    expectedMissionId: state.missionId,
    authority: "explicit", // ou transient si issu d''un trou React — voir §5
  });
  // NE PAS clear missionId ici si authority=transient_loss
  ...
}
```

Le manager **calcule** `desiredState=STOPPED` ; il **ne matérialise plus** le STOP.

Même règle pour `presence_fg_outside_window` (second bypass direct actuel).

C''est **la correction centrale de D5**.

---

## 4. Protection P1 — Self-heal non destructif par défaut

### Supprimer l''implication agressive

```text
lastSentAt=null ∧ lastFixProducedAtMs=null ∧ startedAge>60s
→ STOP + START natif   ❌
```

`NULL` = **absence de preuve de fraîcheur**, pas preuve que la task Location est morte.

### Santé

```text
freshness connue + stale > seuil  → UNHEALTHY
freshness inconnue (null/null)    → UNKNOWN  → pas d''Unregister
panne native positivement prouvée → FAILED   → L2 possible
```

### Recovery à deux niveaux

```text
LEVEL 1 — non destructif (défaut)
  ensure task registered
  ensure FGS
  ensure native foreground
  wake / recheck
  PAS de Location.stopLocationUpdatesAsync

LEVEL 2 — destructive restart
  uniquement si panne native positivement démontrée
  ET sous requestTrackingStop({ authority: "recovery_l2", expectedGeneration })
  ET check gen immédiat pré-natif
```

Signaux L2 acceptables (exemples, à figer en implémentation) :

- task **explicitement** unregistered alors que `desiredState=RUNNING`
- FGS attendu absente **après** L1 + recheck
- erreur native catégorique (permission revoked, etc.)

`startedAge` seul **n''autorise plus** L2.

---

## 5. Protection P1 — `trackingMission → null` non autoritaire

Même sans cause T1 connue, on ne laisse plus :

```text
trackingMission null transitoire
→ cleanup immédiatement destructif
→ clear missionId + STOP natif
```

### Classification

| Classe | Exemples | Effet |
|--------|----------|-------|
| **EXPLICIT STOP** | logout, leave driver, mission terminale confirmée, stop métier explicite | `requestTrackingStop(authority=explicit)` → clear + STOP |
| **TRANSIENT LOSS** | `trackingMission` null, cache/query indisponible, incohérence React | **pas** de clear immédiat ; reconcile ; intention RUNNING conservée ; STOP seulement après confirmation stable |

### Confirmation stable (pas un `setTimeout` arbitraire)

Idéalement :

```text
null observé
→ mark pending_loss(missionId_last, gen)
→ reconcile (cache / since / présence mission bridge)
→ confirm terminal OU confirm same mission revient
→ seulement alors explicit STOP ou cancel pending_loss
```

Un délai seul est un filet de secours, pas le design primaire.

---

## 6. Tests de régression indispensables (avant canary)

| ID | Scénario | Attendu |
|----|----------|---------|
| T1 | STOP gen=N en attente → START N+1 | native Unregister = 0 |
| T2 | manager ineligible → START plus récent | manager STOP abandonné |
| T3 | mission React null transitoire → même mission revient | native STOP = 0 |
| T4 | lastSentAt=null + lastFix=null + startedAge>60s + mission active | self-heal destructif = 0 |
| T5 | vraie mission terminale | STOP natif = 1 |
| T6 | logout / leave driver | STOP natif = 1 |
| T7 | recovery simultané + hook START | pas de storm Register↔Unregister |
| T8 | aucun chemin direct `Location.stopLocationUpdatesAsync` hors owner | lint/arch test fail sinon |

Mocks : compter appels `stopLocationUpdatesAsync` / `stopBackgroundLocationTask`.

---

## 7. Critère canary D5

Pas « le GPS marche ». Preuves explicites :

```text
mission IN_PROGRESS
→ aucun Unregister destructif inattendu

start/stop overlap = 0
storm Register↔Unregister = 0
background-location-task continue
PUT continue
LOC continue
recovery ne contourne plus generation
B2 path (ineligible) ne produit plus stop natif direct
```

Instrumentation minimale canary :

- telemetry `tracking.lifecycle.stop.requested|abandoned|executed`
- `expected_generation` / `actual_generation` / `reason` / `authority`
- compteur native Unregister corrélé JS reason

---

## 8. Hors scope (volontaire)

- Attribution exacte T1 / trou `missionsQuery.data`
- Changement backend / flag prod remote
- Distribution / Play / OTA
- Refactor large hors lifecycle owner

---

## 9. Ordre d''implémentation suggéré (quand CODE = GO)

1. Introduire `requestTrackingStop` + check gen pré-natif (sans changer callers métier)
2. Migrer B2 (`ineligible` + `presence_fg_outside_window`) vers request
3. Couper L2 self-heal sur `startedAge` ; L1 par défaut
4. Hook : classifier EXPLICIT vs TRANSIENT
5. Tests T1–T8 + arch guard
6. Canary interne — **pas** Play

---

## 10. Statut figé

```text
D5 RCA STRUCTURAL        = SUFFICIENT FOR PATCH DESIGN ✅
PATCH DESIGN             = GO ✅
IMPLEMENTATION           = DONE ✅ (GO explicite 2026-08-17)
CODE CHANGE              = OUI (mobile unified-app)
CANARY INTERNE           = NEXT → D5_CANARY_PROTOCOL.md
CANARY VALIDATED         = NON
DISTRIBUTION             = NO-GO ⛔
BACKEND                  = READ-ONLY / GELÉ
```