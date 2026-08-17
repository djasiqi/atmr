# D5 — Disparition locale `missionsQuery.data` + prédicat self-heal (final)

Tip `286737a2`. Artefacts : access bookings, health PG driver 20135.

## Verdict synthétique

```text
LOCAL missionsQuery.data HOLE sans HTTP pré-T_FAIL
= AUCUN MÉCANISME RESTANT FORTEMENT VIABLE ✅/★
  (après élimination sévère)

⇒ T1 (missionId→null via pick)
  reste la *forme* requise pour clear hook
  mais sa *cause locale* n''est plus attribuable
  sur les artefacts courants ★

Q2 self-heal
= RENFORCÉ comme candidat FIRST native Unregister ★
  health @ 19:18:35 : last_fix_age_seconds = NULL
  (= lastFixProducedAtMs bridge null)
  + native_last_fix_age = 17s
  + cascade=0 → stopBackground sans clear
  + si lastSentAt aussi null → branche startedAge >60s
    quasi certaine pour mission longue

PATCH structurel (B2 + STOP destructif depuis React)
= déjà démontré ; NO-GO ship tant que micro-attribution ouverte
```

---

## 1. QueryKey Host — peut-il changer sans leave-driver ?

```text
useDriverMissionsQuery:
  contextId = useActiveDriverContextId()
            = activeContext.context_id si context_type===driver
            sinon null

  queryKey = contextId
    ? ["driver-missions", contextId]
    : ["driver-missions", "disabled"]   // data typiquement undefined
```

| Transition key | Effet data | Possible @ T_FAIL sans leave ? |
|----------------|------------|--------------------------------|
| `driver:X` → `disabled` (contextId null) | **undefined** → T1 | **EXCLUDED** — `setActiveContext(null)` seulement logout / interrupted_logout / terminal revoke |
| `driver:X` → `driver:Y` | miss cache → undefined jusqu''au fetch | **AFFAIBLI** — switchContext multi-driver non observé ; leave company exclu |
| même key, remount observer | cache conservé si query en cache | pas de trou |

**Discriminant queryKey sans HTTP** : théoriquement puissant, **exclu @ T_FAIL** par stabilité session chauffeur (PUT/bookings continuent).

---

## 2. Inventaire mécanismes « data disparaît sans GET /bookings »

| Mécanisme | Touche `driver-missions` ? | @ T_FAIL |
|-----------|----------------------------|----------|
| Full poll `setQueryData(missions)` | oui REPLACE | **EXCLUDED** (0 GET 35→49.975) |
| `/since` reconcile | merge ; vide → keep prev | **EXCLUDED as hole** |
| Socket `setQueryData` updater | patch/unshift | **EXCLUDED** (0 WS mission) |
| Optimistic RELEASE filter | retire id | **EXCLUDED** |
| Optimistic status map | ne retire pas | n/a |
| `invalidateQueries(missions)` | refetch | refetch = GET /bookings **absent** |
| `cancelQueries` | non | n/a |
| `removeQueries` / `clearAllContextCache` | **NON** — filtre `queryKey[0]==="ctx"` seulement | **EXCLUDED** |
| `queryClient.clear()` | oui global | **EXCLUDED** (logout only) |
| `resetQueries(driver-missions)` | — | **AUCUN caller prod** |
| `placeholderData` / `initialData` / `select` | absents | — |
| Host `data ?? []` | **non** — `missionsQuery.data` brut | — |
| GC RQ sous observer actif | non | — |

**Conclusion Q1 cause** : plus de writer local viable pour `data=undefined/[]/sans 38224` dans la fenêtre pré-T_FAIL.

Tension RCA :

```text
clear missionId ⇒ (presque) hook T1
T1 ⇒ data hole local
data hole local ⇒ plus de mécanisme viable observé
⇒ micro-attribution T1 BLOQUÉE sur artefacts ★
```

---

## 3. Q2 — Self-heal / lastSentAt (indépendant)

### Health PG (READ-ONLY) autour de T_FAIL

| recorded_at | tracking_active | fgs | last_fix_age_s | native_last_fix_age_s | trigger |
|-------------|-----------------|-----|-----------------|------------------------|---------|
| 19:18:35.07 | t | t | **NULL** | 17 | (heartbeat) |
| 19:18:35.16 | t | t | **NULL** | 17 | health_monitor_ok |
| 19:18:50.20 | t | **f** | NULL | 6 | (post T_FAIL) |
| 19:18:50.24 | t | f | NULL | 6 | health_monitor:fgs_not_running |

`last_fix_age_seconds` = âge GNSS depuis `snapshot.lastFixProducedAtMs`.  
**NULL @ 19:18:35** ⇒ bridge `lastFixProducedAtMs == null` ~15 s avant Unregister.

### Chaîne anti-zombie (rappel)

```text
si lastSentAt != null → sentAge > 60 ?
sinon si lastFixProducedAtMs != null → fixAge > 60 ?
sinon si trackingStartedAtMs != null → startedAge > 60 ?
```

BG PUT **ne met pas à jour** `lastSentAt`.  
Health prouve **lastFix null** à 35s.  
Si `lastSentAt` aussi null (jamais de `flushPoint` currentHandled) :

```text
startedAge > 60s  (mission IN_PROGRESS depuis des heures)
→ shouldTriggerAntiZombie = TRUE ★
→ cascade=0 → stopBackground("self_heal_restart")
→ FIRST native Unregister SANS clear missionId ★
```

```text
SELF-HEAL as FIRST Unregister = LEADING CONDITIONAL ★★
  (renforcé par last_fix NULL @ 35s)
SELF-HEAL as clear missionId = EXCLUDED ✅
Δ lastSentAt exact = NON OBSERVABLE (pas de colonne health)
mais branche startedAge rend le >60s plausible sans lastSentAt
```

PUT @ 44/45/47 **n''infirment pas** cette branche.

---

## 4. RCA en très peu de lignes (état)

```text
STRUCTUREL DÉMONTRÉ (indépendant du dernier détail)
  B2 STOP unguarded après clear     ✅
  dual authority START/STOP         ✅
  clear writers = STOP family only  ✅
  cascade prod=0                    ✅

Q1 CLEAR missionId
  famille = HOOK T1                 ★★ (forme)
  cause data hole locale            = BLOQUÉE / non attribuée ★

Q2 FIRST NATIVE UNREGISTER
  self_heal stopBackground          = LEADING CONDITIONAL ★★
  hook STOP (si T1 a eu lieu)       = OPEN
  les deux peuvent coexister
  (Unregister self-heal puis clear hook plus tard, ou inverse)

PATCH / DISTRIBUTION                = NO-GO
```

### Implication conception patch (sans ship)

Même sans fermer T1 :

1. **Ne jamais** laisser le manager STOP natif contourner `lifecycleGeneration` (B2).
2. **Ne jamais** traiter un trou React / self-heal restart comme STOP destructif non coordonné avec ownership unique RUNNING/STOPPED.
3. Self-heal `stopBackground` sans tenir compte de `lastSentAt` vs envois BG est un second défaut plausible (prédicat anti-zombie aveugle aux PUT BG).

---

## Artefacts

- `access_bookings_191820_191910.txt`
- PG `driver_device_health_events` 19:18:35 / 19:18:50
- `D5_HOOK_TRANSITION_AND_LASTSENT_AUDIT.md`
- `D5_STOP_BRIDGE_CALLERS_AUDIT.md`