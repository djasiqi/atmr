# P0-E — Rollback PG_FIRST + pm clear témoin (2026-08-17)

```text
PG_FIRST                  = false ✅ (rollback)
OUTBOX                    = true ✅
backend/consumer          = healthy ✅
pm clear ch.liri.operations = Success ✅
RC132 versionCode         = 132 ✅ (binaire inchangé)
ledger serveur            = INCHANGÉ ✅
GLOBAL ENABLE             = NO-GO ⛔
```

## Exécuté

### 1. Rollback flag

```text
TRACKING_PG_FIRST_CANONICAL_ENABLED=false
TRACKING_PERSIST_WITH_OUTBOX=true
recreate backend + tracking-kafka-consumer
PHASE2_ROLLBACK_PASS / POST_ROLLBACK_VERIFY_PASS
```

### 2. pm clear témoin (device only)

```text
adb -s <témoin> shell pm clear ch.liri.operations → Success
→ efface auth / SQLite queue / état local
→ ne touche pas ledger serveur ni AAB RC132
```

App relancée (pid nouveau) ; **re-login 20135 requis**.

## Séquence restante (opérateur)

```text
3. Relancer RC132 (fait) → re-login 20135 → reprendre mission 38243
   → attendre nouvelle tracking_session_id

4. AVANT PG_FIRST — gate pré-canary :
   session status = active ✅
   DLE sur cette session > 0 ✅
   seq monotone ✅
   eid/capture uniques ✅
   PG avance ✅
   PG_FIRST encore false ✅

5. Seulement alors : PG_FIRST=true (fenêtre courte)
6. Attribution PG N → promote → canonical N / TTL≈1200 / REST
7. PG_FIRST=false jusqu’au verdict final
```

**Ne pas réactiver PG-first tant qu’aucune LOC n’est en PG sur la nouvelle session `active`.**

## Gate pré-PG_FIRST (serveur) — 2026-08-17T13:20Z

```text
VERDICT = READY_FOR_PGFIRST ✅
ACTIVE_SESSION = trk_sess_1786972692514_lauam301
ACTIVE_GEN     = 1685
started_at     = 2026-08-17T13:18:12Z  (après pm clear 13:17Z)
DLE_ON_ACTIVE  = 8 (ids 6060–6067, mission 38243)
seq            = 1..8 monotone
eid/capture    = uniques
POST_CLEAR polluted/superseded DLE = 0
PG_FIRST       = false (toujours)
```

→ **re-login / tracking réaligné prouvé côté serveur** (pas seulement UI).  
→ **RE-GO PG_FIRST canary = autorisé** dès GO explicite.
