# P0-E Phase 2 — canary soft-gate (2026-08-17, 2e tentative)

```text
RE-GO PHASE 2              = EXÉCUTÉ
PG_FIRST                   = true (toujours ON)
SOFT-GATE DLQ              = PASS ✅
ATTRIBUTION P5-B           = BLOCKED ⛔ (session superseded)
GLOBAL ENABLE              = NO-GO ⛔
RC132                      = FROZEN ✅
IDEMPOTENCE SERVEUR        = INCHANGÉE ✅
```

## Soft-gate DLQ (remplace DLQ=0)

```text
PG avance continuellement                     ✅ (DLE → 6059+)
nouveaux événements persistés                 ✅
consumer healthy / OUTBOX=true                ✅
traceback                                     = 0
DLQ reasons other than event_id_payload_conflict = 0
échantillons DLQ = eid déjà en PG (post-persist) ✅
```

Scripts : `_p0e_soft_dlq_gate.sh`, `_p0e_soft_dlq_check.py`

## Smoking gun — pourquoi pas de canonical

Les LOC async du témoin partent encore avec :

```text
tracking_session_id = trk_sess_1786968778000_nlh0et7f
session_generation  = 1680
status DB           = superseded
```

Alors que la session **active** est :

```text
trk_sess_1786971820868_fr3ty46h  gen=1683  status=active
DLE sur cette session            = 0
```

Code (Annexe A.3) :

```text
publish_realtime = (status != superseded)
→ _maybe_promote_after_pg return immédiat
→ Redis canonical non écrit (by design)
```

Preuve live :

```text
AUTH_RESOLVE = {session_generation: 1680, status: superseded}
CANON        = {}
TTL          = -2
p5b_promote logs = absents (skip silencieux publish_realtime=false)
```

Donc **P5-B n’est pas invalidé** : le gate promote n’est simplement **jamais atteint** pour des points `superseded`.

## Corrélation demandée (non obtenue)

```text
LOC N sur session ACTIVE
→ PG commit
→ _maybe_promote_after_pg
→ canonical seq=N / session active / TTL≈1200
```

Non observée : 0 LOC sur session active pendant la fenêtre ; flood de LOC sur session superseded (+ retries DLQ post-persist).

## Lien incident mobile

Étend le sous-incident `event_id_payload_mutation` :

```text
ledger mobile continue de flusher une session superseded
pendant qu’une session active existe sans aucun point
→ PG avance sans carte live (pas de promote)
→ pertinent Q1 / confirmation
```

Doc : `docs/ops/gps-p0e-incident-mobile-eid-payload-mutation-2026-08-17.md`

## NEXT

1. Obtenir ≥1 LOC async sur session **active** (`…fr3ty46h` ou plus récente active)  
   - idéalement cycle client propre (`pm clear` + re-login) car `run-as` impossible sur RC132  
   - **pas** de réactivation manuelle serveur d’une session superseded
2. Relancer `_p0e_phase2_active_attribution.py` / soft attribution
3. Puis seulement N+1 + retry muté → canonical reste N+1
4. Si besoin d’arrêter le canary en attendant : rollback `PG_FIRST=false` (image/migration conservées)

## État runtime au moment du rapport

```text
TRACKING_PG_FIRST_CANONICAL_ENABLED = true
TRACKING_PERSIST_WITH_OUTBOX        = true
backend/consumer                    = healthy
```
