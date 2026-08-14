# P0-C-LEDGER — Freeze cause amont + split CLIENT/SERVER

```text
PATCH                      = NO-GO
PARENT                     = gps-p0-c-ledger.md
BODIES                     = gps-p0-c-ledger-bodies-2026-08-14.md
SESSION PIVOT              = trk_sess_1786722711491_2h9hb2ps
```

## Freeze C-LEDGER (figé)

```text
P0-C-LEDGER

queue items valides côté GPS
session_id   = présent
sequence     = présent et monotone (53/54/55 ; session complète 1→55)
generation   = NULL dès l'enqueue (TOUTE la session)

→ backend reçoit un événement incomplet pour le ledger
→ claim Redis SET NX est acquis
→ ledger_ids_missing
→ claim non libéré sur ce chemin
→ retry
→ duplicate_event_id_unproven
→ release
→ reclaim
→ boucle
```

Coords 18:09–18:10 = **vrais nouveaux GNSS** → ne plus les utiliser comme preuve C-NATIVE.

Cause technique immédiate client :

> `createLocalTrackingSession()` rend une session utilisable par l'enqueue avant que son `generation` ne soit obtenu/persisté via le register serveur.

---

## Split conceptuel (pas de patch)

```text
P0-C-LEDGER-CLIENT
session utilisable avant generation
(+ session locale active alors que register n'a pas abouti)

P0-C-LEDGER-SERVER
claim non libéré sur ledger_ids_missing
+ duplicate_unproven avant preuve de persistence
```

Deux défauts **indépendants** à concevoir séparément.

---

## Timeline session `trk_sess_1786722711491_2h9hb2ps`

| T | Instant (local) | Preuve |
|---|-----------------|--------|
| **T0** | 17:51:51.491 | createLocal (ms dans session_id) |
| **T0+1ms** | 17:51:51.492 | **seq=1 enqueue** déjà (`generation=null`) |
| T1–T2 | — | pas de preuve log « register success » ; PG `tracking_sessions` **count=0** |
| T3 | — | generation **jamais** persistée (55/55 items `generation=NULL`) |
| T4 | 18:09:52 | seq **53** enqueue |
| T5 | 18:10:10 | seq **54** |
| T6 | 18:10:29 | seq **55** |
| suite | 18:15+ | watermark HTTP **403** sur cette session |

Stats queue session : **55** items, `ALL_GEN_NULL`, durée ~**18,6 min**, états `rejected×52` + `ingested_non_persisted×3`.

### Discriminant A / B / C

| Scénario | Verdict | Pourquoi |
|----------|---------|----------|
| **A** register in-flight, enqueue non gated | **DESIGN CONFIRMED** (structurel) | seq1 à T0+1ms ; `void registerSessionWithBackend()` ; aucun gate `generation!=null` dans `enqueue` |
| **B** register failed/absent, session reste active | **CONFIRMED (cette session)** | 0 row PG `tracking_sessions` ; generation null pendant 18+ min ; watermark 403 |
| **C** register OK avant seq53 mais generation non réinjectée | **EXCLUDED** | register n’a jamais abouti en PG ; pas de generation serveur connue |

Lien code (read-only) :

```text
beginNewTrackingSession / rotateTrackingSession
  → createLocalTrackingSession()   // generation=null
  → persistSession()
  → void registerSessionWithBackend()  // fire-and-forget
enqueue()
  → ensureSessionFresh()           // ne gate PAS sur generation
  → sessionGeneration: this.sessionGeneration  // peut rester null
```

Si register réussit plus tard, le code **peut** backfiller `item.sessionGeneration` + SQLite — ici cela **n’est jamais arrivé**.

---

## C-NATIVE — nuance (ne pas mélanger)

```text
18:09–18:10  nouveaux GNSS CONFIRMÉS ✅  (items 53–55)
post-18:13   stale reste à investiguer séparément
```

---

## Implémentation

✅ **Implémenté** : freeze cause amont ; split CLIENT/SERVER ; timeline T0–T6 ; verdict **B confirmé** + **A structurel** ; C exclu ; nuance C-NATIVE.  
**Reste à faire** : design séparé CLIENT vs SERVER après GO explicite ; **PATCH NO-GO**.
