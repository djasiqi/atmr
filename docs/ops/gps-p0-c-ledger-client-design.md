# P0-C — Design : C-LEDGER-CLIENT (session ready avant enqueue ledger)

```text
TICKET                     = P0-C-LEDGER-CLIENT
PHASE                      = CLOSED
STATUT                     = CLOSED / PASS ✅ (unitaires + canary)
RCA                        = CONFIRMED (gps-p0-c-ledger-rca.md)
PARENT                     = gps-p0-c-loc-stale-after-pause.md
CANARY                     = gps-c3-ledger-client-canary-2026-08-14.md
SUITE                      = C-LEDGER-SERVER (gps-p0-c-ledger-server-design.md)
SERVER / OBSERVABILITY     = design SERVER READY ; OBSERVABILITY NO-GO
PATCH PROD                 = NO-GO
```

Documents liés :

- [gps-p0-c-ledger-rca.md](gps-p0-c-ledger-rca.md)
- [gps-p0-c-ledger-readiness-2026-08-14.md](gps-p0-c-ledger-readiness-2026-08-14.md)
- [gps-p0-c-ledger-bodies-2026-08-14.md](gps-p0-c-ledger-bodies-2026-08-14.md)
- [gps-p0-c-loc-stale-after-pause.md](gps-p0-c-loc-stale-after-pause.md)

---

## Objectif

Empêcher qu’une **session tracking locale non enregistrée** (`session_generation = null`) produise des items dans la queue ledger / HTTP, ce qui déclenche `ledger_ids_missing`, le cycle claim Redis, et un **HOL** qui masque des fixes GNSS frais.

Invariant central :

> Aucune location ne peut entrer dans la queue ledger tant que `tracking_session_id` **et** `session_generation` ne sont pas valides (non null, session register abouti).

Critère de validation produit (figé) :

> Un item invalide ou non enregistré **ne doit jamais** pouvoir bloquer les positions fraîches suivantes.

### Non-objectifs (cette PR CLIENT)

- Release claim Redis / sémantique ACK serveur → **C-LEDGER-SERVER**
- Nouveau `last_location_fix_at` health → **OBSERVABILITY**
- Changer permissions / cadence GPS / P0-A FGS / P0-B auth
- Purger manuellement les preuves canary Redis/queue

### Anti-pattern refusé

```ts
createLocalTrackingSession(); // generation = null
void registerSessionWithBackend(); // fire-and-forget
await enqueue(fix); // ← INTERDIT tant que generation == null
```

---

## Diagnostic figé (rappel)

```text
trk_sess_1786722711491_2h9hb2ps
T0     createLocal
T0+1ms seq=1 enqueue generation=null
…      55/55 items generation=NULL
PG     tracking_sessions count = 0
→ ledger_ids_missing + HOL + illusion GPS stale
```

Code actuel : `beginNewTrackingSession` / `rotateTrackingSession` → `void register…` ; `enqueue` ne gate pas sur `sessionGeneration`.

---

## États de readiness session (CLIENT)

```text
ABSENT
CREATING
REGISTERING
READY
REGISTER_FAILED
```

| État | Signification | Enqueue ledger |
|------|----------------|----------------|
| `ABSENT` | Pas de `trackingSessionId` | ❌ |
| `CREATING` | ID local alloué, persist session en cours | ❌ |
| `REGISTERING` | `registerTrackingSession` in-flight | ❌ |
| `READY` | `trackingSessionId` + `sessionGeneration` non null (register OK) | ✅ |
| `REGISTER_FAILED` | Register échoué / refusé / context_inactive durable | ❌ |

Transitions :

```text
ABSENT → CREATING → REGISTERING → READY
REGISTERING → REGISTER_FAILED
REGISTER_FAILED → CREATING     (retry explicite / nouvelle session)
READY → CREATING               (rotate / TTL / beginNew — puis re-register)
```

Transitions **interdites** :

```text
CREATING → READY          (sans register)
REGISTERING → enqueue OK
REGISTER_FAILED → READY   (sans nouveau register success)
```

Une session locale **n’est plus « active » pour le ledger** hors `READY`.  
Elle peut exister en mémoire pour retry register, mais **ne doit pas** être traitée comme session tracking exploitable.

---

## Gate enqueue (cœur du design)

Dans `DriverTrackingQueue.enqueue` (et tout chemin équivalent bridge → queue) :

```text
SI sessionReadiness !== READY
  OU !trackingSessionId
  OU sessionGeneration == null
ALORS
  NE PAS créer d’item tracking_queue
  NE PAS incrémenter sequenceCounter ledger
  émettre télémétrie tracking.queue.enqueue_deferred_or_dropped
  retourner selon politique (ci-dessous)
```

### Politique des fixes pendant non-READY

Choisir **une** politique explicite (recommandation design) :

| Option | Comportement | Avantage | Risque |
|--------|--------------|----------|--------|
| **A — Drop observé (recommandé v1)** | Fix non mis en queue ; compteur + dernier fix en mémoire UI | Zéro HOL futur ; simple | Perte points pendant register (secondes) |
| B — Staging hors ledger | Buffer mémoire ring, flush vers queue à READY | Moins de trous | Complexité ; ne doit jamais écrire SQLite ledger |
| C — Enqueue avec flag non-ledger | Items sans gen | **REFUSÉ** — c’est le bug actuel |

**v1 recommandée = A** : pendant `REGISTERING` / `REGISTER_FAILED`, pas d’écriture `tracking_queue`.  
Conserver éventuellement `lastDeferredFix` en RAM pour UI / health (hors scope OBSERVABILITY détaillée).

Durée typique register OK ≪ 1–2 s → perte acceptable vs HOL multi-heures.

---

## Comportement si register échoue

Aujourd’hui : `register_deferred` / `context_inactive` puis **session locale reste utilisable**.

Cible :

```text
register fail (réseau, 4xx/5xx, context_inactive, lease)
→ sessionReadiness = REGISTER_FAILED
→ trackingSessionId peut rester pour diagnostic
→ sessionGeneration reste null
→ enqueue ledger BLOQUÉ
→ télémétrie tracking.session.register_failed
→ retry register avec backoff (sans ouvrir une nouvelle session à chaque fix)
```

Règles :

1. **Ne pas** appeler `createLocalTrackingSession()` à chaque callback Location en échec (évite churn seq=1).
2. Retry register sur : retour réseau, lease restored, timer backoff, foreground resume.
3. Succès register → `READY` + backfill **uniquement** items déjà en file qui auraient `generation=null` **s’ils existent encore** (migration, ci-dessous) ; les **nouveaux** enqueue n’ont lieu qu’après READY.
4. Échec durable (auth/context) → rester `REGISTER_FAILED` ; alignement avec P0-B / lease (pas de contournement auth).

---

## Items historiques `generation=null` (anti-HOL CLIENT)

Même avec gate forward, la queue peut contenir des têtes invalides (canary / devices déjà touchés).

Règle CLIENT obligatoire pour le critère anti-HOL :

```text
À ensureLoaded / flush :
  items avec !trackingSessionId OU sessionGeneration == null
  → marquer ledger_invalid (ou quarantaine dédiée)
  → NE PAS les envoyer sur le chemin HTTP ledger
  → NE PAS les laisser en tête FIFO bloquante
  → permettre le flush des items READY derrière
```

Options d’implémentation (choisir en PR) :

1. **Skip + tombstone local** `state=ledger_invalid` (plus de retry HTTP)  
2. **Quarantine table** SQLite (comme identity quarantine)  
3. **Reorder flush** : sélectionner d’abord items `generation != null`

Interdit : laisser un head `generation=null` en `retry_pending` indéfiniment (état observé canary).

> Note : le SERVER doit quand même corriger le claim leak (design séparé). Le CLIENT ne doit **pas** dépendre du SERVER pour débloquer le HOL local.

---

## API / séquence cible

```text
beginNewTrackingSession()
  → readiness = CREATING
  → createLocalTrackingSession()
  → persistSession()
  → readiness = REGISTERING
  → await registerSessionWithBackend()   // plus de void fire-and-forget pour le chemin « ready »
  → si OK : generation + readiness = READY
  → si KO : readiness = REGISTER_FAILED

enqueue(fix)
  → si readiness != READY : drop/defer observé ; return
  → sinon : item avec session_id + generation + sequence
```

`ensureSessionFresh` / rotate TTL : même contrat — **pas** d’enqueue tant que la nouvelle session n’est pas `READY`.

Offline-first nuance :

- Aujourd’hui le commentaire dit « jamais bloqué par le réseau ».
- Nouveau contrat : **offline ≠ enqueue ledger sans generation**.
- Offline : rester `REGISTER_FAILED` / `REGISTERING` ; capturer en drop observé ou staging RAM ; **pas** SQLite ledger incomplet.
- Au retour réseau : register → READY → reprise enqueue normale.

---

## Télémétrie CLIENT (minimale)

```text
tracking.session.readiness
  { readiness, tracking_session_id, session_generation }

tracking.session.register_failed
  { tracking_session_id, error_code, lease_state }

tracking.queue.enqueue_blocked
  { reason: not_ready|register_failed|generation_null, readiness }

tracking.queue.ledger_invalid_quarantined
  { location_event_id, tracking_session_id, sequence_id }
```

Ne pas utiliser `nfix` health comme preuve de ce patch (sujet OBSERVABILITY).

---

## Tests (avant canary)

| # | Scénario | Attendu |
|---|----------|---------|
| 1 | `beginNew` + register OK | READY ; enqueue possible |
| 2 | enqueue pendant REGISTERING | 0 row queue ; telemetry blocked |
| 3 | register fail | REGISTER_FAILED ; enqueue bloqué |
| 4 | register fail puis succès | READY ; enqueue avec generation |
| 5 | rotate TTL | pas d’enqueue mid-rotate sans READY |
| 6 | file préchargée gen=null en tête | quarantaine/skip ; item READY derrière flushable |
| 7 | pas de fire-and-forget ready | pas de seq à T0+1ms sans generation |

---

## Critères PASS design → implémentation

```text
PASS CLIENT si :
- aucun nouvel item tracking_queue avec sessionGeneration == null
- session non enregistrée n’est jamais « active » pour le ledger
- register fail → comportement explicite REGISTER_FAILED
- items historiques gen=null ne bloquent plus le flush des suivants
- tests 1–7 verts

FAIL si :
- enqueue possible dès createLocal
- void register suffit à « activer » la session
- head invalide reste en retry HTTP indéfini
```

Critère produit global (rappel) :

> Un item invalide ne bloque pas les positions fraîches suivantes.

---

## Indépendance / ordre

| Sujet | Relation |
|-------|----------|
| C-LEDGER-SERVER | **Après** CLIENT ; claim release indispensable mais **ne remplace pas** le gate CLIENT |
| OBSERVABILITY | Après ; ne pas coupler au gate |
| P0-A / P0-B | Ne pas rouvrir |

---

## Décisions

```text
DESIGN                     = READY
IMPLEMENTATION CLIENT      = LIVRÉE
CANARY CLIENT ISOLÉ        = PASS ✅ (gps-c3-ledger-client-canary-2026-08-14.md)
C-LEDGER-SERVER            = NO-GO
OBSERVABILITY              = NO-GO
PATCH PROD                 = NO-GO
PROCHAINE ÉTAPE            = Design C-LEDGER-SERVER
```

---

## Implémentation

✅ **Implémenté** : C-LEDGER-CLIENT runtime + tests 1–7 + **canary CLIENT isolé PASS**.

Fichiers :

- `mobile/unified-app/src/features/driver/services/driverTrackingQueue.ts`
- `mobile/unified-app/src/features/driver/services/driverTrackingBridge.ts`
- `mobile/unified-app/src/features/driver/services/driverTrackingQueue.ledgerClient.test.ts`
- Canary : `docs/ops/gps-c3-ledger-client-canary-2026-08-14.md` · captures `_c3_ledger_client_2026-08-14/`

**Reste à faire** : design C-LEDGER-SERVER (GO explicite).
