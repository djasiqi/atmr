# P0-E — SESSION OWNERSHIP / ROTATION (Q3-A / Q3-B) — 2026-08-17

## Statut figé (HOLD P5-B)

```text
Q2 ROOT                        = CLOSED ✅
P5-B IMPLEMENTATION           = NON INVALIDÉ ✅
P5-B CANARY                   = BLOCKED

PG_FIRST                      = OFF ✅
OUTBOX                        = ON ✅

CLIENT SESSION ROTATION       = OPEN ★★
CLIENT SESSION HANDOFF        = OPEN (affiné) ★★
SUPERSEDED SERVER POLICY      = CORRECT ✅

RE-GO P5-B                    = NO-GO ⛔
GLOBAL ENABLE                 = NO-GO ⛔
RC132                         = FROZEN ✅
```

**Pas de 3e canary PG-first.** Pas d’affaiblissement serveur (promote superseded, idempotence, Redis forcé, pm clear en boucle).

---

## Timeline serveur (témoin driver 20135)

| gen | session | status | started_at (UTC) | DLE |
|-----|---------|--------|------------------|-----|
| 1685 | `…lauam301` | superseded | 13:18:12 | **23** (6060–6083) |
| 1686 | `…3zzbvuqa` | superseded | **13:26:10.603** | **0** |
| 1687 | `…gdnf3xtm` | superseded | **13:26:16.090** | **0** |
| 1689 | `…fl5rv7mx` | active (post) | 13:31:11 | (hors fenêtre canary) |

Double rotation **A → B → C en ~6 s**, puis DLE encore ingestés sur A jusqu’à **13:27:20**.

---

## Q3-A — Qui crée `…gdnf3xtm` ?

### Faits

- Les IDs `trk_sess_<ms>_…` sont créés **uniquement** côté mobile via `createLocalTrackingSession()` puis `registerTrackingSession` (`driverTrackingQueue.ts`).
- Call sites de rotation (`rotateTrackingSessionAwaited`) :
  1. `beginNewTrackingSession` (`begin_new`) — auth recovery idle / authSessionChanged sans backlog
  2. `enqueue` → `ttl_or_missing` (TTL défaut **1800 s** — **exclu** ici : A n’avait ~8 min)
  3. `reconcileAfterSessionConflict` (`session_conflict`)

### Leading ★★★ (code + build RC132)

Flag **ON en prod** : `EXPO_PUBLIC_ENABLE_TRACKING_RESUME_RESYNC=1` (`eas.json` / `.env.production`).

Dans `driverRealtimeBridge.ts` (depuis `4764864a`, juin 2026) :

```text
socket justReconnected
  → releaseSocketEmittedForHttpRetry
  → reconcileAfterSessionConflict()   // TOUJOURS rotate + register
  → flushTrackingQueue
```

**Même sans ACK `session_conflict`**, un reconnect socket déclenche une **nouvelle session active** et supersede la précédente.

Deux reconnects (~6 s) expliquent naturellement :

```text
lauam301 → 3zzbvuqa → gdnf3xtm
```

Note : le resume **foreground** (`hooks.ts`) ne fait qu’un `flushTrackingQueue` — le rotate agressif est le chemin **socket reconnect**.

Autres candidats (à confirmer sur device / télémétrie) :

| Candidat | Plausibilité |
|----------|--------------|
| ACK `session_conflict` réel (batch) | possible ; même chemin `reconcileAfterSessionConflict` |
| `beginNew` auth / bootstrap | faible si backlog pending (preserve → flush only) |
| `startDriverTrackingBridge` / FGS / AppState / mission | **ne crée pas** de tracking_session_id directement |
| `hardRestart` / remote kick | redémarre bridge ; session via queue seulement si autre chemin rotate |
| TTL | exclu sur cette fenêtre |

**Preuve Traefik 13:26:10 / 13:26:16** : deux `POST /tracking/sessions` 200 ; **aucun** `PUT …/location` 409 dans la fenêtre de rotate.  
→ **Q3-A ROOT = ATTRIBUTED** — voir `docs/ops/gps-p0e-q3-attributed-reconnect-rotate-2026-08-17.md`.

**Preuve manquante restante (faible)** : télémétrie device `tracking.session.reconciled` / socket disconnect (package non-debuggable ; ws logs vides). Non bloquant pour attribution face à Traefik + code path.

---

## Q3-B — Pourquoi le producer « reste » sur `lauam301` ?

### Discriminant critique (timestamps client dans `location_event_id`)

`buildQueueId()` = `trk_${nowMs()}_…` au **moment de l’enqueue**.

| seq | location_event_id ms | vs `3zzbvuqa` (1786973170603) |
|-----|----------------------|-------------------------------|
| 8 | 1786972756154 | avant |
| 9 | 1786972776474 | **~6,6 min avant** B |
| 23 | 1786973057866 | **~1,9 min avant** B |

**Tous les DLE 6068–6083 (seq 9–23) encore étiquetés `lauam301` après supersede ont été enqueued AVANT la création de B/C.**  
Ingest serveur tardif (13:26:16–13:27:20) = **drain de file**, pas preuve d’enqueue post-rotation sur A.

### Comportement code (by design)

```text
beginNewTrackingSession / reconcileAfterSessionConflict
  → « N'altère pas les items déjà en file »
  → rebound_count: 0 (explicite)
```

Donc après rotate :

1. File A continue de flusher → serveur voit DLE superseded → `publish_realtime=false` → **pas de promote** (correct).
2. Nouveaux fixes devraient porter B/C **seulement si** `sessionReadiness === READY` après register.
3. Pendant `CREATING` / `REGISTERING` (double rotate 6 s) : enqueue **drop** (`enqueue_blocked`).
4. B et C à **0 DLE** = aucun enqueue réussi sur la session active pendant/après la fenêtre (drops +/ou READY jamais productif +/ou autre rotate).

### Affinage handoff

| Hypothèse | Statut |
|-----------|--------|
| Producer enqueue encore sur A **après** createLocal(B) | **NON prouvé** sur cet échantillon (eid ms &lt; B) |
| Drain multi-session sans rebind → DLE superseded pendant canary | **PROUVÉ** ★★★ |
| Rotate reconnect « gratuit » pendant backlog → active vide | **ATTRIBUTED Q3-A** ✅ |
| Rebind manquant des points futurs (ref mémoire vs file) | ouvert si on capture eid ms **&gt;** B encore sur A |

---

## Gate avant tout futur P5-B (plus strict)

Avant `PG_FIRST=true` :

```text
session X = active
ET DLE arrivent sur X

pendant fenêtre stable ≥ 60–90 s :
  active session id ne change pas
  DLE.session_id = X uniquement
  aucune DLE superseded
  seq augmente
  new eid/capture uniques
  consumer/PG healthy
```

Si rotation mid-fenêtre → **gate FAIL**, ne pas activer PG-first.

Script existant à étendre : `docs/ops/_p0e_pre_pgfirst_active_gate.py` (+ poll stabilité).

---

## Prochaines actions (read-only / fix client hors RC132)

1. **Confirmer Q3-A** : correlér reconnect / `session_conflict` télémétrie au double START 13:26:10 / 13:26:16.
2. **Fix candidat (build &gt; RC132)** : sur reconnect, **ne pas** appeler `reconcileAfterSessionConflict` sans conflit réel — flush / release socket seulement (comme ancien bundle web test).
3. **Handoff** : après rotate volontaire, soit drain+gate avant canary, soit politique explicite « suspend flush superseded » côté client (sans toucher policy serveur).
4. Seulement après gate stabilité 60–90 s → re-GO P5-B court.

---

## ✅ Implémenté (cette passe)

- HOLD P5-B documenté ; priorité basculée vers P0-E SESSION OWNERSHIP.
- Timeline PG sessions 1685–1689 + spans DLE.
- Preuve timestamps enqueue vs rotate (Q3-B = drain attendu).
- **Q3-A ATTRIBUTED** : Traefik POST sessions ×2 @ 13:26:10/:16 sans 409 mid-fenêtre ; path reconnect→reconcile (flag ON).
- Doc fix >132 + tests : `docs/ops/gps-p0e-q3-attributed-reconnect-rotate-2026-08-17.md`
- Gate stabilité 60–90 s : `docs/ops/_p0e_session_stability_gate.py`
- RE-GO / GLOBAL = NO-GO ; RC132 FROZEN.
