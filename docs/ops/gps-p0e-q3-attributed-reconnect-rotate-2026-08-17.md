# P0-E — SESSION OWNERSHIP / ROTATION (Q3) — statut figé 2026-08-17

## Modèle figé

```text
A = session active avec backlog
↓
socket reconnect #1  (resume-resync ON)
↓
reconcileAfterSessionConflict()   // sans preuve serveur de conflit
↓
rotate → B active ; A superseded

~6 s plus tard (throttle reconnect défaut 3 s → 2e passe)
socket reconnect #2
↓
rotate → C active ; B superseded

pendant ce temps :
items déjà en queue pour A → flush immuable
→ PG persiste A superseded
→ pas de promote realtime/canonical ✅ (policy serveur correcte)
```

```text
QUEUE IMMUTABILITY     = comportement attendu ✅
RECONNECT ROTATION     = bug / comportement suspect ★★★
reconnect ≠ conflict   = règle cible >132
```

---

## Statut

```text
P0-E Q2 ROOT             = CLOSED ✅
P5-B                     = NON INVALIDÉ ✅
P5-B CANARY              = BLOCKED

Q3-A reconnect rotation  = ATTRIBUTED ✅ (preuve Traefik + code)
Q3-B post-rotate DLE A   = backlog drain / expected ✅
QUEUE IMMUTABILITY       = expected ✅

SESSION STABILITY GATE   = prêt ✅
  docs/ops/_p0e_session_stability_gate.py

PG_FIRST                 = OFF ✅
RE-GO P5-B               = NO-GO ⛔
GLOBAL ENABLE            = NO-GO ⛔
RC132                    = FROZEN ✅
```

---

## Corrélation 13:26:10 / 13:26:16 (Traefik prod)

Témoin IP `194.230.196.30` (driver 20135) :

| UTC | HTTP | Status | Lecture |
|-----|------|--------|---------|
| 13:23:38–40 | `PUT …/location` | **409** ×6 | hors fenêtre rotate ; ~2,5 min avant ; mobile **ne** mappe pas HTTP 409 → `reconcileAfterSessionConflict` |
| **13:26:10** | `POST …/tracking/sessions` | **200** | register **B** (`3zzbvuqa`) |
| **13:26:16** | `POST …/tracking/sessions` | **200** | register **C** (`gdnf3xtm`) |
| 13:26:x | `PUT …/location` | **202** (nombreux) | drain backlog A ; **aucun 409** dans la fenêtre de rotate |
| 13:26:34 | `POST …/sessions` (autre IP) | 401 | bruit / autre client |
| 13:30:21–23 | `PUT …/location` | 409 ×6 | encore hors lien direct |
| 13:31:11 | `POST …/tracking/sessions` | 200 | register `fl5rv7mx` |

```text
409_IN_ROTATE_WINDOW (13:26:*) = (none)
POST sessions @ 13:26:10 + 13:26:16 = exact match double rotate A→B→C
```

Écart **6 s** entre les deux POST sessions ≈ compatible avec  
`realtime_resync_transition_gate` throttle défaut **3000 ms** : deux reconnects espacés de >3 s → **deux** appels à `reconcileAfterSessionConflict`.

### Discriminant serveur « vrai conflit »

| Signal | Présent à 13:26:10 / :16 ? |
|--------|----------------------------|
| HTTP `409` + `tracking_session_conflict` juste avant register | **NON** |
| Socket ACK `session_conflict: true` | non observé (ws-service logs vides) ; chemin HTTP GPS dominant |
| `POST /tracking/sessions` 200 | **OUI** (×2) |

### ✅ Q3-A ROOT = ATTRIBUTED

```text
CAUSE
  tracking_resume_resync_enabled=1 (EAS/prod)
  + driverRealtimeBridge : justReconnected
    → reconcileAfterSessionConflict()
  = rotate sans preuve serveur de session_conflict

EFFET
  A→B→C ; B/C à 0 DLE ; drain A superseded
  → canary P5-B impossible (_maybe_promote_after_pg skip)
```

Caveat résiduel (faible) : ACK socket `session_conflict` non loggé côté ws — mais le client n’appelle `reconcile` depuis HTTP 409, et la fenêtre 13:26 n’a **aucun** 409. La cause dominante reste reconnect→reconcile.

Détail timeline DLE / enqueue ms : `docs/ops/gps-p0e-session-ownership-q3-2026-08-17.md`.

---

## Correctif candidat (build > RC132)

### Règle

```text
SOCKET RECONNECT / resume-resync
  → releaseSocketEmittedForHttpRetry (si besoin)
  → flush / resync state
  → CONSERVER session courante
  → NE PAS appeler reconcileAfterSessionConflict()

VRAI session_conflict serveur (ACK socket ou contrat HTTP dédié)
  → tombstone ids concernés
  → reconcileAfterSessionConflict / rotate
  → register nouvelle session
```

```text
reconnect ≠ conflict
```

### Barrières supplémentaires

1. **Guard anti-rotate gratuit**  
   Si session locale déjà `READY` + generation non null + aucun flag `explicit_session_conflict` → **interdire** `createLocalTrackingSession()`.

2. **Coalesce rotations** (déjà partiel via `rotateInFlight`)  
   Renforcer : toute 2e demande pendant/juste après rotate → await + no-op si session déjà fraîche ; optionnellement cooldown anti A→B→C (ex. min interval hors conflit explicite).

### Fichiers cibles

- `mobile/unified-app/src/features/driver/services/driverRealtimeBridge.ts` (reconnect)
- `mobile/unified-app/src/features/driver/services/driverTrackingQueue.ts` (guard + coalesce)
- tests listés ci-dessous

---

## Tests indispensables (>132)

```text
socket reconnect unique
  → session id inchangée

2 reconnects rapprochés
  → session id inchangée

foreground resume
  → flush seulement
  → pas de rotate

vrai session_conflict
  → exactement 1 rotate

2 session_conflict concurrents
  → exactement 1 nouvelle session

backlog A + rotate légitime vers B
  → queue A immutable
  → nouvelles positions sur B

aucun session_conflict
  → createLocalTrackingSession impossible depuis reconnect
```

**Test RCA** :

```text
backlog présent + reconnect socket
  → active reste A
  → DLE continuent sur A
  → canonical promotion reste possible (avec PG_FIRST canary ultérieur)
```

---

## NEXT (ops)

1. ~~Corréler reconnect → rotate aux 13:26:10 / :16~~ ✅ Traefik  
2. ~~Implémenter patch >132 (reconnect = flush only) + tests~~ ✅  
   → `docs/ops/gps-p0e-q3-patch-reconnect-ne-conflict-2026-08-17.md`  
3. Ship build **>132** (ex. 133)  
4. Gate stabilité 60–90 s **PASS**  
5. Seulement alors : re-GO P5-B court  

**Ne pas** : promote superseded, affaiblir idempotence, forcer Redis, pm clear en boucle, 3e canary PG-first sur RC132.
