# P0-E Q1 — « GPS connecté · Non confirmé » — RCA (2026-08-17)

## Statut figé

```text
Q1 STATIC ROOT               = ATTRIBUTED ✅
Q1 RUNTIME ROOT              = ONE CAPTURE AWAY ★★★

BUILD 134 QA                 = GO ✅ (production-apk interne)
ACK BEHAVIOR PATCH           = HOLD ⛔
event_id conflict            = séparé
PLAY                         = HOLD ⛔

Q1 UI condition               = ATTRIBUTED STATICALLY ✅
202 → ingested_non_persisted  = CONFIRMED STATICALLY ✅
ingested_non_persisted→error  = CONFIRMED STATICALLY ✅

Q2                           = CLOSED ✅
PG_FIRST (prod)              = ON ✅
```

Réf. build : `docs/ops/gps-p0e-go-build-134-q1-rca-2026-08-17.md`
---

## Discriminant UI (audit statique) — ROOT condition ✅

Fichier : `mobile/unified-app/src/features/driver/services/bridgeAckSemantics.ts`  
Affichage : `DriverDashboardHeader` → `formatBridgeSyncLabel(...)`.

### « Confirmé » exige

```text
seqMatch + eventMatch
+ lastAckError == null
+ lastAckStatus ∈ { accepted, duplicate, ingested, persisted }
+ lastAckAt défini
→ "GPS connecté • Confirmé …"
```

### « Non confirmé » exige

```text
(seqMatch + eventMatch + lastAckError)
OU
(seqMatch + eventMatch + status ∈ { stale, ignored, rejected })
→ "GPS connecté • Non confirmé"
```

### « Mis en file » (async intentionnel)

```text
status === "queued" + seq/event match
→ "GPS connecté • Mis en file …"
```

### Fallback si pas d’ACK corrélé

```text
lastUpdate défini → "GPS connecté • Envoyé …"
sinon → "GPS connecté"
```

---

## Chaîne HTTP 202 → UI (CONFIRMED STATICALLY ✅)

| Étape | Comportement |
|-------|----------------|
| Backend PUT async | `accept_status=accepted_async`, `ack_status=ingested_non_persisted`, `durability=queued_async`, HTTP **202** |
| `api.ts` `sendDriverLocation` | Mappe explicitement → `ack_status: "ingested_non_persisted"` |
| Queue flush | Conserve item (`awaiting_durable_ack`), `persistState=ingested_non_persisted`, poll watermark pour tombstone **ultérieur** |
| Bridge `applyBridgeAckStatus` | Applique le statut HTTP du point courant |
| `resolveBridgeAckFields("ingested_non_persisted")` | **Hors** `BRIDGE_CONFIRMED_*` et ≠ `queued` → `lastAckError = "ack_ingested_non_persisted"`, `lastAckAt = null` |
| `formatBridgeSyncLabel` | erreur + seq/event match → **"Non confirmé"** |

Commentaire bridge : *« Un ACK `queued` (HTTP 202 + Kafka async) est valide »* — mais le client **ne reçoit pas** `queued` ; il reçoit `ingested_non_persisted`.

```text
HYPOTHÈSE Q1-A (leading) — ATTRIBUTED STATICALLY
  202 accepted_async est traité comme ÉCHEC ACK UI
  (pas comme « Mis en file » / QUEUED)
  → UI « Non confirmé » même quand PG persist ensuite ✅
```

Le watermark peut clear le ledger SQLite **sans** repasser `lastAckStatus` à `persisted` pour le point courant → l’UI peut rester bloquée sur l’ACK intermédiaire tant qu’aucun ACK final UI n’arrive.

Nuance runtime (build 133) : `beginBridgeAttempt` remet `lastAckError = null` au début de chaque envoi. Si `applyBridgeAckStatus` ne correle pas le point courant, le label steady-state tombe sur **« Envoyé »** (observé device 17:03 — `GPS connecté • Envoyé 16:54`). La capture T3/T7 doit donc lire **`lastAckStatus` / `lastAckError`**, pas seulement le label header.

---

## Protocole capture runtime T0→T7 (NEXT ★★★)

Témoin de référence (historique PG, session stable) :

```text
event_id = trk_1786978639041_1oxvf8se
seq      = 85
session  = trk_sess_1786977672739_0rzte5pe
```

Pour fermer Q1, capturer **un eid en vol** (nouveau préférable ; le 85 sert de modèle) :

```text
T0 MOBILE
  enqueue seqN / eid X

T1 HTTP
  PUT X → 202 → accepted_async

T2 MAPPING CLIENT
  accepted_async → ingested_non_persisted

T3 BRIDGE
  lastAckStatus = ?
  lastAckError  = ack_ingested_non_persisted   ← smoking gun partiel

T4 UI
  → "GPS connecté • Non confirmé"  (si seq/event match)
  ou QA panel lastAck* si label = Envoyé

T5 BACKEND
  PG contient eid X ✅

T6 WATERMARK / LEDGER
  X devient persisted / cleared ?

T7 UI APRÈS PERSISTENCE
  lastAckStatus / lastAckError évoluent-ils ?
  label devient-il "Confirmé" ?
```

### Smoking gun → Q1 RCA CLOSED

```text
HTTP 202 accepted_async
→ lastAckError = ack_ingested_non_persisted
→ UI Non confirmé (ou QA lastAckError latched)

PUIS

PG persistence = CONFIRMED
ledger/watermark = avancé / cleared
MAIS
aucun nouvel ACK UI
→ lastAckError reste latched (ou se ré-applique à chaque 202)
→ UI ne passe PAS à Confirmé
```

Dans ce cas :

```text
Q1 ROOT
= ACK SEMANTIC MISMATCH CLIENT ★★★

202 async est un succès intermédiaire valide
mais l'UI le transforme en erreur.
```

Correctif conceptuel (après preuve — **pas encore implémenté**) :

```text
accepted_async / ingested_non_persisted ≠ ERROR
→ état = QUEUED / PENDING
→ pas de lastAckError
→ idéalement "GPS connecté • En cours de confirmation"

puis seulement ACK final / watermark fiable :
persisted / accepted / duplicate → "GPS connecté • Confirmé"
```

### Sources de capture (build 133)

| Source | Dispo | Notes |
|--------|-------|-------|
| Header sync label | ✅ uiautomator | Peut être « Envoyé » même si mismatch ACK |
| QA panel `lastAck*` | ✅ après patch local | Voir `DriverTrackingQaPanel` — **requis pour T3/T7** |
| SQLite ledger pull | ❌ | package non-debuggable (`run-as` KO) |
| Traefik PUT status | ✅ | 202 / body si logué |
| PG `driver_location_events` | ✅ Docker | T5 |
| ingest localhost:7242 | ❌ release | `__DEV__` off |

---

## Incident associé — `event_id_payload_conflict` (SÉPARÉ)

```text
À déterminer (pas fusionné dans Q1 ROOT tant que non prouvé) :

202 considéré non final
→ retry de X
→ payload de X muté
→ event_id_payload_conflict
```

Si oui : ACK async mal interprété → retry inutile → bug d’immutabilité payload → conflict.

**Même si le retry est légitime**, réutiliser le même `event_id` avec un payload différent reste un **bug indépendant**. Un `event_id` doit être immuable.

```text
eid payload mutation          = INCIDENT ASSOCIÉ
causal link with ACK retry    = À PROUVER
Ne pas affaiblir le check serveur pour « débloquer » l’UI.
```

---

## Ce qu’il ne faut pas rouvrir

- Redis canonical / carte Q2 (FIXED PROD)
- Affaiblir `event_id_payload_conflict` serveur
- Fix sémantique ACK UI **avant** preuve T7 (PLAY HOLD)

## NEXT

1. Install build **134** sur témoin → smoke QA panel Q1 ACK
2. Capture 1 eid : T3 → PG X → T7 (source de vérité = `lastAck*`)
3. Si smoking gun → `Q1 ROOT = ACK SEMANTIC MISMATCH CLIENT` / RCA CLOSED
4. Alors seulement : patch mapping PENDING + label « En cours de confirmation »
5. Piste conflict mutation en parallèle (séparée)

## ✅ Implémenté

- Attribution statique figée (condition UI + chaîne 202→error)
- Protocole T0→T7 + smoking gun explicite
- Conflict gardé séparé
- Instrumentation QA `lastAck*` (noms capture) — **sans** changer la sémantique produit
- Test unitaire smoking-gun `ingested_non_persisted` → `ack_*` → « Non confirmé »
- GO build interne **134** (`production-apk`)
