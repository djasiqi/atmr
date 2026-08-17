# P0-D — Détail d’implémentation (signatures / comparaison / insertion)

```text
P0-D IMPLEMENTATION DETAIL = FIGÉ ✅
P0-D PATCH SERVER          = DONE ✅ (2026-08-16)
  module : services/tracking/location_idempotency.py
  ingress : routes/driver.py (timestamp → recorded_at)
  persist : persist_with_outbox.py (hash + identité métier)
  tests   : test_location_idempotency_d4.py — D4-T1…T8 PASS
P0-D CANARY                = NO-GO (attendre GO explicite)
MOBILE BUILD 126           = INCHANGÉ
GENERAL DISTRIBUTION       = NO-GO ❌
```

Date : 2026-08-16  
Parents : `D4_SERVER_DESIGN.md`, `D4_HASH_PROVENANCE.md`, `D4_EVENT_ID_PAYLOAD_COMPARE.md`

**Aucune ligne de correctif runtime dans la passe design** — ce document reste la spec.

## ✅ **Implémenté** (patch serveur 2026-08-16)

- `backend/services/tracking/location_idempotency.py`
- `backend/routes/driver.py` — `resolve_client_recorded_at` avant raw.v2
- `backend/services/tracking/persist_with_outbox.py` — `compare_persisted_event`
- `backend/tests/services/test_location_idempotency_d4.py` — T1…T8 PASS (14 tests avec outbox idempotence)
- Hash v1 **sans** `capture_id` ; mismatch hash → identité métier avant DLQ

---

## 1) API interne (signatures)

Module cible proposé (nom indicatif) :

`backend/services/tracking/location_idempotency.py`

```python
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class DuplicateDecision(str, Enum):
    NEW_EVENT = "new_event"
    DUPLICATE_EXACT_HASH = "duplicate_exact_hash"
    DUPLICATE_LEGACY_EQUIVALENT = "duplicate_legacy_equivalent"
    EVENT_ID_PAYLOAD_CONFLICT = "event_id_payload_conflict"


@dataclass(frozen=True)
class LocationIdentity:
    """Identité métier stable — hors transport."""

    driver_id: int
    location_event_id: str
    tracking_session_id: str
    session_generation: int
    sequence_id: int
    recorded_at_canonical: str  # UTC canonique (même helper que F-02)
    latitude_e6: int
    longitude_e6: int
    accuracy_dm: int | None
    speed_dms: int | None
    heading_ddeg: int | None
    # capture_id EXCLU de LocationIdentity pour v1 / legacy


def canonical_location_identity(payload: Mapping[str, Any]) -> LocationIdentity:
    """Construit l'identité métier depuis un payload normalisé (ingress ou row PG)."""
    ...


def legacy_payload_hash(payload: Mapping[str, Any]) -> str:
    """Hash outbox prod actuel : persist_with_outbox._payload_hash (JSON sort_keys).

    Conservé pour fast-path et écriture tant que schema = tracking-event-payload-v1.
    Ne pas y injecter capture_id silencieusement.
    """
    ...


def compare_persisted_event(
    *,
    existing_row: Mapping[str, Any],
    incoming_payload: Mapping[str, Any],
    incoming_hash: str,
) -> DuplicateDecision:
    """
    Précondition : existing_row.location_event_id == incoming.location_event_id
                   (même driver_id).

    NEW_EVENT n'est pas retourné ici — réservé au chemin « eid absent ».
    """
    ...
```

Mapping décisions → comportement persist :

| Decision | Persist | DLQ | reason / status |
|----------|---------|-----|-----------------|
| `NEW_EVENT` | INSERT | non | `persisted` |
| `DUPLICATE_EXACT_HASH` | no-op | non | `duplicate` / `same_event_already_persisted` |
| `DUPLICATE_LEGACY_EQUIVALENT` | no-op | non | `duplicate` / `legacy_business_equivalent` |
| `EVENT_ID_PAYLOAD_CONFLICT` | raise | oui | `event_id_payload_conflict` |

Invariant de sécurité :

```text
même event_id  ≠  automatiquement duplicate
seul l'identité métier (ou hash exact) autorise le soft-duplicate
```

---

## 2) Comparaison métier — règles figées

### 2.1 Champs d’identité logique (égalité stricte après canon)

```text
driver_id
location_event_id
tracking_session_id
session_generation
sequence_id
recorded_at_canonical
```

Tout écart ici → `EVENT_ID_PAYLOAD_CONFLICT` (sauf absences legacy documentées, voir §2.3).

### 2.2 Valeurs GPS — normalisation déterministe, **pas** tolérance géo

```text
INTERDIT : « à moins de 10 m » / haversine pour décider duplicate

AUTORISÉ : égalité après normalisation déterministe partagée
```

Normalisation proposée (alignée F-02 scaled pour éviter float drift) :

| Champ | Représentation identité |
|-------|-------------------------|
| latitude / longitude | `round(x * 1e6)` → `latitude_e6` / `longitude_e6` |
| accuracy | `None` si absent/≤0 ; sinon `round(x * 10)` → `accuracy_dm` |
| speed | `None` si absent/≤0 ; sinon `round(x * 10)` → `speed_dms` |
| heading | `None` si absent ; sinon `round((x % 360) * 10)` → `heading_ddeg` (0 conservé) |

Égalité GPS = égalité des entiers (et des `None`).

But : absorber float ↔ scaled / JSON round-trip **sans** accepter un vrai autre fix sous le même `event_id`.

### 2.3 `capture_id`

```text
HORS LocationIdentity pour tracking-event-payload-v1 / legacy

legacy row sans capture_id
+ retry avec capture_id
+ même identité location/session/sequence/recorded_at/GPS
→ DUPLICATE_LEGACY_EQUIVALENT
```

Si une future canonicalisation inclut `capture_id` dans le hash → **nouveau** label :

```text
tracking-event-payload-v1 = legacy outbox _payload_hash (prod actuel)
tracking-event-payload-v2 = nouvelle canonicalisation (optionnel, plus tard)
```

Ne pas réutiliser silencieusement `v1` pour un nouvel algo.

### 2.4 Hors identité

```text
sent_at, received_at_ms, HTTP arrival, consumer/outbox processing time,
trace_id, is_background (sauf décision produit ultérieure documentée)
```

---

## 3) Point d’insertion A — ingress HTTP (avant raw.v2)

Fichier : `backend/routes/driver.py` — handler `PUT /me/location`

Séquence figée :

```text
HTTP body
  ↓
parse / validate lat/lon
  ↓
recorded_at =
    payload.recorded_at
    OR payload.timestamp      ← AJOUT explicite (aujourd’hui manquant)
    OR payload.ts
    OR reject / 400 si policy fail-closed
    # NE PLUS : datetime.now(UTC) comme défaut silencieux pour idempotence
  ↓
sent_at = payload.sent_at OR datetime.now(UTC)
  ↓
construire ingest_payload → publish raw.v2
```

Effet :

```text
même item SQLite / même event_id
→ recorded_at constant
→ sent_at variable
→ sent_at déjà hors legacy_payload_hash
→ identité stable entre retries
```

Aussi corriger le défaut précoce actuel (~L1769) qui fait `recorded_at = ts or now` **sans** lire `timestamp` — même ordre de priorité.

Batch `/me/locations/batch` : même mapping si le chemin publie raw / outbox (à aligner dans le patch).

---

## 4) Point d’insertion C+D — persist / consumer

### 4.1 Fast-path + legacy (cœur)

Fichier : `backend/services/tracking/persist_with_outbox.py`  
Remplacer le bloc actuel :

```python
if str(existing["event_payload_hash"]) != phash:
    raise PersistConflictError("event_id_payload_conflict")
```

par :

```text
lookup existing (eid + driver_id)
  → charger hash + colonnes métier nécessaires
    (session, gen, seq, recorded_at, lat/lon, accuracy, speed, heading)
    depuis tracking_ingest_events JOIN driver_location_events
    ou SELECT élargi sur LOC (source de vérité GPS)

incoming_hash = legacy_payload_hash(payload_outbox_dict)  # sans capture_id silencieux

decision = compare_persisted_event(existing, incoming, incoming_hash)

DUPLICATE_EXACT_HASH | DUPLICATE_LEGACY_EQUIVALENT
  → return duplicate (pas d’INSERT, pas raise)

EVENT_ID_PAYLOAD_CONFLICT
  → raise PersistConflictError → consumer DLQ
```

### 4.2 Séquence consumer (explicite)

```text
consumer reçoit raw.v2
↓
persist_driver_location_with_outbox_from_kafka
↓
lookup event_id

ABSENT
→ persist normal (NEW_EVENT)
→ store legacy_payload_hash + row métier

PRESENT
→ exact hash check
   match → DUPLICATE_EXACT_HASH
   mismatch → canonical identity comparison
      equivalent → DUPLICATE_LEGACY_EQUIVALENT
      different  → EVENT_ID_PAYLOAD_CONFLICT → DLQ
```

Règle dure post-changement :

```text
Le consumer ne doit jamais DLQ uniquement parce qu’un hash diffère.
```

### 4.3 Chemin F-02 / `ingest_durability`

Hors chemin incident HTTP→raw.v2→outbox, mais **même sémantique** si un jour unifié :

- fast-path hash exact
- sinon identité métier
- pas de DLQ sur seul mismatch hash

Le patch P0-D **prioritaire** = outbox + ingress ; alignement `ingest_durability` = même GO ou follow-up explicitement scopé.

---

## 5) Écriture nouveaux events (v1)

Tant que schema = `tracking-event-payload-v1` :

```text
continuer à stocker legacy_payload_hash (= _payload_hash prod)
dict hashé SANS capture_id (parité prod actuelle)
```

Introduction éventuelle de v2 = ticket séparé + migration de décision, pas scope du premier patch P0-D.

---

## 6) Assertions concrètes D4-T1…T8 (bloquantes avant GO patch)

### D4-T1 — incident cadence 20 s

```text
event_id = X
Location.timestamp = T
sequence = 10
session/gen fixes

PUT #1 13:57:00
PUT #2 13:57:20
PUT #3 13:57:40
PUT #4 13:58:00
PUT #5 13:58:20
PUT #6 13:58:40
(sent_at / arrival différents à chaque fois)

attendu :
  tracking_ingest_events rows for X = 1
  driver_location_events rows for X  = 1
  DLQ messages for X                 = 0
  conflicts                          = 0
  duplicate decisions                = 5
  (1× NEW ou 1er persist + 5× DUPLICATE_*)
```

### D4-T2 — recorded_at HTTP vs Location timestamp

```text
body.timestamp = T (fix)
server clock skew / omitted recorded_at
→ recorded_at effectif = T
→ retry avec sent_at différent → DUPLICATE_*
```

### D4-T3 — hash legacy ≠ incoming hash, identité égale

```text
existing.event_payload_hash = H_legacy (ne match pas recomput tip)
GPS/session/seq/recorded_at_canonical égaux
→ DUPLICATE_LEGACY_EQUIVALENT
→ 0 DLQ
```

### D4-T4 — vrai conflict

```text
même event_id + même session/gen/sequence
lat/lon différents après normalisation e6
→ EVENT_ID_PAYLOAD_CONFLICT
→ DLQ
→ 0 soft-duplicate
```

### D4-T5 — row PG réelle `_payload_hash` prod

```text
seed / fixture = payload qui produit db6ef1ea… (ou golden local)
retry sous code nouveau
→ pas de faux conflict
```

### D4-T6 — capture_id

```text
existing sans capture_id
incoming avec capture_id
identité métier égale
→ DUPLICATE_LEGACY_EQUIVALENT
```

### D4-T7 — nouvel event déterministe

```text
deux constructions successives du même payload normalisé
→ même legacy_payload_hash
→ même LocationIdentity
```

### D4-T8 — smoke repro HOME/BG (intégration)

```text
build mobile 126 inchangé
driver réel ou harness HTTP→Kafka→consumer
après HOME : retries 20s
→ PUT 202
→ raw.v2
→ 0 DLQ conflict pour items déjà acceptés
→ LOC PG continue (nouvelles seq) OU duplicates propres sur retries
```

---

## 7) Ordre de patch recommandé (quand GO PATCH)

1. Module `location_idempotency` + unit tests T3/T4/T5/T6/T7  
2. Ingress `driver.py` mapping `timestamp` → `recorded_at` (T2)  
3. `persist_with_outbox` decision tree (T1)  
4. Vérifier consumer DLQ mapping inchangé sur vrai conflict uniquement  
5. Smoke T8 (prod/staging) — build 126 **inchangé** côté mobile en premier temps  

Serveur-only d’abord ; client (envoyer `recorded_at` explicitement) = renforcement optionnel ultérieur.

---

## 8) Gates

```text
P0-D DIAGNOSTIC              = CLOSED ✅
D4-B CAUSAL                  = CONFIRMED ✅
HASH PROVENANCE              = CLOSED ✅
P0-D DESIGN A/B/C/D          = READY ✅
P0-D IMPLEMENTATION DETAIL   = GO / FIGÉ ✅  (ce doc)
P0-D PATCH                   = NO-GO ❌
GENERAL DISTRIBUTION         = NO-GO ❌
```

Prochain GO utile : **P0-D PATCH** (serveur), uniquement après relecture explicite de ce détail + plan tests T1–T8.
