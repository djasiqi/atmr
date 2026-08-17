# D4 — Provenance hash F-02 / outbox (read-only)

```text
D4-B CAUSAL              = CONFIRMED ✅
HASH PROVENANCE          = CLOSED ✅
P0-D DIAGNOSTIC          = CLOSED ✅
P0-D DESIGN              = READY → D4_SERVER_DESIGN.md
P0-D PATCH               = NO-GO
GENERAL DISTRIBUTION     = NO-GO
```

Date : 2026-08-16  
Échantillon : `trk_1786888628909_kryu2j9y` / hash PG `db6ef1ea…decc6f`

---

## Réponses aux 6 questions

| # | Question | Réponse |
|---|----------|---------|
| 1 | Où le hash raw.v2 est calculé ? | **Nulle part.** `driver.location.raw.v2` ne porte **pas** de `event_payload_hash`. Le message HTTP async publie payload + métadonnées ; le hash naît à la persistance consumer. |
| 2 | Où le hash persisted/PG est calculé ? | Consumer Kafka (`TRACKING_PERSIST_WITH_OUTBOX`) → `persist_driver_location_with_outbox_from_kafka` → `persist_location_event_with_outbox` → **`_payload_hash(payload)`** dans `persist_with_outbox.py`. Stocké dans `tracking_ingest_events` + `driver_location_events`. |
| 3 | Même fonction ? | **Non** (dualité). Chemin Kafka outbox prod = `_payload_hash` (JSON sort_keys du dict outbox). Chemin F-02 « officiel » = `compute_event_payload_hash` / `…_from_point` (entiers scaled) dans `event_payload_hash.py` — utilisé par `ingest_durability` / `/internal/tracking` / ws-service, **pas** par le consumer raw.v2 outbox actuel. |
| 4 | Même canonicalisation ? | **Non.** Outbox : floats bruts + `recorded.isoformat()` + champs session/tenant/source. F-02 : `latitude_e6`, `recorded_at` canonique ms Z, `accuracy_dm`, etc. |
| 5 | Quels champs exacts participent (prod outbox) ? | Voir § Champs. **`sent_at` n’y figure pas.** **`recorded_at` oui** → cause D4-B. |
| 6 | Plusieurs versions historiques du hash ? | **Oui, de facto deux algorithmes** sous le même label `payload_schema_version = "tracking-event-payload-v1"`. Une seule constante schema string ; **pas** de version algorithmique distincte en base. Tip local ajoute encore `capture_id` dans le dict hashé (prod **sans**) → 3ᵉ surface de divergence au prochain deploy. |

---

## Mystère `db6ef1ea…` — RÉSOLU

Rejeu read-only sur prod (`atmr-backend-1`) :

```text
_payload_hash({
  driver_id, company_id, location_event_id,
  tracking_session_id, session_generation, sequence_id,
  latitude, longitude, recorded_at (isoformat),
  location_mode, source="http",
  accuracy_m, speed_mps, heading, mission_id,
  schema_version="tracking-event-payload-v1"
})
= db6ef1eae59f3e175fd9da8ac77f8f7f8fa641d9e61291c7a82251e570decc6f  ✅ MATCH
```

Rejeu F-02 scaled sur les mêmes coords/temps → `583c24e9…` ≠ PG.

Donc l’écart observé en D4 **n’était pas** une corruption ; on avait rehashé avec la **mauvaise** fonction (F-02 au lieu d’outbox).

---

## Pipeline figé (chemin incident)

```text
PUT /driver/me/location
  → recorded_at := now (si absent) ; sent_at := now
  → publish raw.v2  (PAS de hash dans le message)
        ↓
consumer (TRACKING_PERSIST_WITH_OUTBOX)
  → persist_with_outbox._payload_hash(dict)
  → INSERT ledger + LOC  OU  conflict si eid existe + hash ≠
        ↓
PersistKafkaOutboxError(event_id_payload_conflict)
  → DLQ driver.location.dlq.v2
```

Conflit : égalité stricte `existing.event_payload_hash != phash` — **aucune** comparaison métier stable (lat/seq/session).

---

## Champs hash outbox (prod déployé)

Inclus (impactent H) :

- identité : `location_event_id`, `driver_id`, `company_id`
- ledger : `tracking_session_id`, `session_generation`, `sequence_id`
- fix : `latitude`, `longitude`, `accuracy_m`, `speed_mps`, `heading`
- temps métier (problématique retry) : **`recorded_at`**
- contexte : `location_mode`, `mission_id`, `source`, `schema_version`

Exclus :

- **`sent_at`** (déjà OK pour sémantique transport)
- `is_background`, `trace_id`, `received_at_ms`, `queue_item_id`

Algo : `SHA-256( json.dumps(payload, sort_keys=True, separators=(",",":"), default=str) )`

---

## Dualité dangereuse (pré-patch)

| Voie | Module | Algo | Label schema |
|------|--------|------|--------------|
| Kafka HTTP → raw.v2 → outbox (prod incident) | `persist_with_outbox._payload_hash` | JSON dict élargi | `tracking-event-payload-v1` |
| Internal batch / ws F-02 | `event_payload_hash.compute_*` | entiers scaled | `tracking-event-payload-v1` |
| Tip git local (non prod) | outbox + **`capture_id`** dans le dict | JSON dict ≠ prod | même label |

Risque post-deploy tip sans compat :

```text
hash stocké (prod actuel, sans capture_id) = H_legacy
hash tip (avec capture_id)               = H_new
→ faux event_id_payload_conflict sur retries d'événements déjà persistés
```

---

## Sémantiques cibles (design — pas encore patch)

```text
recorded_at  = timestamp du fix Location (stable / event_id)
sent_at      = métadonnée transport (variable ; hors hash)  ← déjà hors outbox hash
```

Invariant :

> Un retry du même événement métier doit toujours produire le même hash canonique.

Design privilégié (à spécifier avant GO) :

| Id | Intent |
|----|--------|
| D4-SERVER-A | `recorded_at` = client Location.timestamp (stable) — map `timestamp` côté ingress et/ou client |
| D4-SERVER-B | `sent_at` transport only — rester hors hash (déjà vrai outbox) |
| D4-SERVER-C | si `event_id` existe → comparer identité métier stable → `duplicate_persisted` vs conflict réel |
| D4-SERVER-D | compat legacy : accepter anciens `_payload_hash` / éventuel F-02 sans faux conflict |

Test d’acceptance incident :

```text
même HTTP location + même event_id + 6 envois @ 20s
→ 1 persistence + 5 duplicates + 0 DLQ + 0 event_id_payload_conflict
```

---

## Requalification officielle P0-D

```text
P0-D
→ D4-B CONFIRMED
→ mutable server timestamps break idempotent retries

FGS DENIED / bannière
→ phénomène secondaire documenté
→ non nécessaire pour expliquer l’absence de LOC PG
```

Gates :

```text
D4-B CAUSAL              = CONFIRMED ✅
HASH PROVENANCE          = DOCUMENTED ✅ (cette note)
P0-D PATCH               = NO-GO
GENERAL DISTRIBUTION     = NO-GO
```

## Artefacts

- `d3c_delivery/d4_hash_provenance_out.txt`
- `d3c_delivery/verify_outbox_hash2.py`
- Rapport causal : `D4_EVENT_ID_PAYLOAD_COMPARE.md`
