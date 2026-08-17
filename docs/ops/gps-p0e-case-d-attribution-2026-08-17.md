# P0-E — Attribution Case D (D1/D2) — 2026-08-17

```text
RC132 / D5 = FROZEN ✅
Frontend Q2 = NE PAS PATCHER (exprime last_known backend)
```

## Verdict

```text
CASE D                         = CONFIRMED ★★
D2 wrong Redis/db/prefix       = EXCLUDED ✅
D1 Redis write path non atteint= CONFIRMED ★★ = Q2 EXACT ROOT

Q2 EXACT ROOT
= HTTP async + Kafka persist_with_outbox
  → PG + Driver OK
  → LocationService.store_location NON appelé
  → last_raw / canonical NON écrits
  → promote_location_candidate gated OFF
    (TRACKING_PG_FIRST_CANONICAL_ENABLED unset/false)
```

---

## Discriminant Redis (D2)

Probe = même client que `ext.redis_client` (backend API) :

```text
host=redis  port=6379  db=0
ping=True  canary_set_ok=True
REDIS_URL_fp = redis://redis:6379/0 (password set)
```

SCAN `*20135*` (50 rounds, count≤80) :

```text
driver:20135:active_tracking_session   ← HTTP session bridge (présent)
driver:20135:health                    ← device health
driver:20135:loc:*                     ← AUCUNE
```

→ pas d’écriture sous un autre namespace pour ce chauffeur. **D2 EXCLUDED.**

---

## Discriminant path PG (LOC 5882)

```text
id=5882 source=http seq=64 mission_live mission_id=38243
payload_schema_version=tracking-event-payload-v1
```

Env effectif (API + consumer) :

```text
TRACKING_INGEST_ASYNC_ENABLED      = true
TRACKING_PERSIST_WITH_OUTBOX       = true
TRACKING_PG_FIRST_CANONICAL_ENABLED= unset/false
KAFKA_ENABLED                      = true
```

### Chaîne réelle RC132

```text
PUT /api/v1/driver/me/location
  → TRACKING_INGEST_ASYNC_ENABLED
  → enqueue_tracking_event (Kafka)
  → HTTP 202
      accept_status=accepted_async
      ack_status=ingested_non_persisted
      durability=queued_async
  ✗ PAS UpdateDriverLocationUseCase / LocationService

Kafka tracking-kafka-consumer
  → TRACKING_PERSIST_WITH_OUTBOX
  → persist_driver_location_with_outbox_from_kafka
  → persist_location_event_with_outbox (PG ledger+events+driver+outbox)
  → COMMIT
  → _maybe_promote_after_pg(...)
       SI is_pg_first_canonical_enabled()   ← FALSE
       → return immédiat  ✗ pas de promote_location_candidate
  → FIN (pas de last_raw non plus — last_raw = LocationService only)
```

Réfs code :

- `backend/routes/driver.py` ~1893–1997 (async return)
- `backend/services/tracking/ingest_consumer.py` ~576–643
- `backend/services/tracking/persist_kafka_outbox.py` `_maybe_promote_after_pg` L59–60 early return
- `backend/services/geolocation/location.py` last_raw HSET (chemin sync non pris)

---

## Chaîne symptôme map (fermée)

```text
RC132 PUT ✅
→ Kafka + PG ✅
→ Driver.last_position ✅
→ Redis loc:canonical ❌ (chemin non atteint)
→ REST db_fallback / last_known
→ frontend « Aucun GPS récent »
```

---

## Q1 (séparé, pont candidat)

ACK HTTP async : `ack_status=ingested_non_persisted` ∉ BRIDGE_CONFIRMED  
→ peut alimenter « Non confirmé » **sans** fusion formelle avec Q2 tant que non prouvé sur device.

```text
Q1 = OPEN ★ (pont async ACK plausible)
```

---

## Fix direction (hors implémentation ici)

Options (décision produit/ops, **sans toucher RC132 mobile**) :

1. Activer `TRACKING_PG_FIRST_CANONICAL_ENABLED` pour que `_maybe_promote_after_pg` écrive canonical après PG  
   — **ne remplit pas last_raw** (toujours LocationService) ; REST lit canonical → suffisant pour Q2 map
2. Ou brancher écriture Redis (last_raw+canonical) dans le chemin outbox Kafka
3. Ou sync ingest pour flotte (régression perf)

Ne pas « corriger » le dashboard pour masquer last_known.

## Artefacts

- `_p0e_d12_redis_fp.py` / `_p0e_d12_redis_fp_out.txt`
- T0 : `_p0e_t0_capture_NOW.txt`
