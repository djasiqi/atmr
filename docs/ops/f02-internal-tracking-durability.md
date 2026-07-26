# F-02 — ACK durable et aucune perte silencieuse

**Statut** : Architecture v6 implémentée.  
**GO production** : bloqué jusqu’aux validations capacité PostgreSQL **et** Redis (§18) PASS + tests §17 PASS.

## Garantie officielle (Option B)

> **Aucune perte silencieuse après acceptation durable par le ws-service** (XADD spool Lua/pipeline confirmé).

Extension E2E mobile (NACK, file locale) = **hors périmètre F-02** (sous-lot séparé).

## Contrat HTTP

| Code | Signification |
|------|---------------|
| **200** | Lot durable ou duplicate (même tenant, hash, schema) — `durability=postgres_committed` |
| **503** | Aucun ACK durable ; résultat nul ou incertain |
| **409** | Conflit déterministe — lot entier en quarantaine DLQ côté ws |

### Réponse 409

```json
{
  "ok": false,
  "batch_id": "…",
  "error_code": "event_id_payload_conflict",
  "conflicting_event_ids": ["…"],
  "durability": "none"
}
```

Codes : `event_id_payload_conflict`, `tenant_mismatch`.

## Matrice ws-service (résumé)

| Résultat | Action |
|----------|--------|
| HTTP 200 + contrat complet + `batch_id` match | Lua ACK idempotent (`XDEL==1` → décrément) |
| HTTP 200 incomplet / mauvais `batch_id` | Pas XACK — retry |
| HTTP **409** | **Lua DLQ lot entier** (`batch_payload_conflict` / `batch_tenant_mismatch`, `force=true`) — **aucun retry** |
| HTTP 400 `batch_id_mismatch` | Pending + circuit breaker |
| HTTP 400/422 validation GPS | DLQ `validation` |
| HTTP 401/403 | Pending + circuit |
| HTTP 429 / 5xx / timeout | Retry |
| Âge ≥ `WS_GPS_MAX_EVENT_AGE_SEC` | DLQ `max_age_exceeded` |
| Saturation spool (pré-XADD) | NACK mobile |
| Saturation DLQ (post-XADD, hors 409 force) | Pending + alerte P0 |

## Hash canonique (entiers scaled)

Module : [`backend/services/tracking/event_payload_hash.py`](../../backend/services/tracking/event_payload_hash.py)  
Copie alignée : [`services/ws-service/event_payload_hash.py`](../../services/ws-service/event_payload_hash.py)

- `latitude_e6` / `longitude_e6` = `int(round(x * 1e6))`
- `accuracy_dm` / `speed_dms` / `heading_ddeg` = `int(round(x * 10))` ou absent
- Rejet : `NaN`, `Infinity`, `-0` normalisé

`batch_id` = SHA-256 JSON `tracking-batch-v1`.

## Ledger PostgreSQL

Tables :

- `tracking_ingest_events` — `UNIQUE (driver_id, location_event_id)`
- `tracking_derived_repair_pending` — UPSERT **dans la TX** principale

Migration : `d07b29c401ea_f02_tracking_ingest_ledger_and_repair`

Position courante chauffeur :

```sql
UPDATE driver SET …
WHERE id = :driver_id
  AND (last_position_update IS NULL OR last_position_update < :recorded_at);
```

(`recorded_at` égal → pas de remplacement.)

## Redis idempotence (accélérateur)

- `done` + ledger absent → **ignorer Redis**, continuer PG
- Pas de HTTP 503 pour contention Redis seule
- `mark_done` post-commit = best-effort

## Spool Redis (Lua)

Fichier : [`services/ws-service/gps_spool.py`](../../services/ws-service/gps_spool.py)

- ACK : décrément compteurs **uniquement si `XDEL == 1`**
- DLQ : index `tracking:ws:dlq:src:{stream_id}` ; double-exec → `already`
- DLQ full (hors force 409) → pending conservé
- Replay : `replay_dlq_entry` — `replay_deadline = first_spooled_at + WS_GPS_MAX_EVENT_AGE_SEC` **immuable**

## Repair Redis canonical

1. UPSERT `tracking_derived_repair_pending` dans la TX
2. Tentative immédiate post-commit
3. Worker Celery `tasks.tracking_repair_tasks.process_derived_repairs` (intervalle `TRACKING_DERIVED_REPAIR_INTERVAL_SEC`)

Une réparation ancienne **ne masque pas** une position Redis plus récente.

Fanout Socket.IO = best-effort (hors garantie) → **F-03** outbox.

## Configuration

| Variable | Valeur F-02 |
|----------|-------------|
| `INTERNAL_TRACKING_DURABILITY_MODE` | `sync_db` |
| `KAFKA_ENABLED` | `false` (Kafka hors frontière ACK) |
| `WS_GPS_SPOOL_BACKEND` | `redis_stream` (prod) / `memory` (tests) |
| `WS_GPS_FLUSH_ENABLED` | `false` → `true` (déploiement séquentiel) |
| `WS_GPS_MAX_EVENT_AGE_SEC` | `85800` |
| `WS_GPS_DLQ_RETENTION_DAYS` | `30` |
| `TRACKING_LEDGER_RETENTION_DAYS` | `45` |
| `TRACKING_DERIVED_REPAIR_INTERVAL_SEC` | `60` |
| `WS_GPS_PEL_MIN_IDLE_MS` | `30000` |

## Déploiement séquentiel

1. Redis spool dédié (AOF `appendfsync always` recommandé sous mesure)
2. Migrations ledger + repair
3. ws F-02 avec **flush off** → vérifier spool
4. backend F-02 → smoke ledger
5. flush on → réconciliation 24 h

Rollback backend : suspendre flush ws. **Jamais** republier ws F-01 si pending F-02.

## Checklist capacité §18 — GO prod bloquant

### PostgreSQL

- [ ] Débit réel ingest (p95 POST `/api/internal/tracking/ingest`)
- [ ] Taille index `uq_tracking_ingest_driver_event` à 45 j
- [ ] WAL / backup-restore avec ledger plein
- [ ] Purge par lots (`TRACKING_LEDGER_RETENTION_DAYS=45`) validée

### Redis spool

- [ ] Débit XADD + flush avec `appendfsync always`
- [ ] Croissance AOF / jour
- [ ] Saturation pending + DLQ (seuils `WS_GPS_SPOOL_*` / `WS_GPS_DLQ_*`)
- [ ] p95 scripts Lua ACK/DLQ/replay
- [ ] Reboot → `reconcile_stats` OK (dérive compteurs)

**Les deux** checklists PASS + tests §17 PASS requis avant GO prod.

## Tests §17

```bash
# Backend
docker exec -T atmr-atmr_api python -m pytest \
  tests/services/test_event_payload_hash_f02.py \
  tests/services/test_ingest_durability_f02.py \
  tests/security/test_internal_tracking_f01.py \
  tests/security/test_internal_tracking_f02.py -q

# ws-service (adapter le conteneur / image)
docker exec -T <ws> python -m pytest \
  tests/test_event_payload_hash_f02.py \
  tests/test_gps_ingest_f01.py \
  tests/test_gps_ingest_f02.py \
  tests/test_gps_spool_f02.py -q
```

## Fichiers clés

- [`backend/routes/internal_tracking.py`](../../backend/routes/internal_tracking.py)
- [`backend/services/tracking/ingest_durability.py`](../../backend/services/tracking/ingest_durability.py)
- [`backend/tasks/tracking_repair_tasks.py`](../../backend/tasks/tracking_repair_tasks.py)
- [`services/ws-service/gps_ingest.py`](../../services/ws-service/gps_ingest.py)
- [`services/ws-service/gps_spool.py`](../../services/ws-service/gps_spool.py)

## Risques résiduels

- Batch splitting / replay sélectif : opérateur only
- Fanout live : best-effort F-02
- Mobile E2E : sous-lot post-F-02
- Capacité ledger : gate GO prod
