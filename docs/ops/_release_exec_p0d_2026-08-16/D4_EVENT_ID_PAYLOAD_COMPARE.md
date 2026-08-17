# D4 — Comparaison event_id DLQ vs accepté (read-only)

```text
CAS          = D4-B ✅ CONFIRMÉ
PATCH        = NO-GO
DATE         = 2026-08-16
DRIVER       = 20135 / mission 38224
EVENT_ID     = trk_1786888628909_kryu2j9y
```

## Verdict

**D4-B** — même item logique (même `location_event_id` = id SQLite, même `sequence_id=10`, mêmes coords),
rejoué en HTTP avec un `recorded_at` / `sent_at` **régénérés à chaque tentative**.

Pas D4-A (pas de nouveau fix / nouvelle sequence sous le même id).  
Pas D4-C comme cause primaire (le hash change pour une raison attendue : `recorded_at` fait partie du schéma `tracking-event-payload-v1`).

## Paire comparée

| Champ | Accepté (raw.v2 offset **4252** → PG) | DLQ pick (raw offset **4258**) |
|-------|----------------------------------------|--------------------------------|
| `location_event_id` | `trk_1786888628909_kryu2j9y` | **identique** |
| `queue_item_id` | = `location_event_id` (item.id) | **identique** (pas de champ dédié Kafka) |
| `sequence_id` | 10 | **10** |
| `tracking_session_id` | `trk_sess_1786888547392_42tpr6tu` | **identique** |
| `session_generation` | 1618 | **1618** |
| `driver_id` / `mission_id` | 20135 / 38224 | **identique** |
| lat / lon / accuracy | 46.2116156 / 6.1262053 / 7.803999900817871 | **bit-identiques** |
| speed / heading | 0.06219… / 0.0 | **identiques** |
| `recorded_at` | `2026-08-16T13:57:08.992849+00:00` | `2026-08-16T13:58:09.908303+00:00` (**+61 s**) |
| `sent_at` | `…13:57:08.997Z` | `…13:58:09.910Z` |
| `event_payload_hash` (PG) | `db6ef1ea…decc6f` | (recomputé DLQ) `146ce55d…d855f` |

## Série Kafka (même eid, partition 1)

Cadence ≈ **20 s** = TaskService Finished / flush BG :

| offset | recorded_at (UTC) | sort |
|--------|-------------------|------|
| 4252 | 13:57:08.992 | **accepté** → ingest + LOC |
| 4253 | 13:57:29.457 | conflict → DLQ |
| 4255 | 13:57:49.703 | conflict |
| 4258 | 13:58:09.908 | conflict (échantillon D4) |
| 4261 | 13:58:30.218 | conflict |
| 4264 | 13:58:50.518 | conflict |

→ **un seul item SQLite**, pas une collision d’enqueue de nouveaux fixes.

## Mécanisme causal (code)

1. Flush HTTP (`driverTrackingQueue`) envoie `sendDriverLocation({ ...item.payload, trackingEventId: item.id })`.
2. Body mobile (`api.ts`) : champ **`timestamp`**, **pas** `recorded_at` / `sent_at`.
3. Ingress (`backend/routes/driver.py`) :
   - si `recorded_at` absent → `ts` **ou `datetime.now(UTC)`** (ignore `timestamp`) ;
   - `sent_at` absent → **`datetime.now(UTC)`**.
4. Hash F-02 inclut **`recorded_at`** (pas `sent_at`) → chaque retry HTTP post-ACK async produit un hash différent sous le **même** `location_event_id`.
5. Consumer : 1ʳᵉ version OK ; suivantes `event_id_payload_conflict` → DLQ ; **pas de nouvelle row** `tracking_ingest_events` / `driver_location_events`.
6. Client reste en `queued_async` / non tombstoné → **réessaie indéfiniment** (~3 PUT / Finished).

Pourquoi surtout après HOME : la task BG force `forceHttpFallback: true`. Le chemin socket pose `recorded_at: item.payload.timestamp` (stable) ; le chemin HTTP actuel **ne propage pas** ce timestamp vers `recorded_at`.

## Requalification P0-D

```text
P0-D initial     "FGS prod binary BG failure"
        ↓
P0-D4            "background tracking events rejected by
                  event-id/payload idempotency conflict"
        ↓
P0-D4 / D4-B     "HTTP retry regenerates recorded_at=now
                  while keeping stable location_event_id"
```

FGS `DENIED` / bannière = problème secondaire possible ; **n’explique plus l’absence de LOC PG** (PUT 202 + Kafka raw continuent).

## Hors scope / notes

- Hash PG `db6ef1ea…` : **résolu** — algo outbox `_payload_hash`, pas F-02 scaled (voir `D4_HASH_PROVENANCE.md`).
- `queue_item_id` absent du message Kafka ; identité file = `location_event_id` / `item.id`.

## Artefacts

- `d3c_delivery/d4_dlq_picked.json`
- `d3c_delivery/d4_raw_eid_scan.txt` / `scan_raw_eid.py`
- `d3c_delivery/compare_d4.py` + `d4_compare_out2.txt`

## Statut gate

```text
P0-A/B/LEDGER          CLOSED ✅
P0-D / D4-B            OPEN (cause classée, fix NO-GO)
PATCH                  NO-GO
GENERAL DISTRIBUTION   NO-GO
```

Prochaine étape **si GO patch** (hors de cette tâche) : propager `timestamp` → `recorded_at` côté client **ou** mapper `timestamp` → `recorded_at` à l’ingress, et/ou traiter soft-duplicate sur conflict après accept ; **ne pas** patcher sans décision explicite.
