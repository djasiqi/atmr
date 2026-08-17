# P0-D D3-C3b — correlation task → payload → enqueue → HTTP → ingest

```text
CYCLE REF     = d3c_delivery HOME 180s (Prod 126, USB RFCW20QC53W)
DRIVER/SESS   = 20135 / trk_sess_1786888547392_42tpr6tu gen=1618
CUT           = dernier LOC PG 2026-08-16T13:57:51.116Z (seq=12)
PATCH         = NO-GO
```

## Limite d'observabilite release

Sur Prod 126 (`__DEV__=false`) :
- `console.info([driver-telemetry]…)` **coupe**
- `sendIngestEvent` **suspendu** (`!__DEV__`)
- `run-as` impossible (non-debuggable)

→ pas de `locations_count` / `enqueue_blocked` en logcat.  
Correlation faite via : `TaskService Finished` + access log HTTP + tables PG + logs Kafka consumer.

Preuve native que Finished ≠ availability-only :  
`LocationTaskConsumer.executeTaskWithLocationBundles` n'appelle `mTask.execute` que si `locationBundles.size > 0` ; `TaskService.Finished` n'apparait qu'apres execution JS.

## Matrice D3-C3b

| Etage | Critere | Verdict |
|-------|---------|---------|
| D3-C3b-1 | Finished + payload locations vide | **RULED OUT** (execute exige bundles > 0 ; HTTP suit) |
| D3-C3b-2 | payload frais + aucun enqueue | **RULED OUT** (flush HTTP present) |
| D3-C3b-3 | enqueue + aucun HTTP | **RULED OUT** |
| D3-C3b-4 | HTTP mais backend refuse / n'ingere pas | **LEADING ★** (voir DLQ) |
| D3-C3b-5 | HTTP+ACK OK mais 0 row PG | **consequence de 4** (pas d'ingest row non plus) |

## Chaine reconstruite (apres 13:57:51Z)

```text
TaskService Finished 'background-location-task'   ✅ ~toutes les 20s
        ↓
PUT /api/v1/driver/me/location                    ✅ HTTP 202 (okhttp)
        ↓  (queued_kafka / tracking_http_accepted_async)
Kafka topic driver.location.raw.v2                ✅ offsets avancent
        ↓
tracking_consumer                                 ❌ DLQ type=event_id_payload_conflict
        ↓
tracking_ingest_events / driver_location_events   ❌ max seq reste 12
```

### Alignement temporel (extrait)

| Device local | UTC | Task Finished | HTTP 202 | DLQ conflict |
|--------------|-----|---------------|----------|--------------|
| 15:57:09/29/50 | 13:57:09/29/50 | oui | oui | oui (partiel) + **LOC seq 10–12 OK** |
| 15:58:10 … 16:00:52 | 13:58:10 … 14:00:52 | oui | ~3 PUT / Finished | **9 DLQ / minute** |
| apres cut | | | | **0 nouvelle row ingest/LOC** |

Apres 13:58Z : **exactement 9 DLQ/minute** ≈ 3 Finished × 3 PUT (flush/retry), tous `event_id_payload_conflict`.

### Preuves PG

```text
tracking_ingest_events  : 12 rows, max sequence_id=12, last 13:57:51Z
driver_location_events  : 12 rows, max sequence_id=12, last 13:57:51Z
tracking_event_outbox   : 12 rows type=persisted, toutes publiees
LOC apres cut           : 0
```

### Preuves HTTP / Kafka

- Access log backend : PUT location **202** continue apres le cut (pas Network Error, pas 4xx/5xx).
- Metric : `tracking_http_accepted_async_total{location_mode="mission_live"}` actif.
- Consumer : `DLQ confirmed … type=event_id_payload_conflict` en continu.

Code : `PayloadConflictError("event_id_payload_conflict")` dans `ingest_durability.py` / `persist_with_outbox.py` — **meme `location_event_id` avec payload divergent** (ou equivalent claim).

## Lecture causale

```text
D3-C3b n'est PAS "task morte" ni "queue morte" ni "HTTP mort".

Rupture = apres admission async :
  client recoit 202 (ACK queued)
  Kafka livre le message
  consumer envoie en DLQ (event_id_payload_conflict)
  → aucune nouvelle sequence ledger / LOC PG

Les Finished post-+46s correspondent a de vraies invocations
avec flush HTTP, pas a de simples onLocationAvailability.
```

Nuance : des DLQ conflict existent **aussi avant** le cut (retries), mais une partie des events passait encore (seq 1–12). Apres le cut, **tout** le flux async de cette session tombe en conflict → LOC plate.

Hypotheses suivantes (toujours read-only / no patch) :
1. Client re-flush les memes `location_event_id` avec payload mute (ts/fields) apres HOME.
2. Double chemin sync+async : event deja persiste puis rejoue async → conflict.
3. Generation d'`event_id` / `captureId` instable sous headless FGS.

## Statut fige

```text
D3-C1        RULED OUT
D3-C2        callback not dead
D3-C3        task invocation alive
D3-C3b       LEADING
  C3b-1..3   RULED OUT
  C3b-4      LEADING ★  (202 + Kafka DLQ event_id_payload_conflict)
  C3b-5      effet (0 row) de C3b-4

PATCH                = NO-GO
GENERAL DISTRIBUTION = NO-GO
```

## Suite

1. Extraire 1–2 messages DLQ (payload event_id + hash) pour un Finished post-cut et comparer a seq<=12.
2. Verifier cote client si le meme `location_event_id` est renvoye avec coords/ts differents (sans patch prod : backup SQLite `driver_tracking_queue_v5.db` si l'utilisateur confirme le dialogue backup, ou build diag).
3. Ne pas patcher ; ne pas whitelister batterie.
