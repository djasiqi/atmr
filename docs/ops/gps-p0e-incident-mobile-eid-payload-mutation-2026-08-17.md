# Incident distinct — mobile rejoue `location_event_id` avec payload muté

```text
STATUS     = FIX IMPLEMENTED tip ✅ — canary device NEXT (build ≥135)
SEVERITY   = P1 ops / bloquant map live + pertinent Q1
DRIVER     = 20135 (témoin)
SEEN_AT    = 2026-08-17 (~12:30–12:35Z et après ; HOME 15:27–15:30Z)
RC132      = FROZEN
BG_FRESHNESS = ATTRIBUTED ✅  docs/ops/gps-p0e-bg-freshness-rca-2026-08-17.md
FIX        = docs/ops/gps-p0e-fix-mobile-event-immutability-2026-08-17.md
```

## Symptôme

```text
même location_event_id
même capture_id
+ recorded_at / lat / lon mutés
→ PUT 202 (Kafka)
→ consumer event_id_payload_conflict
→ DLQ
→ pas de commit PG
```

Preuve DLQ (échantillon) vs ingest déjà persisté :

| Champ | Ingest seq=84 (12:21Z) | Rejeu DLQ (~12:35Z) |
|-------|------------------------|---------------------|
| `location_event_id` | `trk_1786966684481_ec7xbkm7` | **identique** |
| `capture_id` | `cap_msx5tq9t_ah24ckrx2j` | **identique** |
| `recorded_at` | 12:21:35Z | **12:35:10Z** |
| coords | persistées | **légèrement mutées** |

Comportement **serveur = correct** (idempotence P0-D).  
Ne **pas** affaiblir `event_id_payload_conflict` pour débloquer un canary.

## Lien probable Q1 (« GPS connecté · Non confirmé »)

Sur le chemin async, un `202` / `ingested_non_persisted` / `queued_async` **conserve** l’item ledger (`driverTrackingQueue.ts` — branche « Tout le reste (202…) → conserver ») en attendant un ACK durable / watermark.

Si l’item est ensuite **re-PUTé** avec le même `location_event_id` mais un payload GPS rafraîchi :

1. Kafka republie un conflit → DLQ → **jamais de preuve PG**
2. Watermark / confirmation durable ne peut pas arriver pour cet événement
3. UI reste en **Non confirmé**

Hypothèse de travail (à prouver après Q2/P5-B) : boucle  
`awaiting_durable_ack` + retry HTTP avec payload muté.

## Hors scope immédiat

- ~~Pas de patch mobile~~ → **FIX tip** : `gps-p0e-fix-mobile-event-immutability-2026-08-17.md`
- Pas de changement d’idempotence backend
- Canary HOME 120 s après build ≥135 avant reprise Q1

## Contournement ops (témoin seulement)

Nouveau cycle tracking **côté device** (pas purge ledger serveur) :

1. Force-stop `ch.liri.operations` (RC132 non-debuggable → pas de `run-as` rm SQLite)
2. Relance app → nouvelle session tracking observée
3. Gate : `docs/ops/_p0e_gate_clean_session.py`

### Preuve supplémentaire (2026-08-17 Phase 2 canary GO)

Pendant `PG_FIRST=true`, la session pré-gate `lauam301` est devenue **superseded** alors que des DLE continuaient d’y arriver ; la session **active** (`gdnf3xtm`) restait à **0 DLE**.  
→ promote skip (Annexe A.3) → canary P5-B non concluant → rollback `PG_FIRST=false`.

Voir : `docs/ops/gps-p0e-phase2-canary-verdict-2026-08-17.md`


## Références

- Phase 2 inconclusive : `docs/ops/gps-p0e-phase2-canary-2026-08-17.md`
- Idempotence : `backend/services/tracking/location_idempotency.py`
- ACK 202 conserve file : `mobile/unified-app/src/features/driver/services/driverTrackingQueue.ts` (~2092)
