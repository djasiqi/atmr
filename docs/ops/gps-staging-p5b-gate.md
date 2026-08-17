# Gate — Staging P5-B A/B (PR #92)

Candidat applicatif **figé**. Même image pour A et B. Une seule variable change entre A et B.

```text
APPLICATION SHA                = d5694d8e7cec190978098db6eb20f242226784a8
IMAGE                          = docker.io/djasiqi/atmr-backend:sha-d5694d8e7cec
OCI revision                   = d5694d8e7cec190978098db6eb20f242226784a8
Build & Deploy                 = skip_deploy=true (run 31734910371)
Deploy production              = skipped
PRODUCTION                     = NO
MERGE                          = NO
ENFORCE                        = NO
FANOUT                         = NO
CANARY                         = NEXT (GPS réel Android + iOS)
STAGING_P5B_FINAL              = PASS ✅
```

✅ **Implémenté** : `STAGING_P5B_FINAL = PASS` sur image `sha-d5694d8e7cec` (HTTP/outbox + Socket.IO réel). P5-B synthétique **fermé**.

**Reste à faire** : canary GPS réel — voir [gps-canary-real-devices.md](gps-canary-real-devices.md). Pas de merge / prod / enforce / fanout.

## Étape A — baseline nouveau candidat

```text
TRACKING_MISSION_FIREWALL_MODE=observe
TRACKING_INGEST_ASYNC_ENABLED=true
TRACKING_PG_FIRST_CANONICAL_ENABLED=false
```

Objectif : `d5694d8` avec P5-B désactivé n’a rien cassé vs staging observe précédent.

✅ **Implémenté** : image `sha-d5694d8e7cec` recréée (backend + consumer), migration `25ce766952e2` (colonne `capture_id`), mêmes fixtures + `gps_traffic --profile all` (35/35 HTTP 202).

## Étape B — P5-B activé

Une seule variable :

```text
TRACKING_PG_FIRST_CANONICAL_ENABLED=true
```

Backend **et** `tracking-kafka-consumer` recréés. Même image.

✅ **Implémenté** : flag `true` vérifié dans les deux conteneurs. Trafic + preuves `scripts/staging/p5b_proof.py`.

## Critère n°1

```text
CANONICAL_WITHOUT_DLE_PROOF = 0
```

## Étape B-outbox — même image, `TRACKING_PERSIST_WITH_OUTBOX=true`

```text
TRACKING_MISSION_FIREWALL_MODE=observe
TRACKING_INGEST_ASYNC_ENABLED=true
TRACKING_PG_FIRST_CANONICAL_ENABLED=true
TRACKING_PERSIST_WITH_OUTBOX=true
```

✅ **Implémenté** : même image `sha-d5694d8e7cec` (backend + consumer). Nouvelles identités de fixtures. Replay `p5b_proof.py traffic --label B-outbox` + `b-outbox` + `audit-canonical` + `pg-fail-probe`.

Preuves Kafka/HTTP (chemin outbox) :

| Critère | Résultat |
|---------|----------|
| TRAFFIC 35/35 HTTP 202 | PASS |
| DURABILITY ingest/DLE/outbox > 0 | PASS (42 / 42 / 41) |
| CAPTURE_ID HTTP=ingest=DLE=outbox=Redis | PASS |
| ORDER same generation seq 10 puis 9 | PASS (PG persiste 9, canonical reste 10) |
| ORDER old generation superseded | PASS (DLE oui, outbox realtime=0, Redis reste N+1) |
| DUPLICATE | PASS |
| KAFKA lag / DLQ / persist errors / outbox last_error | 0 / 0 / 0 / 0 |
| CANONICAL_WITHOUT_DLE_PROOF | 0 / 6 = PASS |
| SOCKET | INSUFFICIENT (harness, `source != socket_batch`) |

`STAGING_P5B_FINAL` = **PASS** après preuve Socket.IO réelle (client éphémère hors image).

## Étape SOCKET réel — même image, mirror OFF

```text
SOCKET_GPS_INGEST_ENABLED=true
TRACKING_SOCKET_KAFKA_MIRROR_ENABLED=false
TRACKING_PROCESSED_FANOUT_ENABLED=false
```

✅ **Implémenté** : kill-switch vérifié `true` dans le backend. Client Socket.IO éphémère (`scripts/staging/socket_real_proof.py`, image locale deps, **pas** de patch de `sha-d5694d8`). S1–S5 exécutés :

| Case | Résultat |
|------|----------|
| S1 position normale + ACK + PG + Redis + capture_id | PASS |
| S2 same generation seq 10 puis 9 | PASS |
| S3 old session (`session_conflict`) | PASS |
| S4 PG fail (pgbouncer pausé) → Redis inchangé | PASS |
| S5 duplicate même event/capture | PASS |

**Reste à faire** : canary GPS réel Android + iOS (validation terrain). Arrêt définitif des tests synthétiques P5-B. Pas de merge / enforce / prod / fanout sans canary GO.
