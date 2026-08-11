# Exécution prod — Réalignement Kafka + Preuve A (2026-08-11)

**Serveur** : `/srv/atmr`  
**Cible** : `390076efc61ca71332c749a67aff1e6fc7c2d626`  
**Image ID** : `sha256:780a166c04b928d3a24a7f773a83cf1835d03512b9ab1073d87ef395003ecc4d`  
**RepoDigest** : `djasiqi/atmr-backend@sha256:fb919878b7297417c0ed89c01a9f4ffc61dd9dd4c75f394ab227c36c79f41acf`

## Phase 0 — PASS

- Triplet immuable confirmé (backend déjà sain).
- `.env.production` : `GIT_SHA`/`SENTRY_RELEASE`=`390076ef…` ; pas de `BACKEND_IMAGE_REF`.
- Dry-run `ops-tracking-p0-recreate-ingest.sh` OK.
- Baseline : consumer/outbox sur `71732a9d…` ; outbox pending=0.

## Phase 1 — PASS

| Service | OCI | Image ID | GIT_SHA / SENTRY |
| --- | --- | --- | --- |
| backend | 390076ef… | 780a166c… | OK |
| ws-service | 390076ef… | image ws distincte | OK |
| tracking-kafka-consumer | 390076ef… | 780a166c… | OK (script P0) |
| tracking-outbox-publisher | 390076ef… | 780a166c… | OK via `env_file` |

- Fanout/DLQ : stopped (HOLD).
- Outbox pending : 0.
- Lag RAW : partitions à 0 sauf p5 lag=5 (stable) ; processed `ws-service-shared` lag=0.
- Logs 15 min : 0 UniqueViolation / IntegrityError / DNS postgres / Traceback ciblés.

Scripts : `scripts/ops-p0e-kafka-align-phase0.sh` … `phase1c.sh`, `logwatch.sh`.

## Phase 2 — PASS (documenté)

Voir [`gps-p0e-seq3-autopsy.md`](gps-p0e-seq3-autopsy.md).

- `contiguous=2`, `max_seen=23`, **seq=3 absente** ledger/DLE/outbox.
- Classification : **`NON DÉTERMINABLE CÔTÉ SERVEUR`**.
- Invariant #8 : **NON PROUVÉ**.

## Phase 3 — PASS

Fenêtre canary bornée ; chemin sync naturel (HTTP **200**, pas 202) malgré `TRACKING_INGEST_ASYNC_ENABLED=true` (circuit/heartbeat → sync).

**Body HTTP réel** (cache idempotent Redis, eid `trk_1786460937260_eh36urxf`) :

```json
{
  "ack_status": "persisted",
  "durability": "persisted_sync",
  "ledger_persisted": true,
  "ledger_reason": "inserted",
  "location_event_id": "trk_1786460937260_eh36urxf"
}
```

Access log corrélé : `PUT /api/v1/driver/me/location` → `200 673`.

Canary SQL : Preuve A OK + Preuve B `dle_sans_ledger=0`.

## Phase 4 — Gate P0-F

| Critère | État |
| --- | --- |
| consumer = outbox = backend = 390076ef | PASS |
| Preuve A P0-E PASS | PASS |
| Autopsie seq=3 documentée | PASS |
| Sentry interrogeable | **NON AUDITABLE** dans cette session (DSN présent côté env ; pas d’audit events/API effectué) |

**Acceptation écrite du risque Sentry** : risque **NON AUDITABLE** accepté pour la décision d’ouvrir le canary P0-F Mission Live BG ; monitoring Sentry post-canary reste à faire hors bande.

### Décision

**GO Phase suivante (canary P0-F Mission Live BG) : OUI** — sous réserve du risque Sentry accepté ci-dessus.

Kafka alignment : **PASS** (ops).  
P0-E preuve canary : **PASS**.  
GO Phase suivante (historique « NON ») : **levé pour P0-F uniquement**.

## Interdits respectés

Pas d’UPDATE ledger/DLE/outbox rows ; pas de DELETE ; pas de reset offsets ; pas de purge Redis ; pas de compose up global ; pas de déploiement `f8509a4d`.
