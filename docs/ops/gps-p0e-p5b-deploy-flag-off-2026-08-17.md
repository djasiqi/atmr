# P0-E — Déploiement P5-B code (flag OFF) puis canary

```text
DATE                           = 2026-08-17
Q2 RCA                         = CLOSED ✅
ROOT                           = async Kafka → PG sans canonical Redis ✅

P5-B CODE PATH (image prod)    = ABSENT (sha-286737a2362e)
FLAG SEUL                      = NO-OP / NO-GO ✅

PG-FIRST CODE AUDIT            = PASS ✅
PG-FIRST TESTS / gates         = PASS ✅

PHASE 1 NEXT                   = migration capture_id + image P5-B + flag OFF ★
PROD BEHAVIOR (phase 1)        = inchangé côté promote (flag false)
FLAG PROD                      = HOLD ⛔
PHASE 2                        = canary flag ON — après smoke phase 1
RC132 / frontend               = UNCHANGED ✅
```

## Séparation stricte

| Phase | Action | Flag | Attendu |
|-------|--------|------|---------|
| **0** | Migration `25ce766952e2` (`capture_id` nullable) | — | schéma prêt |
| **1** | Deploy image P5-B | **OFF** | API+consumer sains ; **0** promote canonical P5-B |
| **2** | Canary flag ON | ON ciblé | smoking gun Q2 |
| **3** | GO flag global | si canary PASS | — |

## Candidat image (prêt sur serveur)

```text
Prod actuel     = djasiqi/atmr-backend:sha-286737a2362e
Candidat P5-B   = djasiqi/atmr-backend:sha-d5694d8e7cec
  Digest        = sha256:5e58f61bf3393ee3883dff55dd04affe688f7bce71021896fa922d633ef2af00
  Preuve        = STAGING_P5B_FINAL PASS (gps-staging-p5b-gate.md)
  Contenu       = location_candidate ✅ + _maybe_promote_after_pg ✅
  Pull serveur  = DONE ✅ (2026-08-17)
```

## Blocker schéma prod (avant recreate)

```text
alembic prod                      = 9b6638784019
driver_location_events.capture_id = ABSENT ❌
migration requise                 = 25ce766952e2 (nullable)
sinon INSERT outbox P5-B          = casse PG
```

```text
PHASE 1 exécution = en attente GO explicite sur le lot :
  (A) alembic upgrade → 25ce766952e2
  (B) DOCKER_TAG=sha-d5694d8e7cec + GIT_SHA aligné
  (C) TRACKING_PG_FIRST_CANONICAL_ENABLED unset/false (vérifié)
  (D) recreate backend + tracking-kafka-consumer (+ outbox si même image)
  (E) smoke : health, PUT→PG, Redis loc:* toujours vide (flag OFF), pas de régression
```

## Smoking gun phase 2 (flag ON) — plus tard

```text
PUT async → 202 → Kafka → PG → _maybe_promote → loc:canonical
→ REST live → map GPS récent
```

## Q1

```text
Q1 = OPEN — après Q2 canary PASS
```

## Interdit

- Flag ON global / canary avant smoke phase 1 PASS
- Recreate image P5-B **sans** migration capture_id
- Flag ON sur sha-286737a2
- Patch frontend / toucher RC132
