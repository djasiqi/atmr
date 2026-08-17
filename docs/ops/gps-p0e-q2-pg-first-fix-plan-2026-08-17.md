# P0-E Q2 — RCA CLOSED + fix PG-first (canary)

```text
DATE                 = 2026-08-17
Q2 RCA               = CLOSED ✅
ROOT                 = async Kafka → persist_with_outbox sans promotion canonical ✅
PREFERRED FIX        = PG-FIRST CANONICAL PROMOTION ★
FRONTEND PATCH       = NO ✅
MOBILE RC132         = UNCHANGED / FROZEN ✅

PROD FLAG CHANGE     = HOLD ⛔
BACKEND CANARY       = NEXT ★
FLAG-ONLY ON CURRENT IMAGE = NO-GO ❌
```

## Pourquoi pas `flag=true` tout de suite

Image **prod actuelle** (`atmr-backend-1` / kafka-consumer) :

```text
location_candidate.py              = ABSENT ❌
persist_kafka_outbox._maybe_promote = ABSENT ❌
  (fichier prod = 208 lignes, PG commit puis FIN)
```

Workspace local a déjà le chemin P5-B (`_maybe_promote_after_pg` + `promote_location_candidate`).

```text
Activer TRACKING_PG_FIRST_CANONICAL_ENABLED sur l'image actuelle
= NO-OP (aucun code à appeler)
= ne répare pas Q2
```

**NEXT** = déployer le train backend contenant P5-B promote, puis canary flag ON.

---

## Quatre propriétés auditées (module local)

| Propriété | Résultat |
|-----------|----------|
| Monotonie `(session_generation, sequence_id)` — replay N après N+1 | ✅ skip `stale_generation_sequence` |
| TTL canonical = `DRIVER_LOC_TTL_SEC` (défaut 1200) aligné LocationService | ✅ |
| Duplicate / `status!=persisted` → pas de promote | ✅ (contrat `_maybe_promote`) |
| `publish_realtime=False` (session superseded) → pas de promote | ✅ |
| Sync éventuel + consumer | même `promote_location_candidate` + garde gen/seq → pas de régression | ✅ design |

Gates exécutés (overlay fichier, pas flag prod) : `docs/ops/_p0e_pg_first_gates_out.txt` → **ALL_GATES_PASS**.

Tests repo (à rejouer en CI / Docker local) : `backend/tests/services/test_p5b_pg_first_promotion.py`.

---

## Plan canary backend (sans toucher RC132)

```text
1. Deploy backend image incluant :
   - services/tracking/location_candidate.py
   - persist_kafka_outbox._maybe_promote_after_pg
   - sync_ledger_ack._maybe_promote_sync_ledger (si sync fallback)
2. Staging / canary company : TRACKING_PG_FIRST_CANONICAL_ENABLED=true
3. Gates live sur 1 driver :
   - PUT async → PG OK
   - loc:canonical apparaît (seq = LOC, recorded_at OK, TTL≈1200)
   - REST ≠ db_fallback ; location_status live|recent
   - map sans « Aucun GPS récent »
4. Sûreté :
   - seq N+1 puis replay N → canonical reste N+1
   - session superseded / stale → pas de régression
5. Si OK → rollout flag progressif
```

## Après Q2 fixé → Q1

```text
Q1 "Non confirmé" = OPEN
Si ACK reste accepted_async / ingested_non_persisted après PG+canonical
→ sémantique ACK async (distinct de Q2)
```

## Interdit

- Patch frontend map pour masquer `last_known`
- Modifier RC132 / d5-rc-final
- `flag=true` en prod **avant** deploy du code promote
