# Canary C-LEDGER-SERVER isolé — 2026-08-14

```text
CANARY                 = C-LEDGER-SERVER isolé (CLIENT corrigé déjà actif)
ENV                    = docker local atmr_api + Redis/PG réels
DRIVER                 = 1 (company 2) — harness HTTP (vieux client simulé)
HARNESS                = backend/scripts/canary_ledger_server_p0c.py
CAPTURE                = docs/ops/_c3_ledger_server_2026-08-14/
OPTION B               = active (restart API après patch)
MIGRATION              = flask db upgrade → 25ce766952e2 (capture_id)
```

## Freeze

```text
C-LEDGER-CLIENT               CLOSED / PASS ✅
C-LEDGER-SERVER               CLOSED / PASS ✅
SERVER CANARY                 PASS ✅
OBSERVABILITY                 DESIGN READY / PATCH NO-GO
P0-A / P0-B / C3              CLOSED / PASS ✅
PROD PATCH                    NO-GO ❌
```

## Critères bloquants

| Métrique | Résultat |
|----------|----------|
| `orphan_claim_after_invalid_ids` | **0** |
| `duplicate_final_without_persistence` | **0** |
| `double_write` | **0** |
| `HOL_after_invalid_item` | **0** |
| valid LOC PG progression | **YES** |
| old-client `generation=null` | **422 / non-retryable** |
| claim release after invalid | **YES** |

## Matrice S1–S6

| ID | Scénario | Verdict | Preuve |
|----|----------|---------|--------|
| **S1** | Trafic normal (gen valide) | **PASS** | HTTP 200 `persisted` / `persisted_sync` ; 1 row `driver_location_events` ; claim présent |
| **S2** | Vieux client `generation=null` | **PASS** | HTTP **422** `invalid_ledger_ids` `retryable=false` ; claim **absent** après |
| **S3** | Retry même poison | **PASS** | Nouveau **422** non-retryable ; reclaim OK ; **pas** de cycle `duplicate_unproven` ↔ `ledger_ids_missing` |
| **S4** | Duplicate réellement persisté | **PASS** | Resend même `event_id` → `pg_before=1`/`pg_after=1` ; ACK durable sans 2e row |
| **S5** | Claim sans preuve PG | **PASS** | stale → `duplicate_event_id_unproven` + `ingested_non_persisted` + release ; frais → `claim_in_flight` (claim conservé) |
| **S6** | Progression après poison | **PASS** | poison 422+release → event valide suivant `persisted_sync` ; HOL=0 |

## Lifecycle logs (extraits)

Preuves dans `claim_lifecycle_logs.txt` + corrélation ACK/Redis/PG dans `canary_report.json` :

```text
location_event_claim lifecycle=acquired driver_id=… event_id=…
location_event_claim lifecycle=released … reason=invalid_ledger_ids
location_event_claim duplicate_classified=duplicate_event_id_unproven …
location_event_claim duplicate_classified=claim_in_flight …
```

Sur S1/S4, l’idempotence PG (`same_event` / cache) conserve une seule row ; S5 prouve les classifications VERIFY non finales.

## Notes d’exécution

1. **Restart API** obligatoire après patch Option B (gunicorn ne recharge pas le volume).
2. Migration locale `25ce766952e2` (`capture_id`) appliquée — prérequis schéma pour persistence ledger.
3. Harness enregistre la session via `register_tracking_session` (FK `tracking_ingest_events`).
4. Release aussi sur `duplicate_proximity` post-claim et sur exception `location_update_failed` (invariant claim).
5. Aucune purge Redis préventive ; aucun patch mobile ; OBSERVABILITY non touchée.

## Verdict

```text
CANARY C-LEDGER-SERVER     = PASS ✅
BRANCHE SERVER             = CLOSED / PASS ✅
PROCHAIN GO                = OBSERVABILITY design (gps-p0-c-observability-design.md) — PATCH NO-GO
```

## Implémentation

✅ **Implémenté** : canary SERVER isolé S1–S6 PASS ; captures sous `_c3_ledger_server_2026-08-14/` ; métriques bloquantes à 0 ; branche **CLOSED**.  
**Reste à faire** : rien SERVER ; OBSERVABILITY = design only.
