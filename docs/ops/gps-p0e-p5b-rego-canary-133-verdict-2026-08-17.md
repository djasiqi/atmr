# P0-E P5-B re-GO canary (build 133) — verdict 2026-08-17

## Statut

```text
Q3 PATCH CANARY           = VALIDATED ✅
P5-B RE-GO CANARY         = PASS ✅
ATTRIBUTION N / N+1       = PASS ✅
NO REGRESS                = PASS ✅
REST live                 = PASS ✅ (location_status=live ; position_source vide — WARN)

PG_FIRST                  = false (rollback post-canary) ✅
OUTBOX                    = true ✅
GLOBAL ENABLE             = NO-GO ⛔
PLAY                      = HOLD ⛔
```

## Fenêtre

```text
ENABLE  PG_FIRST=true  ~14:50:19Z
CANARY                 ~14:51:04Z → ~14:51:25Z
ROLLBACK PG_FIRST=false ~14:51:25Z
image                  sha-d5694d8e7cec
PIN session            trk_sess_1786977672739_0rzte5pe (gen 1698)
```

## Preuve P5-B

### N

| Champ | PG DLE 6244 | Redis `loc:canonical` |
|-------|-------------|------------------------|
| session | `…0rzte5pe` | `…0rzte5pe` |
| gen | 1698 | 1698 |
| seq | **51** | **51** |
| eid | `trk_1786978265996_8h26rkey` | idem |
| capture | `cap_msxcpymk_40stpzo92e` | idem |
| TTL | — | **1197** (≈1200) |

`ATTRIBUTION_N=PASS` → `P5B_N_PASS`

### N+1

| | |
|--|--|
| DLE | 6245 seq **52** même session |
| canonical | seq **52**, sess `…0rzte5pe`, ttl 1197 |
| régression | non |

`P5B_N1_PASS` → `NO_REGRESS_PASS`

### REST (fleet)

```text
location_status = live
last_seen_seconds = 13
mission_id = 38243
position_source = (vide)  ← WARN, pas db_fallback / last_known
REST_OK
```

## STOP conditions

Aucune déclenchée (pas de rotate, pas de superseded, pas de regress, healthy).

## Artefacts

- Script : `docs/ops/_p0e_p5b_rego_canary.py`
- Run log : `docs/ops/_p0e_p5b_rego_133/run.txt`

## NEXT

```text
P5-B canary court = PASS ✅
→ verdict formel / décision GLOBAL ENABLE = à GO explicite seulement
→ PG_FIRST reste OFF jusqu'à décision
→ Play = HOLD
```
