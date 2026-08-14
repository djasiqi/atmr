# Canary OBSERVABILITY — 2026-08-15

```text
CANARY                 = OBSERVABILITY (mesure / classification)
ENV                    = harness contrôlé (Jest + pytest Docker)
HARNESS                = trackingObservabilityCanary.test.ts
                         + test_driver_device_health (compat backend)
CAPTURE                = docs/ops/_c3_observability_2026-08-15/canary_report.json
MODE                   = non destructif — aucun incident tracking forcé
```

## Freeze

```text
OBSERVABILITY DESIGN          READY ✅
OBSERVABILITY IMPLEMENTATION  IMPLÉMENTÉ ✅
OBSERVABILITY CANARY          PASS ✅
OBSERVABILITY                 CLOSED / PASS ✅

TRACKING FUNCTIONAL PATCH     NO-GO
LEDGER                        CLOSED / PASS ✅
P0-A / P0-B / C3              CLOSED / PASS ✅
P0-C (branche)                CLOSED / PASS ✅
PROD PATCH                    NO-GO ❌
```

## Critères bloquants

| Métrique | Résultat |
|----------|----------|
| fausse classification GNSS avec fix frais | **0** |
| `fix_stale` hors classe GNSS | **0** |
| PIPELINE historique classé GNSS | **0** |
| `Location.timestamp` réellement utilisé | **YES** |
| `task_invoke_age` séparé du fix_age | **YES** |
| régression P0-A / P0-B / ledger | **0** |
| LOC / PUT / persistence inchangé | **YES** |

## Matrice O-C1…O-C6 + historique

| ID | Scénario | Class | fix_stale | Verdict |
|----|----------|-------|-----------|---------|
| **O-C1** | HEALTHY — tout frais | `HEALTHY` | false | **PASS** |
| **O-C2** | PIPELINE — fix frais + queue HOL | `PIPELINE` | false | **PASS** |
| **O-C3** | PERSISTENCE — fix frais + PG retard | `PERSISTENCE` | false | **PASS** |
| **O-C4** | GNSS — Location.timestamp stale | `GNSS` | true | **PASS** |
| **O-C5** | RUNTIME_ONLY — task stale + fix frais | `RUNTIME_ONLY` | false | **PASS** |
| **O-C6** | UNKNOWN — aucun Location | `UNKNOWN` | false | **PASS** |
| **HIST-P0-C** | GNSS frais + enqueue + HOL + persist bloquée | `PIPELINE` | false | **PASS** |

### Vérification historique P0-C (critère principal)

```text
location_fix_age_seconds = 15 (frais)
task_invoke_age_seconds    = 450 (élevé — illusion runtime)
native_last_fix_age       = 450 (= task_invoke, pas GNSS)
health_class               = PIPELINE
fix_stale                 = false
≠ GNSS / fix_stale=true
```

## Compatibilité

```text
native_last_fix_age_seconds = alias de task_invoke_age_seconds
sur O-C1…O-C6 + HIST-P0-C                                 ✅

Backend nouveaux champs prioritaires
(location_fix / task_invoke / observability_class)        ✅

Backend legacy seul (last_fix + native_last_fix)
→ fallback sans exception                                  ✅

Payload minimal → champs vides, pas d’exception            ✅
```

Preuves backend : `tests/services/test_driver_device_health.py`
(`test_canary_observability_backend_compat_*`).

## Notes d’exécution

1. Harness **contrôlé** uniquement — pas d’incident destructif device/prod.
2. Aucun changement fonctionnel tracking / queue / ledger / auth pendant le canary.
3. Classification source de vérité : `trackingObservabilityHealth.ts` (même code que le heartbeat).
4. Rapport machine : `_c3_observability_2026-08-15/canary_report.json`.

## Verdict

```text
CANARY OBSERVABILITY       = PASS ✅
BRANCHE OBSERVABILITY      = CLOSED / PASS ✅
P0-C GLOBAL                = CLOSED / PASS ✅
FREEZE GLOBAL              = gps-p0-global-freeze-2026-08-15.md
```

## Implémentation

✅ **Implémenté** : canary O-C1…O-C6 + HIST-P0-C + compat PASS ; captures JSON ; métriques bloquantes à 0 ; **OBSERVABILITY CLOSED / PASS** ; **P0-C GLOBAL CLOSED / PASS**.  
**Reste à faire** : rien OBSERVABILITY / P0-C ; phase suivante = release/deployment (GO explicite).
