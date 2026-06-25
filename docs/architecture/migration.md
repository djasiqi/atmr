# Migration incrémentale — chaîne GPS (Sprint 1 → 4)

Migration sans big bang via feature flags par couche.

## Sprint 1 — Correction (N1)

| Changement | Flag / config | Rollback |
|------------|---------------|----------|
| Fix `nowIso` flushPoint | Build mobile ≥ 1.0.8 | Queue persistent off |
| Self-heal watch restart | `tracking_self_heal_watch_restart_enabled` | Flag off |
| Circuit breaker | intégré bridge | — |
| Anti-zombie 60 s | self-heal flag | Flag off |
| Kafka partition driver_id | `KAFKA_PARTITION_BY_DRIVER_ID_ENABLED` | env false |
| internal_tracking branché | — | revert route |
| stale_fix_watchdog Celery | `STALE_FIX_WATCHDOG_ENABLED` | disable task |

**Critère merge** : CI green + E2E pipeline + 0 ReferenceError queue.

## Sprint 2 — Résilience (N2)

| Changement | Flag | Rollback |
|------------|------|----------|
| FSM shadow | `tracking_state_machine_enabled` | Flag off |
| Recovery cascade | `tracking_recovery_cascade_enabled` | Flag off |
| Presence / Mission engines séparés | shadow derrière FSM | Flag off |

**Critère** : tests FSM 100 % transitions ; 0 violation INV-2 non RECOVERING/DEGRADED en staging.

## Sprint 3 — Observabilité (N3)

| Changement | Flag | Rollback |
|------------|------|----------|
| TrackingHealthEngine | `TRACKING_HEALTH_ENGINE_ENABLED` | env false |
| ACK par maillon | métriques stage | — |
| Dashboard Grafana | provisioning | revert JSON |
| Frontend merge health | — | revert merge util |

**Critère** : E2E p95 < 1 s staging ; alertes SLO configurées.

## Sprint 4 — Industrialisation (N4)

| Changement | Flag | Rollback |
|------------|------|----------|
| FSM obligatoire | retirer shadow | flag on forcé |
| Suppression legacy bridge | — | branch revert |
| Simulateur flotte CI smoke | workflow | skip job |
| Chaos weekly staging | cron | disable workflow |

**Critère** : pipeline E2E CI nightly green ; simulateur palier 1000 documenté.

## Matrice risques

| Risque | Priorité | Mitigation |
|--------|----------|------------|
| Zombie FGS | P0 | INV-2 anti-zombie + Health Engine |
| ReferenceError queue | P0 | Linter nowIso + test régression |
| Skew Kafka | P1 | Clé driver_id |
| mission_live sans id | P1 | INV-3 garde mobile + métrique backend |
| Cascade agressive | P2 | Plafond 1/30 min |
