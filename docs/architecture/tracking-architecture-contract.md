# Contrat d'architecture — chaîne GPS temps réel ATMR/Lirie

Document de référence pour le flux GPS canonique, les invariants, SLA et compatibilité protocol.

**Produit SoT :** le comportement métier (OFF / BLOCKED / PRESENCE / LIVE, en-service, hors ligne carte) est défini par [`docs/contracts/gps-driver-product-contract.md`](../contracts/gps-driver-product-contract.md). Ce document d’architecture décrit le **comment** technique ; le contrat produit prime en cas d’écart de sémantique.

## Flux canonique (Source of Truth)

```
GPS Provider → Tracking Engine (mobile) → Persistent Queue → Socket.IO batch
  → ACK backend (REAL_ACK) → [HTTP fallback si ACK stale]
  → Backend ingest → Kafka raw.v2 → ingest_consumer
  → Persist canonical → Redis → processed_fanout → Frontend DriverLiveMap
```

**Interdictions** :
- Écriture directe Postgres/Redis hors pipeline persist
- Double capture watch + FGS sans orchestrateur
- `sendLegacyPoint` HTTP en parallèle de la queue sans passer par l'Engine (mode legacy uniquement si flag désactivé)

## Invariants

Voir [`docs/development/invariants.md`](../development/invariants.md) — INV-1 à INV-8.

Vérification : CI (`scripts/architecture/check_tracking_contract.py`) + runtime (`tracking_invariant_violation_total`).

## Matrice des responsabilités

| Composant | Responsabilité | Ne fait PAS |
|-----------|----------------|-------------|
| GPS Provider (OS) | Fix brut | Persister, envoyer réseau |
| Tracking Engine | Capture + normalisation | Écrire Redis/Kafka |
| Persistent Queue | Fiabilité offline | Décider cadence GPS |
| Socket batch | Transport temps réel | Canonicalisation métier |
| Backend ingest | Validation + enqueue Kafka | Affichage dashboard |
| ingest_consumer / persist | Idempotence + Redis canonical | Émission socket directe |
| Fanout | Push dashboard rooms | Modifier coords |
| Frontend | Merge cache live | Lire Postgres temps réel |

## Machine d'état (FSM)

Implémentation : `mobile/unified-app/src/features/driver/tracking/TrackingStateMachine.ts`

États principaux : `IDLE` (OFF), `BLOCKED`, `PRESENCE`, `MISSION_PREPARE`, `MISSION_ACTIVE` / `MISSION_BACKGROUND` (LIVE), `MISSION_RECOVERING`, `MISSION_STOPPING`, `DEGRADED`.

Alignement produit : `IDLE` = hors service / OFF ; `BLOCKED` = en service sans `permissionsReady` ; `PRESENCE` = en service sans mission ; états `MISSION_*` actifs = LIVE.

Flag : `tracking_state_machine_enabled` (shadow puis obligatoire Sprint 4).

## Self-healing cascade

Implémentation : `TrackingRecoveryOrchestrator.ts` — étapes `restart_watch` → `restart_fgs` → `restart_socket` → `restart_engine`.

Plafond : 1 cascade / 30 min. Flag : `tracking_recovery_cascade_enabled`.

Anti-zombie INV-2 : seuil 60 s sans fix → télémétrie `tracking.anti_zombie.triggered` + recovery.

## SLA disponibilité et latence

| Segment | Budget p95 |
|---------|------------|
| GPS → Backend ingest | 300 ms |
| Backend → Kafka raw | 150 ms |
| Consumer → processed | 200 ms |
| Persist → Redis | 50 ms |
| Fanout → Socket | 150 ms |
| Socket → Frontend | 250 ms |
| **Total E2E** | **< 1 s** |

Disponibilité cible flotte : **99,95 %** (positions mission_live reçues dans les 60 s).

## Tracking Protocol

- **v1** : HTTP PUT location (legacy)
- **v2** : Socket batch + persistent queue + REAL_ACK (production actuelle)
- **v3** : correlation IDs étendus + ACK par maillon (rollout progressif)

Politique : rétrocompatibilité v2 minimum 2 sprints après activation v3.

## Migration

Voir [`migration.md`](migration.md).

## Rollback

Registre flags : [`docs/development/feature-flags.md`](../development/feature-flags.md).

Kill-switch critique : `EXPO_PUBLIC_ENABLE_TRACKING_PERSISTENT_QUEUE=0` (mobile), `TRACKING_HEALTH_ENGINE_ENABLED=false` (backend).

## Gate CI (N0)

Checklist automatisée — pas de gate manuelle release :
- Architecture contract tests
- Tests invariants INV-1 à INV-8
- E2E pipeline (`test_tracking_pipeline_e2e.py`)
- Lint + type-check fichiers touchés
