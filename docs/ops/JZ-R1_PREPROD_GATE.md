# JZ-R1 — Gate préprod (instrumentation remote-first)

**Statut :** CODE GATE CLOSED — backend prod **EN COURS** (B1 ✅, B2 en attente dispatch)  
**Branche certifiée :** `release/gps-pilot-5-drivers-20260823` @ `35cbda91929dd6f93363ffd2e21948337b198bf8`  
**Ordre figé :**

```text
1. COMMIT backend ✅
2. COMMIT mobile ✅
3. PREPROD GATE ★ (ce document)
4. BACKEND PROD
5. SM-S911B CANARY
6. OTA ANDROID
7. PREUVE DISTANTE JOZSEF
8. RCA FIRST_STOP
```

**Hors scope :** patch `anti_zombie_fix_stale` / wrong-layer recovery — **NO-GO**.

---

## BACKEND

| Critère | Attendu | Preuve / statut |
|---------|---------|-----------------|
| Migration JSONB nullable | `tracking_pipeline JSONB NULL` sans default | ✅ Migration `14d1b170291f` — `upgrade()` ajoute colonne nullable uniquement |
| Aucun default / backfill | Pas de `UPDATE` ni valeur par défaut | ✅ Migration filtrée — une seule `add_column` |
| Ancien heartbeat toujours 2xx | Payload sans `tracking_pipeline` | ✅ Test `test_ingest_driver_device_health_legacy_without_tracking_pipeline` + route inchangée |
| Nouveau heartbeat persisté | JSON en PG + Redis snapshot | ✅ Test `test_ingest_driver_device_health_persists_tracking_pipeline` ; `ingest_driver_device_health()` |
| Payload taille raisonnable | Borne défensive | ✅ `_parse_tracking_pipeline()` tronque à 64 clés max |

### Commandes préprod backend (Docker local)

```powershell
docker compose exec -T atmr_api pytest tests/services/test_driver_device_health.py -q
docker compose exec -T atmr_api sh -c "cd /app && alembic -c migrations/alembic.ini upgrade head"
```

### SQL post-migration (staging / préprod)

```sql
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'driver_device_health_events'
  AND column_name = 'tracking_pipeline';
```

---

## MOBILE

| Critère | Attendu | Preuve / statut |
|---------|---------|-----------------|
| `collectTrackingPipelineSnapshot()` fail-open | Flag off → `null`, erreurs swallow | ✅ Guard `isFeatureEnabled` + try/catch heartbeat |
| Erreur observability n'empêche jamais le heartbeat | Heartbeat envoyé même si pipeline crash | ✅ `tickHeartbeat()` : pipeline/anomaly dans try/catch séparés |
| Aucun lock/transaction queue/flush | Lecture seule | ✅ Revue code — pas de `enqueue`/`flush`/`restart` ajouté |
| Coût snapshot faible | Pas de boucle lourde | ☐ Mesure SM-S911B requise (voir gate canary) |
| Aucune action recovery/flush/restart | Instrumentation-only | ✅ Revue code + stash WIP bridge non-JZ-R1 |

### Tests automatisés (CI `mobile-tracking-critical`)

```powershell
cd mobile/unified-app
npm run test:gps-critical
# inclut trackingPipelineObservability + trackingPipelineAnomaly
```

---

## FEATURE FLAG

**Clé :** `tracking_pipeline_remote_observability_enabled`

| Source | Comportement |
|--------|--------------|
| Build-time `EXPO_PUBLIC_ENABLE_TRACKING_PIPELINE_REMOTE_OBS=1` | Active instrumentation |
| Absence de variable | **OFF** (opt-in strict) |
| Bootstrap session (`feature_flags` auth) | Override runtime via `runtimeOverrides` |

### Ciblage device / driver

**Constat :** le système actuel **ne permet pas** un ciblage fin par `device_id` ou `driver_id` pour ce flag.

- Les flags mobile passent par :
  1. **Compile-time** `EXPO_PUBLIC_*` (identique pour tous les clients du même bundle OTA)
  2. **Bootstrap auth** : dict global `MOBILE_FEATURE_FLAGS` / `feature_flags` dans la réponse login — **même payload pour tous les chauffeurs** de la session, pas de cohorte par driver_id

**Conséquence :** activation progressive = **canary par build/OTA** (SM-S911B avec env `EXPO_PUBLIC=1`) puis rollout OTA prod avec flag OFF par défaut jusqu'à GO Jozsef via bootstrap global ou nouvelle OTA.

**Rollback :** `EXPO_PUBLIC_ENABLE_TRACKING_PIPELINE_REMOTE_OBS=0` + republish OTA, ou bootstrap `tracking_pipeline_remote_observability_enabled: false`.

---

## Gate canary SM-S911B (obligatoire avant OTA prod)

Comparer **avant / après** instrumentation (flag ON sur canary uniquement) :

```text
□ Cadence P8 / J7 inchangée
□ queue_depth / persistence_lag stable
□ FGS / tracking_active identiques
□ Missions / disponibilité inchangées
□ Heartbeat legacy 2xx (flag OFF) identique au baseline
□ Heartbeat enrichi (flag ON) : tracking_pipeline présent, pas de lag perceptible
□ Simulation ACK stale >120s → 1 anomaly snapshot, cooldown 5 min respecté
□ ACK reprend → événement RECOVERED
```

---

## Critère instrumentation Jozsef (prod)

Jozsef **n'est pas instrumenté** tant que les 4 points ne sont pas présents en PG :

```text
driver_id = 3
ota_update_id = <nouvelle OTA>
pipeline_snapshot_version = 1
tracking_pipeline IS NOT NULL
```

```sql
SELECT recorded_at, ota_update_id,
       tracking_pipeline->>'pipeline_snapshot_version' AS psv,
       tracking_pipeline->>'first_suspect' AS suspect,
       tracking_pipeline IS NOT NULL AS has_pipeline
FROM driver_device_health_events
WHERE driver_id = 3
ORDER BY recorded_at DESC
LIMIT 10;
```

---

## Décision gate

| Étape | Statut |
|-------|--------|
| Phase 1 Gates (run #83 @ `35cbda91`) | ✅ **PASS** — `booking-critical` + `mobile-tracking-critical` + agrégateur |
| Correctif Alembic `down_revision` | ✅ `861b5acea642` → `14d1b170291f` |
| Commits backend + mobile JZ-R1 | ✅ Remote HEAD gate = `35cbda91` |
| B1 pre-deploy prod (2026-08-27) | ✅ Alembic `861b5acea642`, `/ready` 200, colonne `tracking_pipeline` absente (attendu) |
| B2 deploy backend (`Build & Deploy`) | ☐ **AUTO** — dispatché par Phase 1 Gates PASS sur branche release (plus de dispatch manuel requis) |
| B3 post-deploy gate | ☐ **EN ATTENTE** — après B2 |
| Gate canary SM-S911B | ☐ **HOLD** (après B3 PASS) |
| **GO OTA Android prod** | ☐ **HOLD** |

✅ **Implémenté** : scripts ops `scripts/ops/jz-r1-pre-deploy-check.sh` et `scripts/ops/jz-r1-post-deploy-gate.sh` pour capture B1/B3 via SSH.

**Prochain livrable attendu :** push du commit CI auto-deploy sur `release/gps-pilot-5-drivers-20260823` → Phase 1 Gates → Build & Deploy automatique → post-deploy gate B3.

---

## Références

- OTA prep : [`JZ-R1_OTA_PREP.md`](./JZ-R1_OTA_PREP.md)
- Checklist OTA générique : [`gps-gate-b-ota-checklist.md`](./gps-gate-b-ota-checklist.md)
