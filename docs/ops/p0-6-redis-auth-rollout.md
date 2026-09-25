# P0-6 — Rollout Redis auth (cutover sans invalidation des sessions)

**Statut :** Phase A/B **PASS** · Phase C **PASS** (`dual_write` actif) — backfill / auth_primary **NOT STARTED**
**Date :** 2026-09-25
**Dépendances fermées :** P0-1…P0-5 CLOSED · P0-6 CODE CLOSED
**Interdits :** backfill / auth_primary / auth_only tant que Phase D/E non validées

---

## ⛔ Précondition bootstrap (obligatoire)

```text
NE JAMAIS déployer le backend fail-closed avant
redis-auth + AUTH_REDIS_URL.
```

Ordre **obligatoire** (Phase A = précondition au 1er backend P0-6, pas un contrôle post-déploiement) :

```text
1. provisionner redis-auth
2. vérifier health / noeviction / AOF / volume
3. injecter AUTH_REDIS_URL
4. vérifier la connectivité depuis le backend (réseau compose)
5. seulement ensuite déployer le backend compatible migration
6. démarrer en dual_write
```

**MAUVAIS :** deploy nouveau backend → `AUTH_REDIS_URL` absent → startup failure.

Garanties code :

| Garde | Emplacement |
| --- | --- |
| `AUTH_REDIS_URL` requis avant `up backend` | `scripts/deploy-production.sh` |
| `AUTH_REDIS_MIGRATION_MODE` explicite (pas `off`) | idem |
| `wait_redis_auth_ready` (healthy) | idem |
| smoke ping `AUTH_REDIS_URL` via `compose run --no-deps` | idem |
| `depends_on: redis-auth (service_healthy)` | `docker-compose.production.yml` |
| fail-closed startup si URL absente en prod | `security/auth_redis.py` / `app.py` |
| fail-closed si `off` + `AUTH_REDIS_URL` distinct | `assert_prod_migration_mode_not_ambiguous` / `app.py` |

Les scripts de restart (`restart_backend_with_rl_network.sh`, `restart_docker_services.sh`) s’appuient sur `depends_on` compose ; **ne pas** utiliser `up -d --no-deps backend` pour un premier cutover P0-6.

---

## Objectif

Basculer les refresh tokens de `REDIS_URL` (allkeys-lru) vers `redis-auth` / `AUTH_REDIS_URL` (noeviction + AOF + volume) **sans** invalider les sessions mobiles existantes.

Bascule directe vers un Redis auth **vide** = **INTERDITE**.

---

## Modes (`AUTH_REDIS_MIGRATION_MODE`)

Contrat **explicite** (P0-6 final corrective) :

| Mode | READ | WRITE | Usage |
| --- | --- | --- | --- |
| `legacy` | **legacy** (`REDIS_URL` via `get_legacy_redis`) | legacy | jamais `resolve_auth_redis_url()` |
| `dual_write` | **legacy** | legacy **+** redis-auth | Phase C–E |
| `auth_primary` | **redis-auth** (pas de fallback lecture legacy) | redis-auth **+** legacy | Phase F (rollback court) |
| `auth_only` | redis-auth | redis-auth | fin d’observation |
| `off` | client unique historique | idem | **tests/dev uniquement** |

**Production — interdit :**

```text
AUTH_REDIS_MIGRATION_MODE=off|absent
+ AUTH_REDIS_URL défini et distinct de REDIS_URL
→ startup FAIL / deploy refuse
```

Choisir explicitement `legacy|dual_write|auth_primary|auth_only`.
Évite la fenêtre silencieuse `off + AUTH_REDIS_URL` → écritures auth-only involontaires.

Implémentation : `backend/security/auth_redis_migration.py` + `get_auth_redis()` (`auth_redis.py`) + garde `validate_required_env_vars` / `deploy-production.sh`.

Invariant Phase F : **aucun** `auth missing → lire legacy` par requête.

Compensation P0-2 : en `dual_write` / `auth_primary`, `compensate_redis_issuance` applique le rollback sur le proxy **et** miroir legacy + auth bruts (`_mirror_compensation_both_stores`) — pas de R1 orphelin durable sur le secondaire.

---

## Phases

### A — Préparer redis-auth (précondition backend)

1. Démarrer le service `redis-auth` (compose prod) **sans** pointer encore le trafic applicatif.
2. Script read-only :

```bash
docker compose exec -T atmr_api python scripts/p0_6_redis_auth_phase_a_check.py
```

Vérifier : PING, `maxmemory-policy=noeviction`, AOF, volume, healthcheck.

### B — Mesures production (SCAN)

```bash
docker compose exec -T atmr_api python scripts/p0_6_redis_auth_phase_b_size.py
```

Valider / corriger `256mb` selon `recommended_maxmemory_mb` + headroom. **Ne pas dual-write** tant que non validé.

### C — Dual-write

```text
AUTH_REDIS_URL=<redis-auth>
REDIS_URL=<legacy>
AUTH_REDIS_MIGRATION_MODE=dual_write
```

- READ = legacy
- WRITE = les deux
- Échec write redis-auth → métrique `auth_redis_dual_write_errors_total` ; **pas** de logout utilisateur (legacy reste autorité)

✅ **Implémenté (prod 2026-09-25)** : `AUTH_REDIS_MIGRATION_MODE=dual_write` ; backend `sha-96f8cf9ad05f` (inclut C1/C3) healthy ; client `DualWriteRedisClient` ; lecture = legacy ; canary publish+rotate → même état CURRENT/PREVIOUS sur legacy + redis-auth ; `auth_redis_dual_write_errors_total=0` ; pas de déconnexion canary. **Backfill non démarré.**

### D — Backfill (anti-résurrection)

```bash
# Toujours dry-run d'abord
docker compose exec -T atmr_api python scripts/p0_6_redis_auth_backfill.py --dry-run

# Exécution uniquement après validation humaine
docker compose exec -T atmr_api python scripts/p0_6_redis_auth_backfill.py --execute
```

- TTL : `PTTL` legacy préservé (ex. 42s → ~42s, **pas** 300s).
- Revalidation entre GET et SET : si la clé source a changé (rotation concurrente), **pas** de SET stale → `skipped_stale_race` / `rolled_back_stale`.
- Ne compte **pas** uniquement sur le parity check final pour la sûreté intrinsèque.

✅ **Implémenté (prod 2026-09-25)** : dry-run PASS (`scanned=11606`, `errors=0`, auth counts inchangés) puis `--execute` PASS (`copied=11606`, `skipped_stale_race=0`, `errors=0`). Mode reste `dual_write`, lecture legacy. Compteurs post : legacy CURRENT≈1366 / auth CURRENT≈1372 (écart trafic concurrent — Phase E). **Phase E / auth_primary NOT STARTED.**

Note : le binaire prod n'accepte pas `--dry-run` (dry-run = défaut sans `--execute`).

### E — Parité (missing **et** EXTRA)

```bash
docker compose exec -T atmr_api python scripts/p0_6_redis_auth_parity.py
```

Gate avant `auth_primary` :

```text
missing_current = 0
extra_current = 0
mismatched_current = 0
ttl_mismatched_current = 0
missing_previous inexpliqué = 0
extra_previous inexpliqué = 0
mismatched_previous = 0
ttl_mismatched_previous = 0
missing/extra/mismatched revoked = 0
ttl_mismatched_revoked = 0
missing/extra USER_ZSET = 0
mismatched_user_zset_members = 0
mismatched_user_zset_scores = 0
ttl_mismatched_user_zset = 0
```

Les expirations naturelles PREVIOUS sont distinguées (`previous_expired_during_scan`) des erreurs.
`parity_gate_pass()` refuse aussi tout TTL divergent (tolérance `TTL_DRIFT_MS_TOLERANCE`) et tout score ZSET différent (ordre FIFO).

✅ **Exécuté (prod 2026-09-25)** : audit read-only. `GATE_PASS=false` — `extra_current_in_auth=6`. Forensic DB (reclassifié) :
- 3× `revoke_cleanup_gap` (revoked dual ; delete CURRENT conditionné au GET legacy → CURRENT auth orphelin)
- 3× `off_mode_auth_only_issuance` (`AUTH_REDIS_URL` déjà set + `migration_mode=off` → `resolve_auth_redis_url()` → écriture redis-auth seule ; jamais legacy)
- TTL backfill defect : confirmé / secondaire (pas la cause dominante des 6)

**Hors scope P0-6 migration** : 3 lignes `RefreshToken.session_generation=1` vs `MobileDeviceSession` generation >> 1 — anomalie de données à documenter/traiter séparément ; n’explique pas la divergence Redis observée. Pas de cleanup Redis manuel tant que le correctif n’est pas déployé.
Correctif **local** (non déployé) : TTL absolu backfill (`captured_at_mono` / remaining) + revoke CURRENT inconditionnel multi-store + gate parité REVOKED/USER_ZSET. **Auth_primary NO-GO. Pas de DEL manuel.**

**Reste à faire (ops)** : commit + deploy correctif en dual_write, puis rejouer D/E.

### F — Auth primary

```text
AUTH_REDIS_MIGRATION_MODE=auth_primary
```

- READ = redis-auth
- WRITE = redis-auth + legacy (rollback config, **pas** fallback auto par requête)

Documenter la fenêtre de rollback (ex. 24–72 h dual-write legacy) **avant** F.

Puis période d’observation → `auth_only` → retirer writes legacy.

---

## Fail-closed production

- `ENV=production` + `AUTH_REDIS_URL` absent → refus démarrage (sauf `AUTH_REDIS_ALLOW_LEGACY_FALLBACK=1` temporaire cutover)
- Modes `dual_write` / `auth_primary` / `auth_only` : `AUTH_REDIS_URL` obligatoire et **≠** `REDIS_URL`

Tests : `tests/security/test_p0_6_redis_auth_rollout.py` + `test_p0_6_auth_redis.py`

---

## Métriques

| Métrique | Rôle |
| --- | --- |
| `redis_auth_memory_*` / `evicted_keys` | santé redis-auth (invariant evicted=0) |
| `auth_redis_dual_write_errors_total` | write secondaire KO |
| `auth_redis_migration_missing_total` | audit parité |
| `auth_redis_migration_mismatch_total` | audit parité |
| `redis_auth_rejected_writes_total` | OOM |

---

## Canary (manuel, post Phase F)

Session créée **avant** migration → refresh 200, generation+1, grace, lost-response, session-resume, warm recovery.
Session créée **après** activation redis-auth → idem.

---

## Rapport (à remplir à l’exécution prod)

Voir template dans la demande utilisateur « P0-6 PROD ROLLOUT ». État actuel préparation :

```text
P0-1…P0-5          CLOSED
P0-6 CODE          CLOSED

PHASE A             PASS
PHASE B             PASS
PHASE C DUAL_WRITE  PASS
PHASE D BACKFILL    PASS
PHASE E PARITY      FAIL (6 EXTRA CURRENT — evidence retained)

ROOT CAUSES
  3 × revoke_cleanup_gap           CONFIRMED (fix local)
  3 × off_mode_auth_only_issuance  CONFIRMED (fix local: legacy explicite + prod off refuse)
  TTL backfill defect              CONFIRMED / secondary (fix local)
ROOT CAUSE COMPLETE YES

LOCAL CORRECTIVES   TTL + REVOKE + OFF-MODE + PARITY GATE STRICT (TTL/scores)
DEPLOY CORRECTIVE   NO-GO (attendre commit + gate tests verts)
AUTH_PRIMARY        NO-GO

PROD                dual_write / legacy read
manual cleanup      NO
STOP                ACTIVE — avant deploy correctif
```

✅ **Implémenté (local)** : contrat modes explicite (`legacy` → `get_legacy_redis` uniquement) ; refuse prod `off` + `AUTH_REDIS_URL` distinct (`assert_prod_migration_mode_not_ambiguous`, `validate_required_env_vars`, `deploy-production.sh`) ; tests A–F isolation legacy / garde prod / dual_write / auth_primary / auth_only.
**Reste à faire (ops)** : commit → deploy correctif en restant `dual_write` → réconcilier les 6 extras → Phase E → seulement ensuite envisager `auth_primary`.

---

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `security/auth_redis.py` | clients legacy/dédié + `get_auth_redis` migration |
| `security/auth_redis_migration.py` | modes + DualWriteRedisClient + garde prod off |
| `security/auth_redis_rollout_ops.py` | SCAN / size / backfill anti-résurrection / parité EXTRA |
| `security/refresh_redis_rotation.py` | compensation miroir dual store |
| `scripts/deploy-production.sh` | bootstrap redis-auth + mode migration explicite |
| `scripts/p0_6_redis_auth_phase_a_check.py` | Phase A |
| `scripts/p0_6_redis_auth_phase_b_size.py` | Phase B |
| `scripts/p0_6_redis_auth_backfill.py` | Phase D |
| `scripts/p0_6_redis_auth_parity.py` | Phase E |
| `tests/security/test_p0_6_redis_auth_rollout.py` | tests offline + pre-commit gate |
