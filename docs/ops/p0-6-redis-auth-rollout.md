# P0-6 — Rollout Redis auth (cutover sans invalidation des sessions)

**Statut :** préparation code/scripts **READY** — exécution production **NOT STARTED** / **NO-GO**
**Date :** 2026-09-25
**Dépendances fermées :** P0-1…P0-5 CLOSED · P0-6 CODE CLOSED
**Interdits :** commit/push/deploy ; mutation prod ; cutover sans parité ; toucher auth mobile

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
| `wait_redis_auth_ready` (healthy) | idem |
| smoke ping `AUTH_REDIS_URL` via `compose run --no-deps` | idem |
| `depends_on: redis-auth (service_healthy)` | `docker-compose.production.yml` |
| fail-closed startup si URL absente en prod | `security/auth_redis.py` / `app.py` |

Les scripts de restart (`restart_backend_with_rl_network.sh`, `restart_docker_services.sh`) s’appuient sur `depends_on` compose ; **ne pas** utiliser `up -d --no-deps backend` pour un premier cutover P0-6.

---

## Objectif

Basculer les refresh tokens de `REDIS_URL` (allkeys-lru) vers `redis-auth` / `AUTH_REDIS_URL` (noeviction + AOF + volume) **sans** invalider les sessions mobiles existantes.

Bascule directe vers un Redis auth **vide** = **INTERDITE**.

---

## Modes (`AUTH_REDIS_MIGRATION_MODE`)

| Mode | READ | WRITE | Usage |
| --- | --- | --- | --- |
| `off` / `legacy` | client unique (`resolve_auth_redis_url`) | idem | défaut actuel / tests |
| `dual_write` | **legacy** (`REDIS_URL`) | legacy **+** redis-auth | Phase C–E |
| `auth_primary` | **redis-auth** (pas de fallback lecture legacy) | redis-auth **+** legacy | Phase F (rollback court) |
| `auth_only` | redis-auth | redis-auth | fin d’observation |

Implémentation : `backend/security/auth_redis_migration.py` + `get_auth_redis()` (`auth_redis.py`).

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

### E — Parité (missing **et** EXTRA)

```bash
docker compose exec -T atmr_api python scripts/p0_6_redis_auth_parity.py
```

Gate avant `auth_primary` :

```text
missing_current = 0
extra_current = 0
mismatched_current = 0
missing_previous inexpliqué = 0
extra_previous inexpliqué = 0
mismatched_previous = 0
```

Les expirations naturelles PREVIOUS sont distinguées (`previous_expired_during_scan`) des erreurs.

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
P0-6 ROLLOUT PREPARATION: READY (code/scripts/tests offline + pre-commit gate)
P0-6 PROD EXECUTION: NOT READY
DEPLOY: NO-GO
PUSH: NO
```

---

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `security/auth_redis.py` | clients legacy/dédié + `get_auth_redis` migration |
| `security/auth_redis_migration.py` | modes + DualWriteRedisClient |
| `security/auth_redis_rollout_ops.py` | SCAN / size / backfill anti-résurrection / parité EXTRA |
| `security/refresh_redis_rotation.py` | compensation miroir dual store |
| `scripts/deploy-production.sh` | bootstrap redis-auth avant backend |
| `scripts/p0_6_redis_auth_phase_a_check.py` | Phase A |
| `scripts/p0_6_redis_auth_phase_b_size.py` | Phase B |
| `scripts/p0_6_redis_auth_backfill.py` | Phase D |
| `scripts/p0_6_redis_auth_parity.py` | Phase E |
| `tests/security/test_p0_6_redis_auth_rollout.py` | tests offline + pre-commit gate |

✅ **Implémenté** : préparation cutover A–F + pre-commit gate (bootstrap, EXTRA parity, anti-résurrection, compensation dual Redis).
**Reste à faire (ops)** : exécuter A→F en production selon ce runbook, canary, puis `auth_only`.
