# Release readiness — P0 GPS / ledger / observability

```text
DATE                       = 2026-08-15 (Genève)
DOC                        = release-readiness report (dossier unique)
PHASE                      = RELEASE CONTROL PREPARATION
OPTION RELEASE             = B / P0 ONLY ✅
COMMIT FREEZE P0           = GO ✅
SNAPSHOT PROD READ-ONLY    = GO ✅
TAG RELEASE                = après branche propre (NO-GO tant que dirty/mélange)
ALEMBIC PROD               = NO-GO ❌
PROD DEPLOY                = NO-GO ❌
REDIS / KAFKA / QUEUE PURGE = NO-GO ❌
P0 GLOBAL                  = CLOSED / PASS ✅
```

Freeze amont : [gps-p0-global-freeze-2026-08-15.md](gps-p0-global-freeze-2026-08-15.md)

---

## Verdict gates (instantané)

| Gate | Objet | Statut | Commentaire |
|------|--------|--------|-------------|
| **G0** | SHA P0 individuels | **✅** | Freeze cherry-pick |
| **G0** | SHA release unique (TIP RC) | **⏳** | Base connue = `927640a0…` ; branche **NO-GO** jusqu’à GO |
| **G1** | inventaire migration | **✅** | `25ce766952e2` documentée |
| **G1** | prod `alembic current` | **✅ capturé** | `9b6638784019` ; ≠ `25ce766952e2` |
| **G1** | décision upgrade | **⏳** | Dépendance P0 à confirmer ; **ALEMBIC PROD = NO-GO** |
| **G2** | config prod | **✅ capturé** | Flags + compose ; **skew images** consumer/outbox vs API |
| **G3** | N/N-1 | **⏳** | Après branche release |
| **G4** | rollback | **⏳** | `previous-release.json` **absent** ; tags images connus |
| **G5** | checklist monitoring | **✅** | |
| **G5** | baseline prod | **✅ partielle** | `up` + catalogue ; fanout=0 |

```text
G0–G5 tous VERTS  = NON
PROD DEPLOY       = NO-GO ❌
TAG RC            = NO-GO
```

---

## 1. Commits figés (OPTION B)

### 1.1 Contenu P0 figé (cherry-pick order)

```text
P0 RELEASE CONTENT

P0-A             479cd60d560385b8609e9d93b5c50334ce1edd22
P0-B             4cac0fbf455dd203bd44acac3fc7c47c2b573a46
LEDGER-CLIENT    8861667935203048b8b02937a0f1133464b251e7
LEDGER-SERVER    5e2b098ff521952f33e2fca3d3286934aec32615
OBSERVABILITY    e4adfb06bacd1e867839d98c61047b1d1ef4d84a

DOC FREEZE       ba271034
```

```text
OPTION RELEASE                   = B / P0 ONLY ✅
PROCHAIN GO                      = PROD SNAPSHOT READ-ONLY uniquement
release/gps-p0-2026-08-15        = INTERDIT sans prod-current-SHA
TAG RC                           = INTERDIT sans TIP release testé
Base interdite pour la branche   = main | feat/tracking-p0-p7-firewall
```

Note SERVER : swagger ages observability co-localisé dans `driver.py` du commit SERVER (additif API).

### 1.2 Ancres canary / builds

| Branche | Ancre validée | SHA / build | Preuve |
|---------|---------------|-------------|--------|
| **P0-A** | lifecycle native | `479cd60d560385b8609e9d93b5c50334ce1edd22` | [gps-c3-p0a-canary-2026-08-14.md](gps-c3-p0a-canary-2026-08-14.md) |
| **P0-A EAS** | APK staging-canary | build `d85e3254-…` · tag `gps-canary-p0a-2026-08-14` | [gps-android-canary-apk.md](gps-android-canary-apk.md) |
| **P0-B** | presence hydrate | `4cac0fbf…` | [gps-c3-p0b-canary-2026-08-14.md](gps-c3-p0b-canary-2026-08-14.md) |
| **C3 GLOBAL** | A+B | canary device | [gps-c3-ab-canary-2026-08-14.md](gps-c3-ab-canary-2026-08-14.md) |
| **C-LEDGER-CLIENT** | queue | `88616679…` | [gps-c3-ledger-client-canary-2026-08-14.md](gps-c3-ledger-client-canary-2026-08-14.md) |
| **C-LEDGER-SERVER** | Option B | `5e2b098f…` | [gps-c3-ledger-server-canary-2026-08-14.md](gps-c3-ledger-server-canary-2026-08-14.md) |
| **OBSERVABILITY** | ages + class | `e4adfb06…` | [gps-p0-c-observability-canary-2026-08-15.md](gps-p0-c-observability-canary-2026-08-15.md) |
| **Prod documentée (P0-E)** | à re-vérifier live | `390076ef…` (doc 2026-08-11) | snapshot live **manquant** |

### 1.3 ÉTAPE 2 — Snapshot PROD

**FAIT ✅** — [gps-p0-prod-snapshot-2026-08-15.md](gps-p0-prod-snapshot-2026-08-15.md)

```text
prod-current-SHA = 927640a0995a7025edfae3d31802998948a866d5
ALEMBIC_CURRENT  = 9b6638784019
25ce766952e2     = ABSENT
image skew       = API 927640a0 ≠ consumer/outbox 390076ef ; fanout Created
```

### 1.4 ÉTAPE 3 — Branche release (PAS ENCORE — après snapshot)

```text
1. PROD SNAPSHOT READ-ONLY
   → prod-current-SHA, images, alembic current, flags, compose actifs,
     topologie ps (Up/Created), previous-release, baseline

2. git checkout -b release/gps-p0-2026-08-15 <prod-current-SHA>
   (JAMAIS depuis main ni feat/tracking-p0-p7-firewall)

3. Cherry-pick ordre strict :
   479cd60d… P0-A
   4cac0fbf… P0-B
   88616679… LEDGER-CLIENT
   5e2b098f… LEDGER-SERVER
   e4adfb06… OBSERVABILITY

4. Conflits minimaux seulement — aucun refactor / hors-P0

5. Tests sur le TIP exact

6. WT clean

7. TIP = release candidate G0

8. Tag RC ensuite seulement
```

Ce **TIP** de `release/gps-p0-2026-08-15` devient la référence **G0**, pas le working tree mélangé.

---

## 2. Migrations / schéma

### 2.1 Migration tracking nouvelle vs `origin/main`

| Revision | Fichier | Objet | Réversible |
|----------|---------|-------|------------|
| **`25ce766952e2`** | `25ce766952e2_add_nullable_capture_id_to_tracking_.py` | `capture_id` **nullable** sur `driver_location_events` + `tracking_ingest_events` + index non unique `(driver_id, capture_id)` | **OUI** (`downgrade` drop index + column) |

`down_revision` = `9b6638784019`.

### 2.2 Local canary vs besoin production

| Contexte | Rôle de `capture_id` |
|----------|----------------------|
| **Canary ledger SERVER local** | Prérequis **environnement** (harness) — déjà noté hors RCA prod |
| **Release candidate branche** | **Besoin production réel** si on déploie `e14cfbea+` (wire capture_id / P5-B) — sinon risque d’inserts / assertions schéma |
| **OBSERVABILITY seule** | **Aucune** migration Alembic (Redis snapshot fields only) |
| **C-LEDGER-SERVER Option B** | **Aucune** migration dédiée (logique claim Redis + ACK) |

```text
aucun alembic upgrade prod pour l’instant     = OBLIGATOIRE ✅ (respecté)
durée estimée 25ce766952e2                    = courte (ADD COLUMN nullable + CREATE INDEX)
lock risk                                     = faible→modéré selon taille tables / concurrent writes
backward compat code                          = nullable → vieux writers OK si code ignore la colonne
forward compat                                = nouveau code lit/écrit capture_id ; sans migration → FAIL runtime
```

### 2.3 Inventaire complet prod → release

```text
STATUT = INCOMPLET
CAUSE  = flask db current prod non capturé
ACTION = sur serveur, lecture seule :
  docker compose -f docker-compose.production.yml exec -T backend flask db current
  docker compose -f docker-compose.production.yml exec -T backend flask db history -v | head
```

Comparer ensuite `current` → heads du tag candidat ; lister **toutes** les revisions manquantes (pas seulement `capture_id`).

Docker **local** (dev canary, ≠ prod) au moment du rapport :

```text
flask db heads   = 25ce766952e2 (head)
flask db current = 25ce766952e2 (head)
```

Prod : **à capturer séparément** (lecture seule) — ne pas inferer depuis le local.

---

## 3. Configuration production

### 3.1 Sources de vérité (fichiers dépôt — pas secrets)

| Surface | Fichier | Rôle |
|---------|---------|------|
| Compose prod | `docker-compose.production.yml` | images `BACKEND_IMAGE_REF` / `DOCKER_TAG`, workers |
| Deploy | `scripts/deploy-production.sh` | migrations via `migration_exec`, rollback `previous-release.json` |
| Kafka flags | `env.kafka.production.example` | 4+1 flags tracking Kafka |
| Fragments | `scripts/env.production.defaults.fragment`, `scripts/env.production.local.example` | defaults / local secrets |
| Staging refs (≠ prod) | `env.staging.example`, `docker-compose.staging.yml` | firewall / PG-first pour canary |

### 3.2 Flags `TRACKING_*` à inventorier sur prod (lecture seule)

**Ne pas changer** de valeur au moment du snapshot — G2 = constat, pas mutation.

```text
# Kafka (cohérence stricte)
KAFKA_ENABLED
TRACKING_INGEST_ASYNC_ENABLED
TRACKING_PROCESSED_FANOUT_ENABLED
WS_KAFKA_CONSUMER_ENABLED
TRACKING_INGEST_PERSIST_ENABLED
TRACKING_SOCKET_KAFKA_MIRROR_ENABLED

# Programme branche (souvent OFF en prod aujourd’hui — à vérifier)
TRACKING_MISSION_FIREWALL_MODE          # off | observe | enforce | strict
TRACKING_PG_FIRST_CANONICAL_ENABLED     # défaut false
TRACKING_SESSION_REGISTRY_ENFORCED
TRACKING_HEALTH_ENGINE_ENABLED

# Identité release
GIT_SHA / SENTRY_RELEASE / API_GIT_SHA
BACKEND_IMAGE_REF / DOCKER_TAG
```

```text
AUCUNE modification implicite de feature flag
→ release = code + migrations éventuelles
→ flags prod inchangés sauf GO explicite séparé
```

### 3.3 Workers / consumers à aligner (même digest)

```text
backend (API)
tracking-kafka-consumer / ingest
tracking-processed-fanout
celery-worker / celery-beat (si code tracking partagé)
ws-service (si image distincte — WS_SERVICE_IMAGE_REF)
```

Règle P0-E déjà validée : **consumer = outbox = backend = même GIT_SHA**.

### 3.4 G2 statut

```text
G2 = ROUGE — snapshot .env.production + images live manquant
```

---

## 4. Ordre de déploiement (après G0–G5 VERT)

Ordre privilégié (compat N/N-1 d’abord) :

```text
0. Snapshot baseline monitoring + previous-release.json
1. Migration Alembic (si G1 VERT) — capture_id nullable — SANS purge Redis/Kafka
2. Backend API (image digérée) — compatible N-1 mobile
3. Workers / Kafka consumers / fanout (même digest)
4. Smoke LOC / device-health / ledger ACK
5. Mobile / client (EAS production ou OTA selon GO) — en dernier
```

### 4.1 Matrice compat N/N-1 (à valider)

| Mobile \ Backend | Prod actuel (N-1) | Candidat (N) |
|------------------|-------------------|--------------|
| Mobile N-1 (store) | OK (baseline) | **Doit rester OK** : ledger SERVER tolère vieux clients (`422 invalid_ledger_ids` non-retryable, pas de HOL) ; observability champs absents → fallback |
| Mobile N (P0-B+ledger+obs) | **Risque** si backend rollbacké : presence hydrate OK ; ledger CLIENT sans Option B SERVER → HOL possible à nouveau | Cible release |
| Mobile N + backend N | Cible | OK |

```text
G3 PASS si :
- backend N + mobile N-1 : LOC + health sans 5xx storm
- backend N-1 + mobile N : pas de crash ; dégradation ledger documentée acceptable OU mobile N bloqué jusqu’à backend N
```

Recommandation : **backend (+workers) d’abord**, mobile ensuite — cohérent avec ledger SERVER avant client mass-rollout.

---

## 5. Rollback

### 5.1 Image / tag précédent

```text
Source de vérité = /srv/atmr/releases/previous-release.json
Champs           = backend.reference (repo@sha256:…), ws.reference, GIT_SHA
```

Commandes (serveur — **ne pas exécuter** tant que NO-GO) :

```bash
# Rollback images (mécanisme script)
# deploy-production.sh : rollback() → deploy_from_manifest previous-release.json

# Manuel équivalent
export BACKEND_IMAGE_REF="$(jq -r '.backend.reference' /srv/atmr/releases/previous-release.json)"
export WS_SERVICE_IMAGE_REF="$(jq -r '.ws.reference' /srv/atmr/releases/previous-release.json)"
docker pull "$BACKEND_IMAGE_REF"
docker pull "$WS_SERVICE_IMAGE_REF"
# puis up -d aligné compose production (API + consumers même digest)
```

### 5.2 Mobile corrigé × backend rollbacké

| Situation | Comportement attendu |
|-----------|----------------------|
| Mobile N + backend N-1 (sans Option B) | CLIENT peut éviter null generation ; SERVER N-1 peut encore HOL sur poison → **surveiller** `invalid_ledger_ids` / queue depth |
| Mobile N + obs fields + backend N-1 | Health fields ignorés / legacy Redis — **OK** (observability additive) |
| Mobile N-1 + backend N | SERVER Option B protège ; CLIENT vieux peut encore produire null → **422 non-retryable**, pas de claim orphelin |

### 5.3 Migrations

| Revision | Rollback code | Notes |
|----------|---------------|-------|
| `25ce766952e2` | **Réversible** (`downgrade`) | Seulement après rollback code qui n’écrit plus `capture_id` ; drop column = downtime court / lock |

```text
Ne pas downgrade Alembic « à chaud » sans GO explicite migrations.
Préférer : rollback images d’abord ; downgrade schéma seulement si nécessaire.
```

### 5.4 Seuils rollback immédiat (post-deploy)

```text
ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED (orchestration)   spike vs baseline
native_start_error / NATIVE_ERR                                           > 0 soutenu
auth_not_usable (drivers canary)                                          spike
orphan claims Redis                                                       > 0 croissance
invalid_ledger_ids + queue depth explosion                                 HOL revenu
duplicate_event_id_unproven en boucle                                     non drain
LOC persistence rate                                                      chute nette vs baseline
health_class=GNSS avec location_fix_age frais                             > 0 (régression obs)
5xx /me/location ou device-health                                         spike
```

Atteinte d’un seuil → **rollback images** immédiat ; pas de purge Redis « pour nettoyer ».

---

## 6. Surveillance post-deploy

### 6.1 Signaux minimum

```text
ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED
native_start_error
auth_not_usable
tracking.queue.enqueue_blocked
invalid_ledger_ids
duplicate_event_id_unproven
claim_in_flight
orphan claims
queue depth / oldest item age
location_fix_age_seconds
task_invoke_age_seconds
observability_class / health_class
LOC persistence rate
```

### 6.2 Baseline avant / après

```text
T-30min → T0   : capturer baseline (Prometheus / Loki / snaps device-health Redis)
T0 → T+60min   : même requêtes, delta vs baseline
INTERDIT       : FLUSHDB Redis, reset consumer group, delete SQLite queues device
```

Sources utiles : Grafana GPS canary, Sentry `GIT_SHA`, `driver:*:device_health`, logs `location_event_claim`.

### 6.3 G5 statut

```text
G5 = JAUNE — checklist prête ; panels/baseline live non attachés à ce dossier
```

---

## 7. Gates release — critères VERT

```text
G0 PASS  = SHA taguée unique contient A+B+ledger+obs ; CI verte sur tag ; canaries rejoués ou preuves attachées
G1 PASS  = liste migrations prod→tag complète ; 25ce766952e2 classée ; plan upgrade daté ; encore SANS exécution jusqu’au GO deploy
G2 PASS  = snapshot .env.production + compose images + flags TRACKING_* constatés ; aucun flag changé implicitement
G3 PASS  = preuve N/N-1 (staging ou canary prod-like) backend-first
G4 PASS  = previous-release.json présent ; dry-run rollback images documenté/testé
G5 PASS  = baseline capturée + alertes/dashboard prêts sur la liste §6
```

```text
DEPLOY PROD = NO-GO
tant que G0–G5 ne sont pas tous VERTS
```

---

## 8. Checklist préparation (prochaines actions — toujours NO-GO deploy)

```text
[x] Commit / freeze packs P0-B + ledger CLIENT/SERVER + observability (SHAs listés)
[ ] Snapshot live prod lecture seule (SSH) — checklist gps-p0-prod-snapshot-2026-08-15.md
[ ] Branche release/gps-p0-2026-08-15 depuis prod-current-SHA + cherry-pick P0 only
[ ] Diff migrations prod current → tip release (décider 25ce766952e2)
[ ] Capturer baseline monitoring T-30
[ ] Dry-run rollback (lecture previous-release.json + pull image N-1 sans up)
[ ] GO deploy séparé seulement si G0–G5 VERT
```

---

## Implémentation

✅ **Implémenté** : OPTION B tranchée ; packs P0 commités (SHAs) ; dossier readiness + tentative snapshot (SSH bloqué) ; **PROD DEPLOY / ALEMBIC / PURGE = NO-GO**.  
**Reste à faire** : snapshot prod live → branche `release/gps-p0-*` cherry-pick → G0–G5 VERT → GO deploy explicite.
