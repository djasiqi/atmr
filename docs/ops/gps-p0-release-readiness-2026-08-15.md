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
| **G0** | tests / CI / canaries | **JAUNE** | Canaries A/B/C3/ledger/obs **PASS** ; code B/ledger/obs **non commité** → SHA release non figée |
| **G1** | migrations | **JAUNE** | Migration `25ce766952e2` (`capture_id`) listée ; **current prod inconnu** (pas de snapshot live) ; **aucun `alembic upgrade` prod** |
| **G2** | config prod | **ROUGE** | `.env.production` serveur **non lu** (pas de `.local.deploy.env` local) → inventaire à compléter sur serveur |
| **G3** | compat N/N-1 | **JAUNE** | Matrice rédigée ; non vérifiée contre images/tags prod réels |
| **G4** | rollback | **JAUNE** | Procédure documentée via `previous-release.json` ; **non testée** sur ce candidat |
| **G5** | monitoring | **JAUNE** | Checklist prête ; **baseline avant** non capturée |

```text
G0–G5 tous VERTS  = NON
DEPLOY PROD       = NO-GO ❌
```

---

## 1. Commits figés

### 1.1 État working tree (bloquant freeze)

```text
Branche locale     = feat/tracking-p0-p7-firewall
HEAD commité       = 7197914905f3ffb516b21e5893d60d79f8e838fe
origin/main        = 765b81837705d9f6657348d5a166514481a794b2
Ahead origin/main  = ~29 commits (firewall + capture_id + P5-B + CI + …)
Working tree       = DIRTY — P0-B / C-LEDGER / OBSERVABILITY majoritairement non commités
```

**Conséquence** : on ne peut pas encore publier un **release SHA unique** couvrant toute la branche P0 fermée.  
Prochaine action freeze (hors deploy) :

```text
1. Commit packagé P0-B + C-LEDGER-CLIENT + C-LEDGER-SERVER + OBSERVABILITY (+ docs)
2. Tag candidat ex. gps-p0-release-candidate-2026-08-15
3. Recalculer la table SHA ci-dessous
4. Rebuild images backend + mobile sur ce tag uniquement
```

### 1.2 Ancres canary déjà figées (commits / builds)

| Branche | Ancre validée | SHA / build | Preuve |
|---------|---------------|-------------|--------|
| **P0-A** | lifecycle native | `479cd60d560385b8609e9d93b5c50334ce1edd22` | canary [gps-c3-p0a-canary-2026-08-14.md](gps-c3-p0a-canary-2026-08-14.md) |
| **P0-A EAS** | APK staging-canary | build `d85e3254-9f24-43fc-9218-0d281858b960` · tag `gps-canary-p0a-2026-08-14` | [gps-android-canary-apk.md](gps-android-canary-apk.md) |
| **P0-B** | presence hydrate | **WORKING TREE** (`trackingAuthPresence.ts` untracked + diffs session/client) | canary [gps-c3-p0b-canary-2026-08-14.md](gps-c3-p0b-canary-2026-08-14.md) — **pas de SHA git** |
| **C3 GLOBAL** | A+B combiné | Metro/JS sur device au moment canary (pas d’image prod) | [gps-c3-ab-canary-2026-08-14.md](gps-c3-ab-canary-2026-08-14.md) |
| **C-LEDGER-CLIENT** | queue generation null | **WORKING TREE** (`driverTrackingQueue.ts` modifié) | [gps-c3-ledger-client-canary-2026-08-14.md](gps-c3-ledger-client-canary-2026-08-14.md) |
| **C-LEDGER-SERVER** | claim release / Option B | **WORKING TREE** (`driver.py`, `driver_location_dedup.py`, `sync_ledger_ack.py`) | [gps-c3-ledger-server-canary-2026-08-14.md](gps-c3-ledger-server-canary-2026-08-14.md) |
| **OBSERVABILITY** | ages + class | **WORKING TREE** (`trackingObservabilityHealth.ts` untracked + heartbeat/bridge/backend health) | [gps-p0-c-observability-canary-2026-08-15.md](gps-p0-c-observability-canary-2026-08-15.md) |
| **capture_id** (socle) | migration + wire | `e14cfbeab2d5ac4e4c1c755b726982a5cde8fb1e` | commit sur branche |
| **Dernière image prod documentée** (P0-E, 2026-08-11) | align Kafka | `390076efc61ca71332c749a67aff1e6fc7c2d626` | [gps-p0e-kafka-align-execution-2026-08-11.md](gps-p0e-kafka-align-execution-2026-08-11.md) — **à re-vérifier live** |

### 1.3 Diff vs prod actuelle (hypothèse documentaire)

Base prod **documentée** = `390076ef…` (août 2026).  
Candidat local = `HEAD 71979149` + **working tree dirty**.

```text
delta documenté  ≈ 390076ef → 71979149  (+ dirty WT)
contenu delta    = programme tracking P0–P7 firewall / capture_id / P5-B / CI
                 + packs P0-B / ledger / observability non commités
```

**À faire avant G0 VERT** : snapshot live serveur :

```bash
# Sur le serveur (lecture seule) — NE PAS upgrade / NE PAS purge
grep -E '^(GIT_SHA|SENTRY_RELEASE|DOCKER_TAG|BACKEND_IMAGE_REF)=' .env.production
docker compose -f docker-compose.production.yml images
cat /srv/atmr/releases/previous-release.json 2>/dev/null || true
```

### 1.4 Hors périmètre P0 GPS (embarqué si on ship la branche entière)

Commits `origin/main..HEAD` incluent notamment :

- CI coverage / ruff / GitGuardian test passwords
- Dashboard Genève / dispatch heuristics
- Staging observe harness
- PG-first / mission firewall / recovery FSM (programme tracking élargi)

```text
CONFIRMATION « P0-only »              = NON (branche ≠ cherry-pick P0)
OPTION A — ship branche complète      = accepter payload P0–P7 + CI (scope élargi)
OPTION B — release minimale P0        = cherry-pick / PR dédiée A+B+ledger+obs
                                         après commits WT
RECOMMANDATION                        = OPTION B pour limiter le blast radius
```

Tant que l’option n’est pas tranchée : **G0 reste JAUNE**, deploy **NO-GO**.

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
[ ] Commit / tag freeze P0-B + ledger CLIENT/SERVER + observability
[ ] Trancher OPTION A (branche entière) vs OPTION B (cherry-pick P0)
[ ] Snapshot live prod : GIT_SHA, BACKEND_IMAGE_REF, flask db current, flags TRACKING_*
[ ] Diff migrations prod current → tag
[ ] Capturer baseline monitoring T-30
[ ] Dry-run rollback (lecture previous-release.json + pull image N-1 sans up)
[ ] GO deploy séparé (document daté) seulement si G0–G5 VERT
```

---

## Implémentation

✅ **Implémenté** : dossier release-readiness unique (SHAs/ancres, dirty WT, migrations `capture_id` vs canary local, config/flags, ordre deploy, rollback, monitoring, gates G0–G5) ; **PROD DEPLOY = NO-GO**.  
**Reste à faire** : freeze commits → snapshot prod lecture seule → passer G0–G5 au vert → **attendre GO deploy explicite**.
