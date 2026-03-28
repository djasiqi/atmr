# Variables d’environnement — Admin Ops / Platform

## RL — chemins modèles (inférence vs entraînement)

| Variable | Description |
|----------|-------------|
| `RL_INFERENCE_MODEL_PATH` | Fichier `.pth` lu par le générateur de suggestions (API semi-auto). Défaut : `data/ml/dqn_agent_best_v33.pth`. |
| `RL_TRAINING_CHECKPOINT_PATH` | Checkpoint lu au **début** d’un `retrain_dqn_model_task` (Celery). Défaut : `data/rl/models/dqn_best.pth`. |
| `RL_TRAINING_CHECKPOINT_DIR` | Répertoire où le retrain **écrit** les checkpoints `dqn_retrain_<timestamp>.pth` — **pas** le fichier d’inférence. Défaut : `data/rl/models/training`. |
| `ENABLE_RL_POSTOPT` | Si `true`, autorise le post-opt `RLDispatchOptimizer` dans le pipeline (sinon log `RL_POSTOPT_SKIPPED`). Défaut : désactivé côté `FeatureFlags` ; peut être branché via overrides dispatch. |

Le retrain **n’écrit jamais** vers `RL_INFERENCE_MODEL_PATH` : promotion manuelle / CI si besoin.

**Validation staging (axe A)** : voir `docs/RL_STAGING_VALIDATION_CHECKLIST.md` (suggestions, `meta`, post-opt, retrain). Protocole S1–S4 : `docs/RL_STAGING_EXECUTION_PROTOCOL.md`. Smoke HTTP : `python scripts/smoke_rl_staging.py --help` (`--print-report-template` pour le tableau de compte-rendu).

**Shadow mode (L12)** : limiter la cohorte via configuration applicative / `ShadowModeManager` (pas de variable globale obligatoire ici — voir code `shadow_mode/`).

---

| Variable | Description |
|----------|-------------|
| `PLATFORM_API_URL_PROD` | URL de base de l’API production **sans** `/api/v1` (ex. `https://api.lirie.ch`). Vide = prod non monitorée (`monitored: false`). |
| `PLATFORM_API_URL_DEMO` | Idem pour la démo. Vide = demo non monitorée. |
| `PLATFORM_LINK_GRAFANA` | URL Grafana ; toujours renvoyée dans `links`, ou `null` si absent. |
| `PLATFORM_LINK_PROMETHEUS` | Idem Prometheus. |
| `PLATFORM_LINK_ALERTMANAGER` | Idem Alertmanager. |
| `PLATFORM_METADATA_GIT_COMMIT` | Commit Git affiché dans `metadata` de `GET /api/v1/platform/status` (priorité à `GIT_COMMIT_SHA` / `GIT_COMMIT` si non défini). |
| `PLATFORM_METADATA_APP_VERSION` | Version applicative dans `metadata` (sinon `APP_VERSION`). |
| `PLATFORM_STATUS_TIMEOUT_SECONDS` | Timeout HTTP par requête health (défaut `2.5`). |
| `PLATFORM_RUNTIME_REDIS_INFO_TIMEOUT_SECONDS` | Timeout socket pour la collecte **Redis INFO** dans `GET /api/v1/platform/runtime` (défaut `1.5`). Ne modifie pas les timeouts globaux `REDIS_SOCKET_*` du processus. |
| `PLATFORM_RUNTIME_CELERY_INSPECT_TIMEOUT_SECONDS` | Timeout **inspect** Celery (`ping` / `stats`) pour `GET /api/v1/platform/runtime` (défaut `1.5`). |
| `ADMIN_IP_WHITELIST_REQUIRED` | `false` (défaut) ou `true`. En **production** (`FLASK_CONFIG=production`), si `true`, `ADMIN_IP_WHITELIST` doit être **non vide** sinon le backend refuse de démarrer (fail-closed pour la console Admin Ops / `platform/status`). En développement, laisser `false` si la whitelist n’est pas configurée. |

**Sécurité — `GET /api/v1/platform/status`**

- `ADMIN_IP_WHITELIST` : liste d’IPs / CIDR autorisés pour l’endpoint (décorateur `ip_whitelist_required`).
- Si la whitelist est vide, le comportement reste **fail-open** au runtime (hors contrôle ci-dessous), d’où l’intérêt de `ADMIN_IP_WHITELIST_REQUIRED=true` en prod avec une liste renseignée.

---

## Exemple production (serveur Lirie, Traefik)

Les sous-domaines sont ceux définis dans `docker-compose.monitoring.yml` (labels Traefik). À ajouter dans `/srv/atmr/.env.production`, puis recréer le conteneur backend :

```env
PLATFORM_API_URL_PROD=https://api.lirie.ch
PLATFORM_LINK_GRAFANA=https://grafana.lirie.ch
PLATFORM_LINK_PROMETHEUS=https://prometheus.lirie.ch
PLATFORM_LINK_ALERTMANAGER=https://alertmanager.lirie.ch
```

`PLATFORM_API_URL_DEMO` : ne renseigner que si une API démo publique existe (sinon laisser vide).

Après modification : `docker compose -f docker-compose.production.yml up -d backend --force-recreate` depuis `/srv/atmr`.

---

## Déploiement GitHub Actions (`deploy.yml` + `scripts/deploy-production.sh`)

Le fichier `/srv/atmr/.env.production` est **régénéré** à chaque déploiement. Les variables Admin Ops y sont maintenant injectées automatiquement :

| Source | Rôle |
|--------|------|
| Secret **`GRAFANA_ROOT_URL`** | Alimente **`PLATFORM_LINK_GRAFANA`** (même URL que la console Grafana). |
| **Variables** repo (optionnelles) : `PLATFORM_API_URL_PROD`, `PLATFORM_LINK_PROMETHEUS`, `PLATFORM_LINK_ALERTMANAGER`, `PLATFORM_API_URL_DEMO` | Surchargent les URLs ; si vides, le script applique les défauts Lirie (`https://api.lirie.ch`, `https://prometheus.lirie.ch`, `https://alertmanager.lirie.ch`). |

**GitHub → Settings → Secrets and variables → Actions → Variables** (onglet *Variables*, pas *Secrets*) : ajouter seulement si tu dois t’écarter des défauts (autre domaine, pas de démo, etc.).

- `PLATFORM_API_URL_PROD` — ex. `https://api.lirie.ch`
- `PLATFORM_LINK_PROMETHEUS` — ex. `https://prometheus.lirie.ch`
- `PLATFORM_LINK_ALERTMANAGER` — ex. `https://alertmanager.lirie.ch`
- `PLATFORM_API_URL_DEMO` — laisser vide ou URL démo publique sans `/api/v1`

Aucun nouveau **secret** obligatoire : `GRAFANA_ROOT_URL` suffit pour le lien Grafana.
