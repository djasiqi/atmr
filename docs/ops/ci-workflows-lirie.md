# CI/CD GitHub Actions — vue prod GPS (Lirie)

> Phase 1 allègement (2026-06) : workflows non essentiels passés en **manuel** ou **supprimés**.

## Toujours actifs (prod + qualité)

| Workflow | Déclenchement | Rôle |
|---|---|---|
| **Build & Deploy** | Manuel | Deploy app prod |
| **Deploy Kafka (production)** | Manuel | Stack Kafka prod |
| **Backend Tests** | Push/PR `backend/**` | Lint, pytest, migrations, pip-audit |
| **Mobile unified-app** | Push `main` + PR mobile | Lint, Jest, boundaries |
| **Repository integrity** | Push/PR | Anti-fuites IP prod / sentinelle |
| **Secret Pattern Scan** | Push/PR | Bloque clés Google commitées |
| **Security Scan** | Push deps + lundi 3h UTC | pip-audit + Semgrep |

## Manuels uniquement (phase 1)

| Workflow | Quand le lancer |
|---|---|
| **Build & Deploy RL Environ…** | MàJ stack RL |
| **Deploy Demo** | MàJ `www.lirie.ch/demo` |
| **E2E Demo** | Avant release démo commerciale |
| **Frontend Tests** | Retravail dashboard web |
| **Canon multi-surface** | PR multi-surfaces (discipline) |

## PR ciblées (si tu utilises des PR)

| Workflow | Chemins |
|---|---|
| **Architecture Review** | Tracking GPS (driver, ingest, carte live) |

## Supprimés (phase 1)

| Workflow | Raison |
|---|---|
| **CodeQL** (`.github/workflows/codeql.yml`) | Job `if: false` — mort ; alertes via Dependabot / Security tab |
| **Check Broad Exceptions** (`backend/.github/workflows/`) | Doublon de `Backend Tests` → `detect_broad_exceptions.py` |

## Docker Hub + runtime Node 24 (Actions)

- **Ne pas** poser `ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION=true` : ça réactiverait Node 20, déjà déprécié sur les runners.
- `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` reste en place (filet pour les actions JS encore en runtime Node 20).
- Actions Docker alignées **v4** (runtime Node 24 natif) : `setup-qemu-action`, `setup-buildx-action`.
- Login Docker Hub : `scripts/ci/dockerhub-login.sh` (CLI + retry). `docker/login-action@v3` n’a **aucun retry** — un `connection reset by peer` vers `auth.docker.io` (Cloudflare) faisait échouer tout le job.

✅ **Implémenté** : retry 5 tentatives / backoff 8s×n sur RST/EOF/timeout ; 401/403 fatals tout de suite. Workflows : `deploy.yml`, `build-and-deploy-rl.yml`, `deploy-kafka-p0.yml`.

## Hors Actions (GitHub natif)

- **Dependabot Updates** — PR hebdo deps (`.github/dependabot.yml`)
- **Dependency Graph** — vue alertes CVE npm/pip

## Phase 2 (optionnel, plus tard)

- Réactiver **Frontend Tests** en auto sur `frontend/**` quand DriverLiveMap prod stabilisé
- Réactiver **E2E Demo** en auto si `/demo` redevient critique
- Réintroduire **CodeQL** actif si besoin SAST au-delà de Dependabot + Semgrep
