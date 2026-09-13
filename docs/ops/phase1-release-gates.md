# Phase 1 — Gates de déploiement

## Workflows

| Workflow | Rôle |
|---|---|
| [`.github/workflows/phase1-gates.yml`](../../.github/workflows/phase1-gates.yml) | Suites critiques + agrégateur ; push / PR / dispatch ; **sur `release/gps-pilot-5-drivers-20260823` : dispatch automatique de Build & Deploy après PASS** |
| [`.github/workflows/security-gate.yml`](../../.github/workflows/security-gate.yml) | Gate réutilisable (PR + `workflow_call`) : SAST, SCA, secrets HEAD, Trivy fs/IaC, tenant, integrity |
| [`.github/workflows/deploy.yml`](../../.github/workflows/deploy.yml) | **Security Gate sur EXACT SHA** → build → Trivy image CRITICAL → manifeste → deploy |

### `booking-critical`

✅ **Implémenté** : le job installe `backend/requirements-dev.txt` (Flask + stack app), démarre Postgres/Redis, applique les migrations Alembic, puis exécute les tests de caractérisation booking + contrat HTTP. Ne plus installer seulement `pytest` — `tests/conftest.py` importe Flask au chargement.

### `deployment-critical` (compose dry-run)

✅ **Implémenté** : le dry-run `docker compose … config` génère des placeholders `:?required` à la volée via `openssl` (aucune valeur type secret versionnée dans le YAML). Les `env_file: .env.production` du compose prod sont `required: false`.

✅ **Implémenté** : Gitleaks scanné uniquement sur les surfaces déploiement (workflows, compose prod/kafka/monitoring, scripts release/deploy) avec [`.gitleaks.toml`](../../.gitleaks.toml) — évite les 100+ faux positifs d’un scan monorepo `--no-git`.

✅ **Implémenté** : le Security Gate ([`security-gate.yml`](../../.github/workflows/security-gate.yml)) exécute Gitleaks sur **l’arbre HEAD complet** (toujours `--no-git`, allowlist `.gitleaks.toml`). `deploy.yml` l’appelle en `workflow_call` sur le SHA déployé avant tout build.

## Artefacts

- `release-manifest.json` : `source_sha`, digests backend/ws, versions contrat
- `current-release.json` / `previous-release.json` sur le serveur
- Rollback = redéploiement du manifeste précédent (pas `compose down` seul)

## Secrets

- Déployés via `--env-file` (chmod 600), plus via argv positionnels
- Firebase : `0600`

## Images

Compose accepte `BACKEND_IMAGE_REF` / `WS_SERVICE_IMAGE_REF` (`repo@sha256:…`).

## Smoke

[`scripts/smoke_tests.sh`](../../scripts/smoke_tests.sh) : readiness + DB + migrations + schéma booking synthétique (champs internes ignorés).

## Exceptions Trivy

Métadonnées dans [`security/trivy-exceptions.yml`](../../security/trivy-exceptions.yml) ; ignores runtime dans [`.trivyignore`](../../.trivyignore).
