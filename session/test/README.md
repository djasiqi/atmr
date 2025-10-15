# 📁 Audit Complet ATMR — Session Octobre 2025

## 📋 Vue d'ensemble

Ce dossier contient **tous les livrables de l'audit complet** de l'application ATMR (transport médical), réalisé le **15 octobre 2025**.

Les fichiers sont organisés pour faciliter leur consultation, application, et nettoyage ultérieur.

---

## 📂 Structure du Dossier

```
session/test/
├── README.md                   ← Ce fichier
├── REPORT.md                   ← 📊 Rapport exécutif complet
├── tests_plan.md               ← 🧪 Stratégie de tests (pytest, RTL, Cypress)
├── MIGRATIONS_NOTES.md         ← 🗄️ Détails migrations Alembic + rollback
├── DELETIONS.md                ← 🗑️ Fichiers/code morts à supprimer
├── ROADMAP.md                  ← 🗺️ Roadmap 4 semaines détaillée
│
├── patches/                    ← 🔧 Diffs unifiés (correctifs)
│   ├── backend/
│   │   ├── 001_osrm_timeout_retry.diff
│   │   ├── 002_osrm_cache_ttl.diff
│   │   ├── 003_pagination_bookings.diff
│   │   └── 004_solver_early_stop.diff
│   ├── frontend/
│   │   └── 001_unify_api_client.diff
│   ├── infra/
│   └── tests/
│
└── ci/                         ← ⚙️ Workflows GitHub Actions
    ├── backend-tests.yml
    ├── frontend-tests.yml
    └── docker-build.yml
```

---

## 🎯 Livrables Principaux

### 1. **REPORT.md** — Rapport Exécutif

**Contenu** :

- Executive summary (10 points forts/faiblesses)
- Top 20 findings (Impact × Effort, tagués Now/Next/Later)
- Tableau de dette technique
- Carte des dépendances (backend ↔ frontend ↔ services)
- Schéma ERD (Mermaid)
- Synthèse par composant (backend, frontend, mobile, infra)

**Usage** :

```bash
# Lecture
cat session/test/REPORT.md

# Export HTML (optionnel)
pandoc session/test/REPORT.md -o audit_report.html
```

---

### 2. **tests_plan.md** — Plan de Tests Complet

**Contenu** :

- Stratégie pytest backend (fixtures, mocks OSRM, coverage ≥70%)
- Tests frontend React Testing Library + Cypress E2E
- Tests mobile Jest + RNTL
- Mocks MSW, fakeredis, responses
- Intégration CI/CD

**Usage** :

```bash
# Backend
cd backend
pip install pytest pytest-flask pytest-cov fakeredis responses
pytest -v --cov=. --cov-report=html

# Frontend
cd frontend
npm test -- --coverage
npx cypress run
```

---

### 3. **MIGRATIONS_NOTES.md** — Notes Migrations Alembic

**Contenu** :

- Liste des 15 migrations chronologiques
- Risques (timezone, FK sans index, NULL constraints)
- Stratégie de rollback testée
- Graphe de dépendances (Mermaid)
- État des tables (rows estimées, indexes)

**Usage** :

```bash
# Vérifier migration actuelle
cd backend
alembic current

# Tester rollback
alembic downgrade -1
alembic upgrade head

# Voir SQL sans exécuter
alembic upgrade head --sql > migration.sql
```

---

### 4. **DELETIONS.md** — Fichiers Morts

**Contenu** :

- Fichiers backend non utilisés (manage.py, db.py)
- Assets frontend orphelins (avatars, dossiers vides)
- Dépendances npm/pip obsolètes
- Justification grep + diffs de retrait

**Usage** :

```bash
# Vérification manuelle
grep -r "manage.py" backend/ --exclude-dir=__pycache__

# Suppression sécurisée
git rm backend/manage.py
git commit -m "chore: remove deprecated manage.py"
```

---

### 5. **ROADMAP.md** — Roadmap 4 Semaines

**Contenu** :

- Planning détaillé jour par jour (20 jours-homme)
- Semaine 1 : CI/CD + tests backend (P0)
- Semaine 2 : Tests frontend + pagination API (P1)
- Semaine 3 : E2E Cypress + optimisations (P1/P2)
- Semaine 4 : Refacto + audit log (P2)
- Métriques de succès + dépendances critiques

**Usage** :

```bash
# Suivi progrès
# Cocher checklist en fin de chaque semaine
```

---

## 🔧 Patches Unifiés

### Backend

| Patch                          | Description                        | Effort     | Priorité |
| ------------------------------ | ---------------------------------- | ---------- | -------- |
| `001_osrm_timeout_retry.diff`  | Timeout configurable + retry 2x    | XS (1h)    | P0       |
| `002_osrm_cache_ttl.diff`      | Cache Redis TTL 3600s (1h)         | XS (30min) | P1       |
| `003_pagination_bookings.diff` | Pagination /bookings + Link header | S (6h)     | P1       |
| `004_solver_early_stop.diff`   | OR-Tools timeout 120s + early-stop | XS (1h)    | P2       |

### Frontend

| Patch                       | Description                       | Effort | Priorité |
| --------------------------- | --------------------------------- | ------ | -------- |
| `001_unify_api_client.diff` | Fusionner authService → apiClient | M (2j) | P2       |

### Application des Patches

```bash
# Backend
cd backend
patch -p1 < ../session/test/patches/backend/001_osrm_timeout_retry.diff

# Vérifier
git diff

# Commit si OK
git add .
git commit -m "feat: add OSRM timeout/retry configuration"

# Rollback si erreur
patch -R -p1 < ../session/test/patches/backend/001_osrm_timeout_retry.diff
```

---

## ⚙️ Workflows CI/CD

### backend-tests.yml

**Contenu** :

- Lint Ruff
- Tests pytest (postgres + redis services)
- Coverage Codecov
- pip-audit (CVE)
- Migrations check (upgrade + rollback)

### frontend-tests.yml

**Contenu** :

- Lint ESLint
- Tests Jest + coverage
- Build production
- npm audit (CVE)
- E2E Cypress (avec backend)

### docker-build.yml

**Contenu** :

- Build images backend/frontend (multi-arch)
- Trivy vulnerability scanner
- Healthcheck services
- Deploy staging/production

### Installation CI

```bash
# Créer dossier workflows
mkdir -p .github/workflows

# Copier workflows
cp session/test/ci/*.yml .github/workflows/

# Configurer secrets GitHub
# → Settings > Secrets > Actions
# CODECOV_TOKEN
# STAGING_HOST, STAGING_USER, STAGING_SSH_KEY
# SLACK_WEBHOOK_URL (optionnel)

# Push et vérifier
git add .github/workflows/
git commit -m "ci: add GitHub Actions workflows"
git push

# Vérifier dans GitHub Actions tab
```

---

## 📊 Métriques de Succès

| Métrique              | Cible      | État Actuel |
| --------------------- | ---------- | ----------- |
| **Coverage backend**  | ≥70%       | 🔴 0%       |
| **Coverage frontend** | ≥60%       | 🔴 5%       |
| **E2E scénarios**     | 5 passants | 🔴 0        |
| **CI workflows**      | 3 actifs   | 🔴 0        |
| **CVE critiques**     | 0          | 🟡 ?        |
| **Tests flaky**       | <5%        | -           |

---

## 🚀 Quick Start

### 1. Lire le rapport

```bash
less session/test/REPORT.md
# ou ouvrir dans éditeur Markdown
```

### 2. Installer CI/CD (P0)

```bash
cp session/test/ci/*.yml .github/workflows/
git add .github/workflows/
git commit -m "ci: add CI/CD workflows"
git push
```

### 3. Appliquer patches critiques (P0)

```bash
cd backend
patch -p1 < ../session/test/patches/backend/001_osrm_timeout_retry.diff
patch -p1 < ../session/test/patches/backend/002_osrm_cache_ttl.diff

# Tester
pytest -v

# Commit si OK
git add .
git commit -m "feat: OSRM timeout/retry + cache TTL"
```

### 4. Créer tests backend (P0)

```bash
cd backend
pip install pytest pytest-flask pytest-cov fakeredis responses

# Créer fichiers tests selon tests_plan.md
mkdir tests
touch tests/conftest.py tests/test_auth.py tests/test_bookings.py

# Exécuter
pytest -v --cov=. --cov-report=html
```

### 5. Suivre roadmap (4 semaines)

```bash
# Consulter ROADMAP.md
# Cocher checklist fin de chaque semaine
```

---

## 🧹 Nettoyage Post-Audit

**Après implémentation complète**, vous pouvez archiver ce dossier :

```bash
# Option 1: Archiver
mkdir -p archives/
tar -czf archives/audit-2025-10-15.tar.gz session/test/
git rm -r session/test/

# Option 2: Conserver en read-only
chmod -R 444 session/test/

# Option 3: Git branch dédiée
git checkout -b audit/2025-10-15
git add session/test/
git commit -m "docs: audit complet octobre 2025"
git push origin audit/2025-10-15
git checkout main
```

---

## 📞 Support & Questions

- **Issues** : Créer issue GitHub avec tag `[audit]`
- **Slack** : Canal `#tech-audit` (si configuré)
- **Email** : [À compléter]

---

## 📝 Checklist Avant Suppression

- [ ] Tous patches appliqués et testés
- [ ] CI/CD workflows actifs et green
- [ ] Tests backend ≥70% couverture
- [ ] Tests frontend ≥60% couverture
- [ ] E2E Cypress 5 scénarios passants
- [ ] Migrations testées (upgrade + rollback)
- [ ] Fichiers morts supprimés (DELETIONS.md)
- [ ] Documentation README mise à jour
- [ ] Roadmap semaine 1-2 complétée

---

**Date de création** : 15 octobre 2025  
**Révision** : 1.0  
**Auteur** : Audit technique ATMR  
**Validité** : 6 mois (révision avril 2026)
