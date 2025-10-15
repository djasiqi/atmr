# 📑 Index des Livrables — Audit ATMR

## 🎯 Navigation Rapide

### Documents Principaux

| Document                                     | Description                                           | Pages | Priorité |
| -------------------------------------------- | ----------------------------------------------------- | ----- | -------- |
| [REPORT.md](./REPORT.md)                     | 📊 Rapport exécutif complet (findings, ERD, synthèse) | ~80   | **P0**   |
| [ROADMAP.md](./ROADMAP.md)                   | 🗺️ Planning 4 semaines détaillé (jour par jour)       | ~40   | **P0**   |
| [tests_plan.md](./tests_plan.md)             | 🧪 Stratégie tests (pytest, RTL, Cypress)             | ~50   | P1       |
| [MIGRATIONS_NOTES.md](./MIGRATIONS_NOTES.md) | 🗄️ Migrations Alembic + rollback                      | ~35   | P1       |
| [DELETIONS.md](./DELETIONS.md)               | 🗑️ Fichiers/code morts à supprimer                    | ~30   | P2       |

---

## 🔧 Patches (Correctifs Unifiés)

### Backend

| Fichier                                                                        | Changements                        | Impact   | Effort     | Rollback |
| ------------------------------------------------------------------------------ | ---------------------------------- | -------- | ---------- | -------- |
| [001_osrm_timeout_retry.diff](./patches/backend/001_osrm_timeout_retry.diff)   | Timeout configurable + retry 2x    | 🟠 Élevé | XS (1h)    | ✅ Oui   |
| [002_osrm_cache_ttl.diff](./patches/backend/002_osrm_cache_ttl.diff)           | Cache Redis TTL 3600s              | 🟡 Moyen | XS (30min) | ✅ Oui   |
| [003_pagination_bookings.diff](./patches/backend/003_pagination_bookings.diff) | Pagination /bookings + Link header | 🟡 Moyen | S (6h)     | ✅ Oui   |
| [004_solver_early_stop.diff](./patches/backend/004_solver_early_stop.diff)     | OR-Tools timeout 120s + early-stop | 🟡 Moyen | XS (1h)    | ✅ Oui   |

### Frontend

| Fichier                                                                   | Changements                       | Impact   | Effort | Rollback |
| ------------------------------------------------------------------------- | --------------------------------- | -------- | ------ | -------- |
| [001_unify_api_client.diff](./patches/frontend/001_unify_api_client.diff) | Fusionner authService → apiClient | 🟡 Moyen | M (2j) | ✅ Oui   |

**Total patches** : 5  
**Effort total** : ~3 jours  
**Tous rollbackables** : ✅ Oui

---

## ⚙️ Workflows CI/CD

### GitHub Actions

| Workflow                                      | Déclencheurs            | Jobs                             | Durée estimée |
| --------------------------------------------- | ----------------------- | -------------------------------- | ------------- |
| [backend-tests.yml](./ci/backend-tests.yml)   | Push backend/, PR       | lint, test, security, migrations | ~5min         |
| [frontend-tests.yml](./ci/frontend-tests.yml) | Push frontend/, PR      | lint, test, build, security, e2e | ~8min         |
| [docker-build.yml](./ci/docker-build.yml)     | Push main/develop, tags | build, healthcheck, deploy       | ~12min        |

**Total workflows** : 3  
**Couverture** : Lint + Test + Build + Security + Deploy  
**Secrets requis** : 8 (CODECOV*TOKEN, STAGING*\_, PROD\_\_, SLACK\_\*)

---

## 📈 Statistiques Audit

### Findings

- **Total findings** : 20 (classés par Impact × Effort)
- **Priorité P0** : 5 (CI, tests, secrets, indexes, backups)
- **Priorité P1** : 9 (pagination, OSRM, logs PII, E2E)
- **Priorité P2** : 6 (refacto, audit log, monitoring)

### Dette Technique

| Catégorie           | Nombre | Effort Total | Risque Moyen |
| ------------------- | ------ | ------------ | ------------ |
| **Tests manquants** | 3      | M (9j)       | 🔴 Critique  |
| **Performance**     | 4      | S (3j)       | 🟠 Élevé     |
| **Sécurité**        | 3      | S (2j)       | 🔴 Critique  |
| **Refactorisation** | 2      | M (4j)       | 🟡 Moyen     |
| **Infrastructure**  | 4      | S (3j)       | 🟠 Élevé     |

**Total effort estimé** : ~20 jours-homme (4 semaines)

---

## 🗂️ Tables & Migrations

### Migrations Alembic

- **Total migrations** : 15
- **Dernière** : `f3a9c7b8d1e2` (indexes critiques, 2025-10-15)
- **Tables principales** : 30
- **Relations FK** : 45+
- **Indexes composites** : 20+

### Risques Migrations

| Migration                 | Risque                  | Mitigation          |
| ------------------------- | ----------------------- | ------------------- |
| `b15c01673cc4` (timezone) | 🟠 Conversion UTC→naive | Backup DB avant     |
| `f3a9c7b8d1e2` (indexes)  | 🟡 Lent si >100k rows   | Heures creuses      |
| Toutes autres             | 🟢 Faible               | Rollback disponible |

---

## 📚 Ressources Additionnelles

### Documentation Externe

- [pytest Documentation](https://docs.pytest.org/)
- [Cypress Best Practices](https://docs.cypress.io/guides/references/best-practices)
- [OR-Tools VRPTW Guide](https://developers.google.com/optimization/routing/vrp)
- [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)

### Outils Recommandés

```bash
# Backend
pip install ruff pytest pytest-cov pip-audit

# Frontend
npm install --save-dev eslint cypress @testing-library/react

# Infra
docker buildx
trivy
```

---

## 🎓 Glossaire

| Terme            | Définition                                         |
| ---------------- | -------------------------------------------------- |
| **P0/P1/P2**     | Priorités (0=urgent, 1=court terme, 2=moyen terme) |
| **XS/S/M/L**     | Effort (XS<2h, S<1j, M=2-5j, L>1sem)               |
| **RTL**          | React Testing Library                              |
| **MSW**          | Mock Service Worker (mocks API frontend)           |
| **VRPTW**        | Vehicle Routing Problem with Time Windows          |
| **Unified diff** | Format patch standard (git diff)                   |
| **Rollback**     | Annulation migration/changement                    |
| **Flaky test**   | Test instable (passe/échoue aléatoirement)         |

---

## ⏱️ Temps de Lecture Estimé

| Document            | Temps | Audience               |
| ------------------- | ----- | ---------------------- |
| README.md           | 5min  | Tous                   |
| INDEX.md            | 3min  | Tous                   |
| REPORT.md           | 30min | Management + Tech Lead |
| ROADMAP.md          | 20min | Tech Lead + DevOps     |
| tests_plan.md       | 25min | Développeurs           |
| MIGRATIONS_NOTES.md | 15min | Backend + DBA          |
| DELETIONS.md        | 10min | Tech Lead              |
| Patches (ensemble)  | 15min | Développeurs           |

**Total lecture complète** : ~2h

---

## 📞 Contact

- **Tech Lead** : [À compléter]
- **DevOps** : [À compléter]
- **Backend** : [À compléter]
- **Frontend** : [À complérer]

---

**Dernière mise à jour** : 15 octobre 2025, 21:00 UTC+1  
**Version** : 1.0  
**Format** : Markdown (compatible GitHub, GitLab, Notion)
