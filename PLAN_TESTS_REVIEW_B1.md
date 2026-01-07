# 🧪 Plan Tests & Review - Refactoring B1

**Date :** 7 janvier 2025  
**Phase :** 3 & 4 (Tests + Review)  
**Status :** 🔵 **EN COURS**

---

## 🎯 Objectifs

1. **Valider** que tous les tests passent après migration P0+P1+P2
2. **Identifier** les régressions potentielles
3. **Benchmarker** les performances avant/après
4. **Mettre à jour** la documentation projet
5. **Préparer** le code review final

---

## 📋 Plan de Tests (4 niveaux)

### 1️⃣ Tests Unitaires `unified_dispatch/` (14 fichiers)

#### Tests Orchestration (9 fichiers)

```bash
tests/services/unified_dispatch/orchestration/
├── test_initializer.py                     # Validation company/date
├── test_problem_builder.py                 # Construction problème VRPTW
├── test_clustering_manager.py              # Clustering géographique
├── test_pipeline_executor.py               # Exécution pipeline solve
├── test_assignment_applier_wrapper.py      # Application assignments
├── test_dispatch_run_manager.py            # Gestion DispatchRun
├── test_metrics_finalizer.py               # Finalisation métriques
├── test_result_builder.py                  # Construction résultat
├── test_shadow_mode_manager.py             # Shadow mode manager
└── test_utils.py                           # Utilitaires
```

#### Tests Modules (4 fichiers)

```bash
tests/services/unified_dispatch/
├── test_apply.py                           # Application assignments (core)
├── test_apply_post_commit_notifications.py # Notifications post-commit
├── test_apply_skipped_logging.py           # Logs assignments skipped
└── test_settings.py                        # Configuration dispatch
```

**Commande :**

```bash
docker-compose exec api python -m pytest tests/services/unified_dispatch/ -v --tb=short
```

**Critères de succès :**

- ✅ 100% des tests passent (0 failed)
- ✅ Couverture > 80% sur modules migrés
- ✅ Aucune régression fonctionnelle

---

### 2️⃣ Tests d'Intégration Dispatch

#### Tests à exécuter

| Test                        | Objectif                                | Fichier                         |
| --------------------------- | --------------------------------------- | ------------------------------- |
| **Dispatch End-to-End**     | Dispatch complet (load → solve → apply) | `test_booking_dispatch_e2e.py`  |
| **Driver Management E2E**   | Gestion drivers + assignments           | `test_driver_management_e2e.py` |
| **Shadow Mode Integration** | Comparaison baseline vs RL              | `test_shadow_mode_integration`  |
| **ML Pipeline Integration** | Prédiction retards + RL agent           | `test_ml_pipeline.py`           |
| **Metrics Collection**      | Collecte Prometheus + SLO               | `test_metrics_collection.py`    |

**Commande :**

```bash
docker-compose exec api python -m pytest tests/e2e/ -v -m dispatch
```

**Critères de succès :**

- ✅ Dispatch complet fonctionne (company_id=1, day=today)
- ✅ Assignments créés en DB
- ✅ Événements publiés (DriverNewBookingEvent)
- ✅ Métriques Prometheus collectées
- ✅ Shadow mode compare correctement

---

### 3️⃣ Tests E2E Staging

#### Scénarios de test

**Scénario 1 : Dispatch Journalier Normal**

```python
# Conditions
- 50 bookings (9h-18h)
- 10 drivers disponibles
- Pas de contraintes exceptionnelles

# Résultat attendu
- Assignment rate > 90%
- Temps dispatch < 30s
- 0 conflits
```

**Scénario 2 : Dispatch avec Contraintes Complexes**

```python
# Conditions
- 100 bookings (6h-22h)
- 5 drivers (capacité limitée)
- Time windows serrés (15min)

# Résultat attendu
- Assignment rate > 70%
- Temps dispatch < 60s
- Unassigned bookings documentés
```

**Scénario 3 : Shadow Mode Actif**

```python
# Conditions
- Shadow mode activé (company_id=1)
- Baseline vs RL agent

# Résultat attendu
- Métriques comparatives collectées
- Aucune application réelle en shadow
- Redis state correctement géré
```

**Commande :**

```bash
docker-compose exec api python -m pytest tests/e2e/test_performance_e2e.py -v
```

**Critères de succès :**

- ✅ 3 scénarios passent
- ✅ Temps dispatch < SLO (60s p95)
- ✅ Logs structurés disponibles
- ✅ Alerting Prometheus fonctionne

---

### 4️⃣ Benchmark Performance (Avant/Après)

#### Métriques à comparer

| Métrique                           | Avant (v1.0) | Après (v2.0) | Delta |
| ---------------------------------- | ------------ | ------------ | ----- |
| **Import time `unified_dispatch`** | ?            | ?            | ?     |
| **Temps dispatch (p50)**           | ?            | ?            | ?     |
| **Temps dispatch (p95)**           | ?            | ?            | ?     |
| **Mémoire utilisée**               | ?            | ?            | ?     |
| **Temps compilation Python**       | ?            | ?            | ?     |

**Script benchmark :**

```python
# scripts/benchmark_dispatch.py
import time
from services.unified_dispatch.orchestration.dispatch_orchestrator import DispatchOrchestrator

def benchmark_dispatch():
    start = time.time()
    orchestrator = DispatchOrchestrator(company_id=1, day="2025-01-07")
    result = orchestrator.run()
    end = time.time()

    print(f"Durée: {end - start:.2f}s")
    print(f"Assignments: {result.get('applied_count', 0)}")
    print(f"Assignment rate: {result.get('assignment_rate', 0):.2%}")

if __name__ == "__main__":
    benchmark_dispatch()
```

**Commande :**

```bash
docker-compose exec api python scripts/benchmark_dispatch.py
```

**Critères de succès :**

- ✅ Temps dispatch <= v1.0 (pas de régression)
- ✅ Import time réduit (modules mieux organisés)
- ✅ Mémoire stable ou réduite

---

## 📝 Phase Review (Documentation)

### 1️⃣ Mettre à jour `DEPENDENCIES.md`

**Contenu à ajouter :**

```markdown
## Services Internes

### `unified_dispatch` (v2.0.0)

- **Description :** Système d'optimisation et orchestration du dispatch
- **Modules :**

  - `core` : Types, exceptions, configuration
  - `data` : Chargement et préparation données
  - `optimization` : Solver OR-Tools, heuristiques
  - `ml` : ML/RL (prédiction, agent DQN)
  - `metrics` : Prometheus, SLO
  - `validation` : Contraintes métier
  - `shadow_mode` : A/B testing
  - `utils` : Utilitaires transverses
  - `orchestration` : Coordination pipeline
  - `locking` : Verrous distribués (Redis)

- **Dépendances externes :**

  - OR-Tools (VRPTW)
  - PyTorch (RL agent)
  - XGBoost/LightGBM (ML)
  - Redis (locks, cache)
  - PostgreSQL (données)
  - OSRM (routing)

- **Architecture :** `docs/UNIFIED_DISPATCH_ARCHITECTURE.md`
- **Migration :** `docs/UNIFIED_DISPATCH_MIGRATION_GUIDE.md`
```

---

### 2️⃣ Mettre à jour `RUNBOOK.md`

**Sections à ajouter/modifier :**

#### Dispatch Troubleshooting (Section existante)

```markdown
### Problème : Dispatch échoue avec "ModuleNotFoundError"

**Symptôme :**
```

ModuleNotFoundError: No module named 'services.unified_dispatch.solver'

````

**Cause :** Imports non migrés vers nouvelle structure (v2.0)

**Solution :**
1. Consulter `docs/UNIFIED_DISPATCH_MIGRATION_GUIDE.md`
2. Remplacer imports anciens par nouveaux chemins
3. Exemple :
   ```python
   # ❌ AVANT
   from services.unified_dispatch.solver import solve_vrptw

   # ✅ APRÈS
   from services.unified_dispatch.optimization.solver import solve_vrptw
````

4. Relancer le dispatch

**Prévention :** Utiliser les nouveaux imports explicites (voir guide)

````

#### Nouveau: Architecture `unified_dispatch` v2.0
```markdown
### Modules `unified_dispatch`

Le système de dispatch est organisé en 10 modules thématiques :

1. **`core/`** - Types, exceptions, config fondamentale
2. **`data/`** - Chargement bookings/drivers
3. **`optimization/`** - Solver OR-Tools + heuristiques
4. **`ml/`** - Machine Learning & Reinforcement Learning
5. **`metrics/`** - Prometheus, SLO, performance
6. **`validation/`** - Contraintes métier
7. **`shadow_mode/`** - A/B testing production
8. **`utils/`** - Utilitaires (transactions, realtime)
9. **`orchestration/`** - Coordination pipeline
10. **`locking/`** - Verrous distribués (Redis)

**Documentation complète :** `docs/UNIFIED_DISPATCH_ARCHITECTURE.md`
````

---

### 3️⃣ Checklist Code Review

#### Structuration

- [ ] Tous les fichiers sont dans les bons modules (core, data, optimization...)
- [ ] Aucun fichier orphelin à la racine `unified_dispatch/`
- [ ] `__init__.py` bien documentés

#### Imports

- [ ] Tous les imports utilisent les nouveaux chemins explicites
- [ ] Aucun import DEPRECATED (`from services.unified_dispatch import solver`)
- [ ] Pas de cycles d'imports (vérification `basedpyright`)

#### Tests

- [ ] 100% des tests unitaires passent
- [ ] Tests d'intégration dispatch OK
- [ ] Tests E2E staging validés
- [ ] Benchmark performance acceptable

#### Documentation

- [ ] `docs/UNIFIED_DISPATCH_ARCHITECTURE.md` créé ✅
- [ ] `docs/UNIFIED_DISPATCH_MIGRATION_GUIDE.md` créé ✅
- [ ] `DEPENDENCIES.md` mis à jour
- [ ] `RUNBOOK.md` mis à jour
- [ ] `REFACTORING_B1_SUIVI.md` à jour ✅

#### Git

- [ ] Historique Git propre (52 commits tracés)
- [ ] Messages de commit clairs (`[B1-P0]`, `[B1-FIX]`, etc.)
- [ ] Aucun fichier temporaire committé

#### Qualité Code

- [ ] Aucune erreur linter (Ruff, basedpyright)
- [ ] Docstrings à jour
- [ ] Type hints corrects
- [ ] Couverture tests > 80%

---

## 🚀 Plan d'Exécution (Ordre recommandé)

### Jour 1 - Tests Unitaires

1. ✅ Lancer tests `unified_dispatch/` (14 fichiers)
2. ✅ Corriger erreurs si présentes
3. ✅ Vérifier couverture

### Jour 2 - Tests Intégration & E2E

1. ✅ Tests intégration dispatch
2. ✅ Tests E2E staging (3 scénarios)
3. ✅ Benchmark performance
4. ✅ Analyser métriques

### Jour 3 - Documentation

1. ✅ Mettre à jour `DEPENDENCIES.md`
2. ✅ Mettre à jour `RUNBOOK.md`
3. ✅ Finaliser `REFACTORING_B1_SUIVI.md`
4. ✅ Créer rapport final

### Jour 4 - Code Review

1. ✅ Self-review checklist
2. ✅ Review équipe (2+ personnes)
3. ✅ Adresser feedback
4. ✅ Merge sur `main`

---

## 📊 Critères de Validation Finale

| Critère                | Status | Notes |
| ---------------------- | ------ | ----- |
| Tests unitaires 100%   | 🔲     |       |
| Tests intégration OK   | 🔲     |       |
| Tests E2E validés      | 🔲     |       |
| Benchmark ≤ v1.0       | 🔲     |       |
| Documentation complète | 🟡     | 2/4   |
| Aucune erreur linter   | ✅     |       |
| Historique Git propre  | ✅     |       |
| Code review approuvé   | 🔲     |       |

**Légende :** ✅ OK | 🟡 En cours | 🔲 À faire | ❌ Bloquant

---

## 📞 Contacts & Support

- **Responsable refactoring :** [Nom]
- **Slack :** `#dispatch-refactoring`
- **Documentation :** `docs/UNIFIED_DISPATCH_*`

---

**Date de dernière mise à jour :** 7 janvier 2025  
**Version :** 1.0.0
