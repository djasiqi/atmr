# 🏗️ Architecture `unified_dispatch`

**Version :** 2.0.0 (Refactoring B1 - Janvier 2025)  
**Status :** ✅ **Structure complète après refactoring**

---

## 📋 Vue d'Ensemble

Le module `unified_dispatch` est le cœur du système d'optimisation et d'orchestration du dispatch ATMR. Il combine :
- **Optimisation OR-Tools** (VRPTW - Vehicle Routing Problem with Time Windows)
- **Machine Learning & RL** (prédiction retards, agent DQN, A/B testing)
- **Orchestration** (pipeline, locks, shadow mode)
- **Métriques & Validation** (Prometheus, SLO, contraintes métier)

---

## 🗂️ Organisation des Modules (Post-Refactoring B1)

```
unified_dispatch/
├── core/              # Types, exceptions, configuration fondamentale
├── data/              # Chargement et préparation des données
├── optimization/      # Algorithmes d'optimisation (OR-Tools, heuristiques)
├── ml/                # Machine Learning & Reinforcement Learning
├── metrics/           # Métriques Prometheus, SLO, performance
├── validation/        # Contraintes métier, validation assignments
├── shadow_mode/       # A/B testing production vs nouvelle solution
├── utils/             # Utilitaires (transactions, realtime, suggestions)
├── orchestration/     # Coordination du pipeline dispatch
├── locking/           # Gestion des verrous distribués (Redis)
└── docs/              # Documentation et mapping de migration
```

---

## 📦 Modules Détaillés

### 1️⃣ `core/` - Définitions Fondamentales

**Responsabilité :** Types de données, exceptions, configuration globale.

| Fichier            | Description                                      |
| ------------------ | ------------------------------------------------ |
| `types.py`         | Types centraux (DispatchProblem, Assignment...)  |
| `exceptions.py`    | Exceptions personnalisées dispatch               |
| `settings.py`      | Configuration paramètres dispatch                |
| `problem_state.py` | État du problème VRPTW                           |
| `queue.py`         | Structures de files d'attente                    |
| `engine.py`        | Point d'entrée historique (compatibilité)       |

**Dépendances :**
- Aucune dépendance interne (module racine)
- `sqlalchemy`, `pydantic`, `redis`

---

### 2️⃣ `data/` - Chargement et Préparation

**Responsabilité :** Charger les données depuis la DB, les transformer pour OR-Tools.

| Fichier          | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `loader.py`      | Chargement bookings/drivers, création `DispatchProblem` |
| `clustering.py`  | Clustering géographique (DBSCAN)                    |
| `warm_start.py`  | Génération de warm-start pour OR-Tools              |

**Dépendances :**
- `core.types`, `core.settings`
- `models.Booking`, `models.Driver`
- `optimization.heuristics` (baseline, caps)
- `services.osrm_client` (matrices distance/durée)

---

### 3️⃣ `optimization/` - Algorithmes d'Optimisation

**Responsabilité :** Résolution du VRPTW avec OR-Tools, heuristiques, fusion scores.

| Fichier                  | Description                                   |
| ------------------------ | --------------------------------------------- |
| `solver.py`              | Solver OR-Tools principal (VRPTW)             |
| `assignment_applier.py`  | Application des assignments en DB + events    |
| `heuristics.py`          | Heuristiques (distance, baseline, caps)       |
| `pareto_front.py`        | Optimisation multi-objectif (Pareto)          |
| `score_fusion.py`        | Fusion des scores (RL, heuristiques, ML)      |
| `warm_start_tracker.py`  | Tracking gains warm-start                     |
| `solving/`               | Sous-module (contraintes, objectifs, routing) |

**Dépendances :**
- `core.types`, `core.settings`, `core.exceptions`
- `ortools` (VRPTW)
- `ml.delay_predictor`, `ml.rl_optimizer`
- `metrics.prometheus`, `data.loader`
- `validation.constraints`

---

### 4️⃣ `ml/` - Machine Learning & Reinforcement Learning

**Responsabilité :** Prédiction retards, agent DQN, A/B testing, monitoring RL.

| Fichier                | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `rl_optimizer.py`      | Agent DQN pour scoring assignments             |
| `predictor.py`         | Wrapper ML (XGBoost/LightGBM)                  |
| `delay_predictor.py`   | Prédiction probabilité retard                  |
| `rl_kpi_monitor.py`    | Monitoring KPIs agent RL                       |
| `ab_tracking.py`       | Tracking expériences A/B (RL vs baseline)      |
| `ab_router.py`         | Routeur A/B (50/50 RL vs heuristique)          |

**Dépendances :**
- `core.types`, `core.settings`
- `models.RLModel`, `models.DelayPredictionModel`
- `torch`, `xgboost`, `lightgbm`, `scikit-learn`
- `redis` (A/B tracking)

---

### 5️⃣ `metrics/` - Métriques et Monitoring

**Responsabilité :** Collecte métriques Prometheus, SLO, performance, erreurs.

| Fichier           | Description                                    |
| ----------------- | ---------------------------------------------- |
| `dispatch.py`     | Métriques qualité dispatch (assignment_rate...)     |
| `prometheus.py`   | Exposition Prometheus (gauges, counters)       |
| `slo.py`          | Service Level Objectives (latence, disponibilité) |
| `performance.py`  | Métriques perf (temps solver, warm-start...)   |
| `errors.py`       | Tracking erreurs et exceptions                 |
| `osrm_cache.py`   | Métriques cache OSRM (hit rate, taille)        |

**Dépendances :**
- `prometheus_client`
- `core.types`, `core.exceptions`
- `opentelemetry` (optionnel)

---

### 6️⃣ `validation/` - Contraintes Métier

**Responsabilité :** Valider les assignments (capacité, time windows, conflits).

| Fichier            | Description                                  |
| ------------------ | -------------------------------------------- |
| `constraints.py`   | Validation contraintes VRPTW                 |
| `assignment.py`    | Validation assignments (conflits, capacité)  |
| `analysis/`        | Analyse unassigned bookings                  |

**Dépendances :**
- `core.types`, `core.exceptions`
- `models.Booking`, `models.Driver`, `models.Assignment`
- `sqlalchemy`

---

### 7️⃣ `shadow_mode/` - A/B Testing Production

**Responsabilité :** Exécuter le dispatch en shadow mode (sans appliquer) pour comparer.

| Fichier           | Description                                    |
| ----------------- | ---------------------------------------------- |
| `orchestrator.py` | Orchestration shadow mode                      |
| `manager.py`      | Gestion état shadow mode par company           |

**Dépendances :**
- `core.types`
- `metrics.dispatch`, `optimization.solver`
- `redis` (état shadow mode)

---

### 8️⃣ `utils/` - Utilitaires Transverses

**Responsabilité :** Helpers pour transactions, realtime, suggestions autonomes.

| Fichier            | Description                                   |
| ------------------ | --------------------------------------------- |
| `transactions.py`  | Helpers transactions SQLAlchemy + Redis       |
| `realtime.py`      | Optimiseur temps réel (suggestions live)      |
| `suggestions.py`   | Suggestions réactives (réassignation)         |
| `autonomous.py`    | Manager dispatch autonome (décisions auto)    |

**Dépendances :**
- `sqlalchemy`, `redis`
- `core.types`, `optimization.solver`
- `models.*`

---

### 9️⃣ `orchestration/` - Coordination Pipeline

**Responsabilité :** Orchestrer l'ensemble du pipeline dispatch (init → solve → apply → métriques).

| Fichier                        | Description                                 |
| ------------------------------ | ------------------------------------------- |
| `dispatch_orchestrator.py`     | Orchestrateur principal                     |
| `initializer.py`               | Initialisation et validation                |
| `problem_builder.py`           | Construction du problème VRPTW              |
| `clustering_manager.py`        | Gestion clustering géographique             |
| `pipeline_executor.py`         | Exécution pipeline solve                    |
| `assignment_applier_wrapper.py`| Wrapper pour appliquer assignments          |
| `dispatch_run_manager.py`      | Gestion DispatchRun (logs, statut)         |
| `metrics_finalizer.py`         | Finalisation et collecte métriques          |
| `result_builder.py`            | Construction résultat final                 |

**Dépendances :**
- Tous les autres modules (`core`, `data`, `optimization`, `ml`, `metrics`, `validation`, `shadow_mode`, `utils`)
- `models.DispatchRun`, `models.DispatchRunLog`
- `shared.event_bus` (événements domaine)

---

### 🔟 `locking/` - Verrous Distribués

**Responsabilité :** Gérer les locks Redis pour éviter les dispatches concurrents.

| Fichier          | Description                            |
| ---------------- | -------------------------------------- |
| `redis_lock.py`  | Implémentation lock Redis (TTL, renew) |
| `manager.py`     | Manager locks par company + day        |

**Dépendances :**
- `redis`
- `core.exceptions`

---

## 🔄 Flux de Données Principal

```
Celery Task: run_dispatch_task
    ↓
orchestration.DispatchOrchestrator
    ↓
initializer: Validation company/date
    ↓
locking: Acquire Redis Lock
    ↓
problem_builder: Load data via data.loader
    ↓
clustering_manager: Geographic clustering
    ↓
pipeline_executor: Solve VRPTW
    ↓
optimization.solver + ml.rl_optimizer
    ↓
shadow_mode? → shadow_mode.orchestrator: Compare
    ↓
assignment_applier: Apply to DB
    ↓
metrics_finalizer: Prometheus + SLO
    ↓
locking: Release Lock
    ↓
result_builder: Return DispatchResult
```

---

## 📐 Principes de Design

### 1. **Séparation des Responsabilités**
- **core** : Stable, peu de dépendances
- **data** : Lecture seule (DB → modèle)
- **optimization** : Logique pure (pas de side-effects en DB)
- **orchestration** : Coordination (pas de logique métier)

### 2. **Dépendances Unidirectionnelles**
```
orchestration → optimization, ml, metrics, validation
optimization → core, data, ml
ml → core, data
data → core
core → (aucune dépendance interne)
```

### 3. **Testabilité**
- Chaque module est testable indépendamment
- Mocks pour services externes (OSRM, Redis, DB)
- Fixtures réutilisables (`tests/conftest.py`)

### 4. **Observabilité**
- Tous les modules exposent des métriques Prometheus
- Logs structurés (JSON) avec context (company_id, day)
- OpenTelemetry pour tracing distribué

---

## 🚀 Points d'Entrée

### Production
```python
from services.unified_dispatch.orchestration.dispatch_orchestrator import DispatchOrchestrator

orchestrator = DispatchOrchestrator(company_id=1, day="2025-01-07")
result = orchestrator.run()
```

### Tests
```python
from services.unified_dispatch.optimization.solver import solve_vrptw
from services.unified_dispatch.data.loader import load_dispatch_data

problem = load_dispatch_data(company_id=1, day="2025-01-07")
solution = solve_vrptw(problem)
```

---

## 📊 Métriques Clés

- `dispatch_duration_seconds` : Durée totale dispatch
- `dispatch_assignment_rate` : Taux d'assignments réussis
- `dispatch_unassigned_bookings` : Nombre bookings non assignés
- `dispatch_solver_time_seconds` : Temps solver OR-Tools
- `dispatch_ml_prediction_time_seconds` : Temps prédictions ML
- `dispatch_run_status` : Status (success, failure, partial)

---

## 🔒 Conventions de Code

### Imports
```python
# ✅ BON (nouveau, explicite)
from services.unified_dispatch.core.types import DispatchResult
from services.unified_dispatch.optimization.solver import solve_vrptw

# ❌ DEPRECATED (ancien, racine)
from services.unified_dispatch import solver
```

### Nommage
- **Modules** : snake_case (`assignment_applier.py`)
- **Classes** : PascalCase (`DispatchOrchestrator`)
- **Fonctions** : snake_case (`load_dispatch_data`)
- **Constantes** : UPPER_SNAKE_CASE (`MAX_SOLVER_TIME`)

### Docstrings
```python
def solve_vrptw(problem: DispatchProblem, settings: Settings) -> Solution:
    """
    Résout le problème VRPTW avec OR-Tools.
    
    Args:
        problem: Le problème de dispatch à résoudre
        settings: Configuration du solver
        
    Returns:
        Solution optimisée avec assignments
        
    Raises:
        SolverTimeoutError: Si le solver dépasse MAX_SOLVER_TIME
    """
    ...
```

---

## 📚 Documentation Complémentaire

- **Audit complet** : `/AUDIT_TECHNIQUE_COMPLET_2025.md`
- **Suivi refactoring** : `/REFACTORING_B1_SUIVI.md`
- **README principal** : `/README.md`

---

**Date de dernière mise à jour :** 7 janvier 2025  
**Auteur :** Équipe ATMR (Refactoring B1)  
**Version :** 2.0.0-refactor-b1

