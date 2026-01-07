# 📦 Guide de Migration `unified_dispatch` v2.0

**Date :** Janvier 2025  
**Refactoring :** B1 (Réorganisation modulaire)  
**Status :** ✅ **Migration complétée - 38 fichiers migrés**

---

## 🎯 Objectif

Ce guide aide les développeurs à migrer leur code existant vers la nouvelle structure modulaire de `unified_dispatch` après le refactoring B1.

---

## 📋 Résumé des Changements

### Avant (v1.0 - Racine unified_dispatch/)
```python
from services.unified_dispatch import data, solver, validation
from services.unified_dispatch.apply import apply_assignments
from services.unified_dispatch.types import DispatchResult
```

### Après (v2.0 - Structure modulaire)
```python
from services.unified_dispatch.data import loader
from services.unified_dispatch.optimization import solver
from services.unified_dispatch.validation import constraints
from services.unified_dispatch.optimization.assignment_applier import apply_assignments
from services.unified_dispatch.core.types import DispatchResult
```

---

## 🗺️ Mapping Complet (Ancien → Nouveau)

### Core

| Ancien Chemin        | Nouveau Chemin          | Action   |
| -------------------- | ----------------------- | -------- |
| `types.py`           | `core/types.py`         | DÉPLACER |
| `exceptions.py`      | `core/exceptions.py`    | DÉPLACER |
| `settings.py`        | `core/settings.py`      | DÉPLACER |
| `problem_state.py`   | `core/problem_state.py` | DÉPLACER |
| `queue.py`           | `core/queue.py`         | DÉPLACER |
| `engine.py`          | `core/engine.py`        | DÉPLACER |

**Exemples de migration :**
```python
# ❌ AVANT
from services.unified_dispatch.types import DispatchResult
from services.unified_dispatch.exceptions import CompanyNotFoundError
from services.unified_dispatch.settings import Settings

# ✅ APRÈS
from services.unified_dispatch.core.types import DispatchResult
from services.unified_dispatch.core.exceptions import CompanyNotFoundError
from services.unified_dispatch.core.settings import Settings
```

---

### Data

| Ancien Chemin     | Nouveau Chemin        | Action          |
| ----------------- | --------------------- | --------------- |
| `data.py`         | `data/loader.py`      | RENOMMER        |
| `clustering.py`   | `data/clustering.py`  | DÉPLACER        |
| `warm_start.py`   | `data/warm_start.py`  | DÉPLACER        |

**Exemples de migration :**
```python
# ❌ AVANT
from services.unified_dispatch import data
problem = data.load_dispatch_data(company_id=1, day="2025-01-07")

# ✅ APRÈS
from services.unified_dispatch.data import loader
problem = loader.load_dispatch_data(company_id=1, day="2025-01-07")
```

```python
# ❌ AVANT
from services.unified_dispatch.clustering import GeographicClustering

# ✅ APRÈS
from services.unified_dispatch.data.clustering import GeographicClustering
```

---

### Optimization

| Ancien Chemin                | Nouveau Chemin                        | Action   |
| ---------------------------- | ------------------------------------- | -------- |
| `solver.py`                  | `optimization/solver.py`              | DÉPLACER |
| `apply.py`                   | `optimization/assignment_applier.py`  | RENOMMER |
| `heuristics.py`              | `optimization/heuristics.py`          | DÉPLACER |
| `pareto_front.py`            | `optimization/pareto_front.py`        | DÉPLACER |
| `score_fusion.py`            | `optimization/score_fusion.py`        | DÉPLACER |
| `warm_start_gain_tracker.py` | `optimization/warm_start_tracker.py`  | RENOMMER |
| `solving/`                   | `optimization/solving/`               | DÉPLACER |

**Exemples de migration :**
```python
# ❌ AVANT
from services.unified_dispatch.solver import solve_vrptw
from services.unified_dispatch.apply import apply_assignments

# ✅ APRÈS
from services.unified_dispatch.optimization.solver import solve_vrptw
from services.unified_dispatch.optimization.assignment_applier import apply_assignments
```

```python
# ❌ AVANT
from services.unified_dispatch.heuristics import haversine_minutes

# ✅ APRÈS
from services.unified_dispatch.optimization.heuristics import haversine_minutes
```

---

### Validation

| Ancien Chemin            | Nouveau Chemin               | Action   |
| ------------------------ | ---------------------------- | -------- |
| `validation.py`          | `validation/constraints.py`  | RENOMMER |
| `assignment_validator.py`| `validation/assignment.py`   | RENOMMER |
| `analysis/`              | `validation/analysis/`       | DÉPLACER |

**Exemples de migration :**
```python
# ❌ AVANT
from services.unified_dispatch.validation import validate_assignments
from services.unified_dispatch.assignment_validator import validate_assignment

# ✅ APRÈS
from services.unified_dispatch.validation.constraints import validate_assignments
from services.unified_dispatch.validation.assignment import validate_assignment
```

---

### Machine Learning & RL

| Ancien Chemin         | Nouveau Chemin             | Action   |
| --------------------- | -------------------------- | -------- |
| `rl_optimizer.py`     | `ml/rl_optimizer.py`       | DÉPLACER |
| `ml_predictor.py`     | `ml/predictor.py`          | RENOMMER |
| `delay_predictor.py`  | `ml/delay_predictor.py`    | DÉPLACER |
| `rl_kpi_monitor.py`   | `ml/rl_kpi_monitor.py`     | DÉPLACER |
| `rl_ab_tracking.py`   | `ml/ab_tracking.py`        | RENOMMER |
| `ab_router.py`        | `ml/ab_router.py`          | DÉPLACER |

**Exemples de migration :**
```python
# ❌ AVANT
from services.unified_dispatch.rl_optimizer import RLOptimizer
from services.unified_dispatch.delay_predictor import DelayPredictor

# ✅ APRÈS
from services.unified_dispatch.ml.rl_optimizer import RLOptimizer
from services.unified_dispatch.ml.delay_predictor import DelayPredictor
```

---

### Métriques

| Ancien Chemin                    | Nouveau Chemin             | Action   |
| -------------------------------- | -------------------------- | -------- |
| `dispatch_metrics.py`            | `metrics/dispatch.py`      | RENOMMER |
| `dispatch_prometheus_metrics.py` | `metrics/prometheus.py`    | RENOMMER |
| `slo.py`                         | `metrics/slo.py`           | DÉPLACER |
| `performance_metrics.py`         | `metrics/performance.py`   | RENOMMER |
| `error_metrics.py`               | `metrics/errors.py`        | RENOMMER |
| `osrm_cache_metrics.py`          | `metrics/osrm_cache.py`    | RENOMMER |

**Exemples de migration :**
```python
# ❌ AVANT
from services.unified_dispatch.dispatch_metrics import collect_dispatch_metrics
from services.unified_dispatch.dispatch_prometheus_metrics import record_assignment_rate

# ✅ APRÈS
from services.unified_dispatch.metrics.dispatch import collect_dispatch_metrics
from services.unified_dispatch.metrics.prometheus import record_assignment_rate
```

---

### Shadow Mode

| Ancien Chemin                        | Nouveau Chemin                | Action   |
| ------------------------------------ | ----------------------------- | -------- |
| `shadow_mode_orchestrator.py`       | `shadow_mode/orchestrator.py` | RENOMMER |
| `orchestration/shadow_mode_manager.py` | `shadow_mode/manager.py`    | RENOMMER |

**Exemples de migration :**
```python
# ❌ AVANT
from services.unified_dispatch.shadow_mode_orchestrator import ShadowModeOrchestrator

# ✅ APRÈS
from services.unified_dispatch.shadow_mode.orchestrator import ShadowModeOrchestrator
```

---

### Utils

| Ancien Chemin             | Nouveau Chemin           | Action   |
| ------------------------- | ------------------------ | -------- |
| `transaction_helpers.py`  | `utils/transactions.py`  | RENOMMER |
| `realtime_optimizer.py`   | `utils/realtime.py`      | RENOMMER |
| `reactive_suggestions.py` | `utils/suggestions.py`   | RENOMMER |
| `autonomous_manager.py`   | `utils/autonomous.py`    | RENOMMER |

**Exemples de migration :**
```python
# ❌ AVANT
from services.unified_dispatch.transaction_helpers import db_transaction_with_redis
from services.unified_dispatch.realtime_optimizer import RealtimeOptimizer

# ✅ APRÈS
from services.unified_dispatch.utils.transactions import db_transaction_with_redis
from services.unified_dispatch.utils.realtime import RealtimeOptimizer
```

---

## 🔧 Migration Automatique (Script)

Pour les projets avec beaucoup d'imports, utilisez un script de recherche/remplacement :

```python
#!/usr/bin/env python3
"""Script de migration automatique des imports unified_dispatch"""
import re
from pathlib import Path

REPLACEMENTS = {
    # Core
    r'from services\.unified_dispatch\.types import': 'from services.unified_dispatch.core.types import',
    r'from services\.unified_dispatch\.exceptions import': 'from services.unified_dispatch.core.exceptions import',
    r'from services\.unified_dispatch\.settings import': 'from services.unified_dispatch.core.settings import',
    
    # Data
    r'from services\.unified_dispatch import data': 'from services.unified_dispatch.data import loader as data',
    r'from services\.unified_dispatch\.data import': 'from services.unified_dispatch.data.loader import',
    
    # Optimization
    r'from services\.unified_dispatch\.solver import': 'from services.unified_dispatch.optimization.solver import',
    r'from services\.unified_dispatch\.apply import': 'from services.unified_dispatch.optimization.assignment_applier import',
    
    # Validation
    r'from services\.unified_dispatch\.validation import': 'from services.unified_dispatch.validation.constraints import',
    
    # ML
    r'from services\.unified_dispatch\.rl_optimizer import': 'from services.unified_dispatch.ml.rl_optimizer import',
    r'from services\.unified_dispatch\.delay_predictor import': 'from services.unified_dispatch.ml.delay_predictor import',
    
    # Metrics
    r'from services\.unified_dispatch\.dispatch_metrics import': 'from services.unified_dispatch.metrics.dispatch import',
    r'from services\.unified_dispatch\.dispatch_prometheus_metrics import': 'from services.unified_dispatch.metrics.prometheus import',
}

def fix_file(filepath: Path) -> bool:
    content = filepath.read_text(encoding='utf-8')
    original = content
    
    for pattern, replacement in REPLACEMENTS.items():
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        filepath.write_text(content, encoding='utf-8')
        return True
    return False

# Usage: python migrate-imports.py
```

---

## ✅ Checklist de Migration

### Pour chaque fichier Python dans votre codebase :

- [ ] Identifier les imports `from services.unified_dispatch.*`
- [ ] Remplacer par les nouveaux chemins (voir mapping ci-dessus)
- [ ] Vérifier la syntaxe avec `python -m py_compile fichier.py`
- [ ] Exécuter les tests du fichier
- [ ] Commit avec message clair (ex: `[MIGRATION] Mettre à jour imports unified_dispatch`)

### Pour les tests :

- [ ] Mettre à jour les mocks/patches avec les nouveaux chemins
- [ ] Ex: `@patch('services.unified_dispatch.solver.solve_vrptw')` → `@patch('services.unified_dispatch.optimization.solver.solve_vrptw')`

### Pour les adapters (`infrastructure/dispatch/`) :

- [ ] Vérifier que les adapters pointent vers les nouveaux chemins
- [ ] Tester l'intégration avec les bounded contexts (DDD)

---

## 🚨 Erreurs Courantes

### 1. ModuleNotFoundError

```python
# ❌ ERREUR
from services.unified_dispatch.solver import solve_vrptw
# ModuleNotFoundError: No module named 'services.unified_dispatch.solver'

# ✅ SOLUTION
from services.unified_dispatch.optimization.solver import solve_vrptw
```

### 2. ImportError avec anciens noms de fichiers

```python
# ❌ ERREUR
from services.unified_dispatch.apply import apply_assignments
# ModuleNotFoundError: No module named 'services.unified_dispatch.apply'

# ✅ SOLUTION
from services.unified_dispatch.optimization.assignment_applier import apply_assignments
```

### 3. Circular imports

**Problème :** Cycles détectés par `basedpyright` dans `__init__.py`.

**Solution :** Utiliser les imports explicites (nouveaux chemins) pour éviter les cycles via le `__init__.py` racine.

```python
# ❌ ÉVITER (peut créer des cycles)
from services.unified_dispatch import types, solver

# ✅ PRÉFÉRER (imports directs)
from services.unified_dispatch.core import types
from services.unified_dispatch.optimization import solver
```

---

## 📚 Ressources

- **Architecture complète** : `docs/UNIFIED_DISPATCH_ARCHITECTURE.md`
- **Mapping détaillé** : `backend/services/unified_dispatch/docs/MAPPING_REFACTORING_B1.md`
- **Audit technique** : `AUDIT_TECHNIQUE_COMPLET_2025.md`
- **Suivi refactoring** : `REFACTORING_B1_SUIVI.md`

---

## ❓ FAQ

### Q: Les anciens imports continuent de fonctionner ?

**R:** Oui, temporairement. Des exports de compatibilité sont maintenus dans `__init__.py`, mais ils sont **DEPRECATED** et seront supprimés dans v3.0. Migrez dès que possible.

### Q: Comment trouver tous les fichiers à migrer dans mon code ?

**R:** Utilisez `grep` :
```bash
grep -r "from services.unified_dispatch" backend/ --include="*.py"
```

### Q: Les tests passent-ils après migration ?

**R:** Oui, si tous les imports sont correctement mis à jour. Exécutez :
```bash
pytest tests/services/unified_dispatch/ -v
```

### Q: Y a-t-il des breaking changes fonctionnels ?

**R:** **Non**. Le refactoring est purement structurel. Aucune logique métier n'a été modifiée.

---

## 📞 Support

- **Issues GitHub** : [lien vers issues]
- **Slack** : `#dispatch-refactoring`
- **Contact** : équipe-dispatch@atmr.ch

---

**Date de dernière mise à jour :** 7 janvier 2025  
**Version du guide :** 1.0.0

