"""
unified_dispatch - Système d'optimisation et orchestration du dispatch

⚠️ REFACTORING B1 EN COURS (7 janvier 2025)
Ce module a été réorganisé pour améliorer la maintenabilité et réduire la complexité cognitive.

## Nouvelle Structure

```
unified_dispatch/
├── core/           # Types, exceptions, config
├── data/           # Chargement et préparation données
├── optimization/   # Algorithmes OR-Tools, heuristiques
├── ml/             # Machine Learning et RL
├── orchestration/  # Coordination et orchestration
├── metrics/        # Métriques et monitoring
├── validation/     # Validation contraintes
├── shadow_mode/    # A/B testing
├── utils/          # Utilitaires
├── locking/        # Gestion des locks
└── docs/           # Documentation
```

## Migration

Pour les imports, voir `docs/MIGRATION_GUIDE.md`

## Compatibilité Ascendante

Les imports depuis la racine du module sont maintenus pour compatibilité,
mais sont DEPRECATED. Utilisez les nouveaux chemins explicites.

**Ancien (DEPRECATED) :**
```python
from services.unified_dispatch import data, solver
```

**Nouveau (RECOMMANDÉ) :**
```python
from services.unified_dispatch.data import loader
from services.unified_dispatch.optimization import solver
```

---

**Refactoring B1 - Phase 1 (Semaine 1)** : Structure créée
**Status :** 🔵 EN COURS
"""

# pyright: reportImportCycles=false
# Les imports ci-dessous créent des cycles détectés par basedpyright, mais sont nécessaires
# pour maintenir l'API publique du module. Les cycles seront résolus en Semaine 3.
from . import locking  # noqa: I001
from . import orchestration

# ========== Exports Publics (Nouvelle API) ==========

# Note: Ces exports seront complétés lors de la migration (Semaine 2)

# Core
# from .core import types, exceptions, settings, problem_state, queue, engine

# Data
# from .data import loader, clustering, warm_start

# Optimization
# from .optimization import solver, heuristics, pareto_front, score_fusion

# ML
# from .ml import predictor, delay_predictor, rl_optimizer, ab_tracking, ab_router

# Metrics
# from .metrics import dispatch, prometheus, performance, errors, osrm_cache, slo

# Shadow Mode
# from .shadow_mode import orchestrator, manager

# Utils
# from .utils import transactions, realtime, suggestions, autonomous

# Validation
# from .validation import constraints, assignment

# ========== Imports de Compatibilité (DEPRECATED) ==========

# ⚠️ Ces imports maintiennent la compatibilité avec l'ancien code
# Ils seront supprimés après la migration complète de tous les imports
# dans le codebase (Semaine 3-4)

# Imports de compatibilité pour app.py et tasks
from .core import engine
from .core import queue

# TODO: Compléter lors de la migration (Semaine 2)
# from .core.types import *
# from .core.exceptions import *
# from .data.loader import *
# from .optimization.solver import *
# ... autres imports de compatibilité

__all__ = [
    # Modules
    "engine",  # Compatibilité tasks/dispatch_tasks.py
    "locking",
    "orchestration",
    "queue",  # Compatibilité app.py
    # Les autres exports seront ajoutés lors de la migration
]

__version__ = "2.0.0-refactor-b1"
__refactoring_status__ = "Phase 1 - Structure créée (Semaine 1)"
