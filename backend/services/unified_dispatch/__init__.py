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

# Orchestration (déjà existant)
from . import orchestration  # noqa: F401

# Metrics
# from .metrics import dispatch, prometheus, performance, errors, osrm_cache, slo

# Validation
# from .validation import constraints, assignment

# Shadow Mode
# from .shadow_mode import orchestrator, manager

# Utils
# from .utils import transactions, realtime, suggestions, autonomous

# Locking (déjà existant)
from . import locking  # noqa: F401

# ========== Imports de Compatibilité (DEPRECATED) ==========

# ⚠️ Ces imports maintiennent la compatibilité avec l'ancien code
# Ils seront supprimés après la migration complète de tous les imports
# dans le codebase (Semaine 3-4)

# TODO: Compléter lors de la migration (Semaine 2)
# from .core.types import *  # noqa: F401, F403
# from .core.exceptions import *  # noqa: F401, F403
# from .data.loader import *  # noqa: F401, F403
# from .optimization.solver import *  # noqa: F401, F403
# ... autres imports de compatibilité

__all__ = [
    # Modules
    "orchestration",
    "locking",
    # Les autres exports seront ajoutés lors de la migration
]

__version__ = "2.0.0-refactor-b1"
__refactoring_status__ = "Phase 1 - Structure créée (Semaine 1)"
