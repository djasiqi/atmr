"""
Module `dispatch` - Consolidation des services de dispatch et optimisation

Ce module regroupe tous les services liés au dispatch :
- Dispatch unifié (optimisation combinatoire + ML/RL) → `unified_dispatch/`
  (déjà refactorisé B1)
- Agent dispatch (orchestration multi-agents)
- Planning service (planification des courses)
- Auto-réaffectation automatique
- Utilitaires dispatch

## Migration B2 (7 janvier 2025)

Ce module consolide les services dispatch fragmentés :
- `agent_dispatch/` → `dispatch/agent/`
- `planning_service.py` → `dispatch/planning.py`
- `auto_reassignment_service.py` → `dispatch/auto_reassignment.py`
- `dispatch_utils.py` → `dispatch/utils.py`
- `unified_dispatch/` → Reste en place (déjà refactorisé B1, fait partie du
  domaine dispatch)

## Structure

```
dispatch/
├── __init__.py              # Exports publics
├── agent/                   # Agent dispatch (multi-agents)
│   ├── __init__.py
│   ├── orchestrator.py      # Orchestration agents
│   ├── reporting.py         # Reporting agents
│   ├── safety_policy.py     # Politiques de sécurité
│   └── tools.py             # Outils agents
├── planning.py              # Service de planification
├── auto_reassignment.py     # Réaffectation automatique
└── utils.py                 # Utilitaires dispatch

(unified_dispatch/ reste dans services/unified_dispatch/ - déjà refactorisé B1)
```

## Usage

```python
# Imports recommandés (nouveaux)
from services.dispatch.agent.orchestrator import AgentDispatchOrchestrator
from services.dispatch.planning import PlanningService
from services.dispatch.auto_reassignment import AutoReassignmentService
from services.dispatch.utils import dispatch_helper_function

# Unified dispatch (B1) - chemins inchangés
from services.unified_dispatch.orchestration.dispatch_orchestrator import (
    DispatchOrchestrator,
)

# Imports de compatibilité (DEPRECATED, à migrer)
# from services.agent_dispatch.orchestrator import AgentDispatchOrchestrator
# from services.planning_service import PlanningService
# from services.auto_reassignment_service import AutoReassignmentService
```

## Documentation

- Architecture : `docs/DISPATCH_ARCHITECTURE.md`
- Migration : `PLAN_CONSOLIDATION_B2_SERVICES.md`
- Unified Dispatch : `services/unified_dispatch/docs/`

---

**Version :** 1.0.0 (B2 Refactoring)
**Date :** 7 janvier 2025
"""

# ========== Exports publics ==========

# Exports seront ajoutés au fur et à mesure de la migration
# from .planning import PlanningService
# from .auto_reassignment import AutoReassignmentService

__all__ = [
    # Les exports seront ajoutés après migration
]

__version__ = "1.0.0"
__refactoring__ = "B2 - Services Consolidation"
