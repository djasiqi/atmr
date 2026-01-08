"""
Module `ml` - Consolidation des services de Machine Learning et
Reinforcement Learning

Ce module regroupe tous les services liés au Machine Learning et au
Reinforcement Learning :
- Features engineering pour ML
- Monitoring des modèles ML
- Modèles de prédiction (demande, ETA, retards)
- Registry des modèles et métadonnées d'entraînement
- Agent RL (DQN) pour optimisation du dispatch
- Environnement de simulation RL
- Tuning d'hyperparamètres
- Reward shaping et suggestions

## Migration B2 (7 janvier 2025)

Ce module consolide 8+ services fragmentés :
- `ml_features.py` → `ml/features.py`
- `ml_monitoring_service.py` → `ml/monitoring.py`
- `ml/demand_prediction.py` → `ml/models/demand_prediction.py`
- `ml/eta_delay_model.py` → `ml/models/eta_delay.py`
- `ml/model_registry.py` → `ml/models/registry.py`
- `ml/training_metadata_schema.py` → `ml/models/training_metadata.py`
- `rl/` → `ml/rl/` (14 fichiers consolidés en 8)

## Structure

```
ml/
├── __init__.py              # Exports publics
├── features.py              # Feature engineering
├── monitoring.py            # Monitoring modèles ML
├── models/                  # Modèles de prédiction
│   ├── __init__.py
│   ├── demand_prediction.py # Prédiction demande
│   ├── eta_delay.py         # Modèle ETA et retards
│   ├── registry.py          # Registry des modèles
│   └── training_metadata.py # Métadonnées d'entraînement
└── rl/                      # Reinforcement Learning
    ├── __init__.py
    ├── agent.py             # Agent DQN (consolidé)
    ├── networks.py          # Réseaux de neurones
    ├── buffer.py            # Replay buffers
    ├── env.py               # Environnement dispatch
    ├── tuner.py             # Hyperparameter tuning
    ├── rewards.py           # Reward shaping
    ├── logger.py            # RL logging
    └── suggestions.py       # Suggestion generator
```

## Usage

```python
# Imports recommandés (nouveaux)
from services.ml.features import extract_ml_features
from services.ml.monitoring import MLMonitoringService
from services.ml.models.demand_prediction import DemandPredictionModel
from services.ml.models.eta_delay import ETADelayModel
from services.ml.models.registry import ModelRegistry
from services.ml.rl.agent import ImprovedDQNAgent
from services.ml.rl.env import DispatchEnv

# Imports de compatibilité (DEPRECATED, à migrer)
# from services.ml.features import extract_ml_features
# from services.ml.monitoring import MLMonitoringService
# from services.ml.models.demand_prediction import DemandPredictionModel
# from services.ml.rl.improved_dqn_agent import ImprovedDQNAgent
```

## Documentation

- Architecture : `docs/ML_ARCHITECTURE.md`
- Migration : `PLAN_CONSOLIDATION_B2_SERVICES.md`

---

**Version :** 1.0.0 (B2 Refactoring)
**Date :** 7 janvier 2025
"""

# ========== Exports publics ==========

# Exports seront ajoutés au fur et à mesure de la migration
# from .features import extract_ml_features
# from .monitoring import MLMonitoringService
# from .models.registry import ModelRegistry

__all__ = [
    # Les exports seront ajoutés après migration
]

__version__ = "1.0.0"
__refactoring__ = "B2 - Services Consolidation"
