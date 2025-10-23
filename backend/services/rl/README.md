# 🧠 Module Reinforcement Learning - ATMR Dispatch

**Version:** 0.1.0  
**Statut:** Semaine 13-14 ✅ COMPLÈTE | Semaine 15-16 ⏳ PROCHAINE

---

## 🎯 Objectif

Créer un système de dispatch autonome qui **apprend** de l'expérience en utilisant le Reinforcement Learning (Deep Q-Network).

---

## 📁 Structure

```
services/rl/
├── __init__.py              # Module RL
├── dispatch_env.py          # ✅ Environnement Gym (Semaine 13-14)
├── dqn_agent.py             # ⏳ Agent DQN (Semaine 15-16)
├── utils.py                 # ⏳ Utilitaires RL
└── README.md                # Ce fichier
```

---

## 🚀 Quick Start

### 1. Tester l'Environnement

```bash
# Test rapide
docker-compose exec api python scripts/rl/test_env_quick.py

# Tests unitaires
docker-compose exec api pytest tests/rl/test_dispatch_env.py -v
```

### 2. Utiliser dans du Code

```python
from services.rl.dispatch_env import DispatchEnv

# Créer l'environnement
env = DispatchEnv(
    num_drivers=10,
    max_bookings=20,
    simulation_hours=8
)

# Reset
obs, info = env.reset(seed=42)

# Épisode
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # Random (à remplacer par RL agent)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

print(f"Reward: {total_reward}")
```

### 3. Collecter Données Historiques

```bash
docker-compose exec api python scripts/rl/collect_historical_data.py --days 90
```

---

## 📊 État Actuel

### ✅ Semaine 13-14 (COMPLÈTE)

- Environnement Gym fonctionnel
- 23 tests unitaires (100% pass)
- 95.83% de couverture
- Collecte de données opérationnelle

### ⏳ Semaine 15-16 (PROCHAINE)

- Agent DQN avec PyTorch
- Training sur 1000 épisodes
- TensorBoard monitoring
- Sauvegarde checkpoints

---

## 🎓 Concepts RL

### State Space

- **Dimension:** 122 (10 drivers, 20 bookings)
- **Contenu:** Positions, disponibilités, charges, temps, trafic
- **Type:** Box (continuous)

### Action Space

- **Dimension:** 201
- **Type:** Discrete
- **Actions:** 0=wait, 1-200=assignments

### Reward

- **Type:** Scalar (float)
- **Range:** -200 à +150 par step
- **Objectifs:** Temps, distance, satisfaction, équité

---

## 📚 Documentation

- `session/RL/SEMAINE_13-14_GUIDE.md` - Guide complet
- `session/RL/SEMAINE_13-14_COMPLETE.md` - Récapitulatif
- `session/RL/VALIDATION_SEMAINE_13-14.md` - Validation

---

## 🐛 Troubleshooting

### Import Error: "No module named 'gymnasium'"

```bash
docker-compose exec api pip install gymnasium
```

### Tests échouent

```bash
# Vérifier l'installation
docker-compose exec api python -c "import gymnasium; print(gymnasium.__version__)"

# Relancer les tests
docker-compose exec api pytest tests/rl/ -v
```

---

## 👥 Équipe

**ATMR Project - RL Team**  
Semaines 13-19 : Reinforcement Learning POC

---

_Dernière mise à jour: 20 octobre 2025_
