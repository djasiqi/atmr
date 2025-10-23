# 🎯 ROADMAP REINFORCEMENT LEARNING - GUIDE COMPLET

**Période:** Semaines 13-19 (7 semaines)  
**Mise à jour:** 20 octobre 2025

---

## 📊 État Global du Projet RL

| Semaine   | Objectif                    | Statut      | Fichiers | Tests    |
| --------- | --------------------------- | ----------- | -------- | -------- |
| **13-14** | **POC & Environnement Gym** | ✅ **100%** | 8        | 23/23 ✅ |
| **15-16** | Agent DQN PyTorch           | ⏳ 0%       | 0        | 0/0      |
| **17**    | Auto-Tuner Optuna           | ⏳ 0%       | 0        | 0/0      |
| **18**    | Feedback Loop               | ⏳ 0%       | 0        | 0/0      |
| **19**    | Optimisations               | ⏳ 0%       | 0        | 0/0      |

---

## ✅ SEMAINE 13-14 : COMPLÉTÉE

### Résumé

- ✅ **Environnement Gym** : 620 lignes, 95.83% coverage
- ✅ **Tests** : 23 tests, 100% passants
- ✅ **Scripts** : Collecte données, test rapide
- ✅ **Documentation** : 3 fichiers MD complets

### Fichiers Créés

```
✅ backend/services/rl/dispatch_env.py
✅ backend/services/rl/__init__.py
✅ backend/services/rl/README.md
✅ backend/tests/rl/test_dispatch_env.py
✅ backend/scripts/rl/collect_historical_data.py
✅ backend/scripts/rl/test_env_quick.py
✅ backend/requirements-rl.txt
✅ session/RL/ (3 fichiers documentation)
```

### Commandes de Validation

```bash
# Test rapide
docker-compose exec api python scripts/rl/test_env_quick.py

# Tests complets
docker-compose exec api pytest tests/rl/test_dispatch_env.py -v
# ✅ Résultat: 23 passed in 3.87s

# Collecte données (si DB contient données)
docker-compose exec api python scripts/rl/collect_historical_data.py --days 90
```

---

## ⏳ SEMAINE 15-16 : AGENT DQN (À FAIRE)

### Objectif

Créer un agent DQN capable d'apprendre la politique optimale de dispatch.

### Plan d'Action

#### Jour 1-3 : Architecture DQN

**Fichier:** `backend/services/rl/dqn_agent.py` (~800 lignes)

```python
class QNetwork(nn.Module):
    """Réseau de neurones Q(s,a)."""
    def __init__(state_dim, action_dim):
        # Input → FC(512) → ReLU → FC(256) → ReLU → FC(128) → Output

class ReplayBuffer:
    """Buffer d'expérience (capacity=100k)."""
    def push(transition)
    def sample(batch_size)

class DQNAgent:
    """Agent DQN complet."""
    def __init__(state_dim, action_dim, lr=0.001, gamma=0.99)
    def select_action(state, epsilon)
    def train_step()
    def save/load(path)
```

#### Jour 4-7 : Training Loop

**Fichier:** `backend/scripts/rl/train_dqn.py` (~400 lignes)

```python
for episode in range(1000):
    state = env.reset()
    episode_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, next_state, reward, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_step()

        state = next_state
        episode_reward += reward

    # Decay epsilon
    agent.decay_epsilon()

    # Update target network tous les 10 episodes
    if episode % 10 == 0:
        agent.update_target_network()

    # Log à TensorBoard
    writer.add_scalar('Reward/Episode', episode_reward, episode)

    # Sauvegarder checkpoints
    if episode % 100 == 0:
        agent.save(f'models/dqn_ep{episode}.pth')
```

#### Jour 8-14 : Monitoring & Optimisation

**TensorBoard:**

```bash
docker-compose exec api tensorboard --logdir=data/rl/tensorboard
# Ouvrir http://localhost:6006
```

**Métriques à tracker:**

- Reward par épisode
- Loss (Huber/MSE)
- Epsilon (exploration)
- Q-values moyennes
- Taux de complétion
- Distance moyenne

### Livrables Semaine 15-16

- ✅ Agent DQN fonctionnel
- ✅ 1000 épisodes entraînés
- ✅ Modèle sauvegardé
- ✅ Courbes d'apprentissage
- ✅ Tests (20+ tests)
- ✅ Comparaison vs baseline

### KPIs Attendus

- Reward moyen: **+35/step** (vs -105 baseline)
- Taux complétion: **85%** (vs 10% baseline)
- Distance moyenne: **6.5 km**
- Temps entraînement: **6-12h** (GPU)

---

## ⏳ SEMAINE 17 : AUTO-TUNER (À FAIRE)

### Objectif

Optimiser automatiquement les hyperparamètres avec Optuna.

### Plan d'Action

**Fichier:** `backend/scripts/rl/auto_tune.py`

```python
import optuna

def objective(trial):
    # Hyperparamètres
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    gamma = trial.suggest_uniform('gamma', 0.95, 0.999)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    hidden_size = trial.suggest_categorical('hidden', [128, 256, 512])

    # Entraîner agent
    agent = DQNAgent(lr=lr, gamma=gamma, batch_size=batch_size, hidden=hidden_size)
    reward = train_agent(agent, episodes=100)

    return reward

# Optimisation
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
```

### Livrables

- Script auto-tuning
- Configuration optimale
- Rapport d'optimisation
- Gain: +10-15%

---

## ⏳ SEMAINE 18 : FEEDBACK LOOP (À FAIRE)

### Objectif

Apprentissage continu depuis la production.

### Architecture

```
Production Data → Celery Task (hourly) → Replay Buffer → Retrain → A/B Test → Deploy
```

### Fichiers

**1. Task Celery:** `backend/tasks/rl_feedback_task.py`

```python
@celery.task
def collect_production_experiences():
    """Collecte expériences de production."""
    recent = Assignment.query.filter(created_at >= 1h_ago).all()
    for a in recent:
        experience = (state, action, reward, next_state)
        redis.lpush('rl_buffer', pickle.dumps(experience))

@celery.task
def retrain_agent_daily():
    """Ré-entraîne avec nouvelles données."""
    agent.load('current_model.pth')
    new_data = redis.lrange('rl_buffer', 0, -1)
    agent.train_on_batch(new_data)
    agent.save('updated_model.pth')
```

**2. A/B Testing:** `backend/services/rl/ab_dispatcher.py`

```python
def dispatch_with_ab(booking):
    if random.random() < 0.5:
        return rl_agent.dispatch(booking)  # RL
    else:
        return heuristic.dispatch(booking)  # Baseline
```

### Livrables

- Pipeline feedback automatique
- Retraining quotidien
- A/B testing 50/50
- Dashboard monitoring

---

## ⏳ SEMAINE 19 : OPTIMISATIONS (À FAIRE)

### Objectif

Optimiser pour production (latence < 50ms).

### Techniques

**1. Quantification INT8**

```python
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)
# Gain: 4x plus rapide, 4x moins mémoire
```

**2. ONNX Runtime**

```python
torch.onnx.export(model, 'dqn.onnx')
session = ort.InferenceSession('dqn.onnx')
# Gain: 2-3x plus rapide
```

**3. Batch Inference**

```python
states = [get_state(b) for b in bookings]
actions = model(torch.tensor(states)).argmax(dim=1)
# Gain: 10x plus rapide que séquentiel
```

### Livrables

- Modèle optimisé (INT8 + ONNX)
- Service inférence rapide
- Benchmarks
- Déploiement GPU

---

## 📚 Documentation

- **Guide d'utilisation:** `session/RL/SEMAINE_13-14_GUIDE.md`
- **Récapitulatif:** `session/RL/SEMAINE_13-14_COMPLETE.md`
- **Validation:** `session/RL/VALIDATION_SEMAINE_13-14.md`
- **Roadmap globale:** `session/RL/README_ROADMAP_COMPLETE.md`

---

## 🎯 KPIs Finaux Attendus (Semaine 19)

| Métrique     | Baseline | RL Final | Amélioration |
| ------------ | -------- | -------- | ------------ |
| Reward/step  | -105     | +45      | **+143%**    |
| Complétion   | 10%      | 88%      | **+780%**    |
| Distance     | 12 km    | 6.5 km   | **-46%**     |
| Retards      | 40%      | 12%      | **-70%**     |
| Satisfaction | 3.8/5    | 4.6/5    | **+21%**     |
| Latence      | N/A      | 28ms     | ✅           |

---

## 🚀 Commandes Utiles

```bash
# Tests
docker-compose exec api pytest tests/rl/ -v

# Training (Semaine 15-16)
docker-compose exec api python scripts/rl/train_dqn.py --episodes 1000

# Auto-tune (Semaine 17)
docker-compose exec api python scripts/rl/auto_tune.py --trials 50

# Production (Semaine 18-19)
docker-compose exec api python scripts/rl/deploy_agent.py
```

---

_README - Module RL ATMR_  
_Version 0.1.0 - Semaine 13-14 Complète_
