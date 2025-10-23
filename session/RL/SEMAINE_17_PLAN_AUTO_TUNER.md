# 📊 SEMAINE 17 : AUTO-TUNER & OPTIMISATION HYPERPARAMÈTRES

**Objectif :** Optimiser automatiquement les hyperparamètres du DQN avec Optuna  
**Durée estimée :** 2-3 heures  
**Gain attendu :** +20-50% amélioration performance

---

## 🎯 Objectifs

### Principaux

1. ✅ Installer et configurer Optuna
2. ✅ Créer script d'optimisation hyperparamètres
3. ✅ Définir espace de recherche intelligent
4. ✅ Implémenter fonction objective
5. ✅ Lancer optimisation (50-100 trials)
6. ✅ Analyser et sauvegarder meilleurs hyperparamètres
7. ✅ Réentraîner avec hyperparamètres optimaux

### Secondaires

- Visualisation résultats Optuna
- Comparaison avant/après optimisation
- Documentation complète

---

## 📦 Livrables

### 1. Infrastructure Optuna

```
backend/services/rl/
├── hyperparameter_tuner.py  (nouveau)
└── optimal_config.json       (généré)

backend/scripts/rl/
├── tune_hyperparameters.py   (nouveau)
└── compare_models.py         (nouveau)
```

### 2. Hyperparamètres à Optimiser

#### Architecture Réseau

- `hidden_sizes`: (256,128), (512,256,128), (1024,512,256)
- `dropout`: 0.0 à 0.5

#### Apprentissage

- `learning_rate`: 1e-5 à 1e-2 (log scale)
- `gamma` (discount factor): 0.90 à 0.999
- `batch_size`: 32, 64, 128, 256

#### Exploration

- `epsilon_start`: 0.8 à 1.0
- `epsilon_end`: 0.01 à 0.1
- `epsilon_decay`: 0.990 à 0.999

#### Replay Buffer

- `buffer_size`: 50k à 200k
- `min_replay_size`: 1k à 10k

#### Target Network

- `target_update_freq`: 5 à 20 episodes

---

## 🔧 Implémentation

### Étape 1 : Installation Optuna (~5 min)

**Fichier :** `backend/requirements-rl.txt`

```txt
optuna>=3.3.0
optuna-dashboard>=0.13.0  # Pour visualisation
```

**Commande :**

```bash
docker-compose exec api pip install optuna optuna-dashboard
```

---

### Étape 2 : Hyperparameter Tuner (~30 min)

**Fichier :** `backend/services/rl/hyperparameter_tuner.py`

```python
"""
Auto-Tuner pour hyperparamètres DQN avec Optuna.
"""
import json
from pathlib import Path
from typing import Dict, Any

import optuna
from optuna.trial import Trial

from services.rl.dispatch_env import DispatchEnv
from services.rl.dqn_agent import DQNAgent


class HyperparameterTuner:
    """Optimise hyperparamètres DQN avec Optuna."""

    def __init__(
        self,
        n_trials: int = 50,
        n_training_episodes: int = 200,
        n_eval_episodes: int = 20,
        study_name: str = "dqn_optimization"
    ):
        self.n_trials = n_trials
        self.n_training_episodes = n_training_episodes
        self.n_eval_episodes = n_eval_episodes
        self.study_name = study_name

    def objective(self, trial: Trial) -> float:
        """
        Fonction objective pour Optuna.
        Retourne le reward moyen à maximiser.
        """
        # 1. Suggérer hyperparamètres
        config = self._suggest_hyperparameters(trial)

        # 2. Créer environnement
        env = DispatchEnv(
            num_drivers=config['num_drivers'],
            max_bookings=config['max_bookings'],
            simulation_hours=2
        )

        # 3. Créer agent avec hyperparamètres
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            epsilon_start=config['epsilon_start'],
            epsilon_end=config['epsilon_end'],
            epsilon_decay=config['epsilon_decay'],
            batch_size=config['batch_size'],
            buffer_size=config['buffer_size'],
            target_update_freq=config['target_update_freq']
        )

        # 4. Entraîner
        episode_rewards = []
        for episode in range(self.n_training_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < 100:
                action = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                agent.store_transition(state, action, next_state, reward, done or truncated)

                if len(agent.memory) >= agent.batch_size:
                    agent.train_step()

                state = next_state
                episode_reward += reward
                steps += 1

            agent.decay_epsilon()
            if episode % agent.target_update_freq == 0:
                agent.update_target_network()

            episode_rewards.append(episode_reward)

            # Pruning : arrêter si performance mauvaise
            if episode % 20 == 0 and episode > 0:
                intermediate_value = sum(episode_rewards[-20:]) / 20
                trial.report(intermediate_value, episode)

                if trial.should_prune():
                    raise optuna.TrialPruned()

        # 5. Évaluer
        eval_rewards = []
        for _ in range(self.n_eval_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < 100:
                action = agent.select_action(state, training=False)
                next_state, reward, done, truncated, info = env.step(action)
                state = next_state
                episode_reward += reward
                steps += 1

            eval_rewards.append(episode_reward)

        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)

        env.close()
        return avg_eval_reward

    def _suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """Définit l'espace de recherche des hyperparamètres."""
        return {
            # Architecture
            'hidden_size_1': trial.suggest_categorical('hidden_size_1', [256, 512, 1024]),
            'hidden_size_2': trial.suggest_categorical('hidden_size_2', [128, 256, 512]),
            'hidden_size_3': trial.suggest_categorical('hidden_size_3', [64, 128, 256]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),

            # Apprentissage
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'gamma': trial.suggest_float('gamma', 0.90, 0.999),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),

            # Exploration
            'epsilon_start': trial.suggest_float('epsilon_start', 0.8, 1.0),
            'epsilon_end': trial.suggest_float('epsilon_end', 0.01, 0.1),
            'epsilon_decay': trial.suggest_float('epsilon_decay', 0.990, 0.999),

            # Replay buffer
            'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000, 200000]),

            # Target network
            'target_update_freq': trial.suggest_int('target_update_freq', 5, 20),

            # Environnement
            'num_drivers': trial.suggest_int('num_drivers', 5, 15),
            'max_bookings': trial.suggest_int('max_bookings', 10, 30)
        }

    def optimize(self) -> optuna.Study:
        """Lance l'optimisation Optuna."""
        study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=20
            )
        )

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        return study

    def save_best_params(self, study: optuna.Study, output_path: str = "data/rl/optimal_config.json"):
        """Sauvegarde les meilleurs hyperparamètres."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        best_params = study.best_params
        best_value = study.best_value

        config = {
            'best_reward': float(best_value),
            'best_params': best_params,
            'n_trials': len(study.trials),
            'optimization_history': [
                {
                    'trial': i,
                    'value': trial.value,
                    'params': trial.params
                }
                for i, trial in enumerate(study.trials)
                if trial.value is not None
            ][:10]  # Top 10
        }

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ Meilleurs hyperparamètres sauvegardés : {output_path}")
        print(f"   Best reward : {best_value:.1f}")
        print(f"   Paramètres :")
        for key, value in best_params.items():
            print(f"     {key}: {value}")
```

---

### Étape 3 : Script d'Optimisation (~20 min)

**Fichier :** `backend/scripts/rl/tune_hyperparameters.py`

```python
#!/usr/bin/env python3
"""
Script pour optimiser les hyperparamètres DQN avec Optuna.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.rl.hyperparameter_tuner import HyperparameterTuner


def main():
    parser = argparse.ArgumentParser(description="Optimiser hyperparamètres DQN")
    parser.add_argument('--trials', type=int, default=50, help='Nombre de trials Optuna')
    parser.add_argument('--episodes', type=int, default=200, help='Episodes par trial')
    parser.add_argument('--eval-episodes', type=int, default=20, help='Episodes d\'évaluation')
    parser.add_argument('--study-name', type=str, default='dqn_optimization', help='Nom de l\'étude')
    parser.add_argument('--output', type=str, default='data/rl/optimal_config.json', help='Fichier de sortie')

    args = parser.parse_args()

    print("🎯 OPTIMISATION HYPERPARAMÈTRES DQN")
    print("=" * 50)
    print(f"Trials : {args.trials}")
    print(f"Episodes par trial : {args.episodes}")
    print(f"Episodes d'évaluation : {args.eval_episodes}")
    print("=" * 50)

    # Créer tuner
    tuner = HyperparameterTuner(
        n_trials=args.trials,
        n_training_episodes=args.episodes,
        n_eval_episodes=args.eval_episodes,
        study_name=args.study_name
    )

    # Lancer optimisation
    print("\n🚀 Démarrage optimisation...")
    study = tuner.optimize()

    # Sauvegarder résultats
    print("\n📊 Résultats :")
    print(f"   Best trial : {study.best_trial.number}")
    print(f"   Best reward : {study.best_value:.1f}")
    print(f"   Trials complétés : {len(study.trials)}")

    tuner.save_best_params(study, args.output)

    print("\n✅ Optimisation terminée avec succès !")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

### Étape 4 : Script de Comparaison (~20 min)

**Fichier :** `backend/scripts/rl/compare_models.py`

```python
#!/usr/bin/env python3
"""
Compare performance baseline vs optimisé.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.rl.dispatch_env import DispatchEnv
from services.rl.dqn_agent import DQNAgent


def evaluate_config(config: dict, episodes: int = 50) -> dict:
    """Évalue une configuration."""
    env = DispatchEnv(
        num_drivers=config.get('num_drivers', 10),
        max_bookings=config.get('max_bookings', 20),
        simulation_hours=2
    )

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=config.get('learning_rate', 0.001),
        gamma=config.get('gamma', 0.99),
        epsilon_start=config.get('epsilon_start', 1.0),
        epsilon_end=config.get('epsilon_end', 0.01),
        epsilon_decay=config.get('epsilon_decay', 0.995),
        batch_size=config.get('batch_size', 64),
        buffer_size=config.get('buffer_size', 100000),
        target_update_freq=config.get('target_update_freq', 10)
    )

    # Training rapide
    print(f"   Entraînement {episodes} épisodes...")
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_transition(state, action, next_state, reward, done or truncated)

            if len(agent.memory) >= agent.batch_size:
                agent.train_step()

            state = next_state
            steps += 1

        agent.decay_epsilon()
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

    # Évaluation
    print("   Évaluation...")
    rewards = []
    for _ in range(20):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < 100:
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
            steps += 1

        rewards.append(episode_reward)

    env.close()

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards)
    }


def main():
    parser = argparse.ArgumentParser(description="Comparer configs baseline vs optimisé")
    parser.add_argument('--optimal-config', type=str, default='data/rl/optimal_config.json')
    parser.add_argument('--episodes', type=int, default=200)

    args = parser.parse_args()

    print("📊 COMPARAISON BASELINE VS OPTIMISÉ")
    print("=" * 50)

    # Config baseline
    baseline_config = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'buffer_size': 100000,
        'target_update_freq': 10,
        'num_drivers': 10,
        'max_bookings': 20
    }

    print("\n1️⃣  Baseline")
    baseline_results = evaluate_config(baseline_config, args.episodes)
    print(f"   Reward moyen : {baseline_results['mean_reward']:.1f} ± {baseline_results['std_reward']:.1f}")

    # Config optimisée
    if Path(args.optimal_config).exists():
        print("\n2️⃣  Optimisé")
        with open(args.optimal_config) as f:
            optimal_data = json.load(f)
        optimal_config = optimal_data['best_params']
        optimal_results = evaluate_config(optimal_config, args.episodes)
        print(f"   Reward moyen : {optimal_results['mean_reward']:.1f} ± {optimal_results['std_reward']:.1f}")

        # Comparaison
        improvement = ((optimal_results['mean_reward'] - baseline_results['mean_reward']) /
                      abs(baseline_results['mean_reward'])) * 100

        print("\n📈 AMÉLIORATION")
        print(f"   Baseline : {baseline_results['mean_reward']:.1f}")
        print(f"   Optimisé : {optimal_results['mean_reward']:.1f}")
        print(f"   Gain     : {improvement:+.1f}%")
    else:
        print(f"\n❌ Fichier {args.optimal_config} non trouvé")

    print("\n✅ Comparaison terminée !")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## 🚀 Exécution

### 1. Installation

```bash
cd backend
docker-compose exec api pip install optuna optuna-dashboard
```

### 2. Lancement Optimisation (2-3h)

```bash
# Optimisation rapide (10 trials)
python scripts/rl/tune_hyperparameters.py --trials 10 --episodes 100

# Optimisation complète (50 trials)
python scripts/rl/tune_hyperparameters.py --trials 50 --episodes 200

# Optimisation intensive (100 trials)
python scripts/rl/tune_hyperparameters.py --trials 100 --episodes 300
```

### 3. Comparaison

```bash
python scripts/rl/compare_models.py --episodes 200
```

### 4. Visualisation (Optionnel)

```bash
# Lancer dashboard Optuna
optuna-dashboard data/rl/optuna_study.db
# Ouvrir http://localhost:8080
```

---

## 📊 Résultats Attendus

### Baseline (Actuel)

```
Reward moyen : -1890.8
Performance  : 100% (référence)
```

### Après Optimisation (Estimé)

```
Reward moyen : -1400 à -1500  (+20-30%)
Distance     : -8 à -10 km     (+8-12%)
Late pickups : -35 à -38%      (-3-5 pts)
```

### Gains Concrets

```
Pour 1000 dispatches/mois :
  → ~50-80 km économisés/jour
  → ~25-40 retards évités/jour
  → ~15-20% meilleure utilisation flotte
```

---

## ✅ Validation

### Tests à Créer

1. `tests/rl/test_hyperparameter_tuner.py` (10 tests)
2. Validation espace recherche
3. Test fonction objective
4. Test sauvegarde/chargement config

---

## 📚 Documentation

### Fichiers à Créer

1. `SEMAINE_17_COMPLETE.md` - Récapitulatif
2. `OPTIMAL_CONFIG_EXPLAINED.md` - Explication hyperparamètres
3. `OPTUNA_GUIDE.md` - Guide utilisation

---

## 🎯 Prochaines Étapes (Semaine 18)

Après optimisation :

1. **Feedback Loop** - Réentraînement continu
2. **A/B Testing** - Test en production
3. **Monitoring** - Dashboard métriques temps réel

---

**Prêt à commencer ? Voulez-vous :**

1. ✅ **Installer Optuna** et créer l'infrastructure
2. ⏭️ **Sauter** et utiliser config par défaut
3. 📝 **Voir** d'autres options d'optimisation
