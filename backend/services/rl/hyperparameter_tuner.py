# ruff: noqa: T201, DTZ005, W293
# pyright: reportMissingImports=false
"""
Auto-Tuner pour hyperparamètres DQN avec Optuna.

Optimise automatiquement les hyperparamètres du DQN agent
pour maximiser la performance sur l'environnement de dispatch.

Auteur: ATMR Project - RL Team
Date: Octobre 2025
Module: Semaine 17 - Auto-Tuner
"""
import json
from pathlib import Path

import optuna
from optuna.trial import Trial

from services.rl.dispatch_env import DispatchEnv
from services.rl.dqn_agent import DQNAgent


class HyperparameterTuner:
    """
    Optimise hyperparamètres DQN avec Optuna.
    
    Features:
        - Recherche automatique hyperparamètres
        - Pruning des trials non prometteurs
        - Sauvegarde meilleurs paramètres
        - Historique complet des essais
    
    Example:
        >>> tuner = HyperparameterTuner(n_trials=50)
        >>> study = tuner.optimize()
        >>> tuner.save_best_params(study)
    """

    def __init__(
        self,
        n_trials: int = 50,
        n_training_episodes: int = 200,
        n_eval_episodes: int = 20,
        study_name: str = "dqn_optimization",
        storage: str | None = None
    ):
        """
        Initialise le tuner.
        
        Args:
            n_trials: Nombre d'essais Optuna
            n_training_episodes: Episodes d'entraînement par trial
            n_eval_episodes: Episodes d'évaluation par trial
            study_name: Nom de l'étude Optuna
            storage: URL storage Optuna (None = en mémoire)
        """
        self.n_trials = n_trials
        self.n_training_episodes = n_training_episodes
        self.n_eval_episodes = n_eval_episodes
        self.study_name = study_name
        self.storage = storage

        print("🎯 Hyperparameter Tuner initialisé")
        print(f"   Trials: {n_trials}")
        print(f"   Episodes training: {n_training_episodes}")
        print(f"   Episodes eval: {n_eval_episodes}")

    def objective(self, trial: Trial) -> float:
        """
        Fonction objective pour Optuna.
        
        Retourne le reward moyen d'évaluation à maximiser.
        
        Args:
            trial: Trial Optuna
            
        Returns:
            Reward moyen d'évaluation (à maximiser)
        """
        # 1. Suggérer hyperparamètres
        config = self._suggest_hyperparameters(trial)

        # 2. Créer environnement
        env = DispatchEnv(
            num_drivers=config['num_drivers'],
            max_bookings=config['max_bookings'],
            simulation_hours=2  # 2 heures
        )

        # 3. Créer agent avec hyperparamètres suggérés
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
            episode_reward = 0.0
            done = False
            steps = 0

            while not done and steps < 100:
                action = agent.select_action(state, training=True)
                next_state, reward, done, truncated, _ = env.step(action)
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

            # Intermediate reporting pour pruning
            if episode % 20 == 0 and episode > 0:
                intermediate_value = sum(episode_rewards[-20:]) / 20
                trial.report(intermediate_value, episode)

                # Pruning : arrêter si performance clairement mauvaise
                if trial.should_prune():
                    env.close()
                    raise optuna.TrialPruned()

        # 5. Évaluer (mode exploitation pur)
        eval_rewards = []
        for _ in range(self.n_eval_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            done = False
            steps = 0

            while not done and steps < 100:
                action = agent.select_action(state, training=False)
                next_state, reward, done, truncated, _ = env.step(action)
                state = next_state
                episode_reward += reward
                steps += 1

            eval_rewards.append(episode_reward)

        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)

        env.close()
        return avg_eval_reward

    def _suggest_hyperparameters(self, trial: Trial) -> dict:
        """
        Définit l'espace de recherche des hyperparamètres.
        
        Args:
            trial: Trial Optuna
            
        Returns:
            Dictionnaire de configuration suggérée
        """
        # Architecture réseau
        hidden_1 = trial.suggest_categorical('hidden_size_1', [256, 512, 1024])
        hidden_2 = trial.suggest_categorical('hidden_size_2', [128, 256, 512])
        hidden_3 = trial.suggest_categorical('hidden_size_3', [64, 128, 256])

        # S'assurer que les couches décroissent
        # (pas de validation stricte ici, Optuna va tester)

        return {
            # Architecture
            'hidden_sizes': (hidden_1, hidden_2, hidden_3),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),

            # Apprentissage
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'gamma': trial.suggest_float('gamma', 0.90, 0.999),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),

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
        """
        Lance l'optimisation Optuna.
        
        Returns:
            Study Optuna avec résultats
        """
        print("\n🚀 Démarrage optimisation Optuna...")
        print(f"   Study: {self.study_name}")
        print(f"   Trials: {self.n_trials}")

        # Créer pruner pour arrêter trials non prometteurs
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Laisser 5 trials complets avant pruning
            n_warmup_steps=20    # Attendre 20 étapes avant pruning
        )

        # Créer sampler pour exploration efficace
        sampler = optuna.samplers.TPESampler(seed=42)

        # Créer étude
        study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',  # Maximiser le reward
            pruner=pruner,
            sampler=sampler,
            storage=self.storage,
            load_if_exists=True
        )

        # Optimiser
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            catch=(Exception,)  # Continuer même si un trial échoue
        )

        print("\n✅ Optimisation terminée !")
        print(f"   Trials complétés: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"   Trials pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"   Best trial: #{study.best_trial.number}")
        print(f"   Best value: {study.best_value:.1f}")

        return study

    def save_best_params(
        self,
        study: optuna.Study,
        output_path: str = "data/rl/optimal_config.json"
    ) -> None:
        """
        Sauvegarde les meilleurs hyperparamètres.
        
        Args:
            study: Study Optuna
            output_path: Chemin fichier de sortie
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        best_params = study.best_params
        best_value = study.best_value
        best_trial = study.best_trial

        # Tri des trials par valeur
        completed_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)

        config = {
            'best_reward': float(best_value),
            'best_trial_number': best_trial.number,
            'best_params': best_params,
            'n_trials_total': len(study.trials),
            'n_trials_completed': len(completed_trials),
            'n_trials_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'optimization_history': [
                {
                    'trial': t.number,
                    'value': float(t.value) if t.value else None,
                    'params': t.params,
                    'state': t.state.name
                }
                for t in sorted_trials[:10]  # Top 10
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        print(f"\n💾 Meilleurs hyperparamètres sauvegardés: {output_path}")
        print("\n📊 Top 3 configurations:")
        for i, trial in enumerate(sorted_trials[:3], 1):
            print(f"\n{i}. Trial #{trial.number} - Reward: {trial.value:.1f}")
            lr = trial.params.get('learning_rate')
            gamma = trial.params.get('gamma')
            batch = trial.params.get('batch_size')
            print(f"   Learning rate: {lr:.6f}" if lr else "   Learning rate: N/A")
            print(f"   Gamma: {gamma:.4f}" if gamma else "   Gamma: N/A")
            print(f"   Batch size: {batch}" if batch else "   Batch size: N/A")

