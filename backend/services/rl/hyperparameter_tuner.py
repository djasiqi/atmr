# pyright: reportMissingImports=false

# Constantes pour éviter les valeurs magiques
import json
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import Trial

from services.rl.dispatch_env import DispatchEnv
from services.rl.improved_dqn_agent import ImprovedDQNAgent

STEPS_PERCENT = 100
# 20 = 0  # Constante corrigée
EPISODE_ZERO = 0
BEST_VALUE_THRESHOLD = 544

"""Auto-Tuner pour hyperparamètres DQN avec Optuna.

Optimise automatiquement les hyperparamètres du DQN agent
pour maximiser la performance sur l'environnement de dispatch.

Auteur: ATMR Project - RL Team
Date: Octobre 2025
Module: Semaine 17 - Auto-Tuner
"""


class HyperparameterTuner:
    """Optimise hyperparamètres DQN avec Optuna.

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

    def __init__( # pyright: ignore[reportMissingSuperCall]
        self,
        n_trials: int = 50,
        n_training_episodes: int = 200,
        n_eval_episodes: int = 20,
        study_name: str = "dqn_optimization",
        storage: str | None = None
    ):  
        """Initialise le tuner.

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
        print("   Trials: {n_trials}")
        print("   Episodes training: {n_training_episodes}")
        print("   Episodes eval: {n_eval_episodes}")

    def objective(self, trial: Trial) -> float:
        """Fonction objective pour Optuna.

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
            num_drivers=config["num_drivers"],
            max_bookings=config["max_bookings"],
            simulation_hours=2  # 2 heures
        )

        # 3. Créer agent avec hyperparamètres suggérés
        agent = ImprovedDQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            epsilon_start=config["epsilon_start"],
            epsilon_end=config["epsilon_end"],
            epsilon_decay=config["epsilon_decay"],
            batch_size=config["batch_size"],
            buffer_size=config["buffer_size"],
            target_update_freq=config["target_update_freq"],
            use_double_dqn=config.get("use_double_dqn", True),
            use_prioritized_replay=config.get("use_prioritized_replay", True),
            alpha=config.get("alpha", 0.6),
            beta_start=config.get("beta_start", 0.4),
            beta_end=config.get("beta_end", 1),
            tau=config.get("tau", 0.005),
            use_n_step=config.get("use_n_step", True),
            n_step=config.get("n_step", 3),
            n_step_gamma=config.get("n_step_gamma", 0.99),
            use_dueling=config.get("use_dueling", True)
        )

        # 4. Entraîner
        episode_rewards = []
        for episode in range(self.n_training_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < STEPS_PERCENT:
                action = agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                agent.store_transition(
                    state, action, reward, next_state, done or truncated)

                if len(agent.memory) >= agent.batch_size:
                    agent.learn()

                state = next_state
                episode_reward += reward
                steps += 1

            # Décroissance epsilon automatique dans ImprovedDQNAgent

            episode_rewards.append(episode_reward)

            # Intermediate reporting pour pruning
            if episode % 2 == EPISODE_ZERO and episode > EPISODE_ZERO:
                intermediate_value = sum(episode_rewards[-20:]) / 20
                trial.report(intermediate_value, episode)

                # Pruning : arrêter si performance clairement mauvaise
                if trial.should_prune():
                    env.close()
                    raise optuna.TrialPruned

        # 5. Évaluer (mode exploitation pur)
        eval_rewards = []
        for _ in range(self.n_eval_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < STEPS_PERCENT:
                action = agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                state = next_state
                episode_reward += reward
                steps += 1

            eval_rewards.append(episode_reward)

        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)

        env.close()
        return avg_eval_reward

    def _suggest_hyperparameters(self, trial: Trial) -> dict[str, Any]:
        """Définit l'espace de recherche des hyperparamètres étendu.

        Grille étendue pour trouver le triplet gagnant (PER + N-step + Dueling).

        Args:
            trial: Trial Optuna

        Returns:
            Dictionnaire de configuration suggérée

        """
        return {
            # === PARAMÈTRES DE BASE ===
            # Apprentissage
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "gamma": trial.suggest_float("gamma", 0.90, 0.999),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),

            # Exploration
            "epsilon_start": trial.suggest_float("epsilon_start", 0.7, 1),
            "epsilon_end": trial.suggest_float("epsilon_end", 0.1, 0.1),
            "epsilon_decay": trial.suggest_float("epsilon_decay", 0.990, 0.999),

            # Replay buffer
            "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 200000, 500000]),

            # Target network
            "target_update_freq": trial.suggest_int("target_update_freq", 5, 50),

            # === AMÉLIORATIONS AVANCÉES ===
            # Double DQN
            "use_double_dqn": trial.suggest_categorical("use_double_dqn", [True, False]),

            # Prioritized Experience Replay (PER)
            "use_prioritized_replay": trial.suggest_categorical("use_prioritized_replay", [True, False]),
            # Priorité exponentielle
            "alpha": trial.suggest_float("alpha", 0.4, 0.8),
            # Importance sampling début
            "beta_start": trial.suggest_float("beta_start", 0.3, 0.6),
            # Importance sampling fin
            "beta_end": trial.suggest_float("beta_end", 0.8, 1),

            # N-step Learning
            "use_n_step": trial.suggest_categorical("use_n_step", [True, False]),
            # Nombre d'étapes pour N-step
            "n_step": trial.suggest_int("n_step", 2, 5),
            # Gamma pour N-step
            "n_step_gamma": trial.suggest_float("n_step_gamma", 0.95, 0.999),

            # Dueling DQN
            "use_dueling": trial.suggest_categorical("use_dueling", [True, False]),

            # Soft update
            "tau": trial.suggest_float("tau", 0.001, 0.1),  # Soft update rate

            # === ENVIRONNEMENT ===
            "num_drivers": trial.suggest_int("num_drivers", 5, 20),
            "max_bookings": trial.suggest_int("max_bookings", 10, 50)
        }

    def optimize(self) -> optuna.Study:
        """Lance l'optimisation Optuna.

        Returns:
            Study Optuna avec résultats

        """
        print("\n🚀 Démarrage optimisation Optuna...")
        print("   Study: {self.study_name}")
        print("   Trials: {self.n_trials}")

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
            direction="maximize",  # Maximiser le reward
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
        print(
            f"   Trials complétés: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(
            f"   Trials pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print("   Best trial: #{study.best_trial.number}")
        print("   Best value: {study.best_value")

        return study

    def save_best_params(
        self,
        study: optuna.Study,
        output_path: str = "data/rl/optimal_config.json"
    ) -> None:
        """Sauvegarde les meilleurs hyperparamètres.

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
        sorted_trials = sorted(
            completed_trials,
            key=lambda t: t.value,
            reverse=True)

        config = {
            "best_reward": float(best_value),
            "best_trial_number": best_trial.number,
            "best_params": best_params,
            "n_trials_total": len(study.trials),
            "n_trials_completed": len(completed_trials),
            "n_trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "optimization_history": [
                {
                    "trial": t.number,
                    "value": float(t.value) if t.value else None,
                    "params": t.params,
                    "state": t.state.name
                }
                for t in sorted_trials[:10]  # Top 10
            ]
        }

        with Path(output_path, "w", encoding="utf-8").open() as f:
            json.dump(config, f, indent=2)

        print("\n💾 Meilleurs hyperparamètres sauvegardés: {output_path}")
        print("\n📊 Top 3 configurations:")
        for i, trial in enumerate(sorted_trials[:3], 1):
            print(f"\n{i}. Trial #{trial.number} - Reward: {trial.value}")
            lr = trial.params.get("learning_rate")
            gamma = trial.params.get("gamma")
            batch = trial.params.get("batch_size")
            print(
                f"   Learning rate: {lr}" if lr else "   Learning rate: N/A")
            print(f"   Gamma: {gamma}" if gamma else "   Gamma: N/A")
            print(f"   Batch size: {batch}" if batch else "   Batch size: N/A")

        # Log automatique des métriques et comparaisons
        self._log_metrics_and_comparisons(study, sorted_trials)

    def _log_metrics_and_comparisons(
            self, study: optuna.Study, sorted_trials: list[optuna.trial.Trial]) -> None:
        """Log automatique des métriques et résultats de comparaison.

        Args:
            study: Study Optuna
            sorted_trials: Liste des trials triés par performance

        """
        from datetime import UTC, datetime

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        # 1. Sauvegarder métriques détaillées
        metrics_data = {
            "timestamp": timestamp,
            "study_name": self.study_name,
            "n_trials_total": len(study.trials),
            "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "best_value": float(study.best_value),
            "best_trial_number": study.best_trial.number,
            "optimization_duration": None,  # À calculer si disponible
            "trials_detailed": []
        }

        # Ajouter détails de chaque trial
        for trial in sorted_trials:
            trial_data = {
                "trial_number": trial.number,
                "value": float(trial.value) if trial.value else None,
                "state": trial.state.name,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
                "system_attrs": trial.system_attrs
            }
            metrics_data["trials_detailed"].append(trial_data)

        # Sauvegarder métriques
        metrics_path = f"data/rl/metrics_{timestamp}.json"
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

        with Path(metrics_path, "w", encoding="utf-8").open() as f:
            json.dump(metrics_data, f, indent=2)

        print("📊 Métriques détaillées sauvegardées: {metrics_path}")

        # 2. Sauvegarder résultats de comparaison
        comparison_data = {
            "timestamp": timestamp,
            "study_name": self.study_name,
            "comparison_summary": {
                "best_score": float(study.best_value),
                "target_score": 544.3,
                "improvement_over_target": float(study.best_value) - 544.3,
                "improvement_percentage": ((float(study.best_value) - 544.3) / 544.3) * 100,
                "triplet_gagnant_analysis": self._analyze_triplet_gagnant(sorted_trials)
            },
            "top_10trials": [
                {
                    "rank": i + 1,
                    "trial_number": trial.number,
                    "value": float(trial.value) if trial.value else None,
                    "params": trial.params,
                    "features_used": self._extract_features_used(trial.params)
                }
                for i, trial in enumerate(sorted_trials[:10])
            ],
            "feature_analysis": self._analyze_feature_importance(sorted_trials),
            "hyperparameter_ranges": self._get_hyperparameter_ranges()
        }

        # Sauvegarder comparaisons
        comparison_path = f"data/rl/comparison_results_{timestamp}.json"

        with Path(comparison_path, "w", encoding="utf-8").open() as f:
            json.dump(comparison_data, f, indent=2)

        print("📈 Résultats de comparaison sauvegardés: {comparison_path}")

        # 3. Afficher résumé
        print("\n🎯 RÉSUMÉ DE L'OPTIMISATION:")
        print("   Score cible: 544.3")
        print("   Meilleur score: {study.best_value")
        print(
            f"   Amélioration: {float(study.best_value) - 544.3:+.1f} ({comparison_data['comparison_summary']['improvement_percentage']:+.1f}%)")

        if study.best_value >= BEST_VALUE_THRESHOLD + 0.3:
            print("   ✅ OBJECTIF ATTEINT!")
        else:
            print("   ⚠️  Objectif non atteint, continuer l'optimisation")

    def _analyze_triplet_gagnant(self, sorted_trials: list[optuna.trial.Trial]) -> dict[str, Any]:
        """Analyse le triplet gagnant (PER + N-step + Dueling)."""
        triplet_stats = {
            "per_enabled": 0,
            "n_step_enabled": 0,
            "dueling_enabled": 0,
            "all_three_enabled": 0,
            "top_10per_enabled": 0,
            "top_10n_step_enabled": 0,
            "top_10dueling_enabled": 0,
            "top_10all_three_enabled": 0
        }

        # Analyser tous les trials
        for trial in sorted_trials:
            params = trial.params
            per_enabled = params.get("use_prioritized_replay", False)
            n_step_enabled = params.get("use_n_step", False)
            dueling_enabled = params.get("use_dueling", False)

            if per_enabled:
                triplet_stats["per_enabled"] += 1
            if n_step_enabled:
                triplet_stats["n_step_enabled"] += 1
            if dueling_enabled:
                triplet_stats["dueling_enabled"] += 1
            if per_enabled and n_step_enabled and dueling_enabled:
                triplet_stats["all_three_enabled"] += 1

        # Analyser top 10
        top_10 = sorted_trials[:10]
        for trial in top_10:
            params = trial.params
            per_enabled = params.get("use_prioritized_replay", False)
            n_step_enabled = params.get("use_n_step", False)
            dueling_enabled = params.get("use_dueling", False)

            if per_enabled:
                triplet_stats["top_10per_enabled"] += 1
            if n_step_enabled:
                triplet_stats["top_10n_step_enabled"] += 1
            if dueling_enabled:
                triplet_stats["top_10dueling_enabled"] += 1
            if per_enabled and n_step_enabled and dueling_enabled:
                triplet_stats["top_10all_three_enabled"] += 1

        return triplet_stats

    def _extract_features_used(self, params: dict[str, Any]) -> dict[str, Any]:
        """Extrait les features utilisées dans un trial."""
        return {
            "double_dqn": params.get("use_double_dqn", False),
            "prioritized_replay": params.get("use_prioritized_replay", False),
            "n_step": params.get("use_n_step", False),
            "dueling": params.get("use_dueling", False),
            "n_step_value": params.get("n_step", 1),
            "alpha": params.get("alpha", 0.6),
            "tau": params.get("tau", 0.005)
        }

    def _analyze_feature_importance(self, sorted_trials: list[optuna.trial.Trial]) -> dict[str, Any]:
        """Analyse l'importance des features."""
        feature_scores = {
            "double_dqn": {"enabled": [], "disabled": []},
            "prioritized_replay": {"enabled": [], "disabled": []},
            "n_step": {"enabled": [], "disabled": []},
            "dueling": {"enabled": [], "disabled": []}
        }

        for trial in sorted_trials:
            if trial.value is None:
                continue

            params = trial.params
            score = float(trial.value)

            # Double DQN
            if params.get("use_double_dqn", False):
                feature_scores["double_dqn"]["enabled"].append(score)
            else:
                feature_scores["double_dqn"]["disabled"].append(score)

            # PER
            if params.get("use_prioritized_replay", False):
                feature_scores["prioritized_replay"]["enabled"].append(score)
            else:
                feature_scores["prioritized_replay"]["disabled"].append(score)

            # N-step
            if params.get("use_n_step", False):
                feature_scores["n_step"]["enabled"].append(score)
            else:
                feature_scores["n_step"]["disabled"].append(score)

            # Dueling
            if params.get("use_dueling", False):
                feature_scores["dueling"]["enabled"].append(score)
            else:
                feature_scores["dueling"]["disabled"].append(score)

        # Calculer moyennes
        feature_importance = {}
        for feature, scores in feature_scores.items():
            enabled_avg = sum(
                scores["enabled"]) / len(scores["enabled"]) if scores["enabled"] else 0
            disabled_avg = sum(
                scores["disabled"]) / len(scores["disabled"]) if scores["disabled"] else 0

            feature_importance[feature] = {
                "enabled_avg": enabled_avg,
                "disabled_avg": disabled_avg,
                "improvement": enabled_avg - disabled_avg,
                "enabled_count": len(scores["enabled"]),
                "disabled_count": len(scores["disabled"])
            }

        return feature_importance

    def _get_hyperparameter_ranges(self) -> dict[str, Any]:
        """Retourne les plages d'hyperparamètres utilisées."""
        return {
            "learning_rate": {"min": 1e-5, "max": 1e-2, "log": True},
            "gamma": {"min": 0.90, "max": 0.999},
            "batch_size": {"choices": [32, 64, 128, 256]},
            "epsilon_start": {"min": 0.7, "max": 1},
            "epsilon_end": {"min": 0.1, "max": 0.1},
            "epsilon_decay": {"min": 0.990, "max": 0.999},
            "buffer_size": {"choices": [50000, 100000, 200000, 500000]},
            "target_update_freq": {"min": 5, "max": 50},
            "alpha": {"min": 0.4, "max": 0.8},
            "beta_start": {"min": 0.3, "max": 0.6},
            "beta_end": {"min": 0.8, "max": 1},
            "n_step": {"min": 2, "max": 5},
            "n_step_gamma": {"min": 0.95, "max": 0.999},
            "tau": {"min": 0.001, "max": 0.1},
            "num_drivers": {"min": 5, "max": 20},
            "max_bookings": {"min": 10, "max": 50}
        }
