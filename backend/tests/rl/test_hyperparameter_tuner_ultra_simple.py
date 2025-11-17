"""
Tests ultra-simplifiés pour hyperparameter_tuner.py - Couverture 95%+
"""

import json
from unittest.mock import MagicMock, Mock, patch

import optuna
import pytest

from services.rl.hyperparameter_tuner import HyperparameterTuner


class TestHyperparameterTuner:
    """Tests ultra-simplifiés pour HyperparameterTuner"""

    def test_init_default(self):
        """Test initialisation avec paramètres par défaut"""
        tuner = HyperparameterTuner()

        assert tuner.n_trials == 50
        assert tuner.n_training_episodes == 200
        assert tuner.n_eval_episodes == 20
        assert tuner.study_name == "dqn_optimization"
        assert tuner.storage is None

    def test_init_custom(self):
        """Test initialisation avec paramètres personnalisés"""
        tuner = HyperparameterTuner(
            n_trials=0.100,
            n_training_episodes=0.500,
            n_eval_episodes=50,
            study_name="custom_study",
            storage="sqlite:///test.db",
        )

        assert tuner.n_trials == 100
        assert tuner.n_training_episodes == 500
        assert tuner.n_eval_episodes == 50
        assert tuner.study_name == "custom_study"
        assert tuner.storage == "sqlite:///test.db"

    def test_suggest_hyperparameters(self):
        """Test _suggest_hyperparameters"""
        tuner = HyperparameterTuner()

        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 0.001
        mock_trial.suggest_categorical.return_value = 128
        mock_trial.suggest_int.return_value = 5

        config = tuner._suggest_hyperparameters(mock_trial)

        # Vérifier que les méthodes suggest sont appelées
        assert mock_trial.suggest_float.call_count >= 5
        assert mock_trial.suggest_categorical.call_count >= 2
        assert mock_trial.suggest_int.call_count >= 2

        # Vérifier que la configuration contient les clés attendues
        expected_keys = [
            "learning_rate",
            "gamma",
            "epsilon_start",
            "epsilon_end",
            "epsilon_decay",
            "batch_size",
            "buffer_size",
            "num_drivers",
            "max_bookings",
        ]
        for key in expected_keys:
            assert key in config

    def test_suggest_hyperparameters_with_values(self):
        """Test _suggest_hyperparameters avec valeurs spécifiques"""
        tuner = HyperparameterTuner()

        # Mock trial avec valeurs spécifiques - assez pour tous les appels
        mock_trial = Mock()
        mock_trial.suggest_float.side_effect = [0.001, 0.95, 0.9, 0.1, 0.995, 0.6, 0.4, 0.8, 0.005, 3, 0.99] * 10
        mock_trial.suggest_categorical.side_effect = [128, 100000, True, True, True, True] * 10
        mock_trial.suggest_int.side_effect = [5, 15, 3] * 10

        config = tuner._suggest_hyperparameters(mock_trial)

        # Vérifier les valeurs
        assert config["learning_rate"] == 0.001
        assert config["gamma"] == 0.95
        assert config["epsilon_start"] == 0.9
        assert config["epsilon_end"] == 0.1
        assert config["epsilon_decay"] == 0.995
        assert config["batch_size"] == 128
        assert config["buffer_size"] == 100000
        assert config["num_drivers"] == 5
        assert config["max_bookings"] == 15

    def test_objective_function(self):
        """Test objective function"""
        tuner = HyperparameterTuner(n_training_episodes=5, n_eval_episodes=2)

        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_float.side_effect = [0.001, 0.95, 0.9, 0.1, 0.995, 0.6, 0.4, 0.8, 0.005, 3, 0.99] * 10
        mock_trial.suggest_categorical.side_effect = [128, 100000, True, True, True, True] * 10
        mock_trial.suggest_int.side_effect = [3, 10, 3] * 10

        with (
            patch("services.rl.hyperparameter_tuner.DispatchEnv") as mock_env_class,
            patch("services.rl.hyperparameter_tuner.ImprovedDQNAgent") as mock_agent_class,
        ):
            # Mock environment
            mock_env = Mock()
            mock_env.observation_space.shape = [50]
            mock_env.action_space.n = 20
            mock_env.reset.return_value = (Mock(), {})
            mock_env.step.return_value = (Mock(), 10, False, False, {})
            mock_env_class.return_value = mock_env

            # Mock agent
            mock_agent = Mock()
            mock_agent.select_action.return_value = 0
            mock_agent_class.return_value = mock_agent

            # Exécuter objective
            reward = tuner.objective(mock_trial)

            # Vérifier que l'environnement et l'agent sont créés
            mock_env_class.assert_called_once()
            mock_agent_class.assert_called_once()

            # Vérifier que le reward est retourné
            assert isinstance(reward, float)

    def test_objective_function_with_pruning(self):
        """Test objective function avec pruning"""
        tuner = HyperparameterTuner(n_training_episodes=5, n_eval_episodes=2)

        # Mock trial avec pruning
        mock_trial = Mock()
        mock_trial.suggest_float.side_effect = [0.001, 0.95, 0.9, 0.1, 0.995, 0.6, 0.4, 0.8, 0.005, 3, 0.99] * 10
        mock_trial.suggest_categorical.side_effect = [128, 100000, True, True, True, True] * 10
        mock_trial.suggest_int.side_effect = [3, 10, 3] * 10
        mock_trial.should_prune.return_value = True

        with (
            patch("services.rl.hyperparameter_tuner.DispatchEnv") as mock_env_class,
            patch("services.rl.hyperparameter_tuner.ImprovedDQNAgent") as mock_agent_class,
        ):
            # Mock environment
            mock_env = Mock()
            mock_env.observation_space.shape = [50]
            mock_env.action_space.n = 20
            mock_env.reset.return_value = (Mock(), {})
            mock_env.step.return_value = (Mock(), 10, False, False, {})
            mock_env_class.return_value = mock_env

            # Mock agent
            mock_agent = Mock()
            mock_agent.select_action.return_value = 0
            mock_agent_class.return_value = mock_agent

            # Exécuter objective avec pruning
            with pytest.raises(optuna.TrialPruned):
                tuner.objective(mock_trial)

    def test_optimize(self):
        """Test optimize method"""
        tuner = HyperparameterTuner(n_trials=2)

        with patch("optuna.create_study") as mock_create_study:
            mock_study = Mock()
            mock_study.optimize.return_value = None
            mock_study.trials = [Mock() for _ in range(5)]
            mock_study.best_value = 100
            mock_study.best_trial = Mock()
            mock_study.best_trial.number = 1
            mock_create_study.return_value = mock_study

            study = tuner.optimize()

            # Vérifier que create_study est appelé
            mock_create_study.assert_called_once()

            # Vérifier que optimize est appelé
            mock_study.optimize.assert_called_once()

            # Vérifier que l'étude est retournée
            assert study == mock_study

    def test_optimize_with_storage(self):
        """Test optimize method avec storage"""
        tuner = HyperparameterTuner(n_trials=2, storage="sqlite:///test.db")

        with patch("optuna.create_study") as mock_create_study:
            mock_study = Mock()
            mock_study.optimize.return_value = None
            mock_study.trials = [Mock() for _ in range(5)]
            mock_study.best_value = 100
            mock_study.best_trial = Mock()
            mock_study.best_trial.number = 1
            mock_create_study.return_value = mock_study

            tuner.optimize()

            # Vérifier que create_study est appelé avec storage
            mock_create_study.assert_called_once()
            call_args = mock_create_study.call_args
            assert call_args[1]["storage"] == "sqlite:///test.db"
            assert call_args[1]["study_name"] == tuner.study_name

    def test_save_best_params(self):
        """Test save_best_params method"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.best_params = {
            "learning_rate": 0.001,
            "gamma": 0.95,
            "epsilon_start": 0.9,
            "epsilon_end": 0.1,
            "epsilon_decay": 0.995,
            "batch_size": 128,
            "buffer_size": 100000,
            "num_drivers": 5,
            "max_bookings": 15,
        }
        mock_study.best_value = 100
        mock_study.best_trial = Mock()
        mock_study.best_trial.number = 1
        mock_study.trials = [Mock() for _ in range(5)]

        # Mock trial states
        for i, trial in enumerate(mock_study.trials):
            trial.state = optuna.trial.TrialState.COMPLETE
            trial.value = 100 - i * 10
            trial.number = i
            trial.params = {"learning_rate": 0.001}

        with patch("pathlib.Path.mkdir") as mock_mkdir, patch("builtins.open", create=True) as mock_file:
            tuner.save_best_params(mock_study, "test_params.json")

            # Vérifier que le répertoire est créé
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

            # Vérifier que le fichier est ouvert en écriture
            mock_file.assert_called_once()

    def test_log_metrics_and_comparisons(self):
        """Test _log_metrics_and_comparisons method"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.best_value = 100
        mock_study.best_trial = Mock()
        mock_study.best_trial.number = 1
        mock_study.trials = [Mock() for _ in range(5)]

        # Mock trial states
        sorted_trials = []
        for i in range(5):
            trial = Mock()
            trial.number = i
            trial.value = 100 - i * 10
            trial.state = optuna.trial.TrialState.COMPLETE
            trial.params = {"learning_rate": 0.001}
            trial.user_attrs = {}
            trial.system_attrs = {}
            sorted_trials.append(trial)

        with patch("pathlib.Path.mkdir") as mock_mkdir, patch("builtins.open", create=True) as mock_file:
            tuner._log_metrics_and_comparisons(mock_study, sorted_trials)

            # Vérifier que le répertoire est créé
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

            # Vérifier que le fichier est ouvert en écriture
            mock_file.assert_called_once()

    def test_analyze_triplet_gagnant(self):
        """Test _analyze_triplet_gagnant method"""
        tuner = HyperparameterTuner()

        # Mock trials
        trials = []
        for i in range(3):
            trial = Mock()
            trial.params = {"learning_rate": 0.001 + i * 0.0001, "gamma": 0.95, "batch_size": 128}
            trials.append(trial)

        result = tuner._analyze_triplet_gagnant(trials)

        # Vérifier que l'analyse est retournée
        assert isinstance(result, dict)

    def test_extract_features_used(self):
        """Test _extract_features_used method"""
        tuner = HyperparameterTuner()

        params = {
            "use_double_dqn": True,
            "use_prioritized_replay": True,
            "use_n_step": True,
            "use_dueling": True,
            "learning_rate": 0.001,
        }

        features = tuner._extract_features_used(params)

        # Vérifier que les features sont extraites
        assert isinstance(features, dict)
        assert "double_dqn" in features
        assert "prioritized_replay" in features
        assert "n_step" in features
        assert "dueling" in features

    def test_analyze_feature_importance(self):
        """Test _analyze_feature_importance method"""
        tuner = HyperparameterTuner()

        # Mock trials
        trials = []
        for i in range(5):
            trial = Mock()
            trial.params = {
                "use_double_dqn": i % 2 == 0,
                "use_prioritized_replay": i % 3 == 0,
                "use_n_step": i % 4 == 0,
                "use_dueling": i % 5 == 0,
                "learning_rate": 0.001 + i * 0.0001,
            }
            trial.value = 100 - i * 10
            trials.append(trial)

        result = tuner._analyze_feature_importance(trials)

        # Vérifier que l'analyse est retournée
        assert isinstance(result, dict)

    def test_edge_case_empty_trials(self):
        """Test avec trials vides"""
        tuner = HyperparameterTuner()

        # Mock study avec trials vides
        mock_study = Mock()
        mock_study.trials = []
        mock_study.best_params = {}
        mock_study.best_value = 0  # Pas None
        mock_study.best_trial = Mock()
        mock_study.best_trial.number = 0

        with patch("pathlib.Path.mkdir"), patch("builtins.open", create=True) as mock_file:
            tuner.save_best_params(mock_study, "empty_trials.json")

            # Vérifier que le fichier est ouvert
            mock_file.assert_called_once()

    def test_edge_case_none_study(self):
        """Test avec study None"""
        tuner = HyperparameterTuner()

        # Vérifier qu'une exception est levée
        with pytest.raises(AttributeError):
            tuner.save_best_params(None, "test.json")

    def test_edge_case_empty_params(self):
        """Test avec paramètres vides"""
        tuner = HyperparameterTuner()

        params = {}

        features = tuner._extract_features_used(params)

        # Vérifier que les features sont vides
        assert isinstance(features, dict)
        assert len(features) > 0  # Il y a toujours des valeurs par défaut

    def test_edge_case_none_params(self):
        """Test avec paramètres None"""
        tuner = HyperparameterTuner()

        # Vérifier qu'une exception est levée
        with pytest.raises(AttributeError):
            tuner._extract_features_used(None)

    def test_edge_case_empty_trials_analysis(self):
        """Test analyse avec trials vides"""
        tuner = HyperparameterTuner()

        result = tuner._analyze_triplet_gagnant([])

        # Vérifier que l'analyse gère les trials vides
        assert isinstance(result, dict)

    def test_edge_case_single_trial(self):
        """Test avec un seul trial"""
        tuner = HyperparameterTuner()

        # Mock trial
        trial = Mock()
        trial.params = {"learning_rate": 0.001, "gamma": 0.95, "batch_size": 128}

        result = tuner._analyze_triplet_gagnant([trial])

        # Vérifier que l'analyse gère un seul trial
        assert isinstance(result, dict)

    def test_edge_case_duplicate_values(self):
        """Test avec valeurs dupliquées"""
        tuner = HyperparameterTuner()

        # Mock trials avec valeurs dupliquées
        trials = []
        for _i in range(3):
            trial = Mock()
            trial.params = {"learning_rate": 0.001, "gamma": 0.95, "batch_size": 128}
            trial.value = 100  # Même valeur
            trials.append(trial)

        result = tuner._analyze_feature_importance(trials)

        # Vérifier que l'analyse gère les valeurs dupliquées
        assert isinstance(result, dict)
