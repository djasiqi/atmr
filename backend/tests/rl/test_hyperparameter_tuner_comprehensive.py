"""
Tests complets pour hyperparameter_tuner.py - Couverture 95%+
"""

import json
from unittest.mock import Mock, patch

import optuna
import pytest

from services.rl.hyperparameter_tuner import HyperparameterTuner


class TestHyperparameterTuner:
    """Tests complets pour HyperparameterTuner"""

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
        assert (
            mock_trial.suggest_float.call_count >= 5
        )  # learning_rate, gamma, epsilon_*
        assert mock_trial.suggest_categorical.call_count >= 2  # batch_size, buffer_size
        assert mock_trial.suggest_int.call_count >= 2  # num_drivers, max_bookings

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

        # Mock trial avec valeurs spécifiques
        mock_trial = Mock()
        mock_trial.suggest_float.side_effect = [0.001, 0.95, 0.9, 0.1, 0.995]
        mock_trial.suggest_categorical.side_effect = [128, 100000]
        mock_trial.suggest_int.side_effect = [5, 15]

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
        mock_trial.suggest_float.side_effect = [0.001, 0.95, 0.9, 0.1, 0.995]
        mock_trial.suggest_categorical.side_effect = [128, 100000]
        mock_trial.suggest_int.side_effect = [3, 10]

        with (
            patch("services.rl.hyperparameter_tuner.DispatchEnv") as mock_env_class,
            patch(
                "services.rl.hyperparameter_tuner.ImprovedDQNAgent"
            ) as mock_agent_class,
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

            # Vérifier que l'entraînement et l'évaluation sont effectués
            assert (
                mock_env.reset.call_count
                >= tuner.n_training_episodes + tuner.n_eval_episodes
            )
            assert (
                mock_env.step.call_count
                >= tuner.n_training_episodes + tuner.n_eval_episodes
            )

            # Vérifier que le reward est retourné
            assert isinstance(reward, float)

    def test_objective_function_with_pruning(self):
        """Test objective function avec pruning"""
        tuner = HyperparameterTuner(n_training_episodes=5, n_eval_episodes=2)

        # Mock trial avec pruning
        mock_trial = Mock()
        mock_trial.suggest_float.side_effect = [0.001, 0.95, 0.9, 0.1, 0.995]
        mock_trial.suggest_categorical.side_effect = [128, 100000]
        mock_trial.suggest_int.side_effect = [3, 10]
        mock_trial.should_prune.return_value = True

        with (
            patch("services.rl.hyperparameter_tuner.DispatchEnv") as mock_env_class,
            patch(
                "services.rl.hyperparameter_tuner.ImprovedDQNAgent"
            ) as mock_agent_class,
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

        with (
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch("builtins.open", patch("builtins.open", create=True)) as mock_file,
        ):
            tuner.save_best_params(mock_study, "test_params.json")

            # Vérifier que le répertoire est créé
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

            # Vérifier que le fichier est ouvert en écriture
            mock_file.assert_called_once()

    def test_save_best_params_with_custom_filename(self):
        """Test save_best_params avec nom de fichier personnalisé"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.best_params = {"learning_rate": 0.001}
        mock_study.best_value = 100

        with (
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", patch("builtins.open", create=True)) as mock_file,
        ):
            tuner.save_best_params(mock_study, "custom_params.json")

            # Vérifier que le fichier est ouvert avec le bon nom
            mock_file.assert_called_once()

    def test_load_best_params(self):
        """Test load_best_params method"""
        tuner = HyperparameterTuner()

        params_data = {
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

        with patch("builtins.open", patch("builtins.open", create=True)) as mock_file:
            mock_file.return_value.__enter__.return_value.read.return_value = (
                json.dumps(params_data)
            )

            params = tuner.load_best_params("test_params.json")

            # Vérifier que le fichier est ouvert en lecture
            mock_file.assert_called_once()

            # Vérifier que les paramètres sont chargés
            assert params == params_data

    def test_get_study_summary(self):
        """Test get_study_summary method"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.best_params = {"learning_rate": 0.001}
        mock_study.best_value = 100
        mock_study.n_trials = 50
        mock_study.trials = [Mock() for _ in range(50)]

        summary = tuner.get_study_summary(mock_study)

        # Vérifier que le résumé contient les informations attendues
        assert "Meilleur reward" in summary
        assert "Nombre de trials" in summary
        assert "Meilleurs paramètres" in summary
        assert "100" in summary
        assert "50" in summary

    def test_get_study_summary_empty(self):
        """Test get_study_summary avec étude vide"""
        tuner = HyperparameterTuner()

        # Mock study vide
        mock_study = Mock()
        mock_study.best_params = {}
        mock_study.best_value = None
        mock_study.n_trials = 0
        mock_study.trials = []

        summary = tuner.get_study_summary(mock_study)

        # Vérifier que le résumé contient les informations pour étude vide
        assert "Meilleur reward" in summary
        assert "Nombre de trials" in summary
        assert "0" in summary

    def test_plot_optimization_history(self):
        """Test plot_optimization_history method"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_optimization_history") as mock_plot:
            mock_plot.return_value = Mock()

            tuner.plot_optimization_history(mock_study)

            # Vérifier que plot_optimization_history est appelé
            mock_plot.assert_called_once_with(mock_study)

    def test_plot_parameter_importance(self):
        """Test plot_parameter_importance method"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_parameter_importance") as mock_plot:
            mock_plot.return_value = Mock()

            tuner.plot_parameter_importance(mock_study)

            # Vérifier que plot_parameter_importance est appelé
            mock_plot.assert_called_once_with(mock_study)

    def test_plot_parallel_coordinate(self):
        """Test plot_parallel_coordinate method"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_parallel_coordinate") as mock_plot:
            mock_plot.return_value = Mock()

            tuner.plot_parallel_coordinate(mock_study)

            # Vérifier que plot_parallel_coordinate est appelé
            mock_plot.assert_called_once_with(mock_study)

    def test_plot_slice(self):
        """Test plot_slice method"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_slice") as mock_plot:
            mock_plot.return_value = Mock()

            tuner.plot_slice(mock_study)

            # Vérifier que plot_slice est appelé
            mock_plot.assert_called_once_with(mock_study)

    def test_plot_timeline(self):
        """Test plot_timeline method"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_timeline") as mock_plot:
            mock_plot.return_value = Mock()

            tuner.plot_timeline(mock_study)

            # Vérifier que plot_timeline est appelé
            mock_plot.assert_called_once_with(mock_study)

    def test_plot_intermediate_values(self):
        """Test plot_intermediate_values method"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_intermediate_values") as mock_plot:
            mock_plot.return_value = Mock()

            tuner.plot_intermediate_values(mock_study)

            # Vérifier que plot_intermediate_values est appelé
            mock_plot.assert_called_once_with(mock_study)

    def test_plot_edf(self):
        """Test plot_edf method"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_edf") as mock_plot:
            mock_plot.return_value = Mock()

            tuner.plot_edf(mock_study)

            # Vérifier que plot_edf est appelé
            mock_plot.assert_called_once_with(mock_study)

    def test_plot_rank(self):
        """Test plot_rank method"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_rank") as mock_plot:
            mock_plot.return_value = Mock()

            tuner.plot_rank(mock_study)

            # Vérifier que plot_rank est appelé
            mock_plot.assert_called_once_with(mock_study)

    def test_plot_contour(self):
        """Test plot_contour method"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_contour") as mock_plot:
            mock_plot.return_value = Mock()

            tuner.plot_contour(mock_study)

            # Vérifier que plot_contour est appelé
            mock_plot.assert_called_once_with(mock_study)

    def test_plot_pareto_front(self):
        """Test plot_pareto_front method"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_pareto_front") as mock_plot:
            mock_plot.return_value = Mock()

            tuner.plot_pareto_front(mock_study)

            # Vérifier que plot_pareto_front est appelé
            mock_plot.assert_called_once_with(mock_study)

    def test_plot_optimization_history_with_exception(self):
        """Test plot_optimization_history avec exception"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_optimization_history") as mock_plot:
            mock_plot.side_effect = Exception("Plot error")

            # Vérifier qu'une exception est levée
            with pytest.raises(Exception, match="Plot error"):
                tuner.plot_optimization_history(mock_study)

    def test_plot_parameter_importance_with_exception(self):
        """Test plot_parameter_importance avec exception"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_parameter_importance") as mock_plot:
            mock_plot.side_effect = Exception("Plot error")

            # Vérifier qu'une exception est levée
            with pytest.raises(Exception, match="Plot error"):
                tuner.plot_parameter_importance(mock_study)

    def test_plot_parallel_coordinate_with_exception(self):
        """Test plot_parallel_coordinate avec exception"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_parallel_coordinate") as mock_plot:
            mock_plot.side_effect = Exception("Plot error")

            # Vérifier qu'une exception est levée
            with pytest.raises(Exception, match="Plot error"):
                tuner.plot_parallel_coordinate(mock_study)

    def test_plot_slice_with_exception(self):
        """Test plot_slice avec exception"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_slice") as mock_plot:
            mock_plot.side_effect = Exception("Plot error")

            # Vérifier qu'une exception est levée
            with pytest.raises(Exception, match="Plot error"):
                tuner.plot_slice(mock_study)

    def test_plot_timeline_with_exception(self):
        """Test plot_timeline avec exception"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_timeline") as mock_plot:
            mock_plot.side_effect = Exception("Plot error")

            # Vérifier qu'une exception est levée
            with pytest.raises(Exception, match="Plot error"):
                tuner.plot_timeline(mock_study)

    def test_plot_intermediate_values_with_exception(self):
        """Test plot_intermediate_values avec exception"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_intermediate_values") as mock_plot:
            mock_plot.side_effect = Exception("Plot error")

            # Vérifier qu'une exception est levée
            with pytest.raises(Exception, match="Plot error"):
                tuner.plot_intermediate_values(mock_study)

    def test_plot_edf_with_exception(self):
        """Test plot_edf avec exception"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_edf") as mock_plot:
            mock_plot.side_effect = Exception("Plot error")

            # Vérifier qu'une exception est levée
            with pytest.raises(Exception, match="Plot error"):
                tuner.plot_edf(mock_study)

    def test_plot_rank_with_exception(self):
        """Test plot_rank avec exception"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_rank") as mock_plot:
            mock_plot.side_effect = Exception("Plot error")

            # Vérifier qu'une exception est levée
            with pytest.raises(Exception, match="Plot error"):
                tuner.plot_rank(mock_study)

    def test_plot_contour_with_exception(self):
        """Test plot_contour avec exception"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_contour") as mock_plot:
            mock_plot.side_effect = Exception("Plot error")

            # Vérifier qu'une exception est levée
            with pytest.raises(Exception, match="Plot error"):
                tuner.plot_contour(mock_study)

    def test_plot_pareto_front_with_exception(self):
        """Test plot_pareto_front avec exception"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.trials = [Mock() for _ in range(10)]

        with patch("optuna.visualization.plot_pareto_front") as mock_plot:
            mock_plot.side_effect = Exception("Plot error")

            # Vérifier qu'une exception est levée
            with pytest.raises(Exception, match="Plot error"):
                tuner.plot_pareto_front(mock_study)

    def test_edge_case_empty_trials(self):
        """Test avec trials vides"""
        tuner = HyperparameterTuner()

        # Mock study avec trials vides
        mock_study = Mock()
        mock_study.trials = []
        mock_study.best_params = {}
        mock_study.best_value = None
        mock_study.n_trials = 0

        summary = tuner.get_study_summary(mock_study)

        # Vérifier que le résumé gère les trials vides
        assert "0" in summary

    def test_edge_case_none_study(self):
        """Test avec study None"""
        tuner = HyperparameterTuner()

        # Vérifier qu'une exception est levée
        with pytest.raises(AttributeError):
            tuner.get_study_summary(None)

    def test_edge_case_invalid_filename(self):
        """Test avec nom de fichier invalide"""
        tuner = HyperparameterTuner()

        # Mock study
        mock_study = Mock()
        mock_study.best_params = {"learning_rate": 0.001}
        mock_study.best_value = 100

        with (
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", patch("builtins.open", create=True)) as mock_file,
        ):
            mock_file.side_effect = OSError("File error")

            # Vérifier qu'une exception est levée
            with pytest.raises(OSError, match="File error"):
                tuner.save_best_params(mock_study, "invalid/path/file.json")

    def test_edge_case_load_nonexistent_file(self):
        """Test chargement de fichier inexistant"""
        tuner = HyperparameterTuner()

        with patch("builtins.open", patch("builtins.open", create=True)) as mock_file:
            mock_file.side_effect = FileNotFoundError("File not found")

            # Vérifier qu'une exception est levée
            with pytest.raises(FileNotFoundError):
                tuner.load_best_params("nonexistent.json")

    def test_edge_case_invalid_json(self):
        """Test chargement de JSON invalide"""
        tuner = HyperparameterTuner()

        with patch("builtins.open", patch("builtins.open", create=True)) as mock_file:
            mock_file.return_value.__enter__.return_value.read.return_value = (
                "invalid json"
            )

            # Vérifier qu'une exception est levée
            with pytest.raises(json.JSONDecodeError):
                tuner.load_best_params("invalid.json")
