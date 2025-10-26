#!/usr/bin/env python3
"""
Tests pour hyperparameter_tuner.py - couverture de base simplifiée
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from services.rl.hyperparameter_tuner import HyperparameterTuner


class TestHyperparameterTuner:
    """Tests pour la classe HyperparameterTuner."""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut."""
        tuner = HyperparameterTuner()

        assert tuner.n_trials == 50
        assert tuner.n_training_episodes == 200
        assert tuner.n_eval_episodes == 20
        assert tuner.study_name == "dqn_optimization"
        assert tuner.storage is None

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés."""
        tuner = HyperparameterTuner(
            n_trials=0.100,
            n_training_episodes=0.300,
            n_eval_episodes=30,
            study_name="custom_study",
            storage="sqlite:///test.db"
        )

        assert tuner.n_trials == 100
        assert tuner.n_training_episodes == 300
        assert tuner.n_eval_episodes == 30
        assert tuner.study_name == "custom_study"
        assert tuner.storage == "sqlite:///test.db"

    def test_suggest_hyperparameters_basic(self):
        """Test suggestion d'hyperparamètres de base."""
        tuner = HyperparameterTuner()

        # Mock trial avec valeurs fixes
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 0.0001
        mock_trial.suggest_categorical.return_value = 64
        mock_trial.suggest_int.return_value = 10

        config = tuner._suggest_hyperparameters(mock_trial)

        assert isinstance(config, dict)
        assert "learning_rate" in config
        assert "gamma" in config
        assert "batch_size" in config
        assert "epsilon_start" in config
        assert "epsilon_end" in config
        assert "epsilon_decay" in config
        assert "buffer_size" in config
        assert "target_update_freq" in config
        assert "use_double_dqn" in config
        assert "use_prioritized_replay" in config
        assert "alpha" in config
        assert "beta_start" in config
        assert "beta_end" in config
        assert "tau" in config
        assert "use_n_step" in config
        assert "n_step" in config
        assert "n_step_gamma" in config
        assert "use_dueling" in config
        assert "num_drivers" in config
        assert "max_bookings" in config

    def test_suggest_hyperparameters_with_exception(self):
        """Test suggestion d'hyperparamètres avec exception."""
        tuner = HyperparameterTuner()

        # Mock trial qui lève une exception
        mock_trial = Mock()
        mock_trial.suggest_float.side_effect = Exception("Trial error")

        with pytest.raises(Exception):
            tuner._suggest_hyperparameters(mock_trial)

    def test_suggest_hyperparameters_with_none_values(self):
        """Test suggestion d'hyperparamètres avec valeurs None."""
        tuner = HyperparameterTuner()

        # Mock trial avec valeurs None
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = None
        mock_trial.suggest_categorical.return_value = None
        mock_trial.suggest_int.return_value = None

        config = tuner._suggest_hyperparameters(mock_trial)

        assert isinstance(config, dict)
        assert config["learning_rate"] is None
        assert config["gamma"] is None
        assert config["batch_size"] is None
        assert config["epsilon_start"] is None
        assert config["epsilon_end"] is None
        assert config["epsilon_decay"] is None
        assert config["buffer_size"] is None
        assert config["target_update_freq"] is None
        assert config["num_drivers"] is None
        assert config["max_bookings"] is None

    def test_suggest_hyperparameters_with_zero_values(self):
        """Test suggestion d'hyperparamètres avec valeurs zéro."""
        tuner = HyperparameterTuner()

        # Mock trial avec valeurs zéro
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 0.0
        mock_trial.suggest_categorical.return_value = 0
        mock_trial.suggest_int.return_value = 0

        config = tuner._suggest_hyperparameters(mock_trial)

        assert isinstance(config, dict)
        assert config["learning_rate"] == 0.0
        assert config["gamma"] == 0.0
        assert config["batch_size"] == 0
        assert config["epsilon_start"] == 0.0
        assert config["epsilon_end"] == 0.0
        assert config["epsilon_decay"] == 0.0
        assert config["buffer_size"] == 0
        assert config["target_update_freq"] == 0
        assert config["num_drivers"] == 0
        assert config["max_bookings"] == 0

    def test_suggest_hyperparameters_with_negative_values(self):
        """Test suggestion d'hyperparamètres avec valeurs négatives."""
        tuner = HyperparameterTuner()

        # Mock trial avec valeurs négatives
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = -0.0001
        mock_trial.suggest_categorical.return_value = -64
        mock_trial.suggest_int.return_value = -10

        config = tuner._suggest_hyperparameters(mock_trial)

        assert isinstance(config, dict)
        assert config["learning_rate"] == -0.0001
        assert config["gamma"] == -0.0001
        assert config["batch_size"] == -64
        assert config["epsilon_start"] == -0.0001
        assert config["epsilon_end"] == -0.0001
        assert config["epsilon_decay"] == -0.0001
        assert config["buffer_size"] == -64
        assert config["target_update_freq"] == -10
        assert config["num_drivers"] == -10
        assert config["max_bookings"] == -10

    def test_suggest_hyperparameters_with_large_values(self):
        """Test suggestion d'hyperparamètres avec valeurs importantes."""
        tuner = HyperparameterTuner()

        # Mock trial avec valeurs importantes
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 1.0
        mock_trial.suggest_categorical.return_value = 1000
        mock_trial.suggest_int.return_value = 100

        config = tuner._suggest_hyperparameters(mock_trial)

        assert isinstance(config, dict)
        assert config["learning_rate"] == 1.0
        assert config["gamma"] == 1.0
        assert config["batch_size"] == 1000
        assert config["epsilon_start"] == 1.0
        assert config["epsilon_end"] == 1.0
        assert config["epsilon_decay"] == 1.0
        assert config["buffer_size"] == 1000
        assert config["target_update_freq"] == 100
        assert config["num_drivers"] == 100
        assert config["max_bookings"] == 100

    def test_suggest_hyperparameters_with_different_trials(self):
        """Test suggestion d'hyperparamètres avec différents trials."""
        tuner = HyperparameterTuner()

        # Premier trial
        mock_trial1 = Mock()
        mock_trial1.suggest_float.return_value = 0.0001
        mock_trial1.suggest_categorical.return_value = 64
        mock_trial1.suggest_int.return_value = 10

        config1 = tuner._suggest_hyperparameters(mock_trial1)

        # Deuxième trial
        mock_trial2 = Mock()
        mock_trial2.suggest_float.return_value = 0.00001
        mock_trial2.suggest_categorical.return_value = 128
        mock_trial2.suggest_int.return_value = 20

        config2 = tuner._suggest_hyperparameters(mock_trial2)

        assert isinstance(config1, dict)
        assert isinstance(config2, dict)
        assert config1["learning_rate"] != config2["learning_rate"]
        assert config1["gamma"] != config2["gamma"]
        assert config1["batch_size"] != config2["batch_size"]

    def test_suggest_hyperparameters_with_boolean_values(self):
        """Test suggestion d'hyperparamètres avec valeurs booléennes."""
        tuner = HyperparameterTuner()

        # Mock trial avec valeurs booléennes
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 0.0001
        mock_trial.suggest_categorical.return_value = True
        mock_trial.suggest_int.return_value = 10

        config = tuner._suggest_hyperparameters(mock_trial)

        assert isinstance(config, dict)
        assert config["use_double_dqn"] is True
        assert config["use_prioritized_replay"] is True
        assert config["use_n_step"] is True
        assert config["use_dueling"] is True

    def test_suggest_hyperparameters_with_numerical_values(self):
        """Test suggestion d'hyperparamètres avec valeurs numériques."""
        tuner = HyperparameterTuner()

        # Mock trial avec valeurs numériques spécifiques
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 0.6
        mock_trial.suggest_categorical.return_value = 64
        mock_trial.suggest_int.return_value = 3

        config = tuner._suggest_hyperparameters(mock_trial)

        assert isinstance(config, dict)
        assert config["alpha"] == 0.6
        assert config["beta_start"] == 0.6
        assert config["beta_end"] == 0.6
        assert config["tau"] == 0.6
        assert config["n_step"] == 3
        assert config["n_step_gamma"] == 0.6

    def test_suggest_hyperparameters_with_edge_cases(self):
        """Test suggestion d'hyperparamètres avec cas limites."""
        tuner = HyperparameterTuner()

        # Mock trial avec valeurs limites
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 1e-5
        mock_trial.suggest_categorical.return_value = 256
        mock_trial.suggest_int.return_value = 50

        config = tuner._suggest_hyperparameters(mock_trial)

        assert isinstance(config, dict)
        assert config["learning_rate"] == 1e-5
        assert config["gamma"] == 1e-5
        assert config["batch_size"] == 256
        assert config["epsilon_start"] == 1e-5
        assert config["epsilon_end"] == 1e-5
        assert config["epsilon_decay"] == 1e-5
        assert config["buffer_size"] == 256
        assert config["target_update_freq"] == 50
        assert config["num_drivers"] == 50
        assert config["max_bookings"] == 50
