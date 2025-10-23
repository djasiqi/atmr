# ruff: noqa: DTZ001, DTZ003, T201
# pyright: reportMissingImports=false
"""
Tests pour HyperparameterTuner.

Teste l'optimisation automatique des hyperparamètres avec Optuna.
"""
import json
import tempfile
from pathlib import Path

import pytest

from services.rl.hyperparameter_tuner import HyperparameterTuner


class TestHyperparameterTunerCreation:
    """Tests création du tuner."""

    def test_tuner_creation_default(self):
        """Test création tuner avec paramètres par défaut."""
        tuner = HyperparameterTuner()

        assert tuner is not None
        assert tuner.n_trials == 50
        assert tuner.n_training_episodes == 200
        assert tuner.n_eval_episodes == 20
        assert tuner.study_name == "dqn_optimization"

    def test_tuner_creation_custom(self):
        """Test création tuner avec paramètres custom."""
        tuner = HyperparameterTuner(
            n_trials=10,
            n_training_episodes=50,
            n_eval_episodes=5,
            study_name="test_study"
        )

        assert tuner.n_trials == 10
        assert tuner.n_training_episodes == 50
        assert tuner.n_eval_episodes == 5
        assert tuner.study_name == "test_study"


class TestHyperparameterTunerSuggestions:
    """Tests suggestions hyperparamètres."""

    def test_suggest_hyperparameters_structure(self):
        """Test que suggest_hyperparameters retourne la bonne structure."""
        import optuna

        tuner = HyperparameterTuner(n_trials=1)
        study = optuna.create_study()
        trial = study.ask()

        config = tuner._suggest_hyperparameters(trial)

        # Vérifier que toutes les clés sont présentes
        expected_keys = {
            'hidden_sizes', 'dropout', 'learning_rate', 'gamma',
            'batch_size', 'epsilon_start', 'epsilon_end', 'epsilon_decay',
            'buffer_size', 'target_update_freq', 'num_drivers', 'max_bookings'
        }

        assert set(config.keys()) == expected_keys

    def test_suggest_hyperparameters_ranges(self):
        """Test que les hyperparamètres sont dans les bonnes plages."""
        import optuna

        tuner = HyperparameterTuner(n_trials=1)
        study = optuna.create_study()
        trial = study.ask()

        config = tuner._suggest_hyperparameters(trial)

        # Vérifier les ranges
        assert 0.0 <= config['dropout'] <= 0.3
        assert 1e-5 <= config['learning_rate'] <= 1e-2
        assert 0.90 <= config['gamma'] <= 0.999
        assert config['batch_size'] in [32, 64, 128, 256]
        assert 0.8 <= config['epsilon_start'] <= 1.0
        assert 0.01 <= config['epsilon_end'] <= 0.1
        assert 0.990 <= config['epsilon_decay'] <= 0.999
        assert config['buffer_size'] in [50000, 100000, 200000]
        assert 5 <= config['target_update_freq'] <= 20
        assert 5 <= config['num_drivers'] <= 15
        assert 10 <= config['max_bookings'] <= 30


class TestHyperparameterTunerOptimization:
    """Tests optimisation."""

    @pytest.mark.slow
    def test_optimize_minimal(self):
        """Test optimisation minimale (2 trials, 5 episodes)."""
        tuner = HyperparameterTuner(
            n_trials=2,
            n_training_episodes=5,
            n_eval_episodes=2
        )

        study = tuner.optimize()

        assert study is not None
        assert len(study.trials) > 0
        assert study.best_value is not None

    def test_objective_callable(self):
        """Test que objective est callable."""
        tuner = HyperparameterTuner(n_trials=1)

        assert callable(tuner.objective)


class TestHyperparameterTunerSaving:
    """Tests sauvegarde résultats."""

    def test_save_best_params(self):
        """Test sauvegarde des meilleurs paramètres."""
        import optuna

        # Créer un study factice
        study = optuna.create_study(direction='maximize')

        # Simuler un trial
        def objective(trial):
            x = trial.suggest_float('x', 0, 1)
            return x

        study.optimize(objective, n_trials=3)

        # Sauvegarder
        tuner = HyperparameterTuner()
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_path = f.name

        try:
            tuner.save_best_params(study, output_path)

            # Vérifier fichier créé
            assert Path(output_path).exists()

            # Vérifier contenu
            with open(output_path, encoding='utf-8') as f:
                data = json.load(f)

            assert 'best_reward' in data
            assert 'best_params' in data
            assert 'n_trials_total' in data
            assert 'optimization_history' in data

            assert data['n_trials_total'] == 3
            assert len(data['optimization_history']) <= 10  # Top 10

        finally:
            # Cleanup
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_save_best_params_creates_directory(self):
        """Test que save_best_params crée le dossier parent."""
        import optuna

        study = optuna.create_study(direction='maximize')

        def objective(trial):
            return trial.suggest_float('x', 0, 1)

        study.optimize(objective, n_trials=1)

        tuner = HyperparameterTuner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "subdir" / "config.json")

            tuner.save_best_params(study, output_path)

            assert Path(output_path).exists()
            assert Path(output_path).parent.exists()


class TestHyperparameterTunerIntegration:
    """Tests d'intégration."""

    @pytest.mark.slow
    def test_full_workflow_minimal(self):
        """Test workflow complet avec paramètres minimaux."""
        tuner = HyperparameterTuner(
            n_trials=2,
            n_training_episodes=5,
            n_eval_episodes=2,
            study_name="test_workflow"
        )

        # Optimiser
        study = tuner.optimize()

        # Vérifications
        assert len(study.trials) > 0
        assert study.best_trial is not None
        assert study.best_value is not None

        # Sauvegarder
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_path = f.name

        try:
            tuner.save_best_params(study, output_path)

            # Vérifier sauvegarde
            assert Path(output_path).exists()

            with open(output_path, encoding='utf-8') as f:
                data = json.load(f)

            assert data['best_reward'] == study.best_value
            assert data['best_params'] == study.best_params

        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()

