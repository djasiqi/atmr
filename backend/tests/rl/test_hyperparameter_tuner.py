#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Tests de sanity pour HyperparameterTuner.

Valide que l'espace de recherche n'est pas vide et que les bornes sont correctes.
"""

import sys

import optuna
import pytest
from optuna.trial import Trial

from services.rl.hyperparameter_tuner import HyperparameterTuner


class TestHyperparameterTunerSanity:
    """Tests de sanity pour HyperparameterTuner."""

    def test_hyperparameter_space_not_empty(self):
        """Test que l'espace de recherche n'est pas vide."""
        tuner = HyperparameterTuner(n_trials=1)
        
        # Créer un trial mock
        study = optuna.create_study()
        trial = study.ask()

        # Tester la suggestion d'hyperparamètres
        config = tuner._suggest_hyperparameters(trial)

        # Vérifier que tous les paramètres requis sont présents
        required_params = [
            "learning_rate", "gamma", "batch_size",
            "epsilon_start", "epsilon_end", "epsilon_decay",
            "buffer_size", "target_update_freq",
            "use_double_dqn", "use_prioritized_replay",
            "use_n_step", "use_dueling",
            "alpha", "beta_start", "beta_end",
            "n_step", "n_step_gamma", "tau",
            "num_drivers", "max_bookings"
        ]
        
        for param in required_params:
            assert param in config, f"Paramètre manquant: {param}"
            assert config[param] is not None, f"Paramètre {param} est None"

    def test_hyperparameter_bounds_valid(self):
        """Test que les bornes des hyperparamètres sont valides."""
        tuner = HyperparameterTuner(n_trials=1)
        
        # Créer plusieurs trials pour tester les bornes
        study = optuna.create_study()
        
        for _ in range(10):
            trial = study.ask()
            config = tuner._suggest_hyperparameters(trial)

            # Vérifier les bornes des paramètres continus
            assert 1e-5 <= config["learning_rate"] <= 1e-2, "learning_rate hors bornes"
            assert 0.90 <= config["gamma"] <= 0.999, "gamma hors bornes"
            assert 0.7 <= config["epsilon_start"] <= 1.0, "epsilon_start hors bornes"
            assert 0.01 <= config["epsilon_end"] <= 0.1, "epsilon_end hors bornes"
            assert 0.990 <= config["epsilon_decay"] <= 0.999, "epsilon_decay hors bornes"
            assert 0.4 <= config["alpha"] <= 0.8, "alpha hors bornes"
            assert 0.3 <= config["beta_start"] <= 0.6, "beta_start hors bornes"
            assert 0.8 <= config["beta_end"] <= 1.0, "beta_end hors bornes"
            assert 0.95 <= config["n_step_gamma"] <= 0.999, "n_step_gamma hors bornes"
            assert 0.0001 <= config["tau"] <= 0.01, "tau hors bornes"
            
            # Vérifier les bornes des paramètres entiers
            assert 5 <= config["target_update_freq"] <= 50, "target_update_freq hors bornes"
            assert 2 <= config["n_step"] <= 5, "n_step hors bornes"
            assert 5 <= config["num_drivers"] <= 20, "num_drivers hors bornes"
            assert 10 <= config["max_bookings"] <= 50, "max_bookings hors bornes"
            
            # Vérifier les choix catégoriques
            assert config["batch_size"] in [32, 64, 128, 256], "batch_size choix invalide"
            assert config["buffer_size"] in [50000, 100000, 200000, 500000], "buffer_size choix invalide"
            assert config["use_double_dqn"] in [True, False], "use_double_dqn choix invalide"
            assert config["use_prioritized_replay"] in [True, False], "use_prioritized_replay choix invalide"
            assert config["use_n_step"] in [True, False], "use_n_step choix invalide"
            assert config["use_dueling"] in [True, False], "use_dueling choix invalide"

    def test_triplet_gagnant_combinations(self):
        """Test que les combinaisons du triplet gagnant sont possibles."""
        tuner = HyperparameterTuner(n_trials=1)
        
        # Créer plusieurs trials pour trouver le triplet gagnant
        study = optuna.create_study()
        triplet_found = False
        
        for _ in range(50):  # Essayer jusqu'à 50 fois
            trial = study.ask()
            config = tuner._suggest_hyperparameters(trial)
            
            # Vérifier si le triplet gagnant est présent
            if (config["use_prioritized_replay"] and
                config["use_n_step"] and
                config["use_dueling"]):
                triplet_found = True
                break
        
        assert triplet_found, "Triplet gagnant (PER + N-step + Dueling) non trouvé"

    def test_hyperparameter_ranges_consistency(self):
        """Test la cohérence des plages d'hyperparamètres."""
        tuner = HyperparameterTuner(n_trials=1)

        # Obtenir les plages définies
        ranges = tuner._get_hyperparameter_ranges()
        
        # Vérifier que les plages sont cohérentes
        assert ranges["learning_rate"]["min"] < ranges["learning_rate"]["max"]
        assert ranges["gamma"]["min"] < ranges["gamma"]["max"]
        assert ranges["epsilon_start"]["min"] < ranges["epsilon_start"]["max"]
        assert ranges["epsilon_end"]["min"] < ranges["epsilon_end"]["max"]
        assert ranges["epsilon_decay"]["min"] < ranges["epsilon_decay"]["max"]
        assert ranges["alpha"]["min"] < ranges["alpha"]["max"]
        assert ranges["beta_start"]["min"] < ranges["beta_start"]["max"]
        assert ranges["beta_end"]["min"] < ranges["beta_end"]["max"]
        assert ranges["n_step_gamma"]["min"] < ranges["n_step_gamma"]["max"]
        assert ranges["tau"]["min"] < ranges["tau"]["max"]
        
        # Vérifier les plages entières
        assert ranges["target_update_freq"]["min"] < ranges["target_update_freq"]["max"]
        assert ranges["n_step"]["min"] < ranges["n_step"]["max"]
        assert ranges["num_drivers"]["min"] < ranges["num_drivers"]["max"]
        assert ranges["max_bookings"]["min"] < ranges["max_bookings"]["max"]
        
        # Vérifier les choix catégoriques
        assert len(ranges["batch_size"]["choices"]) > 0
        assert len(ranges["buffer_size"]["choices"]) > 0

    def test_feature_extraction(self):
        """Test l'extraction des features utilisées."""
        tuner = HyperparameterTuner(n_trials=1)
        
        # Paramètres de test
        test_params = {
            "use_double_dqn": True,
            "use_prioritized_replay": True,
            "use_n_step": True,
            "use_dueling": True,
            "n_step": 3,
            "alpha": 0.6,
            "tau": 0.0005
        }
        
        features = tuner._extract_features_used(test_params)
        
        assert features["double_dqn"]
        assert features["prioritized_replay"]
        assert features["n_step"]
        assert features["dueling"]
        assert features["n_step_value"] == 3
        assert features["alpha"] == 0.6
        assert features["tau"] == 0.0005

    def test_triplet_gagnant_analysis(self):
        """Test l'analyse du triplet gagnant."""
        tuner = HyperparameterTuner(n_trials=1)
        
        # Créer des trials mock avec différentes configurations
        mock_trials = []
        
        # Trial avec triplet gagnant
        trial1 = optuna.trial.create_trial(
            params={
                "use_prioritized_replay": True,
                "use_n_step": True,
                "use_dueling": True,
                "learning_rate": 0.0001
            },
            value=0.6000
        )
        mock_trials.append(trial1)
        
        # Trial avec seulement PER
        trial2 = optuna.trial.create_trial(
            params={
                "use_prioritized_replay": True,
                "use_n_step": False,
                "use_dueling": False,
                "learning_rate": 0.0001
            },
            value=0.5500
        )
        mock_trials.append(trial2)
        
        # Trial avec seulement N-step
        trial3 = optuna.trial.create_trial(
            params={
                "use_prioritized_replay": False,
                "use_n_step": True,
                "use_dueling": False,
                "learning_rate": 0.0001
            },
            value=0.5200
        )
        mock_trials.append(trial3)
        
        # Analyser le triplet gagnant
        analysis = tuner._analyze_triplet_gagnant(mock_trials)
        
        assert analysis["per_enabled"] == 2  # trial1 et trial2
        assert analysis["n_step_enabled"] == 2  # trial1 et trial3
        assert analysis["dueling_enabled"] == 1  # seulement trial1
        assert analysis["all_three_enabled"] == 1  # seulement trial1

    def test_feature_importance_analysis(self):
        """Test l'analyse d'importance des features."""
        tuner = HyperparameterTuner(n_trials=1)
        
        # Créer des trials mock avec différentes configurations
        mock_trials = []
        
        # Trial avec Double DQN activé
        trial1 = optuna.trial.create_trial(
            params={"use_double_dqn": True, "learning_rate": 0.0001},
            value=0.6000
        )
        mock_trials.append(trial1)
        
        # Trial avec Double DQN désactivé
        trial2 = optuna.trial.create_trial(
            params={"use_double_dqn": False, "learning_rate": 0.0001},
            value=0.5000
        )
        mock_trials.append(trial2)
        
        # Analyser l'importance des features
        importance = tuner._analyze_feature_importance(mock_trials)
        
        assert "double_dqn" in importance
        assert importance["double_dqn"]["enabled_avg"] == 600.0
        assert importance["double_dqn"]["disabled_avg"] == 500.0
        assert importance["double_dqn"]["improvement"] == 100.0
        assert importance["double_dqn"]["enabled_count"] == 1
        assert importance["double_dqn"]["disabled_count"] == 1

    def test_tuner_initialization(self):
        """Test l'initialisation du tuner."""
        tuner = HyperparameterTuner(
            n_trials=0.100,
            n_training_episodes=0.300,
            n_eval_episodes=30,
            study_name="test_study"
        )
        
        assert tuner.n_trials == 100
        assert tuner.n_training_episodes == 300
        assert tuner.n_eval_episodes == 30
        assert tuner.study_name == "test_study"

    def test_reproducibility_seed(self):
        """Test la reproductibilité avec seed."""
        tuner1 = HyperparameterTuner(n_trials=1, study_name="test1")
        tuner2 = HyperparameterTuner(n_trials=1, study_name="test2")
        
        # Créer des études avec le même seed
        study1 = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
        study2 = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
        
        trial1 = study1.ask()
        trial2 = study2.ask()
        
        config1 = tuner1._suggest_hyperparameters(trial1)
        config2 = tuner2._suggest_hyperparameters(trial2)
        
        # Les configurations devraient être identiques avec le même seed
        assert config1 == config2, "Configurations non reproductibles avec le même seed"


def run_sanity_tests():
    """Exécute tous les tests de sanity."""
    print("🧪 Exécution des tests de sanity HyperparameterTuner...")
    
    test_class = TestHyperparameterTunerSanity()
    test_methods = [method for method in dir(test_class) if method.startswith("test_")]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            method = getattr(test_class, method_name)
            method()
            print("   ✅ {method_name}")
            passed += 1
        except Exception:
            print("   ❌ {method_name}: {e}")
            failed += 1
    
    print("\n📊 Résultats: {passed} réussis, {failed} échoués")
    return failed == 0


if __name__ == "__main__":
    success = run_sanity_tests()
    sys.exit(0 if success else 1)
