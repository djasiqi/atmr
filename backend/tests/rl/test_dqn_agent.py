# ruff: noqa: DTZ001, DTZ003
# pyright: reportMissingImports=false
"""
Tests pour l'agent DQN complet.

Teste:
- Création et configuration
- Sélection d'actions (exploration/exploitation)
- Stockage et training
- Save/Load
- Intégration complète
"""
import os
import tempfile

import numpy as np
import pytest
import torch

from services.rl.dqn_agent import DQNAgent


class TestDQNAgentCreation:
    """Tests de création et initialisation."""

    def test_agent_creation(self):
        """Test création de l'agent."""
        agent = DQNAgent(state_dim=122, action_dim=201)

        assert agent.state_dim == 122
        assert agent.action_dim == 201
        assert agent.epsilon == 1.0  # Epsilon initial
        assert len(agent.memory) == 0  # Buffer vide
        assert agent.training_step == 0

    def test_agent_custom_params(self):
        """Test création avec paramètres custom."""
        agent = DQNAgent(
            state_dim=50,
            action_dim=100,
            learning_rate=0.01,
            gamma=0.95,
            epsilon_start=0.8,
            batch_size=32
        )

        assert agent.gamma == 0.95
        assert agent.epsilon == 0.8
        assert agent.batch_size == 32

    def test_agent_device_cpu(self):
        """Test que l'agent fonctionne sur CPU."""
        agent = DQNAgent(state_dim=122, action_dim=201, device='cpu')

        assert agent.device.type == 'cpu'
        assert next(agent.q_network.parameters()).device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_agent_device_cuda(self):
        """Test que l'agent fonctionne sur GPU si disponible."""
        agent = DQNAgent(state_dim=122, action_dim=201, device='cuda')

        assert agent.device.type == 'cuda'


class TestDQNAgentActionSelection:
    """Tests de sélection d'actions."""

    def test_select_action_exploration(self):
        """Test exploration (epsilon=1.0)."""
        agent = DQNAgent(state_dim=122, action_dim=201, epsilon_start=1.0)
        state = np.random.rand(122)

        # Avec epsilon=1.0, devrait explorer (actions aléatoires)
        actions = [agent.select_action(state, training=True) for _ in range(100)]

        # Vérifier variété
        unique_actions = set(actions)
        assert len(unique_actions) > 20  # Beaucoup de variété

    def test_select_action_exploitation(self):
        """Test exploitation (epsilon=0.0)."""
        agent = DQNAgent(state_dim=122, action_dim=201, epsilon_start=0.0)
        agent.q_network.eval()  # Mode évaluation pour déterminisme
        state = np.random.rand(122)

        # Avec epsilon=0.0, devrait exploiter (déterministe)
        actions = [agent.select_action(state, training=True) for _ in range(100)]

        # Toutes les actions devraient être identiques
        unique_actions = set(actions)
        assert len(unique_actions) == 1

    def test_select_action_training_false(self):
        """Test que training=False force exploitation."""
        agent = DQNAgent(state_dim=122, action_dim=201, epsilon_start=1.0)
        agent.q_network.eval()  # Mode évaluation pour déterminisme
        state = np.random.rand(122)

        # Même avec epsilon=1.0, training=False force greedy
        actions = [agent.select_action(state, training=False) for _ in range(100)]

        unique_actions = set(actions)
        assert len(unique_actions) == 1  # Déterministe

    def test_epsilon_decay(self):
        """Test décroissance de epsilon."""
        agent = DQNAgent(
            state_dim=122,
            action_dim=201,
            epsilon_start=1.0,
            epsilon_decay=0.99,
            epsilon_end=0.01
        )

        assert agent.epsilon == 1.0

        # Decay 100 fois
        for _ in range(100):
            agent.decay_epsilon()

        # Epsilon devrait avoir décru
        assert agent.epsilon < 0.5
        assert agent.epsilon >= 0.01  # Pas en dessous de epsilon_end

        # Decay 1000 fois de plus
        for _ in range(1000):
            agent.decay_epsilon()

        # Ne devrait pas descendre sous epsilon_end
        assert agent.epsilon == 0.01


class TestDQNAgentMemory:
    """Tests du stockage et replay buffer."""

    def test_store_transition(self):
        """Test stockage d'une transition."""
        agent = DQNAgent(state_dim=122, action_dim=201)

        state = np.random.rand(122)
        next_state = np.random.rand(122)

        agent.store_transition(state, 1, next_state, 50.0, False)

        assert len(agent.memory) == 1

    def test_store_multiple_transitions(self):
        """Test stockage de plusieurs transitions."""
        agent = DQNAgent(state_dim=122, action_dim=201)

        for i in range(100):
            state = np.random.rand(122)
            next_state = np.random.rand(122)
            agent.store_transition(state, i, next_state, float(i), False)

        assert len(agent.memory) == 100


class TestDQNAgentTraining:
    """Tests de l'entraînement."""

    def test_train_step_no_data(self):
        """Test que train_step retourne 0 si pas assez de données."""
        agent = DQNAgent(state_dim=122, action_dim=201, batch_size=64)

        # Ajouter seulement 10 transitions (< batch_size)
        for i in range(10):
            agent.store_transition(np.random.rand(122), i, np.random.rand(122), 1.0, False)

        loss = agent.train_step()

        assert loss == 0.0
        assert agent.training_step == 0

    def test_train_step_with_data(self):
        """Test training avec suffisamment de données."""
        agent = DQNAgent(state_dim=122, action_dim=201, batch_size=32)

        # Remplir buffer
        for i in range(100):
            state = np.random.rand(122)
            next_state = np.random.rand(122)
            agent.store_transition(state, i % 201, next_state, np.random.rand() * 100, False)

        # Entraîner
        loss = agent.train_step()

        assert loss > 0.0
        assert agent.training_step == 1
        assert len(agent.losses) == 1

    def test_multiple_train_steps(self):
        """Test plusieurs steps d'entraînement."""
        agent = DQNAgent(state_dim=122, action_dim=201, batch_size=32)

        # Remplir buffer
        for i in range(200):
            agent.store_transition(np.random.rand(122), i % 201, np.random.rand(122), 1.0, False)

        # Entraîner 50 fois
        losses = []
        for _ in range(50):
            loss_value = agent.train_step()
            losses.append(loss_value)

        assert agent.training_step == 50
        assert len(agent.losses) == 50
        assert all(loss_val > 0.0 for loss_val in losses)

    def test_target_network_update(self):
        """Test mise à jour du target network."""
        agent = DQNAgent(state_dim=122, action_dim=201)

        # Entraîner le q_network
        for i in range(100):
            agent.store_transition(np.random.rand(122), i % 201, np.random.rand(122), 1.0, False)

        for _ in range(20):
            agent.train_step()

        # Sauvegarder les poids du target avant update
        old_params = [p.clone() for p in agent.target_network.parameters()]

        # Update target
        agent.update_target_network()

        # Les poids devraient avoir changé
        new_params = list(agent.target_network.parameters())

        # Au moins un paramètre doit avoir changé
        changed = any(not torch.equal(old, new) for old, new in zip(old_params, new_params, strict=False))
        assert changed


class TestDQNAgentPersistence:
    """Tests de sauvegarde et chargement."""

    def test_save_and_load(self):
        """Test save/load basique."""
        agent = DQNAgent(state_dim=122, action_dim=201)

        # Entraîner un peu
        for i in range(100):
            agent.store_transition(np.random.rand(122), i % 201, np.random.rand(122), 1.0, False)

        for _ in range(30):
            agent.train_step()

        agent.decay_epsilon()
        agent.decay_epsilon()

        # Sauvegarder
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            save_path = f.name

        agent.save(save_path)

        # Créer nouvel agent et charger
        new_agent = DQNAgent(state_dim=122, action_dim=201)
        new_agent.load(save_path)

        # Vérifier que les métriques sont identiques
        assert new_agent.epsilon == agent.epsilon
        assert new_agent.training_step == agent.training_step
        assert new_agent.episode_count == agent.episode_count

        # Nettoyer
        os.remove(save_path)

    def test_save_checkpoint(self):
        """Test sauvegarde de checkpoint."""
        agent = DQNAgent(state_dim=122, action_dim=201)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = agent.save_checkpoint(
                episode=100,
                avg_reward=250.5,
                path_prefix=tmpdir
            )

            # Vérifier que le fichier existe
            assert os.path.exists(path)
            assert "ep0100" in path
            assert "r250" in path or "r251" in path

    def test_load_nonexistent_file(self):
        """Test que charger un fichier inexistant lève une erreur."""
        agent = DQNAgent(state_dim=122, action_dim=201)

        with pytest.raises(FileNotFoundError):
            agent.load("nonexistent_file.pth")


class TestDQNAgentUtilities:
    """Tests des fonctions utilitaires."""

    def test_get_q_values(self):
        """Test récupération des Q-values."""
        agent = DQNAgent(state_dim=122, action_dim=201)
        state = np.random.rand(122)

        q_values = agent.get_q_values(state)

        assert q_values.shape == (201,)
        assert isinstance(q_values, np.ndarray)

    def test_get_training_info(self):
        """Test récupération des infos de training."""
        agent = DQNAgent(state_dim=122, action_dim=201)

        # Ajouter des données et entraîner
        for i in range(100):
            agent.store_transition(np.random.rand(122), i % 201, np.random.rand(122), 1.0, False)

        for _ in range(10):
            agent.train_step()

        info = agent.get_training_info()

        assert info["training_step"] == 10
        assert info["buffer_size"] == 100
        assert info["epsilon"] == 1.0  # Pas encore de decay

