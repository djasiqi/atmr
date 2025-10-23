#!/usr/bin/env python3
"""
Agent DQN amélioré avec techniques avancées d'apprentissage par renforcement.
"""

import random
from collections import deque

import numpy as np

# pyright: reportMissingImports=false
try:
    import torch
    import torch.nn.functional as F  # noqa: N812
    import torch.optim as optim
except ImportError:
    torch = None
    F = None
    optim = None

from services.rl.improved_q_network import ImprovedQNetwork
from services.rl.replay_buffer import PrioritizedReplayBuffer


class ImprovedDQNAgent:
    """
    Agent DQN amélioré avec techniques avancées.

    Améliorations:
        - Double DQN pour réduire l'overestimation
        - Prioritized Experience Replay
        - Learning rate scheduling
        - Gradient clipping
        - Target network soft update
        - Epsilon decay adaptatif
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 100000,
        target_update_freq: int = 100,
        device: str = "cpu",
        use_double_dqn: bool = True,
        use_prioritized_replay: bool = True,
        alpha: float = 0.6,  # Priorité exponentielle
        beta_start: float = 0.4,  # Importance sampling
        beta_end: float = 1.0,
        tau: float = 0.005,  # Soft update
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.use_double_dqn = use_double_dqn
        self.use_prioritized_replay = use_prioritized_replay
        self.tau = tau

        if torch is None:
            raise ImportError("PyTorch is required but not installed")

        # Réseaux de neurones
        self.q_network = ImprovedQNetwork(state_dim, action_dim).to(device)
        self.target_network = ImprovedQNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimiseur avec scheduler
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.9)

        # Replay buffer
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size, alpha, beta_start, beta_end)
        else:
            self.memory = deque(maxlen=buffer_size)

        # Métriques
        self.training_step = 0
        self.episode_count = 0
        self.losses = []

        print(f"🖥️  Improved DQN Agent using device: {device}")  # noqa: T201
        print("✅ Agent DQN amélioré créé:")  # noqa: T201
        print(f"   State dim: {state_dim}")  # noqa: T201
        print(f"   Action dim: {action_dim}")  # noqa: T201
        print(f"   Paramètres Q-Network: {sum(p.numel() for p in self.q_network.parameters()):,}")  # noqa: T201
        print(f"   Double DQN: {use_double_dqn}")  # noqa: T201
        print(f"   Prioritized Replay: {use_prioritized_replay}")  # noqa: T201

    def select_action(self, state: np.ndarray, valid_actions: list[int] | None = None) -> int:
        """
        Sélectionne une action avec epsilon-greedy et masquage optionnel.
        
        Args:
            state: État actuel
            valid_actions: Liste des actions valides (optionnel)
            
        Returns:
            Action sélectionnée
        """
        if valid_actions is None:
            # Mode standard sans masquage
            if random.random() < self.epsilon:
                return random.randrange(self.action_dim)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

        # Mode avec masquage d'actions
        if random.random() < self.epsilon:
            # Exploration : choisir une action valide aléatoire
            return random.choice(valid_actions)

        # Exploitation : choisir la meilleure action valide
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        # Masquer les actions invalides
        masked_q_values = q_values.clone()
        for i in range(self.action_dim):
            if i not in valid_actions:
                masked_q_values[0, i] = float('-inf')

        return masked_q_values.argmax().item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Stocke une transition dans le replay buffer."""
        if self.use_prioritized_replay:
            # Priorité initiale basée sur l'erreur TD
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_value = self.q_network(state_tensor)[0][action]
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                next_q_value = self.target_network(next_state_tensor).max()
                td_error = abs(reward + self.gamma * next_q_value * (1 - done) - q_value.item())
                priority = td_error + 1e-6  # Éviter les priorités nulles

            if isinstance(self.memory, PrioritizedReplayBuffer):
                self.memory.add(state, action, reward, next_state, done, priority)
        else:
            if isinstance(self.memory, deque):
                self.memory.append((state, action, reward, next_state, done))

    def learn(self) -> float:
        """Apprend à partir d'un batch d'expériences."""
        if len(self.memory) < self.batch_size:
            return 0.0

        # Échantillonnage
        if self.use_prioritized_replay and isinstance(self.memory, PrioritizedReplayBuffer):
            batch, indices, weights = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch, strict=False)
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            if isinstance(self.memory, deque):
                batch = random.sample(list(self.memory), self.batch_size)
                states, actions, rewards, next_states, dones = zip(*batch, strict=False)
                states = torch.FloatTensor(states).to(self.device)
                actions = torch.LongTensor(actions).to(self.device)
                rewards = torch.FloatTensor(rewards).to(self.device)
                next_states = torch.FloatTensor(next_states).to(self.device)
                dones = torch.BoolTensor(dones).to(self.device)
                weights = torch.ones(self.batch_size).to(self.device)
            else:
                return 0.0

        # Calcul des Q-values actuelles
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Calcul des Q-values cibles
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: utiliser le réseau principal pour sélectionner l'action
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            else:
                # DQN standard
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)

            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))

        # Calcul de la loss
        td_errors = current_q_values - target_q_values
        loss = (weights.unsqueeze(1) * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()

        # Mise à jour des priorités
        if self.use_prioritized_replay and isinstance(self.memory, PrioritizedReplayBuffer):
            priorities = torch.abs(td_errors).detach().cpu().numpy().flatten() + 1e-6
            self.memory.update_priorities(indices, priorities)

        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Mise à jour du réseau cible (soft update)
        if self.training_step % self.target_update_freq == 0:
            self._soft_update_target_network()

        self.training_step += 1
        self.losses.append(loss.item())

        return loss.item()

    def _soft_update_target_network(self) -> None:
        """Mise à jour douce du réseau cible."""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters(), strict=False):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self) -> None:
        """Réduit epsilon progressivement."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath: str) -> None:
        """Sauvegarde le modèle."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'losses': self.losses,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'use_double_dqn': self.use_double_dqn,
                'use_prioritized_replay': self.use_prioritized_replay,
                'tau': self.tau,
            }
        }, filepath)

    def load(self, filepath: str) -> None:
        """Charge le modèle."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.losses = checkpoint['losses']
