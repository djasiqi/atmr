# pyright: reportAttributeAccessIssue=false

# Constantes pour éviter les valeurs magiques
import logging
import random
from collections import deque
from typing import Any

import numpy as np

from services.rl.improved_q_network import DuelingQNetwork, ImprovedQNetwork
from services.rl.replay_buffer import PrioritizedReplayBuffer

TARGET_UPDATE_FREQ_ZERO = 0
TUPLE_SIZE_TWO = 2

"""Agent DQN amélioré avec techniques avancées d'apprentissage par renforcement."""


# pyright: reportMissingImports=false
try:
    import torch
    import torch.nn.functional as F
    from torch import optim
except ImportError:
    torch = None
    F = None
    optim = None


# Imports conditionnels pour N-step
try:
    from services.rl.n_step_buffer import NStepBuffer, NStepPrioritizedBuffer
except ImportError:
    NStepBuffer = None
    NStepPrioritizedBuffer = None


class ImprovedDQNAgent:
    """Agent DQN amélioré avec techniques avancées.

    Améliorations:
        - Double DQN pour réduire l'overestimation
        - Prioritized Experience Replay
        - Learning rate scheduling
        - Gradient clipping
        - Target network soft update
        - Epsilon decay adaptatif
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.00001,
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
        tau: float = 0.0005,  # Soft update
        use_n_step: bool = True,  # N-step learning
        n_step: int = 3,  # Nombre d'étapes pour N-step
        n_step_gamma: float = 0.99,  # Gamma pour N-step
        use_dueling: bool = False,  # Dueling DQN
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)  # Ensure action_dim is always an int for randrange()
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
        self.use_n_step = use_n_step
        self.n_step = n_step
        self.n_step_gamma = n_step_gamma
        self.use_dueling = use_dueling

        if torch is None:
            msg = "PyTorch is required but not installed"
            raise ImportError(msg)

        # Vérifier que N-step est disponible si demandé
        if use_n_step and (NStepBuffer is None or NStepPrioritizedBuffer is None):
            msg = "N-step buffers are required but not available"
            raise ImportError(msg)

        # Réseaux de neurones (Dueling ou standard)
        if use_dueling:
            self.q_network = DuelingQNetwork(state_dim, action_dim).to(device)
            self.target_network = DuelingQNetwork(state_dim, action_dim).to(device)
        else:
            self.q_network = ImprovedQNetwork(state_dim, action_dim).to(device)
            self.target_network = ImprovedQNetwork(state_dim, action_dim).to(device)

        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimiseur avec scheduler
        if optim is None:
            msg = "PyTorch optim is required but not available"
            raise ImportError(msg)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.9)

        # Replay buffer (N-step ou standard)
        if use_n_step:
            if NStepPrioritizedBuffer is None or NStepBuffer is None:
                msg = "N-step buffers are required but not available"
                raise ImportError(msg)
            if use_prioritized_replay:
                self.memory = NStepPrioritizedBuffer(buffer_size, n_step, n_step_gamma, alpha, beta_start, beta_end)
            else:
                self.memory = NStepBuffer(buffer_size, n_step, n_step_gamma)
        elif use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size, alpha, beta_start, beta_end)
        else:
            self.memory = deque(maxlen=int(buffer_size))

        # Métriques
        self.training_step = 0
        self.episode_count = 0
        self.losses = []

        print("🖥️  Improved DQN Agent using device: {device}")
        print("✅ Agent DQN amélioré créé:")
        print("   State dim: {state_dim}")
        print("   Action dim: {action_dim}")
        print(f"   Paramètres Q-Network: {sum(p.numel() for p in self.q_network.parameters()):,}")
        print("   Double DQN: {use_double_dqn}")
        print("   Prioritized Replay: {use_prioritized_replay}")
        print("   N-step Learning: {use_n_step} (n={n_step})")
        print("   Dueling DQN: {use_dueling}")

    def select_action(
        self, state: np.ndarray[Any, np.dtype[np.float32]], valid_actions: list[int] | None = None
    ) -> int:
        """Sélectionne une action avec epsilon-greedy et masquage optionnel.

        Args:
            state: État actuel
            valid_actions: Liste des actions valides (optionnel)

        Returns:
            Action sélectionnée

        """
        import time

        start_time = time.time()

        # Import conditionnel du RLLogger
        try:
            from services.rl.rl_logger import get_rl_logger

            rl_logger = get_rl_logger()
            enable_logging = True
        except ImportError:
            enable_logging = False
            rl_logger = None

        action = None
        q_values = None
        is_exploration = False

        if valid_actions is None:
            # Mode standard sans masquage
            if random.random() < self.epsilon:
                action = random.randrange(self.action_dim)
                is_exploration = True
            else:
                if torch is None:
                    msg = "PyTorch is required but not available"
                    raise ImportError(msg)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()

        # Mode avec masquage d'actions
        # Vérification de sécurité : s'assurer que valid_actions n'est pas vide
        elif not valid_actions:
            # Fallback de sécurité : action 0 (wait)
            action = 0
            is_exploration = True
            logging.warning("[ImprovedDQNAgent] valid_actions vide, fallback vers action 0")
        elif random.random() < self.epsilon:
            # Exploration : choisir une action valide aléatoire
            action = random.choice(valid_actions)
            is_exploration = True
        else:
            # Exploitation : choisir la meilleure action valide
            if torch is None:
                msg = "PyTorch is required but not available"
                raise ImportError(msg)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)

            # Masquer les actions invalides
            masked_q_values = q_values.clone()
            for i in range(self.action_dim):
                if i not in valid_actions:
                    masked_q_values[0, i] = float("-inf")

            action = masked_q_values.argmax().item()
            q_values = masked_q_values

        # Calculer la latence
        latency_ms = (time.time() - start_time) * 1000

        # Logging de la décision
        if enable_logging and rl_logger is not None:
            try:
                constraints = {
                    "epsilon": self.epsilon,
                    "is_exploration": is_exploration,
                    "valid_actions": valid_actions,
                    "confidence": 1.0 - self.epsilon if not is_exploration else 0.0,
                }

                metadata = {
                    "agent_type": "ImprovedDQNAgent",
                    "use_double_dqn": self.use_double_dqn,
                    "use_prioritized_replay": self.use_prioritized_replay,
                    "use_n_step": self.use_n_step,
                    "use_dueling": self.use_dueling,
                }

                rl_logger.log_decision(
                    state=state,
                    action=action,
                    q_values=q_values,
                    latency_ms=latency_ms,
                    model_version=f"dqn_v1_{self.state_dim}_{self.action_dim}",
                    constraints=constraints,
                    metadata=metadata,
                )
            except Exception:
                # Ne pas faire échouer la sélection d'action si le logging
                # échoue
                pass

        return action

    def store_transition(
        self,
        state: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
        reward: float,
        next_state: np.ndarray[Any, np.dtype[np.float32]],
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Stocke une transition dans le replay buffer (N-step ou standard)."""
        if self.use_n_step:
            # Utiliser le buffer N-step
            if self.use_prioritized_replay:
                # Calculer l'erreur TD pour la priorité
                if torch is None:
                    msg = "PyTorch is required but not available"
                    raise ImportError(msg)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_value = self.q_network(state_tensor)[0][action]
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                    next_q_value = self.target_network(next_state_tensor).max()
                    td_error = abs(reward + self.gamma * next_q_value * (1 - done) - q_value.item())

                self.memory.add_transition(state, action, reward, next_state, done, info)
            else:
                self.memory.add_transition(state, action, reward, next_state, done, info)
        # Mode standard (1-step)
        elif self.use_prioritized_replay:
            # Priorité initiale basée sur l'erreur TD
            if torch is None:
                msg = "PyTorch is required but not available"
                raise ImportError(msg)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_value = self.q_network(state_tensor)[0][action]
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                next_q_value = self.target_network(next_state_tensor).max()
                td_error = abs(reward + self.gamma * next_q_value * (1 - done) - q_value.item())
                priority = td_error + 1e-6  # Éviter les priorités nulles

            if isinstance(self.memory, PrioritizedReplayBuffer):
                self.memory.add(state, action, reward, next_state, done, priority)
        elif isinstance(self.memory, deque):
            self.memory.append((state, action, reward, next_state, done))

    def learn(self) -> float:
        """Apprend à partir d'un batch d'expériences (N-step ou standard)."""
        if len(self.memory) < self.batch_size:
            return 0.0

        # Échantillonnage selon le type de buffer
        if self.use_n_step:
            # Buffer N-step
            sample_result = self.memory.sample(self.batch_size)
            if isinstance(sample_result, tuple) and len(sample_result) == TUPLE_SIZE_TWO:
                batch, weights = sample_result
            else:
                return 0.0
            if not batch:
                return 0.0

            # Extraire les données des transitions N-step
            if torch is None:
                msg = "PyTorch is required but not available"
                raise ImportError(msg)
            states = torch.FloatTensor([t["state"] for t in batch]).to(self.device)
            actions = torch.LongTensor([t["action"] for t in batch]).to(self.device)
            n_step_rewards = torch.FloatTensor([t["n_step_return"] for t in batch]).to(self.device)
            next_states = torch.FloatTensor([t["next_state"] for t in batch]).to(self.device)
            dones = torch.BoolTensor([t["done"] for t in batch]).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
            # Variables pour compatibilité
            rewards = n_step_rewards  # Utiliser n_step_rewards comme rewards

        elif self.use_prioritized_replay and isinstance(self.memory, PrioritizedReplayBuffer):
            # Buffer PER standard
            batch, indices, weights = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch, strict=False)
            if torch is None:
                msg = "PyTorch is required but not available"
                raise ImportError(msg)
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
            # Variables pour compatibilité
            n_step_rewards = rewards  # Utiliser rewards comme n_step_rewards

            # Stocker indices pour mise à jour des priorités après calcul de la loss
            per_indices = indices

        # Buffer standard (deque)
        elif isinstance(self.memory, deque):
            batch = random.sample(list(self.memory), self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch, strict=False)
            if torch is None:
                msg = "PyTorch is required but not available"
                raise ImportError(msg)
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)
            weights = torch.ones(self.batch_size).to(self.device)
            # Variables pour compatibilité
            n_step_rewards = rewards  # Utiliser rewards comme n_step_rewards
        else:
            return 0.0

        # Calcul des Q-values actuelles
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Calcul des Q-values cibles
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: utiliser le réseau principal pour sélectionner
                # l'action
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            else:
                # DQN standard
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)

            if self.use_n_step:
                # Pour N-step, utiliser les retours N-step calculés
                target_q_values = n_step_rewards.unsqueeze(1) + (
                    self.gamma**self.n_step * next_q_values * ~dones.unsqueeze(1)
                )
            else:
                # Mode standard (1-step)
                target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))

        # Calcul de la loss
        if F is None:
            msg = "PyTorch F is required but not available"
            raise ImportError(msg)
        td_errors = current_q_values - target_q_values
        loss = (weights.unsqueeze(1) * F.mse_loss(current_q_values, target_q_values, reduction="none")).mean()

        # Mise à jour des priorités
        if self.use_prioritized_replay:
            priorities = torch.abs(td_errors).detach().cpu().numpy().flatten() + 1e-6

            if self.use_n_step and hasattr(self.memory, "update_priorities"):
                # Buffer N-step priorisé
                indices = list(range(len(batch)))
                self.memory.update_priorities(indices, priorities)
            elif isinstance(self.memory, PrioritizedReplayBuffer) and "per_indices" in locals():
                # Buffer PER standard - utiliser per_indices stocké plus haut
                self.memory.update_priorities(per_indices, priorities)  # type: ignore

        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Mise à jour du réseau cible (soft update)
        if self.training_step % self.target_update_freq == TARGET_UPDATE_FREQ_ZERO:
            self._soft_update_target_network()

        self.training_step += 1
        self.losses.append(loss.item())

        return loss.item()

    def _soft_update_target_network(self) -> None:
        """Mise à jour douce du réseau cible."""
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters(), strict=False
        ):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self) -> None:
        """Réduit epsilon progressivement."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath: str) -> None:
        """Sauvegarde le modèle."""
        if torch is None:
            msg = "PyTorch is required but not available"
            raise ImportError(msg)
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_step": self.training_step,
                "episode_count": self.episode_count,
                "losses": self.losses,
                "config": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "learning_rate": self.learning_rate,
                    "gamma": self.gamma,
                    "epsilon_end": self.epsilon_end,
                    "epsilon_decay": self.epsilon_decay,
                    "batch_size": self.batch_size,
                    "target_update_freq": self.target_update_freq,
                    "use_double_dqn": self.use_double_dqn,
                    "use_prioritized_replay": self.use_prioritized_replay,
                    "tau": self.tau,
                },
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Charge le modèle."""
        if torch is None:
            msg = "PyTorch is required but not available"
            raise ImportError(msg)
        # nosec B506: Les checkpoints contiennent optimizer state et config, pas seulement des poids
        # Les modèles proviennent de sources internes de confiance uniquement
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]
        self.episode_count = checkpoint["episode_count"]
        self.losses = checkpoint["losses"]
