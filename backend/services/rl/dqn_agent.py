# ruff: noqa: T201
# pyright: reportMissingImports=false
"""
Agent DQN (Deep Q-Network) pour le dispatch autonome.

Implémente:
- Epsilon-greedy exploration/exploitation
- Experience replay avec buffer
- Target network pour stabilité
- Double DQN pour réduire overestimation
- Save/Load de modèles

Auteur: ATMR Project - RL Team
Date: Octobre 2025
Semaine: 15 (Jours 3-5)
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from services.rl.q_network import QNetwork
from services.rl.replay_buffer import ReplayBuffer, Transition


class DQNAgent:
    """
    Agent DQN pour le dispatch de véhicules.

    Features:
        - Epsilon-greedy pour équilibrer exploration/exploitation
        - Experience Replay (réutilise les expériences)
        - Target Network (stabilité d'apprentissage)
        - Double DQN (réduit surestimation des Q-values)
        - Gradient clipping (évite explosions)
        - Checkpoints automatiques

    Hyperparamètres:
        - learning_rate: 0.001 (taux d'apprentissage)
        - gamma: 0.99 (discount factor - importance du futur)
        - epsilon: 1.0 → 0.01 (exploration → exploitation)
        - batch_size: 64 (taille batch pour training)
        - buffer_size: 100k (transitions stockées)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 100000,
        target_update_freq: int = 10,
        device: str | None = None,
    ):
        """
        Initialise l'agent DQN.

        Args:
            state_dim: Dimension de l'espace d'état
            action_dim: Nombre d'actions possibles
            learning_rate: Taux d'apprentissage Adam
            gamma: Discount factor (0-1, importance du futur)
            epsilon_start: Epsilon initial (exploration)
            epsilon_end: Epsilon minimal (exploitation)
            epsilon_decay: Facteur de décroissance de epsilon
            batch_size: Taille du batch pour training
            buffer_size: Capacité du replay buffer
            target_update_freq: Fréquence de mise à jour du target network (episodes)
            device: Device PyTorch ('cpu', 'cuda', ou None pour auto)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device (CPU ou GPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"🖥️  DQN Agent using device: {self.device}")

        # Créer les deux réseaux (Q-network et Target network)
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)

        # Copier les poids initiaux vers le target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Toujours en mode évaluation

        # Optimizer (Adam est un bon choix pour DQN)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Loss function (Huber Loss = robuste aux outliers)
        self.criterion = nn.SmoothL1Loss()

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Tracking des métriques
        self.training_step = 0
        self.episode_count = 0
        self.losses = []

        print("✅ Agent DQN créé:")
        print(f"   State dim: {state_dim}")
        print(f"   Action dim: {action_dim}")
        print(f"   Paramètres Q-Network: {self.q_network.count_parameters():,}")

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Sélectionne une action avec epsilon-greedy.

        Stratégie:
            - Avec probabilité epsilon: action aléatoire (exploration)
            - Avec probabilité 1-epsilon: meilleure action selon Q-network (exploitation)

        Args:
            state: État actuel (numpy array)
            training: Si True, utilise epsilon-greedy, sinon greedy pur

        Returns:
            action: Index de l'action sélectionnée
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: action aléatoire
            return np.random.randint(self.action_dim)
        else:
            # Exploitation: meilleure action selon Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        done: bool
    ):
        """
        Stocke une transition dans le replay buffer.

        Args:
            state: État actuel
            action: Action prise
            next_state: État suivant
            reward: Récompense reçue
            done: Episode terminé ou non
        """
        self.memory.push(state, action, next_state, reward, done)

    def train_step(self) -> float:
        """
        Effectue un pas d'entraînement (backpropagation).

        Algorithme Double DQN:
            1. Sample batch aléatoire du replay buffer
            2. Calculer Q(s, a) actuelles avec q_network
            3. Sélectionner meilleures actions avec q_network
            4. Évaluer ces actions avec target_network (Double DQN)
            5. Calculer target: r + γ * Q_target(s', a_best)
            6. Minimiser loss = (Q_current - target)²
            7. Backpropagation et mise à jour des poids

        Returns:
            loss: Valeur de la loss (pour monitoring)
        """
        # Vérifier qu'il y a assez de données
        if len(self.memory) < self.batch_size:
            return 0.0

        # 1. Sample batch aléatoire
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions, strict=False))

        # 2. Convertir en tensors PyTorch
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # 3. Calculer Q-values actuelles pour les actions prises
        current_q_values = self.q_network(state_batch).gather(
            1, action_batch.unsqueeze(1)
        )

        # 4. Calculer Q-values cibles avec Double DQN
        with torch.no_grad():
            # Double DQN: Sélectionner action avec q_network
            next_actions = self.q_network(next_state_batch).argmax(1, keepdim=True)

            # Évaluer cette action avec target_network
            next_q_values = self.target_network(next_state_batch).gather(1, next_actions)

            # Target: r + γ * Q_target(s', a_best) * (1 - done)
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values.squeeze()

        # 5. Calculer la loss (Huber Loss = robuste)
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        # 6. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # 7. Gradient clipping (évite explosions de gradients)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)

        # 8. Mise à jour des poids
        self.optimizer.step()

        # Tracking
        self.training_step += 1
        self.losses.append(loss.item())

        return loss.item()

    def update_target_network(self):
        """
        Met à jour le target network en copiant les poids du q_network.

        Appelé périodiquement (tous les N episodes) pour stabilité.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """
        Décroît epsilon pour réduire progressivement l'exploration.

        Formule: epsilon = max(epsilon_end, epsilon * epsilon_decay)

        Exemple:
            epsilon_start = 1.0, epsilon_decay = 0.995, epsilon_end = 0.01
            Episode 100: epsilon ≈ 0.60
            Episode 500: epsilon ≈ 0.08
            Episode 1000: epsilon ≈ 0.01
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """
        Sauvegarde le modèle complet (réseaux + optimizer + config).

        Args:
            path: Chemin du fichier .pth
        """
        # Créer le répertoire si nécessaire
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'losses': self.losses[-1000:],  # Garde les 1000 dernières losses
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'gamma': self.gamma,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq
            }
        }, path)

    def load(self, path: str):
        """
        Charge un modèle sauvegardé.

        Args:
            path: Chemin du fichier .pth

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Charger les états des réseaux
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])

        # Charger l'optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Charger les métriques
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.losses = checkpoint.get('losses', [])

        print(f"✅ Modèle chargé depuis: {path}")
        print(f"   Episode: {self.episode_count}")
        print(f"   Training steps: {self.training_step}")
        print(f"   Epsilon: {self.epsilon:.4f}")

    def save_checkpoint(
        self,
        episode: int,
        avg_reward: float,
        path_prefix: str = "data/rl/models"
    ) -> str:
        """
        Sauvegarde un checkpoint avec métadonnées dans le nom.

        Args:
            episode: Numéro de l'épisode
            avg_reward: Reward moyen récent
            path_prefix: Préfixe du chemin

        Returns:
            Chemin du fichier sauvegardé
        """
        os.makedirs(path_prefix, exist_ok=True)

        filename = f"{path_prefix}/dqn_ep{episode:04d}_r{avg_reward:.0f}.pth"
        self.save(filename)

        return filename

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Retourne les Q-values pour toutes les actions (pour debugging/analyse).

        Args:
            state: État (numpy array)

        Returns:
            Q-values (numpy array de taille action_dim)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().squeeze()

    def get_training_info(self) -> dict:
        """
        Retourne des informations sur l'état du training.

        Returns:
            Dictionnaire avec métriques
        """
        avg_loss_100 = np.mean(self.losses[-100:]) if self.losses else 0.0

        return {
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "epsilon": self.epsilon,
            "buffer_size": len(self.memory),
            "avg_loss_100": avg_loss_100,
            "total_losses_tracked": len(self.losses)
        }

