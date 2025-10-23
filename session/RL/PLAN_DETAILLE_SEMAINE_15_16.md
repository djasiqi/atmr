# 📅 PLAN DÉTAILLÉ - SEMAINES 15-16 : AGENT DQN

**Période:** Semaines 15-16 (14 jours)  
**Objectif:** Créer et entraîner un agent DQN capable d'apprendre  
**Prérequis:** ✅ Environnement Gym fonctionnel (Semaine 13-14)

---

## 🎯 Vue d'Ensemble des 2 Semaines

### Semaine 15 : Implémentation DQN

**Objectif:** Créer l'agent DQN complet avec PyTorch

**Livrables:**

- Agent DQN fonctionnel (Q-Network, Replay Buffer, Training)
- Tests unitaires (20+ tests)
- Script de training basique
- Documentation

### Semaine 16 : Entraînement & Optimisation

**Objectif:** Entraîner 1000 épisodes et optimiser

**Livrables:**

- Modèle entraîné (1000 épisodes)
- TensorBoard monitoring
- Comparaison vs baseline
- Rapport d'analyse

---

## 📅 SEMAINE 15 : IMPLÉMENTATION DQN

### 🔷 Jour 1 : Setup & Q-Network (Lundi)

#### Objectifs

1. Installer PyTorch et dépendances
2. Créer le réseau Q-Network
3. Tests unitaires du réseau

#### Tâches Détaillées

**1.1 Installation PyTorch** (30 min)

```bash
# Ajouter à requirements-rl.txt
torch>=2.0.0
tensorboard>=2.13.0

# Installer
docker-compose exec api pip install torch tensorboard
```

**1.2 Créer Q-Network** (2h)

Fichier: `backend/services/rl/q_network.py` (~150 lignes)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Réseau de neurones pour approximer Q(s,a).

    Architecture: Input(122) → FC(512) → FC(256) → FC(128) → Output(201)
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple = (512, 256, 128),
        dropout: float = 0.2
    ):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], action_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialisation Xavier
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Pas d'activation sur output
        return x
```

**1.3 Tests Q-Network** (1h)

Fichier: `backend/tests/rl/test_q_network.py`

```python
def test_q_network_creation():
    """Test création du réseau."""
    net = QNetwork(state_dim=122, action_dim=201)
    assert net is not None

def test_q_network_forward():
    """Test forward pass."""
    net = QNetwork(state_dim=122, action_dim=201)
    state = torch.randn(1, 122)
    q_values = net(state)
    assert q_values.shape == (1, 201)

def test_q_network_batch():
    """Test forward avec batch."""
    net = QNetwork(state_dim=122, action_dim=201)
    states = torch.randn(64, 122)  # Batch de 64
    q_values = net(states)
    assert q_values.shape == (64, 201)
```

#### Livrables Jour 1

- ✅ PyTorch installé
- ✅ `q_network.py` créé et testé
- ✅ 5+ tests unitaires
- ✅ Validation GPU/CPU

---

### 🔷 Jour 2 : Replay Buffer (Mardi)

#### Objectifs

1. Implémenter Replay Buffer
2. Tests du buffer
3. Optimiser la performance

#### Tâches Détaillées

**2.1 Créer Replay Buffer** (2h)

Fichier: `backend/services/rl/replay_buffer.py` (~120 lignes)

```python
from collections import deque, namedtuple
import random
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """
    Experience Replay Buffer.

    Stocke les transitions (s, a, s', r, done) pour apprentissage.
    Permet de ré-utiliser les expériences et casser les corrélations.
    """
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        done: bool
    ):
        """Ajoute une transition."""
        self.buffer.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size: int) -> list:
        """Sample un batch aléatoire."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Vérifie si assez de données pour entraînement."""
        return len(self.buffer) >= min_size
```

**2.2 Tests Replay Buffer** (1h)

Fichier: `backend/tests/rl/test_replay_buffer.py`

```python
def test_buffer_creation():
    buffer = ReplayBuffer(capacity=1000)
    assert len(buffer) == 0

def test_buffer_push():
    buffer = ReplayBuffer(capacity=1000)
    state = np.random.rand(122)
    next_state = np.random.rand(122)
    buffer.push(state, 1, next_state, 50.0, False)
    assert len(buffer) == 1

def test_buffer_sample():
    buffer = ReplayBuffer(capacity=1000)
    # Remplir avec 100 transitions
    for i in range(100):
        buffer.push(np.random.rand(122), i, np.random.rand(122), 1.0, False)

    batch = buffer.sample(32)
    assert len(batch) == 32

def test_buffer_capacity():
    buffer = ReplayBuffer(capacity=10)
    # Ajouter 20 éléments
    for i in range(20):
        buffer.push(np.random.rand(122), i, np.random.rand(122), 1.0, False)

    # Ne devrait garder que les 10 derniers
    assert len(buffer) == 10
```

#### Livrables Jour 2

- ✅ `replay_buffer.py` créé
- ✅ 8+ tests unitaires
- ✅ Performance validée (< 1ms pour sample)

---

### 🔷 Jour 3 : Agent DQN - Structure (Mercredi)

#### Objectifs

1. Créer la classe DQNAgent
2. Implémenter select_action()
3. Implémenter epsilon-greedy

#### Tâches Détaillées

**3.1 Structure Agent DQN** (3h)

Fichier: `backend/services/rl/dqn_agent.py` (~400 lignes)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional

from services.rl.q_network import QNetwork
from services.rl.replay_buffer import ReplayBuffer

class DQNAgent:
    """
    Agent DQN pour le dispatch.

    Implémente:
    - Epsilon-greedy exploration
    - Experience replay
    - Target network
    - Double DQN
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
        device: Optional[str] = None
    ):
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

        # Réseaux Q
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Mode évaluation

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )

        # Loss function (Huber Loss = robuste aux outliers)
        self.criterion = nn.SmoothL1Loss()

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Tracking
        self.training_step = 0
        self.episode_count = 0
        self.losses = []

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> int:
        """
        Sélectionne une action avec epsilon-greedy.

        Args:
            state: État actuel (numpy array)
            training: Si True, utilise epsilon-greedy, sinon greedy pur

        Returns:
            action: Index de l'action (0 à action_dim-1)
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

    def decay_epsilon(self):
        """Décroît epsilon pour réduire l'exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
```

**3.2 Tests Select Action** (1h)

```python
def test_select_action_exploration():
    """Test que l'exploration fonctionne."""
    agent = DQNAgent(state_dim=122, action_dim=201, epsilon_start=1.0)
    state = np.random.rand(122)

    # Avec epsilon=1.0, devrait toujours explorer
    actions = [agent.select_action(state) for _ in range(100)]
    unique_actions = set(actions)

    # Devrait avoir de la variété
    assert len(unique_actions) > 10

def test_select_action_exploitation():
    """Test que l'exploitation fonctionne."""
    agent = DQNAgent(state_dim=122, action_dim=201, epsilon_start=0.0)
    state = np.random.rand(122)

    # Avec epsilon=0.0, devrait toujours exploiter (déterministe)
    actions = [agent.select_action(state) for _ in range(100)]
    unique_actions = set(actions)

    # Devrait être toujours la même action
    assert len(unique_actions) == 1
```

#### Livrables Jour 3

- ✅ `dqn_agent.py` structure créée
- ✅ select_action() implémenté
- ✅ Epsilon-greedy fonctionnel
- ✅ 6+ tests

---

### 🔷 Jour 4 : Agent DQN - Training (Jeudi)

#### Objectifs

1. Implémenter train_step()
2. Implémenter update_target_network()
3. Tests du training

#### Tâches Détaillées

**4.1 Implémenter Training** (3h)

```python
class DQNAgent:
    # ... (suite du Jour 3)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        done: bool
    ):
        """Stocke une transition dans le replay buffer."""
        self.memory.push(state, action, next_state, reward, done)

    def train_step(self) -> float:
        """
        Effectue un pas d'entraînement (backpropagation).

        Returns:
            loss: Valeur de la loss (pour monitoring)
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample batch aléatoire
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convertir en tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # Calcul des Q-values actuelles
        current_q_values = self.q_network(state_batch).gather(
            1, action_batch.unsqueeze(1)
        )

        # Calcul des Q-values cibles (Double DQN)
        with torch.no_grad():
            # Sélection action avec q_network
            next_actions = self.q_network(next_state_batch).argmax(1, keepdim=True)
            # Évaluation avec target_network
            next_q_values = self.target_network(next_state_batch).gather(1, next_actions)
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values.squeeze()

        # Loss (Huber Loss)
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (stabilité)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)

        self.optimizer.step()

        self.training_step += 1
        self.losses.append(loss.item())

        return loss.item()

    def update_target_network(self):
        """Copie q_network vers target_network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
```

**4.2 Tests Training** (1h)

```python
def test_train_step_no_data():
    """Test que train_step retourne 0 si pas assez de données."""
    agent = DQNAgent(state_dim=122, action_dim=201, batch_size=64)
    loss = agent.train_step()
    assert loss == 0.0

def test_train_step_with_data():
    """Test training avec données."""
    agent = DQNAgent(state_dim=122, action_dim=201, batch_size=32)

    # Remplir buffer
    for _ in range(100):
        state = np.random.rand(122)
        next_state = np.random.rand(122)
        agent.store_transition(state, 1, next_state, 50.0, False)

    # Entraîner
    loss = agent.train_step()
    assert loss > 0.0
    assert agent.training_step == 1

def test_target_network_update():
    """Test update du target network."""
    agent = DQNAgent(state_dim=122, action_dim=201)

    # Modifier q_network
    for _ in range(10):
        state = np.random.rand(122)
        agent.store_transition(state, 1, state, 1.0, False)
        agent.train_step()

    # Sauvegarder les poids avant update
    old_params = [p.clone() for p in agent.target_network.parameters()]

    # Update target
    agent.update_target_network()

    # Les poids devraient avoir changé
    new_params = list(agent.target_network.parameters())
    assert not all(torch.equal(old, new) for old, new in zip(old_params, new_params))
```

#### Livrables Jour 4

- ✅ train_step() implémenté
- ✅ Double DQN (stabilité)
- ✅ Gradient clipping
- ✅ 8+ tests

---

### 🔷 Jour 5 : Agent DQN - Persistence (Vendredi)

#### Objectifs

1. Implémenter save/load
2. Créer système de checkpoints
3. Tests de sérialisation

#### Tâches Détaillées

**5.1 Save/Load** (2h)

```python
class DQNAgent:
    # ... (suite)

    def save(self, path: str):
        """
        Sauvegarde le modèle complet.

        Args:
            path: Chemin du fichier .pth
        """
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
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'gamma': self.gamma,
                'batch_size': self.batch_size
            }
        }, path)

    def load(self, path: str):
        """
        Charge un modèle sauvegardé.

        Args:
            path: Chemin du fichier .pth
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.losses = checkpoint.get('losses', [])

    def save_checkpoint(self, episode: int, avg_reward: float, path_prefix: str = "data/rl/models"):
        """Sauvegarde un checkpoint avec métadonnées."""
        import os
        os.makedirs(path_prefix, exist_ok=True)

        filename = f"{path_prefix}/dqn_ep{episode}_r{avg_reward:.0f}.pth"
        self.save(filename)
        return filename
```

**5.2 Tests Save/Load** (1h)

```python
def test_save_and_load(tmp_path):
    """Test sauvegarde et chargement."""
    agent = DQNAgent(state_dim=122, action_dim=201)

    # Entraîner un peu
    for _ in range(50):
        agent.store_transition(np.random.rand(122), 1, np.random.rand(122), 1.0, False)
        agent.train_step()

    # Sauvegarder
    save_path = tmp_path / "agent.pth"
    agent.save(str(save_path))

    # Créer nouvel agent et charger
    new_agent = DQNAgent(state_dim=122, action_dim=201)
    new_agent.load(str(save_path))

    # Vérifier que les paramètres sont identiques
    assert new_agent.epsilon == agent.epsilon
    assert new_agent.training_step == agent.training_step
```

#### Livrables Jour 5

- ✅ save/load implémenté
- ✅ Système checkpoints
- ✅ 4+ tests
- ✅ **Agent DQN complet !**

---

### 🔷 Weekend : Review & Documentation

#### Tâches

- Relire tout le code
- Compléter docstrings
- Créer README pour dqn_agent
- Préparer pour Semaine 16

---

### 🔷 Jour 6 : Script de Training - Setup (Lundi S16)

#### Objectifs

1. Créer le script train_dqn.py
2. Implémenter la boucle principale
3. Logging basique

#### Tâches Détaillées

**6.1 Training Script** (3h)

Fichier: `backend/scripts/rl/train_dqn.py` (~300 lignes)

```python
import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.rl.dispatch_env import DispatchEnv
from services.rl.dqn_agent import DQNAgent

def train_dqn(
    episodes: int = 1000,
    max_steps: int = 100,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    batch_size: int = 64,
    save_interval: int = 100,
    eval_interval: int = 50
):
    """
    Entraîne un agent DQN sur l'environnement de dispatch.

    Args:
        episodes: Nombre d'épisodes d'entraînement
        max_steps: Steps maximum par épisode
        learning_rate: Taux d'apprentissage
        gamma: Discount factor
        batch_size: Taille du batch
        save_interval: Fréquence de sauvegarde (episodes)
        eval_interval: Fréquence d'évaluation (episodes)
    """
    print("="*60)
    print("🚀 ENTRAÎNEMENT AGENT DQN - DISPATCH")
    print("="*60)

    # Créer environnement
    env = DispatchEnv(
        num_drivers=10,
        max_bookings=20,
        simulation_hours=2  # 2h pour training rapide
    )

    # Créer agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size
    )

    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f'data/rl/tensorboard/dqn_{timestamp}')

    # Metrics
    episode_rewards = []
    episode_lengths = []
    best_avg_reward = -float('inf')

    print(f"\n📊 Configuration:")
    print(f"  Episodes: {episodes}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Gamma: {gamma}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {agent.device}")

    # Training loop
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            # Sélectionner action
            action = agent.select_action(state, training=True)

            # Step dans l'environnement
            next_state, reward, done, truncated, info = env.step(action)

            # Stocker transition
            agent.store_transition(state, action, next_state, reward, done or truncated)

            # Entraîner
            loss = agent.train_step()
            episode_loss += loss

            # Mise à jour
            state = next_state
            episode_reward += reward
            steps += 1

        # Decay epsilon
        agent.decay_epsilon()

        # Update target network périodiquement
        if (episode + 1) % agent.target_update_freq == 0:
            agent.update_target_network()

        # Tracking
        agent.episode_count += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Logging
        avg_loss = episode_loss / max(steps, 1)

        writer.add_scalar('Reward/Episode', episode_reward, episode)
        writer.add_scalar('Loss/Episode', avg_loss, episode)
        writer.add_scalar('Epsilon', agent.epsilon, episode)
        writer.add_scalar('Steps/Episode', steps, episode)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward_10 = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{episodes} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Avg(10): {avg_reward_10:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")

        # Évaluation périodique
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_agent(agent, env, episodes=10)
            writer.add_scalar('Reward/Evaluation', eval_reward, episode)

            print(f"\n📈 Evaluation (ep {episode+1}): {eval_reward:.1f}")

            # Sauvegarder meilleur modèle
            if eval_reward > best_avg_reward:
                best_avg_reward = eval_reward
                agent.save("data/rl/models/dqn_best.pth")
                print(f"   ✅ Nouveau meilleur modèle sauvegardé!")

        # Checkpoints
        if (episode + 1) % save_interval == 0:
            agent.save_checkpoint(episode + 1, episode_reward)

    # Fin du training
    writer.close()
    agent.save("data/rl/models/dqn_final.pth")

    print(f"\n✅ ENTRAÎNEMENT TERMINÉ!")
    print(f"   Episodes: {episodes}")
    print(f"   Meilleur reward: {best_avg_reward:.1f}")
    print(f"   Modèle sauvegardé: data/rl/models/dqn_final.pth")

def evaluate_agent(agent, env, episodes: int = 10) -> float:
    """Évalue l'agent sans exploration."""
    rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 100:
            action = agent.select_action(state, training=False)  # Greedy
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

        rewards.append(episode_reward)

    return np.mean(rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    train_dqn(
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size
    )
```

#### Livrables Jour 6

- ✅ `train_dqn.py` créé
- ✅ Training loop complet
- ✅ TensorBoard intégré
- ✅ Fonction d'évaluation

---

### 🔷 Jour 7 : Premier Entraînement (Mardi)

#### Objectifs

1. Lancer le premier training (100 épisodes)
2. Valider que tout fonctionne
3. Analyser les résultats

#### Tâches Détaillées

**7.1 Training Initial** (1h setup + 2h training)

```bash
# Créer dossiers
docker-compose exec api mkdir -p data/rl/models data/rl/tensorboard

# Lancer training (100 épisodes pour test)
docker-compose exec api python scripts/rl/train_dqn.py \
    --episodes 100 \
    --learning-rate 0.001 \
    --gamma 0.99 \
    --batch-size 64
```

**Output attendu:**

```
============================================================
🚀 ENTRAÎNEMENT AGENT DQN - DISPATCH
============================================================

📊 Configuration:
  Episodes: 100
  Learning rate: 0.001
  Gamma: 0.99
  Batch size: 64
  Device: cuda

Episode 10/100 | Reward: -250.5 | Avg(10): -320.1 | Epsilon: 0.951 | Loss: 0.8245
Episode 20/100 | Reward: -180.3 | Avg(10): -215.7 | Epsilon: 0.904 | Loss: 0.6120
...
Episode 100/100 | Reward: +45.2 | Avg(10): +38.5 | Epsilon: 0.606 | Loss: 0.2341

✅ ENTRAÎNEMENT TERMINÉ!
```

**7.2 Analyser TensorBoard** (1h)

```bash
# Lancer TensorBoard
docker-compose exec api tensorboard --logdir=data/rl/tensorboard
# Ouvrir http://localhost:6006
```

**Graphiques à vérifier:**

- Reward/Episode (doit monter)
- Loss/Episode (doit descendre)
- Epsilon (doit décroître)

#### Livrables Jour 7

- ✅ Premier training réussi
- ✅ Modèle sauvegardé
- ✅ Courbes d'apprentissage
- ✅ Validation fonctionnelle

---

### 🔷 Jour 8-9 : Entraînement Long (Mercredi-Jeudi)

#### Objectifs

1. Entraîner 1000 épisodes complets
2. Monitoring continu
3. Sauvegarder checkpoints

#### Tâches Détaillées

**8.1 Training 1000 Episodes** (6-12h sur GPU)

```bash
# Lancer training complet
docker-compose exec api python scripts/rl/train_dqn.py \
    --episodes 1000 \
    --learning-rate 0.001 \
    --gamma 0.99 \
    --batch-size 64 \
    > logs/training_1000ep.log 2>&1 &

# Suivre les logs
docker-compose exec api tail -f logs/training_1000ep.log

# Monitorer avec TensorBoard
tensorboard --logdir=data/rl/tensorboard
```

**8.2 Checkpoints Attendus**

```
data/rl/models/
├── dqn_ep100_r-50.pth
├── dqn_ep200_r120.pth
├── dqn_ep300_r280.pth
├── dqn_ep400_r450.pth
├── dqn_ep500_r680.pth
├── dqn_ep600_r920.pth
├── dqn_ep700_r1100.pth
├── dqn_ep800_r1350.pth
├── dqn_ep900_r1520.pth
├── dqn_ep1000_r1780.pth
├── dqn_best.pth          ← Meilleur modèle
└── dqn_final.pth         ← Dernier modèle
```

#### Livrables Jours 8-9

- ✅ 1000 épisodes entraînés
- ✅ 10 checkpoints sauvegardés
- ✅ Meilleur modèle identifié
- ✅ Logs complets

---

### 🔷 Jour 10 : Script d'Évaluation (Vendredi)

#### Objectifs

1. Créer script d'évaluation
2. Comparer DQN vs Baseline
3. Générer rapport

#### Tâches Détaillées

**10.1 Script Évaluation** (2h)

Fichier: `backend/scripts/rl/evaluate_agent.py` (~200 lignes)

```python
def evaluate_agent_detailed(agent, env, episodes: int = 100):
    """Évalue l'agent sur N épisodes."""
    results = {
        "rewards": [],
        "assignments": [],
        "late_pickups": [],
        "cancellations": [],
        "total_distance": [],
        "completion_rate": []
    }

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 100:
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        # Collecter métriques
        stats = info['episode_stats']
        results["rewards"].append(episode_reward)
        results["assignments"].append(stats["assignments"])
        results["late_pickups"].append(stats["late_pickups"])
        results["cancellations"].append(stats["cancellations"])
        results["total_distance"].append(stats["total_distance"])

        total_bookings = stats["assignments"] + stats["cancellations"]
        completion = stats["assignments"] / total_bookings if total_bookings > 0 else 0
        results["completion_rate"].append(completion)

    # Calculer moyennes
    summary = {
        "avg_reward": np.mean(results["rewards"]),
        "std_reward": np.std(results["rewards"]),
        "avg_assignments": np.mean(results["assignments"]),
        "avg_late_rate": np.mean(results["late_pickups"]) / max(np.mean(results["assignments"]), 1),
        "avg_completion_rate": np.mean(results["completion_rate"]),
        "avg_distance": np.mean(results["total_distance"]) / max(np.mean(results["assignments"]), 1)
    }

    return summary, results

def compare_with_baseline(dqn_results, baseline_results):
    """Compare DQN avec baseline."""
    comparison = {}

    for metric in ["avg_reward", "avg_assignments", "avg_completion_rate", "avg_distance"]:
        dqn_val = dqn_results[metric]
        base_val = baseline_results[metric]
        improvement = ((dqn_val - base_val) / abs(base_val)) * 100

        comparison[metric] = {
            "dqn": dqn_val,
            "baseline": base_val,
            "improvement_pct": improvement
        }

    return comparison
```

**10.2 Évaluer DQN** (1h)

```bash
# Évaluer le meilleur modèle
docker-compose exec api python scripts/rl/evaluate_agent.py \
    --model data/rl/models/dqn_best.pth \
    --episodes 100 \
    --compare-baseline
```

#### Livrables Jour 10

- ✅ Script évaluation créé
- ✅ Comparaison DQN vs baseline
- ✅ Métriques détaillées

---

### 🔷 Jours 11-12 : Visualisation & Analyse (Weekend)

#### Objectifs

1. Créer scripts de visualisation
2. Générer graphiques
3. Analyser comportement de l'agent

#### Tâches Détaillées

**11.1 Script Visualisation** (2h)

Fichier: `backend/scripts/rl/visualize_training.py`

```python
import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_training_curves(log_file: str):
    """Génère les courbes d'apprentissage."""
    # Charger les logs
    with open(log_file, 'rb') as f:
        data = pickle.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Reward
    axes[0, 0].plot(data['rewards'])
    axes[0, 0].plot(moving_average(data['rewards'], 50), 'r-', linewidth=2)
    axes[0, 0].set_title('Reward par Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)

    # Loss
    axes[0, 1].plot(data['losses'])
    axes[0, 1].plot(moving_average(data['losses'], 50), 'r-', linewidth=2)
    axes[0, 1].set_title('Loss par Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)

    # Epsilon
    axes[1, 0].plot(data['epsilons'])
    axes[1, 0].set_title('Epsilon (Exploration)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].grid(True)

    # Completion Rate
    axes[1, 1].plot(data['completion_rates'])
    axes[1, 1].plot(moving_average(data['completion_rates'], 50), 'r-', linewidth=2)
    axes[1, 1].set_title('Taux de Complétion')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Taux')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('data/rl/training_curves.png', dpi=300)
    print("✅ Graphiques sauvegardés: data/rl/training_curves.png")
```

**11.2 Analyser Comportement** (2h)

Fichier: `backend/scripts/rl/analyze_behavior.py`

```python
def analyze_learned_policy(agent, env):
    """Analyse les décisions de l'agent."""

    # Scénarios de test
    scenarios = [
        {
            "name": "Driver proche disponible",
            "driver_distance": 2.0,
            "driver_load": 0,
            "booking_priority": 3
        },
        {
            "name": "Driver loin mais urgent",
            "driver_distance": 8.0,
            "driver_load": 0,
            "booking_priority": 5
        },
        {
            "name": "Trade-off charge vs distance",
            "drivers": [
                {"distance": 3.0, "load": 2},  # Proche mais chargé
                {"distance": 6.0, "load": 0}   # Loin mais dispo
            ],
            "booking_priority": 4
        }
    ]

    for scenario in scenarios:
        # Créer état correspondant
        state = create_state_from_scenario(scenario)

        # Obtenir Q-values
        q_values = agent.q_network(torch.tensor(state))
        best_action = q_values.argmax().item()
        best_q = q_values.max().item()

        print(f"\n📋 Scénario: {scenario['name']}")
        print(f"   Action choisie: {best_action}")
        print(f"   Q-value: {best_q:.2f}")
        print(f"   Top 3 actions: {q_values.topk(3)}")
```

#### Livrables Jours 11-12

- ✅ Graphiques de training
- ✅ Analyse comportement
- ✅ Patterns découverts

---

### 🔷 Jour 13 : Tests Complets (Lundi)

#### Objectifs

1. Tests d'intégration DQN + Env
2. Tests de convergence
3. Tests de performance

#### Tâches Détaillées

**13.1 Tests Intégration** (3h)

Fichier: `backend/tests/rl/test_dqn_integration.py`

```python
def test_full_training_loop():
    """Test training complet (10 episodes)."""
    env = DispatchEnv(num_drivers=5, max_bookings=10, simulation_hours=1)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    for episode in range(10):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, next_state, reward, done)
            agent.train_step()
            state = next_state
            episode_reward += reward

        agent.decay_epsilon()
        if episode % 5 == 0:
            agent.update_target_network()

    # Vérifier que l'agent a appris quelque chose
    assert agent.training_step > 0
    assert len(agent.losses) > 0

def test_agent_improves_over_baseline():
    """Test que l'agent s'améliore vs aléatoire."""
    env = DispatchEnv()
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    # Entraîner 50 episodes
    for _ in range(50):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, next_state, reward, done)
            agent.train_step()
            state = next_state

    # Évaluer (greedy, pas d'exploration)
    total_reward = 0
    for _ in range(10):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward

    avg_reward = total_reward / 10

    # Devrait être mieux que random (environ -100)
    assert avg_reward > -100
```

#### Livrables Jour 13

- ✅ Tests intégration (10+ tests)
- ✅ Validation apprentissage
- ✅ Performance vérifiée

---

### 🔷 Jour 14 : Documentation & Rapport (Mardi)

#### Objectifs

1. Documentation complète du code
2. Rapport d'analyse
3. Guide d'utilisation

#### Tâches Détaillées

**14.1 Documentation** (2h)

Fichiers à créer:

- `backend/services/rl/README_DQN.md`
- `session/RL/SEMAINE_15-16_COMPLETE.md`
- `session/RL/RAPPORT_TRAINING_DQN.md`

**14.2 Rapport d'Analyse** (2h)

Contenu:

```markdown
# Rapport Training DQN

## Résultats

### Courbe d'Apprentissage

- Episode 1-200: Exploration (reward -500 à 0)
- Episode 200-600: Apprentissage (reward 0 à +1000)
- Episode 600-1000: Expert (reward +1000 à +1800)

### Comparaison vs Baseline

| Métrique   | Baseline | DQN   | Amélioration |
| ---------- | -------- | ----- | ------------ |
| Reward     | -2500    | +1780 | +171%        |
| Complétion | 10%      | 87%   | +770%        |
| Distance   | 12km     | 6.5km | -46%         |

### Patterns Découverts

1. Privilégie drivers disponibles en période de pic
2. Équilibre charge automatiquement
3. Anticipe les bookings urgents
```

#### Livrables Jour 14

- ✅ Documentation complète
- ✅ Rapport d'analyse
- ✅ README mis à jour

---

## 📊 Récapitulatif des 2 Semaines

### Fichiers à Créer (Total: 12 fichiers)

#### Code (6 fichiers)

1. `backend/services/rl/q_network.py` (150 lignes)
2. `backend/services/rl/replay_buffer.py` (120 lignes)
3. `backend/services/rl/dqn_agent.py` (400 lignes)
4. `backend/scripts/rl/train_dqn.py` (300 lignes)
5. `backend/scripts/rl/evaluate_agent.py` (200 lignes)
6. `backend/scripts/rl/visualize_training.py` (150 lignes)

#### Tests (3 fichiers)

7. `backend/tests/rl/test_q_network.py` (120 lignes)
8. `backend/tests/rl/test_replay_buffer.py` (100 lignes)
9. `backend/tests/rl/test_dqn_integration.py` (150 lignes)

#### Documentation (3 fichiers)

10. `backend/services/rl/README_DQN.md`
11. `session/RL/SEMAINE_15-16_COMPLETE.md`
12. `session/RL/RAPPORT_TRAINING_DQN.md`

**Total:** ~2,000 lignes de code + doc

---

## ✅ Checklist Complète

### Semaine 15

#### Jour 1 - Q-Network

- [ ] Installer PyTorch
- [ ] Créer q_network.py
- [ ] 5 tests Q-Network
- [ ] Validation GPU/CPU

#### Jour 2 - Replay Buffer

- [ ] Créer replay_buffer.py
- [ ] 8 tests Replay Buffer
- [ ] Optimiser performance

#### Jour 3 - Agent Structure

- [ ] Créer dqn_agent.py
- [ ] select_action() + epsilon-greedy
- [ ] 6 tests action selection

#### Jour 4 - Training

- [ ] train_step() implémenté
- [ ] Double DQN
- [ ] 8 tests training

#### Jour 5 - Persistence

- [ ] save/load modèle
- [ ] Système checkpoints
- [ ] 4 tests sérialisation

### Semaine 16

#### Jour 6 - Training Script

- [ ] train_dqn.py créé
- [ ] TensorBoard intégré
- [ ] Fonction évaluation

#### Jour 7 - Premier Training

- [ ] Training 100 episodes
- [ ] Validation courbes
- [ ] Modèle sauvegardé

#### Jours 8-9 - Training Long

- [ ] Training 1000 episodes
- [ ] 10 checkpoints sauvegardés
- [ ] Monitoring continu

#### Jour 10 - Évaluation

- [ ] Script évaluation créé
- [ ] Comparaison vs baseline
- [ ] Rapport généré

#### Jours 11-12 - Visualisation

- [ ] Graphiques training
- [ ] Analyse comportement
- [ ] Patterns découverts

#### Jour 13 - Tests

- [ ] Tests intégration (10+)
- [ ] Validation apprentissage
- [ ] Performance OK

#### Jour 14 - Documentation

- [ ] Doc complète
- [ ] Rapport final
- [ ] Guide utilisation

---

## 🎯 KPIs de Succès

### Objectifs à Atteindre

| Métrique               | Objectif          | Comment Mesurer   |
| ---------------------- | ----------------- | ----------------- |
| **Agent fonctionne**   | ✅                | Tests passent     |
| **Apprend**            | Reward monte      | TensorBoard       |
| **Converge**           | Reward stable     | Episodes 800-1000 |
| **Mieux que baseline** | +100% reward      | Évaluation        |
| **Rapide**             | < 50ms inférence  | Benchmark         |
| **Stable**             | Pas de divergence | Courbes           |

### Métriques de Training

**Episode 100:**

- Reward: -50 à +50
- Epsilon: ~0.60
- Loss: 0.5-0.8

**Episode 500:**

- Reward: +500 à +800
- Epsilon: ~0.15
- Loss: 0.2-0.4

**Episode 1000:**

- Reward: +1500 à +1800
- Epsilon: ~0.01
- Loss: 0.1-0.2

---

## 🛠️ Outils & Technologies

### Stack Technique

```yaml
Deep Learning:
  - PyTorch 2.0+
  - torch.nn (réseaux)
  - torch.optim (optimisation)

Monitoring:
  - TensorBoard (courbes)
  - Matplotlib (graphiques)
  - Pickle (logs)

Infrastructure:
  - Docker + GPU support
  - CUDA 11.8+ (si GPU)
  - Python 3.11

Stockage:
  - Checkpoints: .pth files
  - Logs: TensorBoard format
  - Métriques: Pickle/JSON
```

### Commandes Clés

```bash
# Installation
docker-compose exec api pip install torch tensorboard matplotlib

# Training
docker-compose exec api python scripts/rl/train_dqn.py --episodes 1000

# Monitoring
tensorboard --logdir=data/rl/tensorboard

# Évaluation
docker-compose exec api python scripts/rl/evaluate_agent.py \
    --model data/rl/models/dqn_best.pth

# Tests
docker-compose exec api pytest tests/rl/ -v
```

---

## 📈 Timeline Estimée

### Semaine 15 (5 jours)

| Jour | Tâche           | Temps | Difficulté |
| ---- | --------------- | ----- | ---------- |
| 1    | Q-Network       | 3h    | ⭐⭐       |
| 2    | Replay Buffer   | 3h    | ⭐         |
| 3    | Agent Structure | 4h    | ⭐⭐⭐     |
| 4    | Training        | 4h    | ⭐⭐⭐     |
| 5    | Persistence     | 3h    | ⭐⭐       |

**Total Semaine 15:** ~17h de développement

### Semaine 16 (5 jours + weekend)

| Jour  | Tâche            | Temps | Difficulté       |
| ----- | ---------------- | ----- | ---------------- |
| 6     | Training Script  | 3h    | ⭐⭐             |
| 7     | Premier Training | 3h    | ⭐               |
| 8-9   | Training 1000ep  | 12h   | ⭐ (automatique) |
| 10    | Évaluation       | 3h    | ⭐⭐             |
| 11-12 | Visualisation    | 4h    | ⭐⭐             |
| 13    | Tests            | 3h    | ⭐⭐             |
| 14    | Documentation    | 3h    | ⭐               |

**Total Semaine 16:** ~31h (dont 12h training automatique)

---

## 🎓 Apprentissages Attendus

### Ce Que l'Agent va Apprendre

**Episodes 1-200 (Débutant):**

```
✅ "Driver proche = mieux que driver loin"
✅ "Booking priorité 5 > priorité 1"
✅ "Ne pas laisser expirer les bookings"
✅ "Assigner > ne rien faire"
```

**Episodes 200-600 (Intermédiaire):**

```
✅ "Distance + priorité ensemble = important"
✅ "Équilibrer charge entre chauffeurs"
✅ "Trafic dense = rester proche"
✅ "Garder ressources si pic à venir"
```

**Episodes 600-1000 (Expert):**

```
✅ "Patterns spatio-temporels (lundi 8h30 = pic zone ouest)"
✅ "Optimisation multi-contraintes parfaite"
✅ "Anticipation séquences d'actions"
✅ "Gestion crise (pénurie drivers)"
```

---

## 🚀 Résultat Final Attendu

### Après 1000 Episodes

**Modèle Entraîné:**

```
data/rl/models/dqn_best.pth
  - Taille: ~3MB
  - Paramètres: ~200,000
  - Performance: +1780 reward/episode
  - Taux complétion: 87%
```

**Capacités:**

- ✅ Décisions optimales en < 50ms
- ✅ Généralise à situations nouvelles
- ✅ Équilibre automatique des objectifs
- ✅ Surpasse heuristique de +112%

**Prêt Pour:**

- ✅ A/B Testing en production
- ✅ Auto-tuning (Semaine 17)
- ✅ Feedback loop (Semaine 18)

---

## 📚 Ressources & Références

### Papers Fondateurs

- **DQN Original:** "Playing Atari with Deep RL" (DeepMind, 2013)
- **Double DQN:** "Deep RL with Double Q-learning" (2015)
- **Prioritized Replay:** "Prioritized Experience Replay" (2016)

### Tutoriels Recommandés

- PyTorch DQN Tutorial
- Spinning Up in Deep RL
- Stable Baselines3 Doc

### Code de Référence

- OpenAI Baselines
- Stable Baselines3
- CleanRL

---

## 🎯 Prochaines Étapes Après Semaine 16

### Semaine 17 : Auto-Tuner

- Optuna pour hyperparams
- 50 trials d'optimisation
- Configuration optimale

### Semaine 18 : Feedback Loop

- Collecte expériences production
- Retraining quotidien
- A/B Testing 50/50

### Semaine 19 : Optimisations

- Quantification INT8
- ONNX Runtime
- Déploiement GPU
- Latence < 50ms

---

## 💡 Conseils Pratiques

### Pour Réussir

**1. Commencer Simple**

- Tester avec 10 drivers, 10 bookings
- Training court (100 episodes) d'abord
- Valider que ça fonctionne

**2. Monitorer Activement**

- TensorBoard ouvert en permanence
- Vérifier courbes toutes les 100 episodes
- Sauvegarder souvent

**3. Déboguer Méthodiquement**

- Si reward ne monte pas → vérifier reward function
- Si loss diverge → réduire learning rate
- Si pas d'apprentissage → vérifier epsilon

**4. Être Patient**

- Les 200 premiers episodes = exploration
- L'apprentissage réel commence après
- Convergence finale vers episode 800

---

## 🎊 Conclusion

### Plan Semaine 15-16 : Clair et Actionnable

**14 jours structurés en:**

- 5 jours implémentation (Semaine 15)
- 5 jours entraînement + analyse (Semaine 16)
- Tests et documentation continus

**Résultat Final:**

- ✅ Agent DQN production-ready
- ✅ Modèle entraîné (1000 episodes)
- ✅ Performance validée (+112% vs baseline)
- ✅ Documentation complète
- ✅ Prêt pour production

**Prêt à commencer l'implémentation ?** 🚀

---

_Plan détaillé - Semaines 15-16_  
_Agent DQN avec PyTorch_  
_Généré le 20 octobre 2025_
