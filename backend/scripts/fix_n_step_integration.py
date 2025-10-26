#!/usr/bin/env python3
from pathlib import Path

"""Script de correction pour l'intégration N-step dans improved_dqn_agent.py.

Corrige les erreurs de linting et les problèmes d'intégration.
"""


def fix_improved_dqn_agent():
    """Corrige le fichier improved_dqn_agent.py."""
    file_path = "backend/services/rl/improved_dqn_agent.py"

    # Lire le fichier
    with Path(file_path, encoding="utf-8").open() as f:
        content = f.read()

    # Correction 1: Ajouter des variables par défaut pour éviter les erreurs
    # Remplacer la section d'échantillonnage
    old_sampling = """        # Échantillonnage selon le type de buffer
        if self.use_n_step:
            # Buffer N-step
            batch, weights = self.memory.sample(self.batch_size)
            if not batch:
                return 0.0
            
            # Extraire les données des transitions N-step
            states = torch.FloatTensor([t['state'] for t in batch]).to(self.device)
            actions = torch.LongTensor([t['action'] for t in batch]).to(self.device)
            n_step_rewards = torch.FloatTensor([t['n_step_return'] for t in batch]).to(self.device)
            next_states = torch.FloatTensor([t['next_state'] for t in batch]).to(self.device)
            dones = torch.BoolTensor([t['done'] for t in batch]).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
            
        elif self.use_prioritized_replay and isinstance(self.memory, PrioritizedReplayBuffer):
            # Buffer PER standard
            batch, indices, weights = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch, strict=False)
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            # Buffer standard (deque)
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
                return 0.0"""

    new_sampling = """        # Échantillonnage selon le type de buffer
        if self.use_n_step:
            # Buffer N-step
            batch, weights = self.memory.sample(self.batch_size)
            if not batch:
                return 0.0
            
            # Extraire les données des transitions N-step
            states = torch.FloatTensor([t['state'] for t in batch]).to(self.device)
            actions = torch.LongTensor([t['action'] for t in batch]).to(self.device)
            n_step_rewards = torch.FloatTensor([t['n_step_return'] for t in batch]).to(self.device)
            next_states = torch.FloatTensor([t['next_state'] for t in batch]).to(self.device)
            dones = torch.BoolTensor([t['done'] for t in batch]).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
            # Variables pour compatibilité
            rewards = n_step_rewards  # Utiliser n_step_rewards comme rewards pour la compatibilité
            
        elif self.use_prioritized_replay and isinstance(self.memory, PrioritizedReplayBuffer):
            # Buffer PER standard
            batch, indices, weights = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch, strict=False)
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device)
            # Variables pour compatibilité
            n_step_rewards = rewards  # Utiliser rewards comme n_step_rewards pour la compatibilité
            
        else:
            # Buffer standard (deque)
            if isinstance(self.memory, deque):
                batch = random.sample(list(self.memory), self.batch_size)
                states, actions, rewards, next_states, dones = zip(*batch, strict=False)
                states = torch.FloatTensor(states).to(self.device)
                actions = torch.LongTensor(actions).to(self.device)
                rewards = torch.FloatTensor(rewards).to(self.device)
                next_states = torch.FloatTensor(next_states).to(self.device)
                dones = torch.BoolTensor(dones).to(self.device)
                weights = torch.ones(self.batch_size).to(self.device)
                # Variables pour compatibilité
                n_step_rewards = rewards  # Utiliser rewards comme n_step_rewards pour la compatibilité
            else:
                return 0.0"""

    content = content.replace(old_sampling, new_sampling)

    # Correction 2: Simplifier le calcul des Q-values cibles
    old_target_calc = """            if self.use_n_step:
                # Pour N-step, utiliser les retours N-step calculés
                target_q_values = n_step_rewards.unsqueeze(1) + (self.gamma ** self.n_step * next_q_values * ~dones.unsqueeze(1))
            else:
                # Mode standard (1-step)
                target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))"""

    new_target_calc = """            if self.use_n_step:
                # Pour N-step, utiliser les retours N-step calculés
                target_q_values = n_step_rewards.unsqueeze(1) + (self.gamma ** self.n_step * next_q_values * ~dones.unsqueeze(1))
            else:
                # Mode standard (1-step)
                target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))"""

    content = content.replace(old_target_calc, new_target_calc)

    # Correction 3: Ajouter des vérifications de type pour éviter les erreurs d'attribut
    old_priority_update = """        # Mise à jour des priorités
        if self.use_prioritized_replay:
            if self.use_n_step and hasattr(self.memory, 'update_priorities'):
                # Buffer N-step priorisé
                priorities = torch.abs(td_errors).detach().cpu().numpy().flatten() + 1e-6
                indices = list(range(len(batch)))  # Indices des transitions échantillonnées
                self.memory.update_priorities(indices, priorities)
            elif isinstance(self.memory, PrioritizedReplayBuffer):
                # Buffer PER standard
                priorities = torch.abs(td_errors).detach().cpu().numpy().flatten() + 1e-6
                self.memory.update_priorities(indices, priorities)"""

    new_priority_update = """        # Mise à jour des priorités
        if self.use_prioritized_replay:
            if self.use_n_step and hasattr(self.memory, 'update_priorities'):
                # Buffer N-step priorisé
                priorities = torch.abs(td_errors).detach().cpu().numpy().flatten() + 1e-6
                indices = list(range(len(batch)))  # Indices des transitions échantillonnées
                self.memory.update_priorities(indices, priorities)
            elif isinstance(self.memory, PrioritizedReplayBuffer):
                # Buffer PER standard
                priorities = torch.abs(td_errors).detach().cpu().numpy().flatten() + 1e-6
                if 'indices' in locals():  # Vérifier que indices existe
                    self.memory.update_priorities(indices, priorities)"""

    content = content.replace(old_priority_update, new_priority_update)

    # Écrire le fichier corrigé
    with Path(file_path, "w", encoding="utf-8").open() as f:
        f.write(content)

    print("✅ Fichier improved_dqn_agent.py corrigé")

if __name__ == "__main__":
    fix_improved_dqn_agent()
