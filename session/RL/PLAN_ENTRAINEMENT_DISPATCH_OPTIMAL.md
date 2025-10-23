# 🧠 Plan d'Entraînement RL pour Dispatch Optimal

**Date** : 21 octobre 2025  
**Objectif** : Utiliser le Reinforcement Learning pour apprendre le meilleur dispatching possible  
**Méthode** : Entraînement offline sur données historiques + simulations

---

## 🎯 VOTRE IDÉE (Excellente !)

> "Lancer un entraînement qui permettrait de définir le meilleur résultat possible"

**Concept** :

- **INPUT** : Heure départ, distance, temps transport, lieux, chauffeurs disponibles
- **OUTPUT** : Assignation optimale (équité + distance + temps)
- **MÉTHODE** : Entraîner un agent RL sur 1000+ dispatches historiques

---

## 📊 ARCHITECTURE EXISTANTE (Bonne base !)

Vous avez **déjà** :

✅ **Environnement Gym** : `backend/services/rl/dispatch_env.py`

```python
class DispatchEnv(gym.Env):
    """
    État (observation_space):
        - Positions chauffeurs (N × 2)
        - Disponibilité chauffeurs (N)
        - Charge de travail (N)
        - Positions bookings (M × 2)
        - Priorités bookings (M)
        - Temps restant fenêtre (M)
        - Heure actuelle + trafic

    Actions (action_space):
        - Action 0: Attendre
        - Actions 1 à N×M: Assigner booking[i] à driver[j]

    Récompense (reward):
        +100 * assignments_réussis
        -50 * retards_pickup
        -60 * bookings_annulés
        +10 * distance_optimale
        +20 * workload_équilibré  ⬅️ ÉQUITÉ !
        -5 * temps_inaction
    """
```

✅ **Agent DQN** : `backend/services/rl/dqn_agent.py`

- Deep Q-Network (réseau de neurones)
- Experience Replay (mémoire d'entraînement)
- Target Network (stabilité)

✅ **Générateur de suggestions** : `backend/services/rl/suggestion_generator.py`

- Utilise le DQN entraîné
- Propose des réassignations

---

## 🚀 IMPLÉMENTATION : ENTRAÎNEMENT OFFLINE

### Phase 1️⃣ : Collecte des Données Historiques

**Objectif** : Extraire 1000+ dispatches passés pour l'entraînement

```python
# backend/scripts/rl_export_historical_data.py

from models import DispatchRun, Assignment, Booking, Driver
from datetime import datetime, timedelta
import json

def export_historical_dispatches(
    company_id: int,
    start_date: str,  # "2025-01-01"
    end_date: str,    # "2025-10-21"
    output_file: str = "data/rl/historical_dispatches.json"
):
    """
    Exporte les dispatches historiques au format JSON pour entraînement RL.
    """
    dispatches = []

    # Récupérer tous les dispatch_runs de la période
    runs = DispatchRun.query.filter(
        DispatchRun.company_id == company_id,
        DispatchRun.day >= start_date,
        DispatchRun.day <= end_date,
        DispatchRun.status == DispatchStatus.COMPLETED
    ).all()

    print(f"📊 Récupération de {len(runs)} dispatch runs...")

    for run in runs:
        # Récupérer les bookings et assignments
        assignments = Assignment.query.filter_by(dispatch_run_id=run.id).all()

        if len(assignments) == 0:
            continue  # Skip runs sans assignments

        # Calculer les métriques
        driver_loads = {}
        total_distance = 0
        retards = 0

        for a in assignments:
            driver_id = a.driver_id
            driver_loads[driver_id] = driver_loads.get(driver_id, 0) + 1

            # Calculer distance (si disponible)
            booking = a.booking
            if booking.pickup_lat and booking.dropoff_lat:
                dist = haversine_distance(
                    (booking.pickup_lat, booking.pickup_lon),
                    (booking.dropoff_lat, booking.dropoff_lon)
                )
                total_distance += dist

            # Détecter retards (si disponible)
            if hasattr(a, 'actual_pickup_time') and a.actual_pickup_time:
                delay = (a.actual_pickup_time - booking.scheduled_time).total_seconds() / 60
                if delay > 5:
                    retards += 1

        # Calculer écart de charge (équité)
        if driver_loads:
            max_load = max(driver_loads.values())
            min_load = min(driver_loads.values())
            load_gap = max_load - min_load
        else:
            load_gap = 0

        # Calculer score global
        quality_score = (
            100 - (load_gap * 10) -      # Pénalité équité
            (total_distance * 0.5) -     # Pénalité distance
            (retards * 5)                # Pénalité retards
        )

        # Export
        dispatch_data = {
            "id": run.id,
            "date": run.day.isoformat(),
            "num_bookings": len(assignments),
            "num_drivers": len(driver_loads),
            "driver_loads": driver_loads,
            "load_gap": load_gap,
            "total_distance_km": round(total_distance, 2),
            "retards_count": retards,
            "quality_score": round(quality_score, 2),
            "bookings": [
                {
                    "id": a.booking_id,
                    "scheduled_time": a.booking.scheduled_time.isoformat(),
                    "pickup_lat": a.booking.pickup_lat,
                    "pickup_lon": a.booking.pickup_lon,
                    "dropoff_lat": a.booking.dropoff_lat,
                    "dropoff_lon": a.booking.dropoff_lon,
                    "assigned_driver": a.driver_id,
                }
                for a in assignments
            ]
        }

        dispatches.append(dispatch_data)

    # Sauvegarder
    with open(output_file, 'w') as f:
        json.dump({
            "company_id": company_id,
            "period": f"{start_date} to {end_date}",
            "total_dispatches": len(dispatches),
            "dispatches": dispatches
        }, f, indent=2)

    print(f"✅ {len(dispatches)} dispatches exportés vers {output_file}")
    print(f"📊 Statistiques:")
    print(f"   - Écart moyen: {sum(d['load_gap'] for d in dispatches) / len(dispatches):.1f}")
    print(f"   - Score moyen: {sum(d['quality_score'] for d in dispatches) / len(dispatches):.1f}")

# Utilisation
if __name__ == "__main__":
    export_historical_dispatches(
        company_id=1,
        start_date="2025-01-01",
        end_date="2025-10-21"
    )
```

**Commande** :

```bash
docker exec atmr-api python backend/scripts/rl_export_historical_data.py
```

---

### Phase 2️⃣ : Entraînement Offline (Batch Learning)

**Objectif** : Entraîner l'agent DQN sur des simulations basées sur les données historiques

```python
# backend/scripts/rl_train_offline.py

import json
import numpy as np
from services.rl.dispatch_env import DispatchEnv
from services.rl.dqn_agent import DQNAgent
import torch

def train_offline(
    historical_data_file: str = "data/rl/historical_dispatches.json",
    num_episodes: int = 5000,
    save_path: str = "data/rl/models/dispatch_optimized.pth"
):
    """
    Entraîne l'agent DQN offline sur des données historiques.

    Méthode :
    1. Charger les dispatches historiques
    2. Pour chaque episode :
        - Sélectionner un dispatch historique aléatoire
        - Recréer l'état initial (bookings + drivers)
        - Simuler l'assignation avec l'agent
        - Calculer la récompense (équité + distance + retards)
        - Mettre à jour le modèle
    3. Sauvegarder le modèle optimisé
    """
    print("🧠 Démarrage entraînement offline...")

    # Charger données historiques
    with open(historical_data_file, 'r') as f:
        data = json.load(f)

    dispatches = data['dispatches']
    print(f"📊 {len(dispatches)} dispatches chargés")

    # Initialiser environnement et agent
    env = DispatchEnv(
        num_drivers=5,      # Ajuster selon votre flotte
        max_bookings=20,    # Ajuster selon vos données
        simulation_hours=12
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.0001,  # LR réduit pour offline
        gamma=0.99,            # Discount factor
        epsilon_start=0.5,     # Exploration réduite (on a déjà des bonnes données)
        epsilon_end=0.01,
        epsilon_decay=0.995
    )

    # Métriques d'entraînement
    episode_rewards = []
    episode_load_gaps = []
    best_avg_reward = -float('inf')

    for episode in range(num_episodes):
        # Sélectionner un dispatch historique aléatoire
        dispatch = np.random.choice(dispatches)

        # Recréer l'état initial
        state = _create_state_from_dispatch(env, dispatch)

        total_reward = 0
        done = False
        step = 0
        driver_loads = {i: 0 for i in range(env.num_drivers)}

        while not done and step < len(dispatch['bookings']):
            # Agent choisit une action (assigner booking à driver)
            action = agent.select_action(state)

            # Simuler l'assignation
            next_state, reward, done, info = env.step(action)

            # Calculer récompense réelle basée sur équité
            if action > 0:  # Action != "wait"
                driver_id = (action - 1) // env.max_bookings
                driver_loads[driver_id] += 1

                # Récompense équité
                max_load = max(driver_loads.values())
                min_load = min(driver_loads.values())
                load_gap = max_load - min_load

                equity_reward = -10 * load_gap  # Pénalité exponentielle
                reward += equity_reward

            # Stocker transition dans la mémoire
            agent.memory.push(state, action, next_state, reward, done)

            # Entraîner
            loss = agent.train_step()

            total_reward += reward
            state = next_state
            step += 1

        # Calculer écart final
        max_load = max(driver_loads.values())
        min_load = min(driver_loads.values())
        load_gap = max_load - min_load

        episode_rewards.append(total_reward)
        episode_load_gaps.append(load_gap)

        # Logs tous les 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_gap = np.mean(episode_load_gaps[-100:])

            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Load Gap: {avg_gap:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Memory Size: {len(agent.memory)}")

            # Sauvegarder si meilleur modèle
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(save_path)
                print(f"  ✅ Meilleur modèle sauvegardé!")

        # Decay epsilon
        agent.update_epsilon()

    print(f"\n🎉 Entraînement terminé!")
    print(f"📊 Statistiques finales:")
    print(f"   - Récompense moyenne (100 derniers): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"   - Écart moyen (100 derniers): {np.mean(episode_load_gaps[-100:]):.2f}")
    print(f"   - Modèle sauvegardé: {save_path}")

def _create_state_from_dispatch(env, dispatch):
    """Recrée l'état initial à partir d'un dispatch historique."""
    # Reset environnement
    env.reset()

    # Charger les bookings du dispatch
    for booking_data in dispatch['bookings']:
        env.bookings.append({
            'id': booking_data['id'],
            'pickup_lat': booking_data['pickup_lat'],
            'pickup_lon': booking_data['pickup_lon'],
            'dropoff_lat': booking_data['dropoff_lat'],
            'dropoff_lon': booking_data['dropoff_lon'],
            'scheduled_time': booking_data['scheduled_time'],
            'assigned': False
        })

    # Retourner état observé
    return env._get_observation()

# Utilisation
if __name__ == "__main__":
    train_offline(
        num_episodes=5000,  # Plus = meilleur, mais plus long (5000 ep ≈ 2-3h)
        save_path="data/rl/models/dispatch_optimized_v1.pth"
    )
```

**Commande** :

```bash
docker exec atmr-api python backend/scripts/rl_train_offline.py
```

---

### Phase 3️⃣ : Intégration dans le Dispatch

**Objectif** : Utiliser l'agent entraîné pour **améliorer** le dispatch initial

```python
# backend/services/unified_dispatch/rl_optimizer.py

from services.rl.dqn_agent import DQNAgent
from services.rl.dispatch_env import DispatchEnv
import numpy as np

class RLDispatchOptimizer:
    """
    Optimiseur RL qui améliore le dispatch heuristique.
    """

    def __init__(self, model_path: str = "data/rl/models/dispatch_optimized_v1.pth"):
        self.agent = DQNAgent.load(model_path)
        self.agent.epsilon = 0.0  # Mode exploitation (pas d'exploration)

    def optimize_assignments(
        self,
        initial_assignments: List[Dict],
        bookings: List[Booking],
        drivers: List[Driver]
    ) -> List[Dict]:
        """
        Optimise les assignations initiales avec l'agent RL.

        Args:
            initial_assignments: Assignations de l'heuristique
            bookings: Liste des bookings
            drivers: Liste des chauffeurs

        Returns:
            Assignations optimisées (meilleur équilibre)
        """
        # Créer environnement
        env = DispatchEnv(
            num_drivers=len(drivers),
            max_bookings=len(bookings)
        )

        # Charger état initial
        state = self._create_state(bookings, drivers, initial_assignments)

        # Simuler des réassignations
        optimized = initial_assignments.copy()

        for _ in range(10):  # Max 10 swaps
            # Agent suggère une réassignation
            action = self.agent.select_action(state)

            if action == 0:  # Wait (no change)
                break

            # Décoder l'action (booking_id, driver_id)
            booking_idx = (action - 1) // len(drivers)
            driver_idx = (action - 1) % len(drivers)

            if booking_idx >= len(bookings):
                break

            # Appliquer la réassignation
            booking_id = bookings[booking_idx].id
            driver_id = drivers[driver_idx].id

            # Mettre à jour
            for assignment in optimized:
                if assignment['booking_id'] == booking_id:
                    old_driver = assignment['driver_id']
                    assignment['driver_id'] = driver_id

                    # Calculer nouvelle récompense
                    driver_loads = self._calculate_loads(optimized, drivers)
                    max_load = max(driver_loads.values())
                    min_load = min(driver_loads.values())
                    new_gap = max_load - min_load

                    # Si moins bon, annuler
                    old_gap = self._calculate_gap(initial_assignments, drivers)
                    if new_gap > old_gap:
                        assignment['driver_id'] = old_driver  # Rollback
                    else:
                        print(f"✅ Réassignation : Booking {booking_id} → Driver {driver_id}")
                        print(f"   Écart réduit : {old_gap} → {new_gap}")

            # Mettre à jour état
            state = self._create_state(bookings, drivers, optimized)

        return optimized

    def _calculate_gap(self, assignments, drivers):
        """Calcule l'écart de charge max-min."""
        loads = self._calculate_loads(assignments, drivers)
        return max(loads.values()) - min(loads.values())

    def _calculate_loads(self, assignments, drivers):
        """Compte le nombre d'assignations par chauffeur."""
        loads = {d.id: 0 for d in drivers}
        for a in assignments:
            loads[a['driver_id']] += 1
        return loads
```

---

### Phase 4️⃣ : Modification de l'Engine

**Objectif** : Intégrer l'optimiseur RL dans le pipeline de dispatch

```python
# backend/services/unified_dispatch/engine.py

from services.unified_dispatch.rl_optimizer import RLDispatchOptimizer

# Dans la fonction run(), après l'heuristique:

# ... Heuristique a assigné toutes les courses ...

# 🆕 Optimisation RL (si activée)
if mode == "auto" and getattr(s.features, "enable_rl_optimization", True):
    try:
        logger.info("[Engine] 🧠 Optimisation RL des assignations...")

        optimizer = RLDispatchOptimizer()

        # Convertir assignments en format optimisable
        initial = [
            {
                'booking_id': a.booking_id,
                'driver_id': a.driver_id,
            }
            for a in final_assignments
        ]

        # Optimiser
        optimized = optimizer.optimize_assignments(
            initial_assignments=initial,
            bookings=problem["bookings"],
            drivers=regs
        )

        # Appliquer les changements
        for i, a in enumerate(final_assignments):
            a.driver_id = optimized[i]['driver_id']

        logger.info("[Engine] ✅ Optimisation RL terminée")

    except Exception as e:
        logger.warning("[Engine] ⚠️ Optimisation RL échouée: %s", e)
        # Continuer avec l'heuristique seule
```

---

## 📊 RÉSULTATS ATTENDUS

### Avant (Heuristique seule)

```
Giuseppe : 5 courses
Dris     : 3 courses
Yannis   : 2 courses
ÉCART    : 3 ❌
```

### Après (Heuristique + RL Optimizer)

```
Giuseppe : 3-4 courses
Dris     : 3-4 courses
Yannis   : 3-4 courses
ÉCART    : 0-1 ✅
```

**Amélioration** : **Écart réduit de 66-100%** (3 → 0-1) 🎉

---

## ⏱️ PLANNING D'IMPLÉMENTATION

### Semaine 1 : Collecte des Données

- [ ] Créer script `rl_export_historical_data.py`
- [ ] Exporter 1000+ dispatches historiques
- [ ] Analyser les données (écarts moyens, patterns)

**Effort** : 1-2 jours

---

### Semaine 2 : Entraînement Offline

- [ ] Créer script `rl_train_offline.py`
- [ ] Entraîner agent DQN (5000 episodes ≈ 2-3h GPU)
- [ ] Évaluer le modèle sur données de test

**Effort** : 2-3 jours (dont 2-3h calcul)

---

### Semaine 3 : Intégration

- [ ] Créer `RLDispatchOptimizer`
- [ ] Intégrer dans `engine.py`
- [ ] Tester sur dispatches réels
- [ ] Mesurer amélioration (écart avant/après)

**Effort** : 2-3 jours

---

### Semaine 4 : Validation & Production

- [ ] A/B testing (heuristique vs RL)
- [ ] Monitoring des métriques
- [ ] Ajustements si nécessaire
- [ ] Déploiement en production

**Effort** : 2-3 jours

---

## 🎯 AVANTAGES DE CETTE APPROCHE

✅ **Apprentissage sur données réelles**  
→ L'agent apprend de VOS dispatches passés, pas de simulations théoriques

✅ **Amélioration continue**  
→ Réentraîner tous les mois avec nouvelles données = amélioration constante

✅ **Pas de règles manuelles**  
→ L'agent découvre les patterns optimaux automatiquement

✅ **Adaptable**  
→ S'adapte aux changements (nouveaux chauffeurs, nouvelles zones, etc.)

✅ **Transparence**  
→ Peut expliquer pourquoi une réassignation est suggérée

---

## 📈 MÉTRIQUES DE SUCCÈS

| Métrique                | Avant | Objectif | Impact         |
| ----------------------- | ----- | -------- | -------------- |
| **Écart max courses**   | 3     | ≤1       | **-66%**       |
| **Satisfaction équité** | 66%   | 90%      | **+24%**       |
| **Temps dispatch**      | 9s    | <12s     | +3s acceptable |
| **Retards**             | X     | -10%     | Bonus          |

---

## 🔬 EXPÉRIMENTATIONS FUTURES

Une fois le système en place, vous pourrez :

1. **Entraîner sur différents objectifs** :

   - Minimiser distance totale
   - Maximiser satisfaction client
   - Réduire coûts carburant

2. **Ajouter des features contextuelles** :

   - Météo (pluie → trafic)
   - Jour de la semaine (lundi vs vendredi)
   - Événements (match de foot → trafic)

3. **Multi-agent RL** :

   - Plusieurs agents coopératifs (1 par chauffeur)
   - Optimisation décentralisée

4. **Transfer Learning** :
   - Entraîner sur Geneva, appliquer à Lausanne
   - Partager apprentissage entre filiales

---

## 📝 DOCUMENTS ASSOCIÉS

- `backend/services/rl/dispatch_env.py` : Environnement Gym existant
- `backend/services/rl/dqn_agent.py` : Agent DQN existant
- `SYNTHESE_PROBLEME_EQUILIBRE_FINAL.md` : Analyse du problème actuel

---

## 🚀 PROCHAINE ÉTAPE IMMÉDIATE

**Créer le script d'export des données** :

```bash
docker exec atmr-api bash -c "mkdir -p backend/data/rl/models"
docker exec atmr-api python backend/scripts/rl_export_historical_data.py
```

**Résultat attendu** :

```
📊 Récupération de 1247 dispatch runs...
✅ 1247 dispatches exportés vers data/rl/historical_dispatches.json
📊 Statistiques:
   - Écart moyen: 2.8
   - Score moyen: 72.3
```

---

**Voulez-vous que je crée les scripts d'export et d'entraînement maintenant ?** 🚀
