# pyright: reportAttributeAccessIssue=false
"""
Tests pour l'environnement Gym de dispatch.

Teste:
- Création et initialisation
- Reset et step
- Logique d'assignment
- Calcul de rewards
- Episodes complets
"""

import numpy as np
import pytest

from services.rl.dispatch_env import DispatchEnv


class TestDispatchEnvBasics:
    """Tests basiques de l'environnement."""

    def test_env_creation(self):
        """Test création environnement avec paramètres par défaut."""
        env = DispatchEnv()

        assert env.num_drivers == 10
        assert env.max_bookings == 20
        assert env.simulation_hours == 8
        assert env.observation_space.shape[0] == 10 * 4 + 20 * 4 + 2  # 122
        assert env.action_space.n == 10 * 20 + 1  # 201

    def test_env_creation_custom_params(self):
        """Test création avec paramètres custom."""
        env = DispatchEnv(num_drivers=5, max_bookings=10, simulation_hours=4)

        assert env.num_drivers == 5
        assert env.max_bookings == 10
        assert env.simulation_hours == 4
        assert env.observation_space.shape[0] == 5 * 4 + 10 * 4 + 2  # 62
        assert env.action_space.n == 5 * 10 + 1  # 51

    def test_env_reset(self):
        """Test reset de l'environnement."""
        env = DispatchEnv(num_drivers=5, max_bookings=10)
        obs, info = env.reset(seed=42)

        # Vérifier l'observation
        assert obs.shape == env.observation_space.shape
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

        # Vérifier les infos
        assert "episode_stats" in info
        assert "current_time" in info
        assert "active_bookings" in info
        assert "available_drivers" in info

        # Vérifier l'état initial
        assert env.current_time == 0
        assert len(env.drivers) == 5
        assert len(env.bookings) >= 3  # Au moins 3 bookings au départ
        assert all(d["available"] for d in env.drivers)

    def test_env_reset_reproducibility(self):
        """Test que reset avec seed donne des résultats identiques."""
        env1 = DispatchEnv(num_drivers=5, max_bookings=10)
        env2 = DispatchEnv(num_drivers=5, max_bookings=10)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        assert np.array_equal(obs1, obs2)

    def test_observation_bounds(self):
        """Test que l'observation reste dans les limites raisonnables."""
        env = DispatchEnv()
        obs, _ = env.reset(seed=42)

        # Vérifier que pas de NaN ou Inf
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

        # Vérifier que les valeurs normalisées sont dans [-1, 1] ou [0, 1]
        # (certaines peuvent être > 1 si non normalisées, mais pas trop grandes)
        assert np.all(np.abs(obs) < 100)


class TestDispatchEnvActions:
    """Tests des actions et steps."""

    def test_step_wait_action(self):
        """Test action 0 (wait)."""
        env = DispatchEnv(num_drivers=5, max_bookings=10)
        obs, _ = env.reset(seed=42)

        next_obs, reward, terminated, truncated, info = env.step(0)

        # Vérifier les retours
        assert next_obs.shape == obs.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Action wait devrait donner une petite pénalité
        assert reward <= 0

        # Le temps devrait avancer
        assert env.current_time == 5  # 5 minutes par step

    def test_step_valid_assignment(self):
        """Test assignment valide d'un booking à un driver."""
        env = DispatchEnv(num_drivers=5, max_bookings=10)
        _obs, info = env.reset(seed=42)

        # S'assurer qu'il y a au moins un booking
        initial_bookings = info["active_bookings"]
        assert initial_bookings > 0

        # Action 1 = assigner booking 0 à driver 0
        _next_obs, reward, _terminated, _truncated, info = env.step(1)

        # Une assignation valide devrait donner une récompense positive (généralement)
        # (peut être négative si très mauvais assignment, mais c'est ok)
        assert reward != 0  # Au minimum, pas de récompense nulle

        # Le nombre de bookings actifs devrait diminuer ou rester pareil
        # (peut rester pareil si de nouveaux bookings sont générés)
        assert info["active_bookings"] <= initial_bookings + 5  # Max

    def test_step_invalid_action(self):
        """Test action invalide (hors limites)."""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset(seed=42)

        # Action hors limites
        invalid_action = env.action_space.n + 10
        _next_obs, reward, _terminated, _truncated, _info = env.step(invalid_action)

        # Devrait donner une pénalité
        assert reward == -10.0

    def test_step_already_assigned(self):
        """Test assignment d'un booking déjà assigné."""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset(seed=42)

        # Premier assignment
        env.step(1)  # Assigner booking 0 à driver 0

        # Essayer de réassigner le même booking
        _next_obs, reward, _terminated, _truncated, _info = env.step(1)

        # Devrait donner une pénalité pour action invalide
        assert reward <= 0


class TestDispatchEnvRewards:
    """Tests de la fonction de récompense."""

    def test_late_pickup_penalty(self):
        """Test pénalité pour pickup en retard."""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset(seed=42)

        # Créer une situation de retard
        if env.bookings:
            booking = env.bookings[0]
            booking["time_window_end"] = env.current_time + 1  # Fenêtre très courte
            booking["pickup_lat"] = env.center_lat + 0.5  # Loin
            booking["pickup_lon"] = env.center_lon + 0.5

        driver = env.drivers[0]
        driver["lat"] = env.center_lat
        driver["lon"] = env.center_lon

        # Assignment qui causera un retard
        reward = env._assign_booking(driver, booking)

        # Devrait avoir une pénalité (reward < 50)
        # Note: Peut varier selon la distance exacte
        assert reward < 50  # Sans retard, reward de base est 50

    def test_optimal_distance_bonus(self):
        """Test bonus pour distance optimale."""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset(seed=42)

        if env.bookings:
            booking = env.bookings[0]
            booking["time_window_end"] = env.current_time + 60  # Beaucoup de temps
            # Mettre le booking très proche du driver
            booking["pickup_lat"] = env.center_lat + 0.01
            booking["pickup_lon"] = env.center_lon + 0.01

        driver = env.drivers[0]
        driver["lat"] = env.center_lat
        driver["lon"] = env.center_lon

        reward = env._assign_booking(driver, booking)

        # Devrait avoir un bonus (reward > 50)
        assert reward > 50

    def test_high_priority_bonus(self):
        """Test bonus pour priorité haute."""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset(seed=42)

        if env.bookings:
            booking = env.bookings[0]
            booking["priority"] = 5  # Priorité maximale
            booking["time_window_end"] = env.current_time + 60

        driver = env.drivers[0]
        reward = env._assign_booking(driver, booking)

        # Devrait inclure un bonus de priorité
        assert reward >= 50  # Au minimum le reward de base

    def test_booking_expiration_penalty(self):
        """Test pénalité pour bookings expirés."""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset(seed=42)

        if env.bookings:
            # Créer un booking sur le point d'expirer
            booking = env.bookings[0]
            booking["time_remaining"] = 3  # Va expirer au prochain step
            booking["assigned"] = False
            booking["priority"] = 5

        # Avancer le temps sans assigner
        for _ in range(2):
            env.step(0)  # Wait

        # Le booking devrait avoir expiré et causé une pénalité
        assert env.episode_stats["cancellations"] > 0


class TestDispatchEnvEpisode:
    """Tests d'épisodes complets."""

    def test_full_episode_random(self):
        """Test épisode complet avec actions aléatoires."""
        env = DispatchEnv(num_drivers=5, max_bookings=10, simulation_hours=1)
        _obs, _ = env.reset(seed=42)

        total_reward = 0.0
        steps = 0
        terminated = False

        while not terminated and steps < 100:
            action = env.action_space.sample()
            _obs, reward, terminated, _truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        assert steps > 0
        assert terminated or steps == 100
        assert "episode_stats" in info

        print("\n📊 Random Episode Results:")
        print("  Steps: {steps}")
        print("  Total reward: {total_reward")
        print("  Assignments: {info['episode_stats']['assignments']}")
        print("  Cancellations: {info['episode_stats']['cancellations']}")

    def test_full_episode_greedy(self):
        """Test épisode avec stratégie greedy (toujours assigner)."""
        env = DispatchEnv(num_drivers=5, max_bookings=10, simulation_hours=1)
        _obs, _ = env.reset(seed=42)

        total_reward = 0.0
        steps = 0
        terminated = False

        while not terminated and steps < 100:
            # Stratégie simple: toujours prendre action 1 (premier assignment possible)
            action = 1
            _obs, reward, terminated, _truncated, _info = env.step(action)
            total_reward += reward
            steps += 1

        assert steps > 0
        print("\n📊 Greedy Episode Results:")
        print("  Steps: {steps}")
        print("  Total reward: {total_reward")
        print("  Assignments: {info['episode_stats']['assignments']}")

    def test_episode_terminates_correctly(self):
        """Test que l'épisode se termine au bon moment."""
        env = DispatchEnv(num_drivers=5, max_bookings=10, simulation_hours=1)
        env.reset(seed=42)

        terminated = False
        steps = 0

        while not terminated and steps < 200:
            action = env.action_space.sample()
            _obs, _reward, terminated, _truncated, info = env.step(action)
            steps += 1

        # L'épisode devrait se terminer autour de 60 minutes / 5 min par step = 12 steps
        # (mais peut varier légèrement)
        assert 10 <= steps <= 15  # Marge de tolérance
        assert terminated
        assert info["current_time"] >= 60  # Au moins 1 heure écoulée


class TestDispatchEnvHelpers:
    """Tests des fonctions helper."""

    def test_calculate_distance(self):
        """Test calcul de distance haversine."""
        env = DispatchEnv()

        # Distance Genève centre à Genève aéroport (~5km)
        distance = env._calculate_distance(
            46.2044,
            6.1432,  # Centre
            46.2381,
            6.1090,  # Aéroport
        )

        # Devrait être autour de 4-5 km
        assert 4.0 < distance < 6.0

        # Distance nulle (même point)
        distance_zero = env._calculate_distance(46.2044, 6.1432, 46.2044, 6.1432)
        assert distance_zero < 0.0001

    def test_traffic_density_peaks(self):
        """Test que le trafic a des pics aux bonnes heures."""
        env = DispatchEnv()
        env.reset()

        # 8h-9h: pic du matin
        env.current_time = 30  # 8h30
        assert env._get_traffic_density() > 0.7

        # 17h-18h: pic du soir
        env.current_time = 540  # 17h00
        assert env._get_traffic_density() > 0.7

        # 14h: normal
        env.current_time = 360  # 14h00
        assert env._get_traffic_density() < 0.5

    def test_booking_generation_rate_varies(self):
        """Test que le taux de génération varie selon l'heure."""
        env = DispatchEnv()
        env.reset()

        # Pic du matin
        env.current_time = 30
        rate_peak = env._get_booking_generation_rate()

        # Heure creuse
        env.current_time = 360
        rate_off = env._get_booking_generation_rate()

        assert rate_peak > rate_off

    def test_episode_bonus_calculation(self):
        """Test calcul du bonus de fin d'épisode."""
        env = DispatchEnv(num_drivers=5)
        env.reset()

        # Simuler des stats parfaites
        env.episode_stats["assignments"] = 20
        env.episode_stats["cancellations"] = 0
        env.episode_stats["total_distance"] = 80.0  # 4km en moyenne

        # Équilibrer les loads
        for driver in env.drivers:
            driver["load"] = 4  # Tous égaux

        bonus = env._calculate_episode_bonus()

        # Devrait être positif avec de bonnes stats
        assert bonus > 0

        print("\n🎁 Episode bonus: {bonus")


class TestDispatchEnvRender:
    """Tests du rendu."""

    def test_render_human_mode(self):
        """Test render en mode human."""
        env = DispatchEnv(render_mode="human")
        env.reset(seed=42)

        # Ne devrait pas crasher
        env.render()

        # Faire quelques steps et render
        for _ in range(3):
            env.step(env.action_space.sample())
            env.render()

        assert True  # Si on arrive ici, le render fonctionne

    def test_close(self):
        """Test fermeture de l'environnement."""
        env = DispatchEnv()
        env.reset()
        env.close()
        # Ne devrait pas crasher
        assert True


# Test d'intégration complet
def test_realistic_scenario():
    """Test scénario réaliste complet."""
    print("\n" + "=" * 60)
    print("🧪 TEST SCÉNARIO RÉALISTE")
    print("=" * 60)

    env = DispatchEnv(num_drivers=8, max_bookings=15, simulation_hours=2, render_mode="human")

    _obs, info = env.reset(seed=0.123)
    print("\n✅ Environnement initialisé")
    print("  Drivers: {info['available_drivers']}")
    print("  Bookings: {info['active_bookings']}")

    env.render()

    total_reward = 0.0
    steps = 0
    terminated = False

    # Simuler une stratégie simple: nearest driver
    while not terminated and steps < 50:
        # Action aléatoire (à remplacer par une vraie politique plus tard)
        action = env.action_space.sample()

        _obs, reward, terminated, _truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Render tous les 5 steps
        if steps % 5 == 0:
            env.render()

    env.render()  # Final state

    print("\n📊 RÉSULTATS FINAUX:")
    print("  Steps: {steps}")
    print("  Reward total: {total_reward")
    print("  Reward moyen/step: {total_reward/steps")
    print("  Assignments: {info['episode_stats']['assignments']}")
    print("  Retards: {info['episode_stats']['late_pickups']}")
    print("  Annulations: {info['episode_stats']['cancellations']}")
    print("  Distance totale: {info['episode_stats']['total_distance']")

    if info["episode_stats"]["assignments"] > 0:
        (info["episode_stats"]["total_distance"] / info["episode_stats"]["assignments"])
        print("  Distance moyenne: {avg_distance")

    print("=" * 60)

    assert steps > 0
    assert total_reward != 0
