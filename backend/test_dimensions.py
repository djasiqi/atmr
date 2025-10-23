#!/usr/bin/env python3
"""Test des dimensions de l'environnement."""

from services.rl.dispatch_env import DispatchEnv


def test_dimensions():
    """Test différentes combinaisons de paramètres."""

    print("Test des dimensions de l'environnement:")  # noqa: T201
    print("Recherche de drivers=X, bookings=Y qui donnent state=166, actions=115")  # noqa: T201
    print()  # noqa: T201

    for drivers in [3, 4, 5]:
        for bookings in [19, 20, 21, 22, 23, 24, 25]:
            try:
                env = DispatchEnv(num_drivers=drivers, max_bookings=bookings)
                obs, _ = env.reset()
                state_dim = obs.shape[0]
                action_dim = env.action_space.n

                print(f"drivers={drivers}, bookings={bookings} → state={state_dim}, actions={action_dim}")  # noqa: T201

                if state_dim == 166 and action_dim == 115:
                    print(f"🎯 TROUVÉ! drivers={drivers}, bookings={bookings}")  # noqa: T201
                    return drivers, bookings

            except Exception as e:
                print(f"❌ Erreur avec drivers={drivers}, bookings={bookings}: {e}")  # noqa: T201

    print("❌ Aucune combinaison trouvée")  # noqa: T201
    return None, None

if __name__ == "__main__":
    test_dimensions()
