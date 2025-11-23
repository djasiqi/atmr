#!/usr/bin/env python3

# Constantes pour éviter les valeurs magiques
STATE_DIM_THRESHOLD = 166
ACTION_DIM_THRESHOLD = 115

"""Résolution des équations pour trouver les paramètres corrects."""


def solve_dimensions():
    """Résout les équations pour trouver drivers et bookings."""
    print("Résolution des équations:")
    print("drivers * 4 + bookings * 4 + 2 = 166")
    print("drivers * bookings + 1 = 115")
    print()

    for drivers in range(1, 20):
        for bookings in range(1, 50):
            state_dim = drivers * 4 + bookings * 4 + 2
            action_dim = drivers * bookings + 1

            if state_dim == STATE_DIM_THRESHOLD and action_dim == ACTION_DIM_THRESHOLD:
                print("🎯 Solution trouvée: drivers={drivers}, bookings={bookings}")
                print("   Vérification: state={state_dim}, actions={action_dim}")
                return drivers, bookings

    print("❌ Aucune solution entière trouvée")

    # Testons quelques combinaisons proches
    print("\nCombinaisons proches:")
    for drivers in [3, 4, 5, 6]:
        for bookings in [19, 20, 21, 22, 23, 24, 25]:
            state_dim = drivers * 4 + bookings * 4 + 2
            action_dim = drivers * bookings + 1
            print(
                (
                    "drivers={drivers}, bookings={bookings} → "
                    "state={state_dim}, actions={action_dim}"
                )
            )
    return None


if __name__ == "__main__":
    solve_dimensions()
