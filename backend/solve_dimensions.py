#!/usr/bin/env python3
"""Résolution des équations pour trouver les paramètres corrects."""

def solve_dimensions():
    """Résout les équations pour trouver drivers et bookings."""

    print("Résolution des équations:")  # noqa: T201
    print("drivers * 4 + bookings * 4 + 2 = 166")  # noqa: T201
    print("drivers * bookings + 1 = 115")  # noqa: T201
    print()  # noqa: T201

    for drivers in range(1, 20):
        for bookings in range(1, 50):
            state_dim = drivers * 4 + bookings * 4 + 2
            action_dim = drivers * bookings + 1

            if state_dim == 166 and action_dim == 115:
                print(f"🎯 Solution trouvée: drivers={drivers}, bookings={bookings}")  # noqa: T201
                print(f"   Vérification: state={state_dim}, actions={action_dim}")  # noqa: T201
                return drivers, bookings

    print("❌ Aucune solution entière trouvée")  # noqa: T201

    # Testons quelques combinaisons proches
    print("\nCombinaisons proches:")  # noqa: T201
    for drivers in [3, 4, 5, 6]:
        for bookings in [19, 20, 21, 22, 23, 24, 25]:
            state_dim = drivers * 4 + bookings * 4 + 2
            action_dim = drivers * bookings + 1
            print(f"drivers={drivers}, bookings={bookings} → state={state_dim}, actions={action_dim}")  # noqa: T201

if __name__ == "__main__":
    solve_dimensions()
