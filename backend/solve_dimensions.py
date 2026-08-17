#!/usr/bin/env python3
"""Résolution des équations pour trouver les paramètres corrects."""

from __future__ import annotations

# Constantes pour éviter les valeurs magiques
STATE_DIM_THRESHOLD = 166
ACTION_DIM_THRESHOLD = 115
MAX_DRIVERS_SEARCH = 20
MAX_BOOKINGS_SEARCH = 50
NEARBY_DRIVERS = (3, 4, 5, 6)
NEARBY_BOOKINGS = (19, 20, 21, 22, 23, 24, 25)


def compute_dims(drivers: int, bookings: int) -> tuple[int, int]:
    """Retourne (state_dim, action_dim) pour un couple drivers/bookings."""
    state_dim = drivers * 4 + bookings * 4 + 2
    action_dim = drivers * bookings + 1
    return state_dim, action_dim


def format_nearby_combinations(
    drivers_values: tuple[int, ...] = NEARBY_DRIVERS,
    bookings_values: tuple[int, ...] = NEARBY_BOOKINGS,
) -> str:
    """Liste quelques combinaisons proches (quand aucune solution exacte)."""
    lines = ["", "Combinaisons proches:"]
    for drivers in drivers_values:
        for bookings in bookings_values:
            state_dim, action_dim = compute_dims(drivers, bookings)
            lines.append(
                f"drivers={drivers}, bookings={bookings} → "
                f"state={state_dim}, actions={action_dim}"
            )
    return "\n".join(lines)


def solve_dimensions(
    *,
    state_target: int = STATE_DIM_THRESHOLD,
    action_target: int = ACTION_DIM_THRESHOLD,
    max_drivers: int = MAX_DRIVERS_SEARCH,
    max_bookings: int = MAX_BOOKINGS_SEARCH,
) -> tuple[int, int] | None:
    """Résout les équations pour trouver drivers et bookings."""
    print("Résolution des équations:")
    print("drivers * 4 + bookings * 4 + 2 = 166")
    print("drivers * bookings + 1 = 115")
    print()

    for drivers in range(1, max_drivers):
        for bookings in range(1, max_bookings):
            state_dim, action_dim = compute_dims(drivers, bookings)
            if state_dim == state_target and action_dim == action_target:
                print(f"🎯 Solution trouvée: drivers={drivers}, bookings={bookings}")
                print(f"   Vérification: state={state_dim}, actions={action_dim}")
                return drivers, bookings

    print("❌ Aucune solution entière trouvée")
    print(format_nearby_combinations())
    return None


def main() -> None:
    solve_dimensions()


if __name__ == "__main__":
    main()
