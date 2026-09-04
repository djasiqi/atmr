"""Couverture de ``solve_dimensions`` (recherche drivers/bookings)."""

from __future__ import annotations

from solve_dimensions import (
    ACTION_DIM_THRESHOLD,
    STATE_DIM_THRESHOLD,
    compute_dims,
    format_nearby_combinations,
    main,
    solve_dimensions,
)


def test_compute_dims_et_solution_connue(capsys):
    assert compute_dims(3, 38) == (STATE_DIM_THRESHOLD, ACTION_DIM_THRESHOLD)
    found = solve_dimensions()
    assert found == (3, 38)
    out = capsys.readouterr().out
    assert "Solution trouvée" in out
    assert "drivers=3" in out
    assert "bookings=38" in out


def test_aucune_solution_et_main(capsys):
    nearby = format_nearby_combinations(drivers_values=(3,), bookings_values=(19,))
    assert "drivers=3, bookings=19" in nearby

    missing = solve_dimensions(
        state_target=1, action_target=1, max_drivers=3, max_bookings=3
    )
    assert missing is None
    out = capsys.readouterr().out
    assert "Aucune solution entière" in out
    assert "Combinaisons proches" in out

    main()
    assert "Solution trouvée" in capsys.readouterr().out
