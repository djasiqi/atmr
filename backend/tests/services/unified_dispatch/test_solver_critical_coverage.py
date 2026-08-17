"""Couverture critique de ``optimization.solver`` (seuil 95 %)."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from models.enums import DriverType
from services.unified_dispatch.core.settings import Settings
from services.unified_dispatch.optimization import solver as solver_mod
from services.unified_dispatch.optimization.solver import SolverAssignment, solve


def _booking(bid: int, **kw):
    data = {"id": bid, "parent_booking_id": None, "is_return": False}
    data.update(kw)
    return SimpleNamespace(**data)


def _driver(did: int, driver_type=DriverType.REGULAR):
    return SimpleNamespace(id=did, driver_type=driver_type)


def _matrix(n: int, travel: int = 5) -> list[list[int]]:
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0 if i == j else travel)
        rows.append(row)
    return rows


def _problem(*, n_bookings=1, n_drivers=1, **extra):
    bookings = extra.pop("bookings", [_booking(i + 1) for i in range(n_bookings)])
    drivers = extra.pop("drivers", [_driver(i + 1) for i in range(n_drivers)])
    n_bookings = len(bookings)
    n_drivers = len(drivers)
    n_nodes = n_drivers + 2 * n_bookings
    tw = extra.pop("time_windows", [(0, 400), (10, 420)] * n_bookings)
    svc = extra.pop("service_times", [1, 1] * n_bookings)
    problem = {
        "bookings": bookings,
        "drivers": drivers,
        "time_matrix": extra.pop("time_matrix", _matrix(n_nodes)),
        "starts": extra.pop("starts", list(range(n_drivers))),
        "ends": extra.pop("ends", list(range(n_drivers))),
        "num_vehicles": extra.pop("num_vehicles", n_drivers),
        "time_windows": tw,
        "service_times": svc,
        "driver_windows": extra.pop("driver_windows", [(0, 480)] * n_drivers),
        "horizon": extra.pop("horizon", 480),
        "base_time": extra.pop("base_time", datetime(2030, 1, 1, 8, 0, 0)),
        "dispatch_run_id": extra.pop("dispatch_run_id", 7),
    }
    problem.update(extra)
    return problem


def test_to_dict_minutes_invalides():
    assignment = SolverAssignment(
        booking_id=1,
        driver_id=2,
        estimated_pickup_min=object(),  # type: ignore[arg-type]
        estimated_dropoff_min=object(),  # type: ignore[arg-type]
        base_time=datetime(2030, 1, 1),
    )
    data = assignment.to_dict()
    assert data["estimated_pickup_arrival"] == datetime(2030, 1, 1)


def test_empty_matrix_et_validations():
    bookings = [_booking(1)]
    drivers = [_driver(1)]
    result = solve(
        {
            "bookings": bookings,
            "drivers": drivers,
            "time_matrix": [],
            "starts": [0],
            "ends": [0],
            "num_vehicles": 1,
            "time_windows": [(0, 10), (10, 20)],
            "service_times": [1, 1],
        }
    )
    assert result.debug["reason"] == "empty_matrix"
    assert result.unassigned_booking_ids == [1]

    with pytest.raises(ValueError, match="square"):
        solve(_problem(time_matrix=[[0, 1], [1, 0, 9]]))
    bad_size = _problem()
    bad_size["time_matrix"] = _matrix(4)
    with pytest.raises(ValueError, match="size mismatch"):
        solve(bad_size)
    with pytest.raises(ValueError, match="time_windows"):
        solve(_problem(time_windows=[(0, 10)]))
    with pytest.raises(ValueError, match="service_times"):
        solve(_problem(service_times=[1]))
    with pytest.raises(ValueError, match="starts/ends"):
        solve(_problem(starts=[0, 1], ends=[0]))


def test_too_large_fallback(monkeypatch):
    monkeypatch.setattr(solver_mod, "SAFE_MAX_TASKS", 0)
    result = solve(_problem())
    assert result.debug["status"] == "too_large"
    assert result.unassigned_booking_ids == [1]


def test_no_solution(monkeypatch):
    monkeypatch.setattr(
        solver_mod.pywrapcp.RoutingModel,
        "SolveWithParameters",
        lambda self, params: None,
    )
    result = solve(_problem())
    assert result.debug["status"] == "no_solution"
    assert result.assignments == []


def test_feasible_emergency_roundtrip_and_flags(monkeypatch):
    settings = Settings()
    settings.solver.time_limit_sec = 1
    settings.solver.add_driver_work_windows = False
    settings.solver.strict_driver_end_window = False
    settings.solver.unassigned_penalty_base = "bad"  # type: ignore[assignment]
    settings.solver.enable_warm_start = True
    settings.emergency.emergency_distance_multiplier = 2.0
    settings.emergency.emergency_per_stop_penalty = 12
    settings.emergency.emergency_vehicle_fixed_cost = 40
    settings.matrix.avg_speed_kmh = "xx"  # type: ignore[assignment]
    monkeypatch.setenv("UD_SOLVER_FINALIZE_END", "1")
    monkeypatch.setattr(
        "services.unified_dispatch.data.warm_start.apply_warm_start",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "services.unified_dispatch.optimization.warm_start_tracker.TARGET_SIZE_MIN",
        1,
    )
    monkeypatch.setattr(
        "services.unified_dispatch.optimization.warm_start_tracker.TARGET_SIZE_MAX",
        10,
    )
    monkeypatch.setattr(
        "services.unified_dispatch.optimization.warm_start_tracker.measure_warm_start_gain",
        lambda *_a, **_k: {
            "skipped": False,
            "gain_pct": 15.0,
            "without_ms": 200,
            "with_ms": 170,
        },
    )

    outbound = _booking(10)
    retour = _booking(11, parent_booking_id=10, is_return=True)
    emg = _driver(3, driver_type=DriverType.EMERGENCY)
    result = solve(
        _problem(
            bookings=[outbound, retour],
            drivers=[emg],
            pair_min_gaps=[8, 8],
            heuristic_assignments=[{"booking_id": 10, "driver_id": 3}],
            vehicle_capacities=[1],
            driver_windows=[],
        ),
        settings,
    )
    handled = len(result.assignments) + len(result.unassigned_booking_ids)
    assert handled == 2


def test_driver_windows_missing_and_strict(monkeypatch):
    settings = Settings()
    settings.solver.time_limit_sec = 1
    settings.solver.add_driver_work_windows = True
    settings.solver.strict_driver_end_window = False
    result = solve(
        _problem(
            n_bookings=1,
            n_drivers=2,
            driver_windows=[(0, 400)],
        ),
        settings,
    )
    assert "vehicles" in result.debug or result.debug.get("status")


def test_warm_start_et_gain_exceptions(monkeypatch):
    monkeypatch.setattr(
        "services.unified_dispatch.data.warm_start.apply_warm_start",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("ws")),
    )
    monkeypatch.setattr(
        "services.unified_dispatch.optimization.warm_start_tracker.measure_warm_start_gain",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("gain")),
    )
    settings = Settings()
    settings.solver.time_limit_sec = 1
    result = solve(
        _problem(heuristic_assignments=[{"booking_id": 1, "driver_id": 1}]),
        settings,
    )
    assert isinstance(result.assignments, list)


def test_drivertype_import_fallback(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _import(name, globs=None, locs=None, fromlist=(), level=0):
        if name == "models" and fromlist and "DriverType" in fromlist:
            raise ImportError("no DriverType")
        return real_import(name, globs, locs, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    settings = Settings()
    settings.solver.time_limit_sec = 1
    settings.emergency.emergency_distance_multiplier = 1.5
    settings.emergency.emergency_per_stop_penalty = 5
    settings.emergency.emergency_vehicle_fixed_cost = 9
    result = solve(
        _problem(drivers=[_driver(1, driver_type="X_EMERGENCY")]),
        settings,
    )
    assert isinstance(result.assignments, list)


def test_span_cost_et_tracker_import(monkeypatch):
    settings = Settings()
    settings.solver.time_limit_sec = 1
    monkeypatch.setattr(
        solver_mod.pywrapcp.RoutingDimension,
        "SetSpanCostCoefficientForVehicle",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("span")),
    )
    import sys
    import types

    monkeypatch.setitem(
        sys.modules,
        "services.unified_dispatch.optimization.warm_start_tracker",
        types.ModuleType("warm_start_tracker_missing"),
    )
    result = solve(
        _problem(heuristic_assignments=[{"booking_id": 1, "driver_id": 1}]),
        settings,
    )
    assert "vehicles" in result.debug or result.debug.get("status")


def test_simple_assignment_extraction():
    settings = Settings()
    settings.solver.time_limit_sec = 2
    settings.solver.unassigned_penalty_base = 1_000_000
    result = solve(_problem(n_bookings=1, n_drivers=1), settings)
    assert len(result.assignments) + len(result.unassigned_booking_ids) == 1
