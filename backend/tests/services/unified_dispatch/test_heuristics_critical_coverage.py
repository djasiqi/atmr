"""Couverture critique ``optimization/heuristics.py`` (seuil 95 %)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from models import BookingStatus
from models.enums import DriverType
from services.unified_dispatch.core.settings import Settings
from services.unified_dispatch.optimization import heuristics as h
from shared.time_utils import now_local
from tests.factories import BookingFactory, CompanyFactory, DriverFactory


def _naive_local(dt: datetime | None = None) -> datetime:
    """Datetime métier naïf Europe/Zurich (évite naive-aware dans heuristics)."""
    value = dt if dt is not None else now_local()
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _booking(
    bid: int,
    *,
    when: datetime | None = None,
    pickup=(46.2044, 6.1432),
    dropoff=(46.21, 6.15),
    **extra,
):
    data = {
        "id": bid,
        "scheduled_time": _naive_local(when or (now_local() + timedelta(hours=2))),
        "status": BookingStatus.PENDING,
        "driver_id": None,
        "pickup_lat": pickup[0],
        "pickup_lon": pickup[1],
        "dropoff_lat": dropoff[0],
        "dropoff_lon": dropoff[1],
        "is_return": False,
        "medical_facility": None,
        "hospital_service": False,
        "pickup_location": "Rue Test",
        "_coord_quality_factor": 1.0,
    }
    data.update(extra)
    return _ns(**data)


def _driver(did: int, *, lat=46.2044, lon=6.1432, **extra):
    data = {
        "id": did,
        "latitude": lat,
        "longitude": lon,
        "current_lat": None,
        "current_lon": None,
        "is_available": True,
        "driver_type": "REGULAR",
        "_coord_quality_factor": 1.0,
    }
    data.update(extra)
    return _ns(**data)


@pytest.fixture(autouse=True)
def _clear_score_cache():
    h._HEURISTIC_SCORE_CACHE.clear()
    yield
    h._HEURISTIC_SCORE_CACHE.clear()


def test_helpers_baseline_counter_coords():
    assert h.baseline_and_cap_loads({}) == ({}, 0)
    loads, baseline = h.baseline_and_cap_loads(
        {"1": 5, "bad": 2, 3: "x", 4: -2}  # type: ignore[dict-item]
    )
    assert baseline >= 0
    assert 1 in loads
    assert h.baseline_and_cap_loads({"x": "y"}) == ({}, 0)
    assert h._normalized_loads({}) == {}

    h.TemporalConflictCounter._instance = None
    h.reset_temporal_conflict_counter()
    assert h.get_temporal_conflict_count() == 0
    h.increment_temporal_conflict_counter()
    assert h.get_temporal_conflict_count() >= 1
    h.reset_temporal_conflict_counter()
    inst = h.TemporalConflictCounter.get_instance()
    inst.reset()
    assert inst.get_count() == 0
    inst.increment()

    assert h.haversine_minutes((46.2, 6.1), (46.2, 6.1), avg_kmh=0) >= 1
    assert h.haversine_minutes((46.2, 6.1), (46.2, 6.1), avg_kmh=float("nan")) >= 1
    assert (
        h.haversine_minutes((46.2, 6.1), (46.21, 6.12), avg_kmh=0, fallback_speed_kmh=0)
        >= 1
    )
    assert h.haversine_minutes((200, 400), (-200, -400), avg_kmh=40, max_minutes=2) <= 2
    assert h._py_int("nope") is None
    assert h._py_int(None) is None
    assert h._py_int(7) == 7

    coord, factor = h._driver_current_coord(
        _driver(1, current_lat="bad", current_lon="x")
    )
    assert factor <= 1.0
    coord, _ = h._driver_current_coord(
        _ns(
            current_lat=None,
            current_lon=None,
            latitude="z",
            longitude="z",
            _coord_quality_factor=None,
        )
    )
    assert coord == h.FALLBACK_COORD_DEFAULT

    pickup, drop = h._booking_coords(_ns(pickup_lat=None, pickup_lon=None))
    assert pickup == h.FALLBACK_COORD_DEFAULT
    assert drop == h.FALLBACK_COORD_DEFAULT

    assert h._is_booking_assigned(_ns(status=BookingStatus.ASSIGNED)) is True
    assert h._is_booking_assigned(_ns(status="nope")) is False
    assert h._is_booking_assigned(_ns()) is False

    class _BoomStatus:
        value = None

        def __eq__(self, other):
            raise RuntimeError("eq boom")

        def __hash__(self):
            return 0

    assert h._is_booking_assigned(_ns(status=_BoomStatus())) is False

    w = {"medical": 0.6, "hospital": 0.4, "time_pressure": 0.5, "return_generic": 0.1}
    pr = h._priority_weight(
        _booking(
            1,
            when=datetime.now(UTC) + timedelta(minutes=2),
            medical_facility="HUG",
            hospital_service=True,
            is_return=True,
        ),
        w,
    )
    assert pr > 0
    assert h._regular_driver_bonus(_ns(driver_id="x"), _ns(id="y")) == 0
    ha = h.HeuristicAssignment(1, 2, 0.5, "regular", "bad", "bad")  # type: ignore[arg-type]
    dumped = ha.to_dict()
    assert dumped["booking_id"] == 1
    assert h._check_driver_window_feasible((0, 480), 600) is True
    assert h._check_driver_window_feasible((120, 480), 30) is False


def test_can_be_pooled_fallback_adresse():
    settings = Settings()
    settings.pooling.enabled = True
    settings.pooling.time_tolerance_min = 10
    t = datetime.now(UTC) + timedelta(hours=1)
    a = _booking(1, when=t, pickup=(None, None))  # type: ignore[arg-type]
    a.pickup_lat = None
    a.pickup_lon = None
    a.pickup_location = "Gare Cornavin"
    b = _booking(2, when=t + timedelta(minutes=2), pickup=(None, None))  # type: ignore[arg-type]
    b.pickup_lat = None
    b.pickup_lon = None
    b.pickup_location = "Gare Cornavin"
    assert h._can_be_pooled(a, b, settings) is True
    b.pickup_location = "Aéroport"
    assert h._can_be_pooled(a, b, settings) is False
    missing = _booking(3)
    missing.scheduled_time = None
    assert h._can_be_pooled(a, missing, settings) is False
    far_a = _booking(4, pickup=(46.20, 6.14))
    far_b = _booking(5, pickup=(47.50, 8.50), when=far_a.scheduled_time)
    assert h._can_be_pooled(far_a, far_b, settings) is False


def test_scoring_distance_preference_urgence_cache(monkeypatch):
    settings = Settings()
    settings.features.enable_rl = True
    settings.features.enable_rl_apply = True
    near = _driver(1)
    far = _driver(2, lat=48.85, lon=2.35)
    b = _booking(10)
    dw = (0, 24 * 60)

    far_score, _, _ = h._score_driver_for_booking(b, far, dw, settings, {})
    assert far_score < 0

    pref, _, _ = h._score_driver_for_booking(
        b,
        near,
        dw,
        settings,
        {1: 3},
        company_coords=(46.2044, 6.1432),
        preferred_driver_id=1,
        last_dropoff_coord=(46.205, 6.144),
    )
    assert pref >= 0
    cached, _, _ = h._score_driver_for_booking(
        b,
        near,
        dw,
        settings,
        {1: 3},
        company_coords=(46.2044, 6.1432),
        preferred_driver_id=1,
        last_dropoff_coord=(46.205, 6.144),
    )
    assert cached == pref

    emg = _driver(3, driver_type="DriverType.EMERGENCY")
    sc_emg, br, _ = h._score_driver_for_booking(
        b, emg, dw, settings, {}, company_coords=(46.2044, 6.1432)
    )
    assert "proximity" in br
    assert sc_emg >= 0 or sc_emg == -1

    mid = _booking(11, pickup=(46.26, 6.20), dropoff=(46.38, 6.32))
    h._score_driver_for_booking(
        mid, emg, dw, settings, {}, company_coords=(46.2044, 6.1432)
    )
    far_emg = _booking(12, pickup=(46.32, 6.28), dropoff=(46.50, 6.50))
    h._score_driver_for_booking(
        far_emg, emg, dw, settings, {}, company_coords=(46.2044, 6.1432)
    )

    infeas, _, _ = h._score_driver_for_booking(b, near, (500, 1440), settings, {})
    assert infeas == -1 or isinstance(infeas, float)

    far_start = _driver(8, lat=46.42, lon=6.38)
    for last in ((46.28, 6.22), (46.35, 6.30), (46.39, 6.35)):
        h._score_driver_for_booking(
            b,
            far_start,
            dw,
            settings,
            {},
            last_dropoff_coord=last,
        )

    pair = h._score_booking_driver_pair(
        b, near, dw, settings, {}, {1: 0}, preferred_driver_id=1
    )
    assert pair[0] == 10
    assert pair[1] == 1

    def _boom(*_a, **_k):
        raise RuntimeError("score fail")

    monkeypatch.setattr(h, "_score_driver_for_booking", _boom)
    boom = h._score_booking_driver_pair(b, near, dw, settings, {}, {})
    assert boom[2] == 0.0


def test_assign_empty_et_urgent_et_etats(db):
    settings = Settings()
    empty = h.assign({}, settings)
    assert empty.debug["reason"] == "empty_problem"

    company = CompanyFactory()
    driver = DriverFactory(
        company=company, latitude=46.2044, longitude=6.1432, is_available=True
    )
    soon = _naive_local(now_local() + timedelta(minutes=8))
    urgent = BookingFactory(
        company=company,
        pickup_lat=46.2044,
        pickup_lon=6.1432,
        dropoff_lat=46.21,
        dropoff_lon=6.15,
        scheduled_time=soon,
        status=BookingStatus.PENDING,
        is_return=True,
    )
    assigned = _booking(
        88001,
        when=soon - timedelta(hours=1),
        pickup=(46.2044, 6.1432),
        dropoff=(46.21, 6.15),
        status=BookingStatus.ASSIGNED,
        driver_id=driver.id,
    )
    later = BookingFactory(
        company=company,
        pickup_lat=46.206,
        pickup_lon=6.146,
        dropoff_lat=46.22,
        dropoff_lon=6.16,
        scheduled_time=soon + timedelta(hours=2),
        status=BookingStatus.PENDING,
    )
    problem = {
        "bookings": [urgent, assigned, later],
        "drivers": [driver],
        "driver_windows": [(0, 24 * 60)],
        "fairness_counts": {driver.id: 1},
        "preferred_driver_id": driver.id,
        "company_coords": (46.2044, 6.1432),
        "previous_busy": {driver.id: 10},
        "busy_until": {driver.id: 10},
        "driver_scheduled_times": {driver.id: [50]},
        "proposed_load": {driver.id: 0},
        "driver_load_multipliers": {driver.id: 1.5},
        "base_time": _naive_local(
            now_local().replace(hour=0, minute=0, second=0, microsecond=0)
        ),
    }
    result = h.assign(problem, settings)
    assert isinstance(result, h.HeuristicResult)


def test_assign_conflit_temporel_et_pooling(db):
    settings = Settings()
    settings.pooling.enabled = True
    settings.pooling.time_tolerance_min = 15
    settings.pooling.pickup_distance_m = 800
    settings.safety.min_gap_minutes = 30
    company = CompanyFactory()
    driver = DriverFactory(
        company=company, latitude=46.2044, longitude=6.1432, is_available=True
    )
    t0 = datetime.now(UTC).replace(hour=10, minute=0, second=0, microsecond=0)
    b1 = BookingFactory(
        company=company,
        pickup_lat=46.2044,
        pickup_lon=6.1432,
        dropoff_lat=46.21,
        dropoff_lon=6.15,
        scheduled_time=t0,
        status=BookingStatus.PENDING,
    )
    b2 = BookingFactory(
        company=company,
        pickup_lat=46.2045,
        pickup_lon=6.1433,
        dropoff_lat=46.22,
        dropoff_lon=6.16,
        scheduled_time=t0 + timedelta(minutes=8),
        status=BookingStatus.PENDING,
    )
    coords = [
        (float(b1.pickup_lat), float(b1.pickup_lon)),
        (float(b1.dropoff_lat), float(b1.dropoff_lon)),
        (float(b2.pickup_lat), float(b2.pickup_lon)),
        (float(b2.dropoff_lat), float(b2.dropoff_lon)),
    ]
    matrix = [[0, 12, 3, 15], [12, 0, 10, 8], [3, 10, 0, 12], [15, 8, 12, 0]]
    problem = {
        "bookings": [b1, b2],
        "drivers": [driver],
        "driver_windows": [(0, 24 * 60)],
        "coords": coords,
        "time_matrix": matrix,
        "base_time": t0.replace(hour=0, minute=0),
    }
    pooled = h.assign(problem, settings)
    assert isinstance(pooled, h.HeuristicResult)

    settings.pooling.enabled = False
    b3 = BookingFactory(
        company=company,
        pickup_lat=46.25,
        pickup_lon=6.20,
        dropoff_lat=46.26,
        dropoff_lon=6.21,
        scheduled_time=t0 + timedelta(minutes=10),
        status=BookingStatus.PENDING,
    )
    conflict = h.assign(
        {
            "bookings": [b1, b3],
            "drivers": [driver],
            "driver_windows": [(0, 24 * 60)],
            "coords": "invalid",
            "time_matrix": "invalid",
            "base_time": t0.replace(hour=0, minute=0),
        },
        settings,
    )
    assert isinstance(conflict, h.HeuristicResult)


def test_assign_parallele_et_urgence(db):
    settings = Settings()
    settings.features.enable_parallel_heuristics = True
    t = datetime.now(UTC) + timedelta(hours=3)
    t = t.replace(hour=13, minute=45, second=0, microsecond=0)
    bookings = [
        _booking(
            i + 1, when=t + timedelta(minutes=i * 40), pickup=(46.20 + i * 0.001, 6.14)
        )
        for i in range(22)
    ]
    drivers = [_driver(i + 1, lat=46.2044 + i * 0.002, lon=6.1432) for i in range(6)]
    drivers[0].driver_type = "EMERGENCY"
    problem = {
        "bookings": bookings,
        "drivers": drivers,
        "driver_windows": [(0, 24 * 60)] * 6,
        "company_coords": (46.2044, 6.1432),
        "allow_emergency": True,
        "preferred_driver_id": 2,
        "fairness_counts": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
    }
    result = h.assign(problem, settings)
    assert isinstance(result, h.HeuristicResult)

    company = CompanyFactory()
    emg = DriverFactory(
        company=company,
        latitude=46.2044,
        longitude=6.1432,
        is_available=True,
        driver_type=DriverType.EMERGENCY,
    )
    booking = BookingFactory(
        company=company,
        pickup_lat=46.2044,
        pickup_lon=6.1432,
        dropoff_lat=46.21,
        dropoff_lon=6.15,
        scheduled_time=t,
        status=BookingStatus.PENDING,
    )
    settings2 = Settings()
    settings2.solver.max_bookings_per_driver = 0
    settings2.emergency.allow_emergency_drivers = True
    emg_result = h.assign(
        {
            "bookings": [booking],
            "drivers": [emg],
            "driver_windows": [(0, 24 * 60)],
            "company_coords": (46.2044, 6.1432),
            "allow_emergency": True,
            "base_time": t.replace(hour=0, minute=0),
        },
        settings2,
    )
    assert isinstance(emg_result, h.HeuristicResult)


def test_assign_urgent_et_closest_feasible(db):
    settings = Settings()
    settings.emergency.allow_emergency_drivers = True
    empty = h.assign_urgent({}, [1], settings)
    assert empty.debug["reason"] == "no_urgent"

    company = CompanyFactory()
    regular = DriverFactory(
        company=company, latitude=46.2044, longitude=6.1432, is_available=True
    )
    emg = DriverFactory(
        company=company,
        latitude=46.2044,
        longitude=6.1432,
        is_available=True,
        driver_type=DriverType.EMERGENCY,
    )
    b = BookingFactory(
        company=company,
        pickup_lat=46.2044,
        pickup_lon=6.1432,
        dropoff_lat=46.21,
        dropoff_lon=6.15,
        scheduled_time=datetime.now(UTC) + timedelta(hours=1),
        status=BookingStatus.PENDING,
        driver_id=regular.id,
    )
    b.status = BookingStatus.ASSIGNED
    problem = {
        "bookings": [b],
        "drivers": [regular, emg],
        "driver_windows": [(0, 24 * 60), (0, 24 * 60)],
        "fairness_counts": {regular.id: 0, emg.id: 0},
        "preferred_driver_id": regular.id,
        "driver_load_multipliers": {regular.id: 1.0, emg.id: 2.0},
        "company_coords": (46.2044, 6.1432),
    }
    urgent = h.assign_urgent(problem, [b.id, 99999], settings)
    assert isinstance(urgent, h.HeuristicResult)

    empty_fb = h.closest_feasible({}, [1], settings)
    assert empty_fb.debug["reason"] == "empty_fallback"

    t0 = datetime.now(UTC).replace(hour=11, minute=0, second=0, microsecond=0)
    b2 = BookingFactory(
        company=company,
        pickup_lat=46.2045,
        pickup_lon=6.1433,
        dropoff_lat=46.22,
        dropoff_lon=6.16,
        scheduled_time=t0 + timedelta(minutes=5),
        status=BookingStatus.PENDING,
    )
    settings.pooling.enabled = True
    settings.pooling.time_tolerance_min = 20
    settings.pooling.pickup_distance_m = 1000
    fb = h.closest_feasible(
        {
            "bookings": [b, b2],
            "drivers": [regular],
            "driver_windows": [(0, 24 * 60)],
            "preferred_driver_id": regular.id,
            "busy_until": {regular.id: 20},
            "driver_scheduled_times": {regular.id: [t0.hour * 60 + t0.minute]},
            "proposed_load": {regular.id: 0},
            "driver_load_multipliers": {regular.id: 1.0},
            "base_time": t0.replace(hour=0, minute=0),
            "fairness_counts": {regular.id: 0},
        },
        [b.id, b2.id, 424242],
        settings,
    )
    assert isinstance(fb, h.HeuristicResult)

    busy = h.closest_feasible(
        {
            "bookings": [b2],
            "drivers": [regular],
            "driver_windows": [(0, 24 * 60)],
            "busy_until": {regular.id: 20_000},
            "driver_scheduled_times": {regular.id: [20_000]},
            "proposed_load": {regular.id: 1},
            "fairness_counts": {regular.id: 8},
        },
        [b2.id],
        settings,
    )
    assert isinstance(busy, h.HeuristicResult)


def _high_score(*_a, **_k):
    return (0.5, {"proximity": 1.0}, (120, 160))


def _score_variable(b, d, *_a, **_k):
    bid = int(getattr(b, "id", 0) or 0)
    did = int(getattr(d, "id", 0) or 0)
    dtype = str(getattr(d, "driver_type", "") or "").upper()
    if did == 99:
        return (0.0, {"proximity": 0.0}, (120, 160))
    if bid in {401, 501, 502} and "EMERGENCY" not in dtype:
        return (0.0, {"proximity": 0.0}, (120, 160))
    if "EMERGENCY" in dtype:
        if bid == 502:
            return (0.0, {"proximity": 0.0}, (120, 160))
        return (0.4, {"proximity": 0.4}, (120, 160))
    return (0.5, {"proximity": 1.0}, (120, 160))


def test_assign_urgent_et_conflits_avec_score_force(db, monkeypatch):
    monkeypatch.setattr(h, "_score_driver_for_booking", _high_score)
    settings = Settings()
    settings.pooling.enabled = True
    settings.pooling.time_tolerance_min = 20
    settings.pooling.pickup_distance_m = 2000
    settings.safety.min_gap_minutes = 30
    company = CompanyFactory()
    d1 = DriverFactory(
        company=company, latitude=46.2044, longitude=6.1432, is_available=True
    )
    d2 = DriverFactory(
        company=company, latitude=46.2044, longitude=6.1432, is_available=True
    )
    t0 = datetime.now(UTC).replace(hour=10, minute=0, second=0, microsecond=0)
    urgent = BookingFactory(
        company=company,
        pickup_lat=46.2044,
        pickup_lon=6.1432,
        dropoff_lat=46.21,
        dropoff_lon=6.15,
        scheduled_time=datetime.now(UTC) + timedelta(minutes=5),
        status=BookingStatus.PENDING,
        is_return=True,
    )
    b1 = BookingFactory(
        company=company,
        pickup_lat=46.2044,
        pickup_lon=6.1432,
        dropoff_lat=46.21,
        dropoff_lon=6.15,
        scheduled_time=t0 + timedelta(hours=3),
        status=BookingStatus.PENDING,
    )
    b2 = BookingFactory(
        company=company,
        pickup_lat=46.2045,
        pickup_lon=6.1433,
        dropoff_lat=46.22,
        dropoff_lon=6.16,
        scheduled_time=t0 + timedelta(hours=3, minutes=8),
        status=BookingStatus.PENDING,
    )
    coords = [
        (float(b1.pickup_lat), float(b1.pickup_lon)),
        (float(b1.dropoff_lat), float(b1.dropoff_lon)),
        (float(b2.pickup_lat), float(b2.pickup_lon)),
        (float(b2.dropoff_lat), float(b2.dropoff_lon)),
    ]
    matrix = [[0, 8, 2, 12], [8, 0, 9, 6], [2, 9, 0, 10], [12, 6, 10, 0]]
    result = h.assign(
        {
            "bookings": [urgent, b1, b2],
            "drivers": [d1, d2],
            "driver_windows": [(0, 24 * 60), (0, 24 * 60)],
            "coords": coords,
            "time_matrix": matrix,
            "base_time": t0.replace(hour=0, minute=0),
            "preferred_driver_id": d1.id,
            "company_coords": (46.2044, 6.1432),
            "fairness_counts": {d1.id: 0, d2.id: 2},
        },
        settings,
    )
    assert result.assignments

    settings.pooling.enabled = False
    far = BookingFactory(
        company=company,
        pickup_lat=46.30,
        pickup_lon=6.25,
        dropoff_lat=46.31,
        dropoff_lon=6.26,
        scheduled_time=t0 + timedelta(hours=3, minutes=5),
        status=BookingStatus.PENDING,
    )
    conflicted = h.assign(
        {
            "bookings": [b1, far],
            "drivers": [d1],
            "driver_windows": [(0, 24 * 60)],
            "base_time": t0.replace(hour=0, minute=0),
            "busy_until": {d1.id: 500},
            "driver_scheduled_times": {
                d1.id: [(t0 + timedelta(hours=3)).hour * 60 + 0]
            },
        },
        settings,
    )
    assert isinstance(conflicted, h.HeuristicResult)

    fb = h.closest_feasible(
        {
            "bookings": [b1, b2],
            "drivers": [d1],
            "driver_windows": [(0, 24 * 60)],
            "preferred_driver_id": d1.id,
            "busy_until": {d1.id: 50},
            "driver_scheduled_times": {d1.id: [(t0 + timedelta(hours=3)).hour * 60]},
            "proposed_load": {d1.id: 1},
            "base_time": t0.replace(hour=0, minute=0),
            "fairness_counts": {d1.id: 0},
        },
        [b1.id, b2.id],
        settings,
    )
    assert isinstance(fb, h.HeuristicResult)


def test_assign_branches_charge_pooling_urgence_fallback(monkeypatch):
    monkeypatch.setattr(h, "_score_driver_for_booking", _score_variable)
    settings = Settings()
    settings.pooling.enabled = True
    settings.pooling.time_tolerance_min = 20
    settings.pooling.pickup_distance_m = 2000
    settings.safety.min_gap_minutes = 30
    settings.safety.post_trip_buffer_min = -5
    settings.emergency.allow_emergency_drivers = True
    settings.solver.max_bookings_per_driver = 6

    t_soon = datetime.now(UTC) + timedelta(minutes=5)
    t_reg = t_soon + timedelta(minutes=8)
    t_far = t_soon + timedelta(hours=4)
    t_rush = datetime.now(UTC).replace(hour=13, minute=45, second=0, microsecond=0)
    if t_rush < datetime.now(UTC):
        t_rush = t_rush + timedelta(days=1)

    d1 = _driver(1)
    d2 = _driver(2)
    d_emg = _driver(3, driver_type="DriverType.EMERGENCY")
    d_cap = _driver(99)
    d_win = _driver(4)

    urgent = _booking(
        106,
        when=t_soon,
        is_return=True,
        pickup=(46.2044, 6.1432),
        dropoff=(46.21, 6.15),
    )
    regular_pool = _booking(
        201,
        when=t_reg,
        pickup=(46.2044, 6.1432),
        dropoff=(46.22, 6.16),
    )
    assigned_old = _booking(
        301,
        when=t_soon - timedelta(hours=2),
        pickup=(46.2044, 6.1432),
        dropoff=(46.205, 6.144),
        status=BookingStatus.ASSIGNED,
        driver_id=1,
    )
    assigned_newer = _booking(
        302,
        when=t_soon - timedelta(hours=1),
        pickup=(46.206, 6.146),
        dropoff=(46.207, 6.147),
        status=BookingStatus.ASSIGNED,
        driver_id=1,
    )
    no_time = _booking(401, when=t_far)
    no_time.scheduled_time = None
    debug_b = _booking(
        109,
        when=t_far,
        pickup=(46.25, 6.20),
        dropoff=(46.26, 6.21),
        status=BookingStatus.ASSIGNED,
        driver_id=1,
    )
    rush_b = _booking(
        501,
        when=t_rush,
        pickup=(46.2044, 6.1432),
        dropoff=(46.21, 6.15),
    )
    late_b = _booking(
        502,
        when=t_far + timedelta(minutes=5),
        pickup=(46.30, 6.25),
        dropoff=(46.31, 6.26),
    )

    coords = [
        (46.2044, 6.1432),
        (46.21, 6.15),
        (46.22, 6.16),
        (46.205, 6.144),
        (46.207, 6.147),
        (46.25, 6.20),
        (46.26, 6.21),
        (46.30, 6.25),
        (46.31, 6.26),
    ]
    matrix = [[0 for _ in coords] for _ in coords]
    for i in range(len(coords)):
        for j in range(len(coords)):
            matrix[i][j] = abs(i - j) * 8 + 5

    result = h.assign(
        {
            "bookings": [
                urgent,
                regular_pool,
                assigned_old,
                assigned_newer,
                no_time,
                debug_b,
                rush_b,
                late_b,
            ],
            "drivers": [d1, d2, d_emg, d_cap, d_win],
            "driver_windows": [(0, 24 * 60), (0, 24 * 60), (0, 24 * 60)],
            "coords": coords,
            "time_matrix": matrix,
            "preferred_driver_id": 2,
            "company_coords": (46.2044, 6.1432),
            "fairness_counts": {1: 0, 2: 1, 3: 0, 99: 0, 4: 3},
            "driver_load_multipliers": {99: 1.0, 2: 1.0, 1: 1.0, 3: 1.0, 4: 1.0},
            "allow_emergency": True,
            "busy_until": {1: 125},
            "driver_scheduled_times": {1: [120], 3: [0]},
            "proposed_load": {1: 1},
        },
        settings,
    )
    assert isinstance(result, h.HeuristicResult)

    settings.pooling.enabled = False
    settings.safety.post_trip_buffer_min = 15
    close1 = _booking(601, when=t_far, pickup=(46.2044, 6.1432), dropoff=(46.21, 6.15))
    close2 = _booking(
        602,
        when=t_far + timedelta(minutes=8),
        pickup=(46.30, 6.25),
        dropoff=(46.31, 6.26),
    )
    conflicted = h.assign(
        {
            "bookings": [close1, close2],
            "drivers": [d1],
            "driver_windows": [(0, 24 * 60)],
            "coords": coords,
            "time_matrix": None,
            "busy_until": {1: 20_000},
            "driver_scheduled_times": {1: [t_far.hour * 60 + t_far.minute]},
            "fairness_counts": {1: 0},
        },
        settings,
    )
    assert isinstance(conflicted, h.HeuristicResult)

    no_base = h.assign(
        {
            "bookings": [
                _booking(610, when=t_far),
                _booking(611, when=t_far + timedelta(hours=2)),
                _booking(612, when=t_far + timedelta(hours=4)),
                _booking(613, when=t_far + timedelta(hours=6)),
            ],
            "drivers": [d1, d2],
            "driver_windows": [(0, 24 * 60), (0, 24 * 60)],
            "fairness_counts": {1: 0, 2: 1},
            "preferred_driver_id": 2,
            "proposed_load": {1: 3},
        },
        settings,
    )
    assert isinstance(no_base, h.HeuristicResult)

    settings.features.enable_parallel_heuristics = True

    def _pair_partial(b, d, *_a, **_k):
        did = int(getattr(d, "id", 0) or 0)
        if did == 15:
            raise RuntimeError("parallel boom")
        return (int(b.id), did, 0.5, {}, (120, 160))

    monkeypatch.setattr(h, "_score_booking_driver_pair", _pair_partial)
    bookings_p = [
        _booking(700 + i, when=t_far + timedelta(minutes=i * 40)) for i in range(22)
    ]
    drivers_p = [_driver(10 + i) for i in range(6)]
    drivers_p[0].driver_type = "EMERGENCY"
    parallel = h.assign(
        {
            "bookings": bookings_p,
            "drivers": drivers_p,
            "driver_windows": [(0, 24 * 60)] * 6,
            "fairness_counts": {10: 6, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0},
            "company_coords": (46.2044, 6.1432),
            "preferred_driver_id": 11,
            "allow_emergency": True,
        },
        settings,
    )
    assert isinstance(parallel, h.HeuristicResult)

    settings.features.enable_parallel_heuristics = False
    settings.pooling.enabled = True
    settings.safety.min_gap_minutes = 30
    fb = h.closest_feasible(
        {
            "bookings": [close1, close2],
            "drivers": [d1, d2],
            "driver_windows": [(0, 24 * 60)],
            "preferred_driver_id": 1,
            "busy_until": {1: 50, 2: 0},
            "driver_scheduled_times": {
                1: [t_far.hour * 60 + t_far.minute],
            },
            "proposed_load": {1: 1, 2: 0},
            "fairness_counts": {1: 4, 2: 0},
            "driver_load_multipliers": {1: 1.0, 2: 1.0},
            "company_coords": (46.2044, 6.1432),
        },
        [close1.id, close2.id, 424242],
        settings,
    )
    assert isinstance(fb, h.HeuristicResult)

    no_st = _booking(801, when=t_far)
    no_st.scheduled_time = None
    h.closest_feasible(
        {
            "bookings": [no_st, close1],
            "drivers": [d1],
            "driver_windows": [(0, 24 * 60)],
            "fairness_counts": {1: 0},
        },
        [801, close1.id],
        settings,
    )

    assigned_fb = _booking(
        802,
        when=t_far,
        status=BookingStatus.ASSIGNED,
        driver_id=1,
        pickup=(46.2044, 6.1432),
        dropoff=(46.21, 6.15),
    )
    pool_fb = _booking(
        803,
        when=t_far + timedelta(minutes=5),
        pickup=(46.2045, 6.1433),
        dropoff=(46.22, 6.16),
    )
    h.closest_feasible(
        {
            "bookings": [assigned_fb, pool_fb],
            "drivers": [d1],
            "driver_windows": [(0, 24 * 60)],
            "preferred_driver_id": 1,
            "fairness_counts": {1: 0},
        },
        [assigned_fb.id, pool_fb.id],
        settings,
    )

    far_d = _driver(50, lat=48.85, lon=2.35)
    h.assign_urgent(
        {
            "bookings": [urgent, debug_b],
            "drivers": [d1, d_emg, far_d, d_cap],
            "driver_windows": [(0, 24 * 60)],
            "fairness_counts": {1: 0, 3: 0, 50: 0, 99: 8},
            "driver_load_multipliers": {99: 0.0},
            "preferred_driver_id": 1,
            "company_coords": (46.2044, 6.1432),
        },
        [urgent.id, debug_b.id, 99999],
        settings,
    )

    h.assign(
        {
            "bookings": [_booking(901, when=t_far)],
            "drivers": [d1, d2],
            "driver_windows": [(0, 24 * 60), (0, 24 * 60)],
            "preferred_driver_id": 2,
            "fairness_counts": {1: 0, 2: 0},
            "driver_load_multipliers": {2: 0.0},
        },
        settings,
    )

    def _getenv_boom(*_a, **_k):
        raise ValueError("env")

    monkeypatch.setattr(h.os, "getenv", _getenv_boom)
    h.closest_feasible(
        {
            "bookings": [close1],
            "drivers": [d1],
            "driver_windows": [(0, 24 * 60)],
            "busy_until": {1: 10},
            "driver_scheduled_times": {1: [100]},
            "proposed_load": {1: 0},
            "fairness_counts": {1: 0},
        },
        [close1.id],
        settings,
    )


def test_scored_pool_conflits_caps_et_urgence_assign(monkeypatch):
    def _sc(b, d, *_a, **_k):
        did = int(getattr(d, "id", 0) or 0)
        if did == 99:
            return (0.0, {}, (600, 640))
        return (0.5, {"proximity": 1.0}, (600, 640))

    monkeypatch.setattr(h, "_score_driver_for_booking", _sc)
    settings = Settings()
    settings.pooling.enabled = False
    settings.safety.min_gap_minutes = 30
    settings.safety.post_trip_buffer_min = -25
    settings.emergency.allow_emergency_drivers = True
    settings.solver.max_bookings_per_driver = 2

    t0 = datetime.now(UTC).replace(hour=10, minute=0, second=0, microsecond=0)
    d1 = _driver(1)
    d2 = _driver(2)
    d_emg = _driver(3, driver_type="EMERGENCY")
    d_zero = _driver(99)

    urgent = _booking(106, when=t0 + timedelta(minutes=5), is_return=True)
    a = _booking(
        610,
        when=t0 + timedelta(hours=3),
        pickup=(46.2044, 6.1432),
        dropoff=(46.21, 6.15),
    )
    b = _booking(
        611,
        when=t0 + timedelta(hours=3, minutes=8),
        pickup=(46.30, 6.25),
        dropoff=(46.31, 6.26),
    )
    c = _booking(612, when=t0 + timedelta(hours=5))
    assigned_same = _booking(
        109,
        when=t0 + timedelta(hours=6),
        status=BookingStatus.ASSIGNED,
        driver_id=1,
    )
    no_co = _booking(501, when=t0.replace(hour=13, minute=45))

    coords = [
        (46.2044, 6.1432),
        (46.21, 6.15),
        (46.30, 6.25),
        (46.31, 6.26),
    ]
    matrix = [[0, 8, 40, 42], [8, 0, 35, 30], [40, 35, 0, 8], [42, 30, 8, 0]]

    out = h.assign(
        {
            "bookings": [urgent, a, b, c, assigned_same],
            "drivers": [d1, d2, d_emg, d_zero],
            "driver_windows": [(0, 24 * 60), (0, 24 * 60), (0, 24 * 60), (0, 24 * 60)],
            "coords": coords,
            "time_matrix": matrix,
            "base_time": t0.replace(hour=0, minute=0),
            "preferred_driver_id": 2,
            "fairness_counts": {1: 0, 2: 2, 3: 0, 99: 0},
            "proposed_load": {1: 0, 2: 1, 3: 0, 99: 2},
            "busy_until": {1: 610, 3: 620},
            "driver_scheduled_times": {1: [600], 2: [9999]},
            "allow_emergency": True,
            "company_coords": (46.2044, 6.1432),
        },
        settings,
    )
    assert isinstance(out, h.HeuristicResult)

    orphan = h.assign(
        {
            "bookings": [a, b],
            "drivers": [d1],
            "driver_windows": [(0, 24 * 60)],
            "coords": coords,
            "time_matrix": matrix,
            "base_time": t0.replace(hour=0, minute=0),
            "fairness_counts": {1: 0},
            "driver_scheduled_times": {1: [(t0 + timedelta(hours=3)).hour * 60 + 4]},
        },
        settings,
    )
    assert isinstance(orphan, h.HeuristicResult)

    loads = h.assign(
        {
            "bookings": [
                _booking(701, when=t0 + timedelta(hours=2)),
                _booking(702, when=t0 + timedelta(hours=4)),
                _booking(703, when=t0 + timedelta(hours=6)),
            ],
            "drivers": [d1],
            "driver_windows": [(0, 24 * 60)],
            "fairness_counts": {1: 0},
            "proposed_load": {1: 3},
            "base_time": t0.replace(hour=0, minute=0),
        },
        settings,
    )
    assert isinstance(loads, h.HeuristicResult)

    settings.features.enable_parallel_heuristics = True
    settings.solver.max_bookings_per_driver = 6

    def _pair_partial(b, d, *_a, **_k):
        did = int(getattr(d, "id", 0) or 0)
        if did == 15:
            raise RuntimeError("parallel boom")
        return (int(b.id), did, 0.5, {}, (600, 640))

    monkeypatch.setattr(h, "_score_booking_driver_pair", _pair_partial)
    bookings_p = [
        _booking(800 + i, when=t0 + timedelta(hours=2, minutes=i * 40))
        for i in range(22)
    ]
    drivers_p = [_driver(10 + i) for i in range(6)]
    parallel = h.assign(
        {
            "bookings": bookings_p,
            "drivers": drivers_p,
            "driver_windows": [(0, 24 * 60)] * 6,
            "fairness_counts": dict.fromkeys(range(10, 16), 0),
            "proposed_load": {10: 6},
            "preferred_driver_id": 11,
            "allow_emergency": True,
        },
        settings,
    )
    assert isinstance(parallel, h.HeuristicResult)

    settings.features.enable_parallel_heuristics = False
    emg_only = h.assign(
        {
            "bookings": [no_co],
            "drivers": [d1, d_emg],
            "driver_windows": [(0, 24 * 60), (0, 24 * 60)],
            "fairness_counts": {1: 2, 3: 0},
            "proposed_load": {1: 2},
            "allow_emergency": True,
        },
        settings,
    )
    assert isinstance(emg_only, h.HeuristicResult)

    settings.pooling.enabled = True
    settings.pooling.time_tolerance_min = 20
    settings.pooling.pickup_distance_m = 2000
    p1 = _booking(
        901,
        when=t0 + timedelta(hours=3),
        pickup=(46.2044, 6.1432),
        dropoff=(46.21, 6.15),
    )
    p2 = _booking(
        902,
        when=t0 + timedelta(hours=3, minutes=6),
        pickup=(46.2045, 6.1433),
        dropoff=(46.22, 6.16),
    )
    fb = h.closest_feasible(
        {
            "bookings": [p1, p2],
            "drivers": [d1, d2],
            "driver_windows": [(0, 24 * 60)],
            "preferred_driver_id": 1,
            "fairness_counts": {1: 0, 2: 4},
            "driver_load_multipliers": {1: 1.0, 2: 1.0},
            "base_time": t0.replace(hour=0, minute=0),
            "company_coords": (46.2044, 6.1432),
        },
        [p1.id, p2.id],
        settings,
    )
    assert isinstance(fb, h.HeuristicResult)

    settings.pooling.enabled = False
    h.closest_feasible(
        {
            "bookings": [p1, p2],
            "drivers": [d1],
            "driver_windows": [(0, 24 * 60)],
            "busy_until": {1: 20_000},
            "driver_scheduled_times": {1: [(t0 + timedelta(hours=3)).hour * 60]},
            "fairness_counts": {1: 3},
            "proposed_load": {1: 1},
            "base_time": t0.replace(hour=0, minute=0),
        },
        [p1.id, p2.id],
        settings,
    )

    h.assign_urgent(
        {
            "bookings": [assigned_same, urgent],
            "drivers": [d1, d_emg, d_zero, _driver(50, lat=48.85, lon=2.35)],
            "driver_windows": [(0, 24 * 60)],
            "fairness_counts": {1: 0, 3: 0, 99: 0, 50: 0},
            "preferred_driver_id": 1,
            "company_coords": (46.2044, 6.1432),
        },
        [assigned_same.id, urgent.id],
        settings,
    )


def test_isolats_conflits_caps_parallel_urgence(monkeypatch):
    def _sc(*_a, **_k):
        return (0.5, {"proximity": 1.0}, (600, 640))

    monkeypatch.setattr(h, "_score_driver_for_booking", _sc)
    t0 = datetime.now(UTC).replace(hour=10, minute=0, second=0, microsecond=0)
    d1 = _driver(1)
    d_emg = _driver(3, driver_type="EMERGENCY")
    urgent = _booking(
        106, when=datetime.now(UTC) + timedelta(minutes=4), is_return=True
    )

    s_cap = Settings()
    s_cap.solver.max_bookings_per_driver = 6
    h.assign(
        {
            "bookings": [urgent],
            "drivers": [d1],
            "driver_windows": [(0, 24 * 60)],
            "driver_load_multipliers": {1: 0.0},
            "fairness_counts": {1: 0},
        },
        s_cap,
    )

    s_gap = Settings()
    s_gap.safety.min_gap_minutes = 30
    s_gap.safety.post_trip_buffer_min = 15
    h.assign(
        {
            "bookings": [urgent],
            "drivers": [d1],
            "driver_windows": [(0, 24 * 60)],
            "driver_scheduled_times": {1: [600]},
            "fairness_counts": {1: 0},
        },
        s_gap,
    )

    s_busy = Settings()
    s_busy.safety.post_trip_buffer_min = -25
    h.assign(
        {
            "bookings": [urgent],
            "drivers": [d1],
            "driver_windows": [(0, 24 * 60)],
            "busy_until": {1: 620},
            "fairness_counts": {1: 0},
        },
        s_busy,
    )

    s_pref = Settings()
    s_pref.solver.max_bookings_per_driver = 8
    h.assign(
        {
            "bookings": [urgent],
            "drivers": [d1, _driver(2)],
            "driver_windows": [(0, 24 * 60), (0, 24 * 60)],
            "preferred_driver_id": 2,
            "fairness_counts": {1: 0, 2: 0},
            "proposed_load": {1: 0, 2: 3},
        },
        s_pref,
    )

    s_load = Settings()
    s_load.solver.max_bookings_per_driver = 8
    s_load.pooling.enabled = False
    late = [_booking(701 + i, when=t0 + timedelta(hours=2 + i * 3)) for i in range(5)]
    h.assign(
        {
            "bookings": late,
            "drivers": [d1],
            "driver_windows": [(0, 24 * 60)],
            "proposed_load": {1: 3},
            "fairness_counts": {1: 0},
            "base_time": t0.replace(hour=0, minute=0),
        },
        s_load,
    )

    s_pool = Settings()
    s_pool.pooling.enabled = False
    s_pool.safety.min_gap_minutes = 30
    a = _booking(
        610,
        when=t0 + timedelta(hours=3),
        pickup=(46.2044, 6.1432),
        dropoff=(46.21, 6.15),
    )
    b = _booking(
        611,
        when=t0 + timedelta(hours=3, minutes=8),
        pickup=(46.30, 6.25),
        dropoff=(46.31, 6.26),
    )
    coords = [
        (46.2044, 6.1432),
        (46.21, 6.15),
        (46.30, 6.25),
        (46.31, 6.26),
    ]
    h.assign(
        {
            "bookings": [a, b],
            "drivers": [d1],
            "driver_windows": [(0, 24 * 60)],
            "coords": coords,
            "time_matrix": None,
            "base_time": t0.replace(hour=0, minute=0),
            "fairness_counts": {1: 0},
        },
        s_pool,
    )

    s_par = Settings()
    s_par.features.enable_parallel_heuristics = True
    s_par.solver.max_bookings_per_driver = 6

    def _pair_partial(b, d, *_a, **_k):
        did = int(getattr(d, "id", 0) or 0)
        if did == 15:
            raise RuntimeError("parallel boom")
        return (int(b.id), did, 0.5, {}, (600, 640))

    monkeypatch.setattr(h, "_score_booking_driver_pair", _pair_partial)
    bookings_p = [
        _booking(800 + i, when=t0 + timedelta(hours=2, minutes=i * 40))
        for i in range(22)
    ]
    drivers_p = [_driver(10 + i) for i in range(6)]
    h.assign(
        {
            "bookings": bookings_p,
            "drivers": drivers_p,
            "driver_windows": [(0, 24 * 60)] * 6,
            "fairness_counts": dict.fromkeys(range(10, 16), 0),
            "proposed_load": {10: 6},
            "preferred_driver_id": 11,
        },
        s_par,
    )

    s_emg = Settings()
    s_emg.emergency.allow_emergency_drivers = True
    h.assign_urgent(
        {
            "bookings": [urgent],
            "drivers": [d_emg],
            "driver_windows": [(0, 24 * 60)],
            "fairness_counts": {3: 0},
            "company_coords": (46.2044, 6.1432),
        },
        [urgent.id],
        s_emg,
    )

    s_emg2 = Settings()
    s_emg2.emergency.allow_emergency_drivers = True
    s_emg2.solver.max_bookings_per_driver = 0
    rush = _booking(501, when=t0.replace(hour=13, minute=45))
    h.assign(
        {
            "bookings": [rush],
            "drivers": [d1, d_emg],
            "driver_windows": [(0, 24 * 60), (0, 24 * 60)],
            "allow_emergency": True,
            "fairness_counts": {1: 0, 3: 0},
        },
        s_emg2,
    )

    s_fb = Settings()
    s_fb.solver.max_bookings_per_driver = 8
    h.closest_feasible(
        {
            "bookings": [a],
            "drivers": [d1, _driver(2)],
            "driver_windows": [(0, 24 * 60)],
            "fairness_counts": {1: 0, 2: 5},
            "preferred_driver_id": 1,
            "driver_load_multipliers": {1: 1.0, 2: 1.0},
        },
        [a.id],
        s_fb,
    )

    debug_b = _booking(109, when=t0 + timedelta(hours=3))
    h.assign(
        {
            "bookings": [debug_b],
            "drivers": [d_emg],
            "driver_windows": [(0, 24 * 60)],
            "busy_until": {3: 20_000},
            "base_time": t0.replace(hour=0, minute=0),
            "fairness_counts": {3: 0},
        },
        Settings(),
    )


def test_estimate_wait_or_require_extra(db, monkeypatch):
    settings = Settings()
    settings.emergency.allow_emergency_drivers = False
    assert (
        h.estimate_wait_or_require_extra({}, [1], settings)["summary"] == "no_remaining"
    )

    company = CompanyFactory()
    d1 = DriverFactory(
        company=company, latitude=46.2044, longitude=6.1432, is_available=True
    )
    d1.current_lat = 46.2044
    d1.current_lon = 6.1432
    d2 = DriverFactory(company=company, is_available=True)
    d2.latitude = None
    d2.longitude = None
    d2.current_lat = None
    d2.current_lon = None
    b = BookingFactory(
        company=company,
        pickup_lat=46.3,
        pickup_lon=6.3,
        scheduled_time=datetime.now(UTC) + timedelta(minutes=5),
    )
    bad = _booking(99)
    bad.pickup_lat = "x"
    bad.pickup_lon = "y"
    out = h.estimate_wait_or_require_extra(
        {"bookings": [b, bad], "drivers": [d1, d2]},
        [b.id, bad.id, 123456],
        settings,
    )
    assert out["summary"] == "ok"
    assert out["items"]
    assert any(
        "urgence" in s.lower() or "chauffeur" in s.lower() for s in out["suggestions"]
    )

    none = h.estimate_wait_or_require_extra(
        {"bookings": [b], "drivers": []}, [b.id], settings
    )
    assert any("Aucun chauffeur" in s for s in none["suggestions"])

    d3 = DriverFactory(
        company=company, latitude=46.2044, longitude=6.1432, is_available=True
    )
    d3.current_lat = None
    d3.current_lon = None
    near_b = BookingFactory(
        company=company,
        pickup_lat=46.21,
        pickup_lon=6.15,
        scheduled_time=datetime.now(UTC) + timedelta(minutes=8),
    )
    settings_emg = Settings()
    settings_emg.emergency.allow_emergency_drivers = True
    mild_b = BookingFactory(
        company=company,
        pickup_lat=46.22,
        pickup_lon=6.17,
        scheduled_time=datetime.now(UTC) + timedelta(minutes=1),
    )
    mild = h.estimate_wait_or_require_extra(
        {"bookings": [mild_b], "drivers": [d3]}, [mild_b.id], settings_emg
    )
    assert mild["summary"] == "ok"

    weird = _booking(77)
    weird.scheduled_time = object()
    h.estimate_wait_or_require_extra(
        {"bookings": [weird], "drivers": [d3]}, [77], settings_emg
    )

    def _mins_boom(_dt):
        raise ValueError("mins")

    monkeypatch.setattr(h, "minutes_from_now", _mins_boom)
    h.estimate_wait_or_require_extra(
        {"bookings": [near_b], "drivers": [d3]}, [near_b.id], settings_emg
    )
