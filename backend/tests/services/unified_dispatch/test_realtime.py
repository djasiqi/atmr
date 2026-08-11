"""Tests RealtimeOptimizer (sévérité, cache, batch OSRM, orchestration)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from cachetools import TTLCache

from services.unified_dispatch.utils import realtime
from services.unified_dispatch.utils.realtime import (
    SEVERITY_CRITICAL_THRESHOLD,
    SEVERITY_HIGH_THRESHOLD,
    SEVERITY_MEDIUM_THRESHOLD,
    OptimizationOpportunity,
    RealtimeOptimizer,
)
from services.unified_dispatch.utils.suggestions import Suggestion


@pytest.fixture
def optimizer(app, monkeypatch):
    """RealtimeOptimizer isolé (pas de ML/DB lourde à l'init)."""
    monkeypatch.setattr(realtime, "SuggestionEngine", MagicMock)
    monkeypatch.setattr(realtime, "DelayPredictor", MagicMock)
    monkeypatch.setattr(
        realtime,
        "get_eta_delay_model",
        lambda: MagicMock(
            predict=MagicMock(
                return_value=SimpleNamespace(
                    probability_delay=0.0, predicted_delay_minutes=0
                )
            )
        ),
    )
    monkeypatch.setattr(
        realtime,
        "get_auto_reassignment_service",
        MagicMock,  # factory → nouvelle instance
    )

    with realtime._DELAY_CACHE_LOCK:
        realtime._DELAY_CALCULATION_CACHE.clear()
    with realtime._optimizers_lock:
        realtime._active_optimizers.clear()

    opt = RealtimeOptimizer(company_id=42, check_interval_seconds=1, app=app)
    opt.suggestion_engine = MagicMock()
    opt.eta_delay_model = MagicMock(
        predict=MagicMock(
            return_value=SimpleNamespace(
                probability_delay=0.0, predicted_delay_minutes=0
            )
        )
    )
    yield opt

    opt._running = False
    with realtime._DELAY_CACHE_LOCK:
        realtime._DELAY_CALCULATION_CACHE.clear()
    with realtime._optimizers_lock:
        realtime._active_optimizers.clear()


def _booking(**kwargs: Any) -> SimpleNamespace:
    defaults = {
        "id": 1,
        "is_urgent": False,
        "medical_facility": None,
        "scheduled_time": datetime(2026, 8, 12, 10, 0, tzinfo=UTC),
        "pickup_lat": 46.52,
        "pickup_lon": 6.63,
        "created_at": None,
        "updated_at": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _driver(**kwargs: Any) -> SimpleNamespace:
    defaults = {
        "id": 10,
        "current_lat": 46.53,
        "current_lon": 6.64,
        "latitude": 46.53,
        "longitude": 6.64,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _assignment(**kwargs: Any) -> SimpleNamespace:
    defaults = {"id": 100, "booking_id": 1, "driver_id": 10}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _suggestion(action: str = "reassign") -> Suggestion:
    return Suggestion(
        action=action,
        priority="medium",
        message="test",
        estimated_gain_minutes=5,
        auto_applicable=False,
    )


# ---------------------------------------------------------------------------
# Phase 2 — sévérité
# ---------------------------------------------------------------------------


class TestSeverity:
    def test_normal_boundaries(self, optimizer):
        b = _booking()
        assert optimizer._determine_severity(5, b) == "low"
        assert optimizer._determine_severity(9, b) == "low"
        assert optimizer._determine_severity(10, b) == "medium"
        assert optimizer._determine_severity(19, b) == "medium"
        assert optimizer._determine_severity(20, b) == "high"
        assert optimizer._determine_severity(29, b) == "high"
        assert optimizer._determine_severity(30, b) == "critical"
        assert SEVERITY_MEDIUM_THRESHOLD == 10
        assert SEVERITY_HIGH_THRESHOLD == 20
        assert SEVERITY_CRITICAL_THRESHOLD == 30

    def test_urgent_medical(self, optimizer):
        urgent = _booking(is_urgent=True)
        assert optimizer._determine_severity(5, urgent) == "high"
        assert optimizer._determine_severity(10, urgent) == "critical"
        medical = _booking(medical_facility="HUG")
        assert optimizer._determine_severity(7, medical) == "high"
        assert optimizer._determine_severity(12, medical) == "critical"

    def test_advance_always_low(self, optimizer):
        b = _booking(is_urgent=True)
        assert optimizer._determine_severity(0, b) == "low"
        assert optimizer._determine_severity(-30, b) == "low"
        assert optimizer._determine_severity(-5, _booking()) == "low"


# ---------------------------------------------------------------------------
# Phase 3 — cache / délai individuel
# ---------------------------------------------------------------------------


class TestIndividualDelayCache:
    def test_gps_path_and_cache_hit_miss_expiry(self, optimizer, monkeypatch):
        fake_time = [0.0]
        cache = TTLCache(maxsize=100, ttl=10, timer=lambda: fake_time[0])
        monkeypatch.setattr(realtime, "_DELAY_CALCULATION_CACHE", cache)

        now = datetime(2026, 8, 12, 10, 0, tzinfo=UTC)
        monkeypatch.setattr(realtime, "now_local", lambda: now)
        monkeypatch.setattr(
            "services.unified_dispatch.data.calculate_eta",
            lambda *_a, **_k: 600,  # 10 min ETA
        )

        assignment = _assignment()
        booking = _booking(scheduled_time=now)  # delay ≈ ETA/60 = 10
        driver = _driver()

        d1 = optimizer._calculate_realtime_delay(assignment, booking, driver)
        assert d1 == 10
        d2 = optimizer._calculate_realtime_delay(assignment, booking, driver)
        assert d2 == 10  # hit

        # miss après expiration
        fake_time[0] = 11.0
        monkeypatch.setattr(
            "services.unified_dispatch.data.calculate_eta",
            lambda *_a, **_k: 1200,  # 20 min
        )
        d3 = optimizer._calculate_realtime_delay(assignment, booking, driver)
        assert d3 == 20

    def test_no_gps_time_based(self, optimizer, monkeypatch):
        now = datetime(2026, 8, 12, 10, 20, tzinfo=UTC)
        monkeypatch.setattr(realtime, "now_local", lambda: now)
        assignment = _assignment()
        booking = _booking(
            scheduled_time=datetime(2026, 8, 12, 10, 0, tzinfo=UTC),
            pickup_lat=None,
            pickup_lon=None,
        )
        driver = _driver(
            current_lat=None, current_lon=None, latitude=None, longitude=None
        )
        delay = optimizer._calculate_realtime_delay(assignment, booking, driver)
        # 20 min de retard + éventuel buffer
        assert delay >= 15


# ---------------------------------------------------------------------------
# Phase 4 — batch OSRM
# ---------------------------------------------------------------------------


class TestBatchOsrm:
    def test_diagonal_mapping_regression(self, optimizer, monkeypatch):
        """sources [0,2,4] / dest [1,3,5] → ETA = diagonale 100/200/300."""
        now = datetime(2026, 8, 12, 10, 0, tzinfo=UTC)
        monkeypatch.setattr(realtime, "now_local", lambda: now)

        items = []
        for i in range(3):
            a = _assignment(id=100 + i, booking_id=1 + i, driver_id=10 + i)
            b = _booking(
                id=1 + i,
                scheduled_time=now,
                pickup_lat=46.5 + i * 0.01,
                pickup_lon=6.6 + i * 0.01,
            )
            d = _driver(
                id=10 + i,
                current_lat=46.51 + i * 0.01,
                current_lon=6.61 + i * 0.01,
            )
            items.append((a, b, d))

        def session_get(model, pk):
            name = getattr(model, "__name__", str(model))
            for _a, b, d in items:
                if "Booking" in name and pk == b.id:
                    return b
                if "Driver" in name and pk == d.id:
                    return d
            return None

        monkeypatch.setattr(realtime.db.session, "get", session_get)
        monkeypatch.setattr(
            realtime,
            "_table",
            lambda **_k: {
                "code": "Ok",
                "durations": [
                    [100, 900, 900],
                    [900, 200, 900],
                    [900, 900, 300],
                ],
            },
        )

        with realtime._DELAY_CACHE_LOCK:
            realtime._DELAY_CALCULATION_CACHE.clear()

        optimizer._calculate_delays_batch([a for a, _, _ in items], now)

        # Avec scheduled=now, delay_min = int(eta/60) → 1, 3, 5
        delays = sorted(realtime._DELAY_CALCULATION_CACHE.values())
        assert delays == [1, 3, 5]

    def test_batch_table_raise_no_propagate_cache_empty(self, optimizer, monkeypatch):
        now = datetime(2026, 8, 12, 10, 0, tzinfo=UTC)
        a = _assignment()
        b = _booking(scheduled_time=now)
        d = _driver()

        def session_get(model, pk):
            name = getattr(model, "__name__", str(model))
            if "Booking" in name:
                return b
            if "Driver" in name:
                return d
            return None

        monkeypatch.setattr(realtime.db.session, "get", session_get)

        def boom(**_k):
            raise RuntimeError("osrm down")

        monkeypatch.setattr(realtime, "_table", boom)
        with realtime._DELAY_CACHE_LOCK:
            realtime._DELAY_CALCULATION_CACHE.clear()

        # Ne doit pas propager
        optimizer._calculate_delays_batch([a, a, a], now)
        assert len(realtime._DELAY_CALCULATION_CACHE) == 0

    def test_batch_no_durations(self, optimizer, monkeypatch):
        now = datetime(2026, 8, 12, 10, 0, tzinfo=UTC)
        a = _assignment()
        b = _booking(scheduled_time=now)
        d = _driver()
        monkeypatch.setattr(
            realtime.db.session,
            "get",
            lambda model, pk: b if "Booking" in getattr(model, "__name__", "") else d,
        )
        monkeypatch.setattr(
            realtime, "_table", lambda **_k: {"code": "Ok", "durations": []}
        )
        optimizer._calculate_delays_batch([a, a, a], now)
        assert len(realtime._DELAY_CALCULATION_CACHE) == 0

    def test_orchestration_batch_fail_still_calls_individual(
        self, optimizer, monkeypatch
    ):
        """Échec batch silencieux → analyse individuelle toujours possible."""
        now = datetime(2026, 8, 12, 10, 0, tzinfo=UTC)
        monkeypatch.setattr(realtime, "now_local", lambda: now)

        with realtime._DELAY_CACHE_LOCK:
            realtime._DELAY_CALCULATION_CACHE.clear()

        # Batch échoue sans propager
        def batch_boom(*_a, **_k):
            raise RuntimeError("batch fail")

        monkeypatch.setattr(realtime, "_table", batch_boom)
        a = _assignment()
        b = _booking(scheduled_time=now)
        d = _driver()
        monkeypatch.setattr(
            realtime.db.session,
            "get",
            lambda model, pk: b if "Booking" in getattr(model, "__name__", "") else d,
        )
        optimizer._calculate_delays_batch([a, a, a], now)
        assert len(realtime._DELAY_CALCULATION_CACHE) == 0

        # Chemin individuel reste opérationnel
        called = {"n": 0}
        real = optimizer._calculate_realtime_delay

        def wrap(*args, **kwargs):
            called["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(optimizer, "_calculate_realtime_delay", wrap)
        monkeypatch.setattr(
            "services.unified_dispatch.data.calculate_eta",
            lambda *_a, **_k: 900,
        )
        optimizer.suggestion_engine.generate_suggestions_for_assignment.return_value = [
            _suggestion()
        ]
        monkeypatch.setattr(realtime.db.session, "add", lambda *_: None)
        monkeypatch.setattr(realtime.db.session, "commit", lambda: None)
        monkeypatch.setattr(realtime.db.session, "rollback", lambda: None)

        booking = _booking(scheduled_time=now - timedelta(minutes=20))
        monkeypatch.setattr(
            realtime.db.session,
            "get",
            lambda model, pk: (
                booking if "Booking" in getattr(model, "__name__", "") else d
            ),
        )
        opp = optimizer._analyze_assignment(_assignment())
        assert called["n"] >= 1
        assert opp is not None
        assert opp.severity in ("low", "medium", "high", "critical")


# ---------------------------------------------------------------------------
# Phase 5 — orchestration
# ---------------------------------------------------------------------------


class TestOrchestration:
    def test_no_assignments_no_opportunities(self, optimizer, monkeypatch):
        monkeypatch.setattr(
            realtime.BookingRepository,
            "find_for_day",
            lambda self, cid, d: [],
        )
        monkeypatch.setattr(
            realtime,
            "day_local_bounds",
            lambda *_: (datetime.now(UTC), datetime.now(UTC)),
        )
        assert optimizer.check_current_assignments("2026-08-12") == []

    def test_notify_only_high_critical(self, optimizer, monkeypatch):
        notified = []
        monkeypatch.setattr(
            realtime,
            "notify_dispatcher_optimization_opportunity",
            notified.append,
        )
        low = OptimizationOpportunity(
            assignment_id=1,
            booking_id=1,
            driver_id=1,
            current_delay_minutes=-20,
            severity="low",
            suggestions=[_suggestion("add_booking")],
            detected_at=datetime.now(UTC),
        )
        high = OptimizationOpportunity(
            assignment_id=2,
            booking_id=2,
            driver_id=2,
            current_delay_minutes=25,
            severity="high",
            suggestions=[_suggestion()],
            detected_at=datetime.now(UTC),
        )
        optimizer._notify_opportunities([low, high])
        assert len(notified) == 1

    def test_advance_suggestions_but_no_critical_notify(self, optimizer, monkeypatch):
        notified = []
        monkeypatch.setattr(
            realtime,
            "notify_dispatcher_optimization_opportunity",
            notified.append,
        )
        booking = _booking()
        assert optimizer._determine_severity(-25, booking) == "low"
        sugg = [_suggestion("add_booking")]
        opp = OptimizationOpportunity(
            assignment_id=1,
            booking_id=1,
            driver_id=1,
            current_delay_minutes=-25,
            severity="low",
            suggestions=sugg,
            detected_at=datetime.now(UTC),
        )
        assert opp.suggestions
        optimizer._notify_opportunities([opp])
        assert notified == []

    def test_start_monitoring_twice(self, optimizer, monkeypatch):
        monkeypatch.setattr(
            realtime.threading,
            "Thread",
            lambda **kwargs: MagicMock(start=MagicMock(), join=MagicMock()),
        )
        optimizer.start_monitoring()
        assert optimizer._running is True
        optimizer.start_monitoring()  # no-op
        assert optimizer._running is True
        optimizer.stop_monitoring()
        assert optimizer._running is False

    def test_loop_exception_continues(self, optimizer, monkeypatch):
        calls = {"n": 0}

        def boom():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            optimizer._running = False
            return []

        monkeypatch.setattr(optimizer, "check_current_assignments", boom)
        monkeypatch.setattr(
            realtime.time,
            "sleep",
            lambda *_: None,
        )
        optimizer._running = True
        optimizer._app = MagicMock()
        optimizer._app.app_context = MagicMock(
            return_value=MagicMock(
                __enter__=lambda *_: None,
                __exit__=lambda *_: None,
            )
        )
        optimizer._monitoring_loop()
        assert calls["n"] >= 2

    def test_module_helpers(self, app, monkeypatch, optimizer):
        monkeypatch.setattr(realtime, "SuggestionEngine", MagicMock)
        monkeypatch.setattr(realtime, "DelayPredictor", MagicMock)
        monkeypatch.setattr(realtime, "get_eta_delay_model", MagicMock)
        monkeypatch.setattr(realtime, "get_auto_reassignment_service", MagicMock)
        monkeypatch.setattr(
            realtime.threading,
            "Thread",
            lambda **kwargs: MagicMock(start=MagicMock(), join=MagicMock()),
        )
        with realtime._optimizers_lock:
            realtime._active_optimizers.clear()

        opt = realtime.start_optimizer_for_company(99, check_interval=1, app=app)
        assert realtime.get_optimizer_for_company(99) is opt
        realtime.start_optimizer_for_company(99, app=app)  # reuse
        realtime.stop_optimizer_for_company(99)
        assert realtime.get_optimizer_for_company(99) is None

    def test_get_status_and_opportunities(self, optimizer):
        optimizer._opportunities = [
            OptimizationOpportunity(
                assignment_id=1,
                booking_id=1,
                driver_id=1,
                current_delay_minutes=40,
                severity="critical",
                suggestions=[],
                detected_at=datetime.now(UTC),
            )
        ]
        st = optimizer.get_status()
        assert st["company_id"] == 42
        assert st["critical_count"] == 1
        assert len(optimizer.get_current_opportunities()) == 1

    def test_overload_redistribute_path(self, optimizer, monkeypatch):
        # Couvrir _detect_overloaded_drivers avec 2 retards
        now = datetime(2026, 8, 12, 10, 0, tzinfo=UTC)
        monkeypatch.setattr(realtime, "now_local", lambda: now)
        asgs = []
        for i in range(2):
            asgs.append(_assignment(id=200 + i, booking_id=20 + i, driver_id=50))

        def session_get(model, pk):
            name = getattr(model, "__name__", "")
            if "Booking" in name:
                return _booking(
                    id=pk,
                    scheduled_time=now - timedelta(minutes=20),
                )
            if "Driver" in name:
                return _driver(id=50)
            return None

        monkeypatch.setattr(realtime.db.session, "get", session_get)
        monkeypatch.setattr(
            optimizer,
            "_calculate_realtime_delay",
            lambda *_a, **_k: 20,
        )
        opps = optimizer._detect_overloaded_drivers(asgs)
        assert isinstance(opps, list)
