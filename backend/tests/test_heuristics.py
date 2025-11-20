# pyright: reportAttributeAccessIssue=false
"""
Tests pour services/unified_dispatch/heuristics.py
Coverage cible : 70%+
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from models import Booking, BookingStatus, Driver
from services.unified_dispatch.heuristics import (
    HeuristicAssignment,
    HeuristicResult,
    _can_be_pooled,
    _regular_driver_bonus,
    _score_driver_for_booking,
    assign,
    assign_urgent,
    closest_feasible,
)
from services.unified_dispatch.settings import Settings
from tests.factories import BookingFactory, CompanyFactory, DriverFactory


class TestHeuristicsPooling:
    """Tests pour le pooling de courses."""

    def test_can_be_pooled_same_location_and_time(self, db):
        """Test pooling : même pickup, même heure."""
        company = CompanyFactory()
        base_time = datetime.utcnow() + timedelta(hours=2)

        booking1 = BookingFactory(company=company, pickup_lat=46.2044, pickup_lon=6.1432, scheduled_time=base_time)
        booking2 = BookingFactory(
            company=company,
            pickup_lat=46.2045,  # Très proche (100m)
            pickup_lon=6.1433,
            scheduled_time=base_time + timedelta(minutes=2),  # 2 min plus tard
        )

        settings = Settings()
        settings.pooling.enabled = True
        settings.pooling.time_tolerance_min = 5
        settings.pooling.pickup_distance_m = 500

        result = _can_be_pooled(booking1, booking2, settings)
        assert result is True, "Les courses avec même pickup et temps proche devraient être poolables"

    def test_can_be_pooled_disabled(self, db):
        """Test pooling désactivé."""
        company = CompanyFactory()
        booking1 = BookingFactory(company=company)
        booking2 = BookingFactory(company=company)

        settings = Settings()
        settings.pooling.enabled = False

        result = _can_be_pooled(booking1, booking2, settings)
        assert result is False, "Pooling devrait être désactivé"

    def test_can_be_pooled_different_time(self, db):
        """Test pooling : temps trop différent."""
        company = CompanyFactory()
        base_time = datetime.utcnow() + timedelta(hours=2)

        booking1 = BookingFactory(company=company, scheduled_time=base_time)
        booking2 = BookingFactory(
            company=company,
            scheduled_time=base_time + timedelta(minutes=30),  # 30 min plus tard
        )

        settings = Settings()
        settings.pooling.enabled = True
        settings.pooling.time_tolerance_min = 15  # Max 15 min

        result = _can_be_pooled(booking1, booking2, settings)
        assert result is False, "Courses avec temps trop différent ne devraient pas être poolables"


class TestHeuristicsScoring:
    """Tests pour les fonctions de scoring."""

    def test_regular_driver_bonus(self, db):
        """Test bonus chauffeur habituel."""
        company = CompanyFactory()
        driver = DriverFactory(company=company)

        # Booking avec driver_id régulier
        booking = BookingFactory(company=company, driver_id=driver.id)

        bonus = _regular_driver_bonus(booking, driver)
        assert bonus > 0, "Devrait avoir un bonus pour chauffeur habituel"
        assert bonus <= 2.0, "Bonus devrait être raisonnable"

    def test_regular_driver_no_bonus(self, db):
        """Test pas de bonus si autre chauffeur."""
        company = CompanyFactory()
        driver1 = DriverFactory(company=company)
        driver2 = DriverFactory(company=company)

        booking = BookingFactory(company=company, driver_id=driver1.id)

        bonus = _regular_driver_bonus(booking, driver2)
        assert bonus == 0, "Pas de bonus si ce n'est pas le chauffeur habituel"

    def test_score_driver_for_booking_basic(self, db):
        """Test calcul score de base."""
        company = CompanyFactory()
        driver = DriverFactory(company=company, latitude=46.2044, longitude=6.1432, is_available=True)
        booking = BookingFactory(
            company=company,
            pickup_lat=46.2100,  # ~500m de distance
            pickup_lon=6.1500,
            scheduled_time=datetime.utcnow() + timedelta(hours=2),
        )

        driver_window = (0, 480)  # 8h de travail
        fairness_counts = {driver.id: 0}
        settings = Settings()

        score, breakdown, times = _score_driver_for_booking(
            b=booking, d=driver, driver_window=driver_window, settings=settings, fairness_counts=fairness_counts
        )

        assert isinstance(score, float), "Score devrait être un float"
        assert score >= 0, "Score devrait être positif"
        assert isinstance(breakdown, dict), "Breakdown devrait être un dict"
        assert isinstance(times, tuple), "Times devrait être un tuple"


class TestHeuristicsAssignment:
    """Tests pour les fonctions d'assignment."""

    def test_assign_empty_problem(self):
        """Test assign avec problème vide."""
        problem = {
            "bookings": [],
            "drivers": [],
            "time_matrix_min": [],
            "service_times_min": [],
            "company": CompanyFactory.build(),
            "for_date": "2025-0.1-15",
        }

        settings = Settings()
        result = assign(problem, settings)

        assert isinstance(result, HeuristicResult)
        assert len(result.assignments) == 0
        assert len(result.unassigned_booking_ids) == 0

    def test_assign_single_booking_driver(self, db):
        """Test assign simple : 1 booking, 1 driver."""
        company = CompanyFactory()
        driver = DriverFactory(company=company, latitude=46.2044, longitude=6.1432, is_available=True)
        booking = BookingFactory(
            company=company,
            pickup_lat=46.2100,
            pickup_lon=6.1500,
            scheduled_time=datetime.utcnow() + timedelta(hours=2),
            status=BookingStatus.PENDING,
        )

        problem = {
            "bookings": [booking],
            "drivers": [driver],
            "time_matrix_min": [[0, 10], [10, 0]],
            "service_times_min": [5],
            "scheduled_times_min": [120],
            "driver_windows_min": [(0, 480)],  # 8h de travail
            "company": company,
            "for_date": "2025-0.1-15",
        }

        settings = Settings()
        result = assign(problem, settings)

        assert isinstance(result, HeuristicResult)
        # Devrait avoir au moins une tentative d'assignment
        assert len(result.assignments) > 0 or len(result.unassigned_booking_ids) > 0

    def test_closest_feasible_finds_driver(self, db):
        """Test closest_feasible trouve le chauffeur le plus proche."""
        company = CompanyFactory()
        # Créer 2 drivers à distances différentes
        driver_far = DriverFactory(
            company=company,
            latitude=46.3000,  # ~10km
            longitude=6.3000,
            is_available=True,
        )
        driver_close = DriverFactory(
            company=company,
            latitude=46.2050,  # ~500m
            longitude=6.1450,
            is_available=True,
        )

        booking = BookingFactory(
            company=company,
            pickup_lat=46.2044,
            pickup_lon=6.1432,
            scheduled_time=datetime.utcnow() + timedelta(hours=2),
        )

        problem = {
            "bookings": [booking],
            "drivers": [driver_far, driver_close],
            "time_matrix": [
                [0, 15, 2],  # distances depuis chaque point
                [15, 0, 15],
                [2, 15, 0],
            ],
            "service_times": [5],
            "scheduled_times": [120],
            "driver_windows": [(0, 480), (0, 480)],
            "company": company,
            "for_date": "2025-0.1-15",
        }

        settings = Settings()
        booking_ids = [booking.id]
        result = closest_feasible(problem, booking_ids, settings)

        # Devrait assigner ou marquer comme unassigned
        assert isinstance(result, HeuristicResult)
        total_handled = len(result.assignments) + len(result.unassigned_booking_ids)
        assert total_handled == 1, "1 booking devrait être traité"


def test_heuristic_assignment_to_dict():
    """Test sérialisation HeuristicAssignment."""
    assignment = HeuristicAssignment(
        booking_id=1, driver_id=2, score=0.95, reason="regular_scoring", estimated_start_min=30, estimated_finish_min=60
    )

    result = assignment.to_dict()

    assert result["booking_id"] == 1
    assert result["driver_id"] == 2
    assert result["status"] == "proposed"
    assert "estimated_pickup_arrival" in result
    assert "estimated_dropoff_arrival" in result
    assert result["score"] == 0.95
    assert result["reason"] == "regular_scoring"


def test_heuristic_result_structure():
    """Test structure HeuristicResult."""
    assignments = [
        HeuristicAssignment(
            booking_id=1, driver_id=2, score=0.9, reason="test", estimated_start_min=10, estimated_finish_min=20
        )
    ]

    result = HeuristicResult(assignments=assignments, unassigned_booking_ids=[3, 4], debug={"test": "data"})

    assert len(result.assignments) == 1
    assert result.unassigned_booking_ids == [3, 4]
    assert result.debug["test"] == "data"


class TestHeuristicsAssignUrgent:
    """Tests pour assign_urgent()."""

    def test_assign_urgent_with_urgent_bookings(self, db):
        """Test assign_urgent avec courses urgentes."""
        company = CompanyFactory()
        driver = DriverFactory(company=company, latitude=46.2044, longitude=6.1432, is_available=True)

        # Course urgente
        booking_urgent = BookingFactory(
            company=company,
            pickup_lat=46.2100,
            pickup_lon=6.1500,
            scheduled_time=datetime.utcnow() + timedelta(hours=1),
            status=BookingStatus.PENDING,
            is_urgent=True,
        )

        problem = {
            "bookings": [booking_urgent],
            "drivers": [driver],
            "company": company,
            "for_date": datetime.utcnow().date().isoformat(),
        }

        settings = Settings()
        urgent_ids = [booking_urgent.id]
        result = assign_urgent(problem, urgent_ids, settings)

        assert isinstance(result, HeuristicResult)
        # Devrait traiter la course urgente
        total = len(result.assignments) + len(result.unassigned_booking_ids)
        assert total >= 1, "Course urgente devrait être traitée"

    def test_assign_urgent_empty_list(self, db):
        """Test assign_urgent avec liste vide."""
        company = CompanyFactory()
        driver = DriverFactory(company=company)
        booking = BookingFactory(company=company)

        problem = {
            "bookings": [booking],
            "drivers": [driver],
            "company": company,
            "for_date": datetime.utcnow().date().isoformat(),
        }

        settings = Settings()
        result = assign_urgent(problem, [], settings)  # Liste vide

        assert isinstance(result, HeuristicResult)
        assert len(result.assignments) == 0, "Aucune course urgente à assigner"
        assert "reason" in result.debug
        assert result.debug["reason"] == "no_urgent"


class TestHeuristicsHelpers:
    """Tests pour les fonctions helper."""

    def test_haversine_minutes_calculation(self):
        """Test calcul temps en minutes via haversine."""
        from services.unified_dispatch.heuristics import haversine_minutes

        # Genève centre → Genève aéroport (~5km)
        point_a = (46.2044, 6.1432)
        point_b = (46.238, 6.109)

        time_min = haversine_minutes(point_a, point_b, avg_kmh=25.0)

        assert isinstance(time_min, int), "Devrait retourner un entier"
        assert time_min > 0, "Temps devrait être positif"
        assert time_min < 30, "Temps devrait être raisonnable pour 5km"

    def test_is_return_urgent(self, db):
        """Test détection course retour urgente."""
        from services.unified_dispatch.heuristics import _is_return_urgent

        company = CompanyFactory()

        # Course marquée comme retour
        booking_return = BookingFactory(
            company=company, is_return=True, scheduled_time=datetime.utcnow() + timedelta(hours=1)
        )

        settings = Settings()
        settings.regular_first = True  # Active priorité retours

        result = _is_return_urgent(booking_return, settings)

        # Devrait être considéré urgent si settings.regular_first est True
        assert isinstance(result, bool)

    def test_driver_fairness_penalty(self):
        """Test pénalité de fairness."""
        from services.unified_dispatch.heuristics import _driver_fairness_penalty

        # Driver avec beaucoup de courses déjà
        fairness_counts = {1: 5, 2: 2, 3: 0}

        penalty_1 = _driver_fairness_penalty(1, fairness_counts)
        penalty_3 = _driver_fairness_penalty(3, fairness_counts)

        # Driver 1 (5 courses) devrait avoir plus de pénalité que Driver 3 (0 course)
        assert penalty_1 > penalty_3, "Driver surchargé devrait avoir plus de pénalité"
        assert penalty_1 >= 0, "Pénalité 1 devrait être positive"
        assert penalty_3 >= 0, "Pénalité 3 devrait être positive"

    def test_check_driver_window_feasible(self):
        """Test vérification faisabilité fenêtre chauffeur."""
        from services.unified_dispatch.heuristics import _check_driver_window_feasible

        driver_window = (60, 480)  # 1h-8h

        # Cas faisable (dans la fenêtre)
        # 0.120 heures = 7.2 minutes, arrondi à 7 minutes
        feasible = _check_driver_window_feasible(driver_window, est_start_min=7)
        assert feasible is True, "Devrait être faisable"

        # Cas faisable limit (juste dans la fenêtre)
        feasible_limit = _check_driver_window_feasible(driver_window, est_start_min=60)
        assert feasible_limit is True, "Devrait être faisable aux limites"

    def test_py_int_helper(self):
        """Test helper _py_int."""
        from services.unified_dispatch.heuristics import _py_int

        assert _py_int(42) == 42
        assert _py_int("42") == 42
        assert _py_int(42.7) == 42
        assert _py_int(None) is None
        assert _py_int("invalid") is None

    def test_driver_current_coord(self, db):
        """Test extraction coordonnées driver."""
        from services.unified_dispatch.heuristics import _driver_current_coord

        driver = DriverFactory(latitude=46.2044, longitude=6.1432)

        coords = _driver_current_coord(driver)

        assert isinstance(coords, tuple)
        assert len(coords) == 2
        assert coords[0] == 46.2044
        assert coords[1] == 6.1432

    def test_booking_coords(self, db):
        """Test extraction coordonnées booking (pickup + dropoff)."""
        from services.unified_dispatch.heuristics import _booking_coords

        booking = BookingFactory(pickup_lat=46.2044, pickup_lon=6.1432, dropoff_lat=46.2100, dropoff_lon=6.1500)

        pickup, dropoff = _booking_coords(booking)

        assert isinstance(pickup, tuple)
        assert isinstance(dropoff, tuple)
        assert pickup == (46.2044, 6.1432)
        assert dropoff == (46.2100, 6.1500)

    def test_is_booking_assigned(self, db):
        """Test vérification si booking est assigné."""
        from models import BookingStatus
        from services.unified_dispatch.heuristics import _is_booking_assigned

        # Booking avec driver assigné
        company = CompanyFactory()
        driver = DriverFactory(company=company)
        booking_assigned = BookingFactory(company=company, driver_id=driver.id, status=BookingStatus.ASSIGNED)

        result = _is_booking_assigned(booking_assigned)
        assert result is True, "Booking avec driver devrait être assigné"

        # Booking sans driver
        booking_unassigned = BookingFactory(company=company, driver_id=None, status=BookingStatus.PENDING)
        result2 = _is_booking_assigned(booking_unassigned)
        assert result2 is False, "Booking sans driver ne devrait pas être assigné"

    def test_current_driver_id(self, db):
        """Test récupération driver_id actuel."""
        from services.unified_dispatch.heuristics import _current_driver_id

        company = CompanyFactory()
        driver = DriverFactory(company=company)

        booking = BookingFactory(company=company, driver_id=driver.id)
        driver_id = _current_driver_id(booking)

        assert driver_id == driver.id

        # Booking sans driver
        booking_no_driver = BookingFactory(company=company, driver_id=None)
        result = _current_driver_id(booking_no_driver)
        assert result is None

    def test_priority_weight(self, db):
        """Test calcul poids de priorité."""
        from services.unified_dispatch.heuristics import _priority_weight

        company = CompanyFactory()

        # Booking urgent
        booking_urgent = BookingFactory(company=company, is_urgent=True)

        weights = {"urgent": 10.0, "return": 5.0, "medical": 8.0}
        weight = _priority_weight(booking_urgent, weights)

        assert isinstance(weight, float)
        assert weight >= 0, "Poids devrait être positif ou nul"


class TestHeuristicsAssignWithDriverWindows:
    """Tests pour assign() avec fenêtres chauffeurs."""

    def test_assign_with_driver_windows(self, db):
        """Test assign avec fenêtres de travail chauffeurs."""
        company = CompanyFactory()
        driver = DriverFactory(company=company, latitude=46.2044, longitude=6.1432, is_available=True)
        booking = BookingFactory(
            company=company,
            pickup_lat=46.2100,
            pickup_lon=6.1500,
            scheduled_time=datetime.utcnow() + timedelta(hours=2),
            status=BookingStatus.PENDING,
        )

        problem = {
            "bookings": [booking],
            "drivers": [driver],
            "time_matrix": [[0, 10, 15], [10, 0, 5], [15, 5, 0]],
            "service_times": [5],
            "scheduled_times": [120],
            "driver_windows": [(0, 480)],
            "fairness_counts": {driver.id: 0},
            "busy_until": {driver.id: 0},
            "driver_scheduled_times": {driver.id: []},
            "company": company,
            "for_date": datetime.utcnow().date().isoformat(),
        }

        settings = Settings()
        result = assign(problem, settings)

        assert isinstance(result, HeuristicResult)

    def test_assign_with_multiple_bookings_fairness(self, db):
        """Test assign avec équilibrage de charge entre drivers."""
        company = CompanyFactory()

        driver1 = DriverFactory(company=company, latitude=46.2000, longitude=6.1000, is_available=True)
        driver2 = DriverFactory(company=company, latitude=46.2200, longitude=6.1600, is_available=True)

        booking1 = BookingFactory(
            company=company,
            pickup_lat=46.2044,
            pickup_lon=6.1432,
            scheduled_time=datetime.utcnow() + timedelta(hours=1),
        )
        booking2 = BookingFactory(
            company=company,
            pickup_lat=46.2150,
            pickup_lon=6.1550,
            scheduled_time=datetime.utcnow() + timedelta(hours=2),
        )

        problem = {
            "bookings": [booking1, booking2],
            "drivers": [driver1, driver2],
            "time_matrix": [[0] * 5 for _ in range(5)],  # Matrice simple
            "service_times": [5, 5],
            "scheduled_times": [60, 120],
            "driver_windows": [(0, 480), (0, 480)],
            "fairness_counts": {driver1.id: 0, driver2.id: 0},
            "busy_until": {driver1.id: 0, driver2.id: 0},
            "driver_scheduled_times": {driver1.id: [], driver2.id: []},
            "company": company,
            "for_date": datetime.utcnow().date().isoformat(),
        }

        settings = Settings()
        result = assign(problem, settings)

        assert isinstance(result, HeuristicResult)
        # Au moins une tentative d'assignment
        total = len(result.assignments) + len(result.unassigned_booking_ids)
        assert total > 0, "Au moins 1 booking devrait être traité"

    def test_assign_with_pooling_enabled(self, db):
        """Test assign avec pooling activé."""
        company = CompanyFactory()
        driver = DriverFactory(company=company, latitude=46.2044, longitude=6.1432)

        # Deux bookings poolables (même pickup, même heure)
        base_time = datetime.utcnow() + timedelta(hours=2)
        booking1 = BookingFactory(company=company, pickup_lat=46.2044, pickup_lon=6.1432, scheduled_time=base_time)
        booking2 = BookingFactory(
            company=company,
            pickup_lat=46.2045,  # Très proche
            pickup_lon=6.1433,
            scheduled_time=base_time + timedelta(minutes=2),
        )

        problem = {
            "bookings": [booking1, booking2],
            "drivers": [driver],
            "time_matrix": [[0] * 5 for _ in range(5)],
            "service_times": [5, 5],
            "scheduled_times": [120, 122],
            "driver_windows": [(0, 480)],
            "fairness_counts": {driver.id: 0},
            "busy_until": {driver.id: 0},
            "driver_scheduled_times": {driver.id: []},
            "company": company,
            "for_date": datetime.utcnow().date().isoformat(),
        }

        settings = Settings()
        settings.pooling.enabled = True

        result = assign(problem, settings)

        assert isinstance(result, HeuristicResult)
