# pyright: reportAttributeAccessIssue=false
"""
Tests pour services/unified_dispatch/solver.py
Coverage cible : 70%+
Tests pour OR-Tools VRPTW (Vehicle Routing Problem with Time Windows)
"""

from datetime import datetime, timedelta

import pytest

from services.unified_dispatch.settings import Settings
from services.unified_dispatch.solver import (
    SAFE_MAX_NODES,
    SAFE_MAX_TASKS,
    SAFE_MAX_VEH,
    SolverAssignment,
    SolverResult,
    solve,
)
from tests.factories import BookingFactory, CompanyFactory, DriverFactory


class TestSolverDataclasses:
    """Tests pour les structures de données du solver."""

    def test_solver_assignment_structure(self):
        """Test structure SolverAssignment."""
        base_time = datetime.utcnow()
        assignment = SolverAssignment(
            booking_id=1,
            driver_id=2,
            reason="solver",
            route_index=0,
            estimated_pickup_min=30,
            estimated_dropoff_min=60,
            base_time=base_time,
            dispatch_run_id=0.100,
        )

        assert assignment.booking_id == 1
        assert assignment.driver_id == 2
        assert assignment.reason == "solver"
        assert assignment.route_index == 0
        assert assignment.estimated_pickup_min == 30
        assert assignment.estimated_dropoff_min == 60
        assert assignment.base_time == base_time
        assert assignment.dispatch_run_id == 100

    def test_solver_assignment_to_dict(self):
        """Test sérialisation SolverAssignment."""
        base_time = datetime.utcnow()
        assignment = SolverAssignment(
            booking_id=1,
            driver_id=2,
            estimated_pickup_min=30,
            estimated_dropoff_min=60,
            base_time=base_time,
            dispatch_run_id=0.100,
        )

        result = assignment.to_dict()

        assert result["booking_id"] == 1
        assert result["driver_id"] == 2
        assert result["status"] == "proposed"
        assert "estimated_pickup_arrival" in result
        assert "estimated_dropoff_arrival" in result
        assert result["reason"] == "solver"
        assert result["route_index"] == 0
        assert result["dispatch_run_id"] == 100

    def test_solver_result_structure(self):
        """Test structure SolverResult."""
        assignments = [SolverAssignment(booking_id=1, driver_id=2, estimated_pickup_min=10, estimated_dropoff_min=20)]

        result = SolverResult(
            assignments=assignments, unassigned_booking_ids=[3, 4], debug={"solver_time_ms": 150, "status": "optimal"}
        )

        assert len(result.assignments) == 1
        assert result.unassigned_booking_ids == [3, 4]
        assert result.debug["solver_time_ms"] == 150
        assert result.debug["status"] == "optimal"


class TestSolverEmptyProblems:
    """Tests pour les cas limites et problèmes vides."""

    def test_solve_empty_problem(self):
        """Test solver avec problème complètement vide."""
        problem = {
            "bookings": [],
            "drivers": [],
            "time_matrix": [],
            "num_vehicles": 0,
            "starts": [],
            "ends": [],
            "service_times": [],
            "time_windows": [],
            "driver_windows": [],
            "capacities": {"passengers": [], "wheelchairs": [], "stretchers": []},
            "demands": {"passengers": [], "wheelchairs": [], "stretchers": []},
            "horizon": 480,
            "base_time": datetime.utcnow(),
            "company": None,
            "for_date": "2025-0.1-15",
        }

        settings = Settings()
        result = solve(problem, settings)

        assert isinstance(result, SolverResult)
        assert len(result.assignments) == 0, "Aucun assignment pour problème vide"
        assert len(result.unassigned_booking_ids) == 0, "Aucune course à assigner"
        assert "status" in result.debug, "Debug info devrait contenir le statut"

    def test_solve_no_drivers(self, db):
        """Test solver sans chauffeurs disponibles."""
        company = CompanyFactory()
        booking = BookingFactory(company=company)

        problem = {
            "bookings": [booking],
            "drivers": [],  # Pas de drivers
            "time_matrix": [[0]],
            "num_vehicles": 0,
            "starts": [],
            "ends": [],
            "service_times": [5],
            "time_windows": [(60, 120)],
            "driver_windows": [],
            "capacities": {"passengers": [], "wheelchairs": [], "stretchers": []},
            "demands": {"passengers": [1], "wheelchairs": [0], "stretchers": [0]},
            "horizon": 480,
            "base_time": datetime.utcnow(),
            "company": company,
            "for_date": "2025-0.1-15",
        }

        settings = Settings()
        result = solve(problem, settings)

        assert isinstance(result, SolverResult)
        assert len(result.assignments) == 0, "Aucun assignment sans drivers"
        assert booking.id in result.unassigned_booking_ids, "Booking devrait être non assigné"

    def test_solve_no_bookings(self, db):
        """Test solver sans courses à assigner."""
        company = CompanyFactory()
        driver = DriverFactory(company=company)

        problem = {
            "bookings": [],  # Pas de bookings
            "drivers": [driver],
            "time_matrix": [[0]],
            "num_vehicles": 1,
            "starts": [0],
            "ends": [0],
            "service_times": [],
            "time_windows": [],
            "driver_windows": [(0, 480)],
            "capacities": {"passengers": [4], "wheelchairs": [1], "stretchers": [0]},
            "demands": {"passengers": [], "wheelchairs": [], "stretchers": []},
            "horizon": 480,
            "base_time": datetime.utcnow(),
            "company": company,
            "for_date": "2025-0.1-15",
        }

        settings = Settings()
        result = solve(problem, settings)

        assert isinstance(result, SolverResult)
        assert len(result.assignments) == 0, "Aucun assignment sans bookings"
        assert len(result.unassigned_booking_ids) == 0


class TestSolverBasicScenarios:
    """Tests pour les scénarios de base."""

    def test_solve_single_booking_single_driver(self, db):
        """Test solver cas simple : 1 booking, 1 driver."""
        company = CompanyFactory()
        driver = DriverFactory(company=company, latitude=46.2044, longitude=6.1432)
        booking = BookingFactory(
            company=company,
            pickup_lat=46.2100,
            pickup_lon=6.1500,
            scheduled_time=datetime.utcnow() + timedelta(hours=2),
        )

        problem = {
            "bookings": [booking],
            "drivers": [driver],
            "time_matrix": [
                [0, 10, 15],  # depot
                [10, 0, 5],  # pickup
                [15, 5, 0],  # dropoff
            ],
            "num_vehicles": 1,
            "starts": [0],
            "ends": [0],
            "service_times": [5],  # 5 min de service au pickup
            "time_windows": [(60, 180)],  # Fenêtre 1h-3h
            "driver_windows": [(0, 480)],  # 8h de travail
            "capacities": {"passengers": [4], "wheelchairs": [1], "stretchers": [0]},
            "demands": {"passengers": [1], "wheelchairs": [0], "stretchers": [0]},
            "horizon": 480,
            "base_time": datetime.utcnow(),
            "company": company,
            "for_date": "2025-0.1-15",
        }

        settings = Settings()
        result = solve(problem, settings)

        assert isinstance(result, SolverResult)
        # Soit assigned, soit unassigned (dépend de la faisabilité)
        total_handled = len(result.assignments) + len(result.unassigned_booking_ids)
        assert total_handled == 1, "1 booking devrait être traité"

    def test_solve_multiple_bookings_multiple_drivers(self, db):
        """Test solver avec plusieurs bookings et drivers."""
        company = CompanyFactory()

        driver1 = DriverFactory(company=company, latitude=46.2000, longitude=6.1000)
        driver2 = DriverFactory(company=company, latitude=46.2200, longitude=6.1600)

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

        # Matrice simplifiée : depot + 4 points (2 pickups + 2 dropoffs)
        problem = {
            "bookings": [booking1, booking2],
            "drivers": [driver1, driver2],
            "time_matrix": [
                [0, 10, 15, 12, 18],  # depot
                [10, 0, 10, 8, 12],  # pickup1
                [15, 10, 0, 15, 5],  # dropoff1
                [12, 8, 15, 0, 10],  # pickup2
                [18, 12, 5, 10, 0],  # dropoff2
            ],
            "num_vehicles": 2,
            "starts": [0, 0],
            "ends": [0, 0],
            "service_times": [5, 5],
            "time_windows": [(30, 120), (60, 180)],
            "driver_windows": [(0, 480), (0, 480)],
            "capacities": {"passengers": [4, 4], "wheelchairs": [1, 1], "stretchers": [0, 0]},
            "demands": {"passengers": [1, 1], "wheelchairs": [0, 0], "stretchers": [0, 0]},
            "horizon": 480,
            "base_time": datetime.utcnow(),
            "company": company,
            "for_date": "2025-0.1-15",
        }

        settings = Settings()
        result = solve(problem, settings)

        assert isinstance(result, SolverResult)
        total_handled = len(result.assignments) + len(result.unassigned_booking_ids)
        assert total_handled == 2, "2 bookings devraient être traités"


class TestSolverConstraints:
    """Tests pour les contraintes du solver."""

    def test_solve_respects_time_windows(self, db):
        """Test que le solver respecte les fenêtres horaires."""
        company = CompanyFactory()
        driver = DriverFactory(company=company)
        booking = BookingFactory(company=company)

        # Fenêtre très stricte : 2h-2h05
        # ✅ FIX: Le solver attend 2 * len(bookings) fenêtres (pickup + dropoff pour chaque booking)
        problem = {
            "bookings": [booking],
            "drivers": [driver],
            "time_matrix": [[0, 10, 15], [10, 0, 5], [15, 5, 0]],
            "num_vehicles": 1,
            "starts": [0],
            "ends": [0],
            "service_times": [5],
            "time_windows": [(120, 125), (180, 185)],  # ✅ FIX: Pickup (120-125) + Dropoff (180-185)
            "driver_windows": [(0, 480)],
            "capacities": {"passengers": [4], "wheelchairs": [1], "stretchers": [0]},
            "demands": {"passengers": [1], "wheelchairs": [0], "stretchers": [0]},
            "horizon": 480,
            "base_time": datetime.utcnow(),
            "company": company,
            "for_date": "2025-0.1-15",
        }

        settings = Settings()
        result = solve(problem, settings)

        # Si assigné, devrait respecter la fenêtre
        if result.assignments:
            assignment = result.assignments[0]
            assert 120 <= assignment.estimated_pickup_min <= 125, "Devrait respecter time window"

    def test_solve_respects_capacity(self, db):
        """Test que le solver respecte les capacités."""
        company = CompanyFactory()

        # Driver avec petite capacité
        driver = DriverFactory(company=company)

        # 2 bookings qui dépassent la capacité si assignés ensemble
        booking1 = BookingFactory(company=company)
        booking2 = BookingFactory(company=company)

        problem = {
            "bookings": [booking1, booking2],
            "drivers": [driver],
            "time_matrix": [
                [0, 5, 10, 8, 12],
                [5, 0, 8, 6, 10],
                [10, 8, 0, 10, 5],
                [8, 6, 10, 0, 8],
                [12, 10, 5, 8, 0],
            ],
            "num_vehicles": 1,
            "starts": [0],
            "ends": [0],
            "service_times": [5, 5],
            "time_windows": [(60, 120), (70, 130)],
            "driver_windows": [(0, 480)],
            "capacities": {"passengers": [1], "wheelchairs": [0], "stretchers": [0]},  # Capacité = 1
            "demands": {"passengers": [1, 1], "wheelchairs": [0, 0], "stretchers": [0, 0]},  # Demande = 2
            "horizon": 480,
            "base_time": datetime.utcnow(),
            "company": company,
            "for_date": "2025-0.1-15",
        }

        settings = Settings()
        result = solve(problem, settings)

        # Au maximum 1 booking assigné (capacité limitée)
        assert len(result.assignments) <= 1, "Ne devrait pas dépasser la capacité"


class TestSolverLimits:
    """Tests pour les limites de sécurité."""

    def test_solve_max_nodes_limit_exists(self):
        """Test que les limites MAX_NODES sont définies."""
        assert SAFE_MAX_NODES > 0, "SAFE_MAX_NODES devrait être défini"
        assert SAFE_MAX_NODES == 800, "Valeur par défaut devrait être 800"

    def test_solve_max_tasks_limit_exists(self):
        """Test que les limites MAX_TASKS sont définies."""
        assert SAFE_MAX_TASKS > 0, "SAFE_MAX_TASKS devrait être défini"
        assert SAFE_MAX_TASKS == 250, "Valeur par défaut devrait être 250"

    def test_solve_max_vehicles_limit_exists(self):
        """Test que les limites MAX_VEH sont définies."""
        assert SAFE_MAX_VEH > 0, "SAFE_MAX_VEH devrait être défini"
        assert SAFE_MAX_VEH == 120, "Valeur par défaut devrait être 120"


class TestSolverDebugInfo:
    """Tests pour les informations de debug."""

    def test_solve_returns_debug_info(self, db):
        """Test que le solver retourne des infos de debug."""
        company = CompanyFactory()
        driver = DriverFactory(company=company)
        booking = BookingFactory(company=company)

        problem = {
            "bookings": [booking],
            "drivers": [driver],
            "time_matrix": [[0, 10, 15], [10, 0, 5], [15, 5, 0]],
            "num_vehicles": 1,
            "starts": [0],
            "ends": [0],
            "service_times": [5],
            "time_windows": [(60, 120)],
            "driver_windows": [(0, 480)],
            "capacities": {"passengers": [4], "wheelchairs": [1], "stretchers": [0]},
            "demands": {"passengers": [1], "wheelchairs": [0], "stretchers": [0]},
            "horizon": 480,
            "base_time": datetime.utcnow(),
            "company": company,
            "for_date": "2025-0.1-15",
        }

        settings = Settings()
        result = solve(problem, settings)

        assert "status" in result.debug, "Debug devrait contenir le statut"
        assert "for_date" in result.debug, "Debug devrait contenir la date"
        # Peut aussi contenir solver_time_ms, num_vehicles, etc.

    def test_solve_handles_infeasible_problem(self, db):
        """Test que le solver gère les problèmes non faisables."""
        company = CompanyFactory()
        driver = DriverFactory(company=company)
        booking = BookingFactory(company=company)

        # Problème impossible : fenêtre déjà passée
        problem = {
            "bookings": [booking],
            "drivers": [driver],
            "time_matrix": [[0, 10, 15], [10, 0, 5], [15, 5, 0]],
            "num_vehicles": 1,
            "starts": [0],
            "ends": [0],
            "service_times": [5],
            "time_windows": [(-60, -30)],  # Dans le passé !
            "driver_windows": [(0, 480)],
            "capacities": {"passengers": [4], "wheelchairs": [1], "stretchers": [0]},
            "demands": {"passengers": [1], "wheelchairs": [0], "stretchers": [0]},
            "horizon": 480,
            "base_time": datetime.utcnow(),
            "company": company,
            "for_date": "2025-0.1-15",
        }

        settings = Settings()
        result = solve(problem, settings)

        # Problème infaisable → booking non assigné
        assert booking.id in result.unassigned_booking_ids, "Booking impossible devrait être non assigné"
        assert len(result.assignments) == 0, "Aucun assignment pour problème infaisable"
