# tests/test_preferred_driver.py
import pytest
from datetime import datetime, timedelta
from services.unified_dispatch import heuristics
from services.unified_dispatch.settings import Settings
from models import Booking, Driver, Company


@pytest.fixture
def mock_booking():
    b = Booking()
    b.id = 100
    b.pickup_lat = 46.2044
    b.pickup_lon = 6.1432
    b.dropoff_lat = 46.2100
    b.dropoff_lon = 6.1500
    b.scheduled_time = datetime.now() + timedelta(hours=2)
    b.is_return = False
    b.time_confirmed = True
    return b


@pytest.fixture
def mock_drivers():
    d1 = Driver()
    d1.id = 1
    d1.latitude = 46.2044
    d1.longitude = 6.1432
    d1.is_active = True
    d1.is_available = True
    
    d2 = Driver()
    d2.id = 2  # Chauffeur préféré
    d2.latitude = 46.2050
    d2.longitude = 6.1435
    d2.is_active = True
    d2.is_available = True
    
    return [d1, d2]


@pytest.fixture
def mock_company():
    c = Company()
    c.id = 1
    c.latitude = 46.2044
    c.longitude = 6.1432
    return c


def test_preferred_driver_selected_when_feasible(mock_booking, mock_drivers, mock_company):
    """Test: Chauffeur préféré sélectionné quand faisable."""
    problem = {
        "bookings": [mock_booking],
        "drivers": mock_drivers,
        "driver_windows": [(0, 1440), (0, 1440)],
        "fairness_counts": {1: 0, 2: 0},
        "preferred_driver_id": 2,
        "base_time": datetime.now(),
        "company_coords": (46.2044, 6.1432),
    }
    
    result = heuristics.assign(problem, settings=Settings())
    
    assert len(result.assignments) == 1
    assert result.assignments[0].driver_id == 2, "Chauffeur préféré #2 doit être sélectionné"
    assert result.assignments[0].booking_id == 100
    assert result.unassigned_booking_ids == []


def test_preferred_driver_fallback_when_unavailable(mock_booking, mock_drivers, mock_company):
    """Test: Fallback si chauffeur préféré indisponible (fenêtre TW)."""
    problem = {
        "bookings": [mock_booking],
        "drivers": mock_drivers,
        "driver_windows": [(0, 1440), (600, 800)],  # Chauffeur #2 indisponible après 10h
        "fairness_counts": {1: 0, 2: 0},
        "preferred_driver_id": 2,
        "base_time": datetime.now(),
        "company_coords": (46.2044, 6.1432),
    }
    
    # Ajuster scheduled_time pour que ça tombe après la fenêtre du chauffeur #2
    mock_booking.scheduled_time = datetime.now() + timedelta(hours=12)
    
    result = heuristics.assign(problem, settings=Settings())
    
    assert len(result.assignments) == 1
    assert result.assignments[0].driver_id == 1, "Chauffeur #1 doit être sélectionné en fallback"
    assert result.unassigned_booking_ids == []


def test_preferred_driver_in_closest_feasible(mock_booking, mock_drivers, mock_company):
    """Test: preferred_driver_id appliqué dans closest_feasible fallback."""
    problem = {
        "bookings": [mock_booking],
        "drivers": mock_drivers,
        "driver_windows": [(0, 1440), (0, 1440)],
        "fairness_counts": {1: 0, 2: 0},
        "preferred_driver_id": 2,
        "base_time": datetime.now(),
        "company_coords": (46.2044, 6.1432),
        "busy_until": {},
        "driver_scheduled_times": {},
        "proposed_load": {},
    }
    
    result = heuristics.closest_feasible(problem, [100], settings=Settings())
    
    assert len(result.assignments) == 1
    assert result.assignments[0].driver_id == 2, "closest_feasible doit respecter preferred_driver_id"
    assert result.unassigned_booking_ids == []


def test_preferred_driver_tie_break(mock_booking, mock_drivers, mock_company):
    """Test: Chauffeur préféré gagne en cas d'égalité de score."""
    # Simuler deux chauffeurs avec scores identiques (proximité similaire)
    problem = {
        "bookings": [mock_booking],
        "drivers": mock_drivers,
        "driver_windows": [(0, 1440), (0, 1440)],
        "fairness_counts": {1: 0, 2: 0},
        "preferred_driver_id": 2,
        "base_time": datetime.now(),
        "company_coords": (46.2044, 6.1432),
    }
    
    result = heuristics.assign(problem, settings=Settings())
    
    assert len(result.assignments) == 1
    # Avec bonus +3.0, le chauffeur préféré doit gagner même si l'autre est légèrement plus proche
    assert result.assignments[0].driver_id == 2
    assert result.assignments[0].score > 3.0, "Score doit inclure le bonus préférence"

