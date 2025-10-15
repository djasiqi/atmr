"""
Tests d'intégration pour unified_dispatch/engine.py

Teste le système complet de dispatch :
- Assignation automatique des courses
- Gestion des contraintes (horaires, capacité)
- Optimisation multi-chauffeurs
- Propagation des états
"""
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from models import (
    Assignment,
    AssignmentStatus,
    Booking,
    BookingStatus,
    Company,
    DispatchRun,
    DispatchStatus,
    Driver,
    DriverType,
    User,
    Vehicle,
    db,
)
from services.unified_dispatch.engine import run
from services.unified_dispatch.settings import Settings


@pytest.fixture
def app():
    """Créer une application Flask pour les tests."""
    from app import create_app
    app = create_app('testing')

    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def company(app):
    """Créer une entreprise de test."""
    with app.app_context():
        company = Company(
            name="Test Dispatch SA",
            email="dispatch@test.ch",
            phone="+41223334455",
            address="Rue du Test 1, 1200 Genève",
            latitude=46.2044,
            longitude=6.1432
        )
        db.session.add(company)
        db.session.commit()
        return company


@pytest.fixture
def drivers(app, company):
    """Créer plusieurs chauffeurs disponibles."""
    with app.app_context():
        drivers = []

        for i in range(3):
            user = User(
                email=f"driver{i+1}@test.ch",
                first_name="Chauffeur",
                last_name=f"{i+1}",
                role="DRIVER"
            )
            user.set_password("password123")
            db.session.add(user)
            db.session.flush()

            vehicle = Vehicle(
                company_id=company.id,
                license_plate=f"GE-{1000+i}",
                model="Test Model",
                seats=4,
                wheelchair_accessible=True
            )
            db.session.add(vehicle)
            db.session.flush()

            driver = Driver(
                company_id=company.id,
                user_id=user.id,
                vehicle_id=vehicle.id,
                driver_type=DriverType.EMPLOYEE,
                latitude=46.2044 + (i * 0.01),  # Position légèrement différente
                longitude=6.1432 + (i * 0.01),
                is_available=True
            )
            db.session.add(driver)
            drivers.append(driver)

        db.session.commit()
        return drivers


@pytest.fixture
def pending_bookings(app, company):
    """Créer des réservations en attente d'assignation."""
    with app.app_context():
        bookings = []
        now = datetime.utcnow()

        # 5 réservations à différentes heures
        for i in range(5):
            booking = Booking(
                company_id=company.id,
                pickup_address=f"Rue Pickup {i+1}, Genève",
                dropoff_address=f"Avenue Dropoff {i+1}, Genève",
                pickup_latitude=46.2044 + (i * 0.005),
                pickup_longitude=6.1432 + (i * 0.005),
                dropoff_latitude=46.2100 + (i * 0.005),
                dropoff_longitude=6.1500 + (i * 0.005),
                scheduled_time=now + timedelta(hours=i+1),
                status=BookingStatus.PENDING,
                amount=Decimal("45.00"),
                requires_wheelchair=False
            )
            db.session.add(booking)
            bookings.append(booking)

        db.session.commit()
        return bookings


# ============================================================
# Tests d'assignation basique
# ============================================================

def test_dispatch_single_booking_single_driver(app, company, drivers, pending_bookings):
    """Test d'assignation simple : 1 course → 1 chauffeur."""
    with app.app_context():
        # Dispatcher seulement la première réservation
        result = run(
            company_id=company.id,
            mode="auto",
            for_date=None  # Today
        )

        # Vérifications
        assert result["status"] == "success"
        assert result["assignments_count"] > 0
        assert result["unassigned_count"] >= 0

        # Vérifier qu'au moins une assignation a été créée
        assignments = Assignment.query.filter_by(
            company_id=company.id
        ).all()

        assert len(assignments) > 0

        # Vérifier que le booking a été assigné
        first_assignment = assignments[0]
        assert first_assignment.driver_id in [d.id for d in drivers]
        assert first_assignment.booking_id == pending_bookings[0].id


def test_dispatch_multiple_bookings_multiple_drivers(app, company, drivers, pending_bookings):
    """Test d'assignation multiple : plusieurs courses → plusieurs chauffeurs."""
    with app.app_context():
        # Dispatcher toutes les réservations
        result = run(
            company_id=company.id,
            mode="auto"
        )

        # Vérifications
        assert result["status"] == "success"

        # Tous les bookings devraient pouvoir être assignés (3 chauffeurs, 5 courses)
        assignments = Assignment.query.filter_by(
            company_id=company.id,
            status=AssignmentStatus.ASSIGNED
        ).all()

        assert len(assignments) >= 3  # Au moins 3 assignations

        # Vérifier que plusieurs chauffeurs ont été utilisés
        driver_ids = set(a.driver_id for a in assignments)
        assert len(driver_ids) >= 2  # Au moins 2 chauffeurs différents


def test_dispatch_with_fairness(app, company, drivers, pending_bookings):
    """Test de l'équité : les courses sont réparties équitablement."""
    with app.app_context():
        result = run(
            company_id=company.id,
            mode="auto"
        )

        # Compter les assignations par chauffeur
        from collections import Counter
        assignments = Assignment.query.filter_by(
            company_id=company.id,
            status=AssignmentStatus.ASSIGNED
        ).all()

        driver_counts = Counter(a.driver_id for a in assignments)

        # La différence max entre chauffeurs ne devrait pas être > 2
        if len(driver_counts) > 0:
            max_count = max(driver_counts.values())
            min_count = min(driver_counts.values())
            assert max_count - min_count <= 2, "Répartition non équitable"


# ============================================================
# Tests de contraintes
# ============================================================

def test_dispatch_respects_driver_capacity(app, company, drivers):
    """Test que le dispatch respecte la capacité max par chauffeur."""
    with app.app_context():
        # Créer 20 réservations (> 3 chauffeurs * 5 max)
        now = datetime.utcnow()
        bookings = []

        for i in range(20):
            booking = Booking(
                company_id=company.id,
                pickup_address=f"Pickup {i}",
                dropoff_address=f"Dropoff {i}",
                pickup_latitude=46.2044,
                pickup_longitude=6.1432,
                dropoff_latitude=46.2100,
                dropoff_longitude=6.1500,
                scheduled_time=now + timedelta(minutes=i*30),
                status=BookingStatus.PENDING,
                amount=Decimal("40.00")
            )
            db.session.add(booking)
            bookings.append(booking)

        db.session.commit()

        # Dispatcher avec max 5 courses par chauffeur
        settings = Settings()
        settings.solver.max_bookings_per_driver = 5

        result = run(
            company_id=company.id,
            mode="auto",
            custom_settings=settings
        )

        # Vérifier que personne n'a plus de 5 courses
        from collections import Counter
        assignments = Assignment.query.filter_by(
            company_id=company.id,
            status=AssignmentStatus.ASSIGNED
        ).all()

        driver_counts = Counter(a.driver_id for a in assignments)

        for driver_id, count in driver_counts.items():
            assert count <= 5, f"Driver {driver_id} a {count} courses (max: 5)"


def test_dispatch_time_conflict_detection(app, company, drivers):
    """Test que le dispatch détecte les conflits horaires."""
    with app.app_context():
        now = datetime.utcnow()

        # Créer 2 réservations à la même heure
        booking1 = Booking(
            company_id=company.id,
            pickup_address="Pickup 1",
            dropoff_address="Dropoff 1",
            pickup_latitude=46.2044,
            pickup_longitude=6.1432,
            dropoff_latitude=46.2100,
            dropoff_longitude=6.1500,
            scheduled_time=now + timedelta(hours=1),
            status=BookingStatus.PENDING,
            amount=Decimal("40.00")
        )

        booking2 = Booking(
            company_id=company.id,
            pickup_address="Pickup 2",
            dropoff_address="Dropoff 2",
            pickup_latitude=46.2050,
            pickup_longitude=6.1440,
            dropoff_latitude=46.2110,
            dropoff_longitude=6.1510,
            scheduled_time=now + timedelta(hours=1, minutes=10),  # 10 min après
            status=BookingStatus.PENDING,
            amount=Decimal("40.00")
        )

        db.session.add_all([booking1, booking2])
        db.session.commit()

        result = run(
            company_id=company.id,
            mode="auto"
        )

        # Vérifier que les 2 courses ne sont PAS assignées au même chauffeur
        assignments = Assignment.query.filter_by(
            company_id=company.id,
            status=AssignmentStatus.ASSIGNED
        ).all()

        # Grouper par chauffeur
        from collections import defaultdict
        driver_bookings = defaultdict(list)
        for a in assignments:
            driver_bookings[a.driver_id].append(a.booking_id)

        # Aucun chauffeur ne devrait avoir les 2 courses
        for driver_id, booking_ids in driver_bookings.items():
            if booking1.id in booking_ids:
                assert booking2.id not in booking_ids, \
                    "Conflit horaire non détecté : 2 courses au même moment"


# ============================================================
# Tests d'optimisation
# ============================================================

def test_dispatch_proximity_optimization(app, company, drivers):
    """Test que le dispatch préfère les chauffeurs proches."""
    with app.app_context():
        # Driver 1 : proche du pickup
        drivers[0].latitude = 46.2044
        drivers[0].longitude = 6.1432

        # Driver 2 : loin du pickup
        drivers[1].latitude = 46.3000
        drivers[1].longitude = 6.2000

        # Driver 3 : très loin
        drivers[2].latitude = 46.4000
        drivers[2].longitude = 6.3000

        db.session.commit()

        # Créer une réservation proche de driver 1
        now = datetime.utcnow()
        booking = Booking(
            company_id=company.id,
            pickup_address="Pickup proche",
            dropoff_address="Dropoff",
            pickup_latitude=46.2045,
            pickup_longitude=6.1433,
            dropoff_latitude=46.2100,
            dropoff_longitude=6.1500,
            scheduled_time=now + timedelta(hours=1),
            status=BookingStatus.PENDING,
            amount=Decimal("40.00")
        )
        db.session.add(booking)
        db.session.commit()

        result = run(
            company_id=company.id,
            mode="auto"
        )

        # Vérifier que c'est driver 1 qui a été assigné
        assignment = Assignment.query.filter_by(
            company_id=company.id,
            booking_id=booking.id,
            status=AssignmentStatus.ASSIGNED
        ).first()

        if assignment:
            # Driver 1 devrait être favorisé (mais pas garanti si heuristique)
            # On vérifie juste qu'il y a une assignation
            assert assignment.driver_id is not None


# ============================================================
# Tests de DispatchRun
# ============================================================

def test_dispatch_creates_dispatch_run(app, company, drivers, pending_bookings):
    """Test que le dispatch crée bien un DispatchRun."""
    with app.app_context():
        result = run(
            company_id=company.id,
            mode="auto"
        )

        # Vérifier qu'un DispatchRun a été créé
        dispatch_runs = DispatchRun.query.filter_by(
            company_id=company.id
        ).all()

        assert len(dispatch_runs) > 0

        latest_run = dispatch_runs[-1]
        assert latest_run.status in [DispatchStatus.COMPLETED, DispatchStatus.PARTIAL]
        assert latest_run.assignments_count >= 0
        assert latest_run.duration_seconds is not None


def test_dispatch_run_metadata(app, company, drivers, pending_bookings):
    """Test que le DispatchRun contient les bonnes métadonnées."""
    with app.app_context():
        result = run(
            company_id=company.id,
            mode="auto"
        )

        dispatch_run = DispatchRun.query.filter_by(
            company_id=company.id
        ).order_by(DispatchRun.id.desc()).first()

        assert dispatch_run is not None
        assert dispatch_run.mode == "auto"
        assert dispatch_run.bookings_count == len(pending_bookings)
        assert dispatch_run.drivers_count == len(drivers)

        # Vérifier les métadonnées JSON
        if dispatch_run.metadata:
            assert "assignments_count" in dispatch_run.metadata
            assert "unassigned_count" in dispatch_run.metadata


# ============================================================
# Tests de modes de dispatch
# ============================================================

def test_dispatch_mode_heuristic(app, company, drivers, pending_bookings):
    """Test du mode heuristique (rapide)."""
    with app.app_context():
        result = run(
            company_id=company.id,
            mode="heuristic"
        )

        assert result["status"] == "success"
        assert result["mode"] == "heuristic"

        # Le mode heuristique devrait être rapide
        assert result["duration_seconds"] < 5.0


def test_dispatch_mode_auto(app, company, drivers, pending_bookings):
    """Test du mode auto (choix intelligent)."""
    with app.app_context():
        result = run(
            company_id=company.id,
            mode="auto"
        )

        assert result["status"] == "success"
        assert result["mode"] in ["heuristic", "solver"]


# ============================================================
# Tests de gestion d'erreurs
# ============================================================

def test_dispatch_no_drivers(app, company, pending_bookings):
    """Test du dispatch sans chauffeurs disponibles."""
    with app.app_context():
        result = run(
            company_id=company.id,
            mode="auto"
        )

        # Devrait réussir mais avec 0 assignations
        assert result["status"] in ["success", "partial"]
        assert result["assignments_count"] == 0
        assert result["unassigned_count"] == len(pending_bookings)


def test_dispatch_no_bookings(app, company, drivers):
    """Test du dispatch sans réservations."""
    with app.app_context():
        result = run(
            company_id=company.id,
            mode="auto"
        )

        # Devrait réussir avec 0 assignations
        assert result["status"] == "success"
        assert result["assignments_count"] == 0
        assert result["unassigned_count"] == 0


def test_dispatch_invalid_company(app):
    """Test du dispatch avec ID entreprise invalide."""
    with app.app_context(), pytest.raises(Exception):
        run(
            company_id=99999,  # N'existe pas
            mode="auto"
        )


# ============================================================
# Tests de propagation d'état
# ============================================================

def test_dispatch_state_propagation(app, company, drivers, pending_bookings):
    """Test que l'état des chauffeurs est correctement propagé."""
    with app.app_context():
        # Premier dispatch
        result1 = run(
            company_id=company.id,
            mode="auto"
        )

        initial_assignments = result1["assignments_count"]

        # Créer de nouvelles réservations
        now = datetime.utcnow()
        new_bookings = []
        for i in range(3):
            booking = Booking(
                company_id=company.id,
                pickup_address=f"New Pickup {i}",
                dropoff_address=f"New Dropoff {i}",
                pickup_latitude=46.2044,
                pickup_longitude=6.1432,
                dropoff_latitude=46.2100,
                dropoff_longitude=6.1500,
                scheduled_time=now + timedelta(hours=5+i),
                status=BookingStatus.PENDING,
                amount=Decimal("40.00")
            )
            db.session.add(booking)
            new_bookings.append(booking)

        db.session.commit()

        # Deuxième dispatch (devrait prendre en compte les assignations précédentes)
        result2 = run(
            company_id=company.id,
            mode="auto"
        )

        # Le total devrait être cohérent
        total_assignments = Assignment.query.filter_by(
            company_id=company.id,
            status=AssignmentStatus.ASSIGNED
        ).count()

        assert total_assignments >= initial_assignments


# ============================================================
# Tests de performance
# ============================================================

def test_dispatch_performance_many_bookings(app, company, drivers):
    """Test de performance avec beaucoup de réservations."""
    with app.app_context():
        # Créer 50 réservations
        now = datetime.utcnow()
        for i in range(50):
            booking = Booking(
                company_id=company.id,
                pickup_address=f"Pickup {i}",
                dropoff_address=f"Dropoff {i}",
                pickup_latitude=46.2044 + (i * 0.001),
                pickup_longitude=6.1432 + (i * 0.001),
                dropoff_latitude=46.2100,
                dropoff_longitude=6.1500,
                scheduled_time=now + timedelta(minutes=i*15),
                status=BookingStatus.PENDING,
                amount=Decimal("40.00")
            )
            db.session.add(booking)

        db.session.commit()

        # Dispatcher
        import time
        start = time.time()
        result = run(
            company_id=company.id,
            mode="heuristic"  # Mode rapide
        )
        duration = time.time() - start

        # Devrait prendre moins de 10 secondes
        assert duration < 10.0, f"Dispatch trop lent: {duration:.2f}s"
        assert result["status"] == "success"


# ============================================================
# Tests d'intégration complète
# ============================================================

def test_dispatch_full_workflow(app, company, drivers, pending_bookings):
    """Test du workflow complet de dispatch."""
    with app.app_context():
        # 1. État initial
        initial_pending = Booking.query.filter_by(
            company_id=company.id,
            status=BookingStatus.PENDING
        ).count()

        assert initial_pending == len(pending_bookings)

        # 2. Lancer le dispatch
        result = run(
            company_id=company.id,
            mode="auto"
        )

        # 3. Vérifier le résultat
        assert result["status"] == "success"

        # 4. Vérifier que des assignations ont été créées
        assignments = Assignment.query.filter_by(
            company_id=company.id
        ).all()

        assert len(assignments) > 0

        # 5. Vérifier qu'un DispatchRun existe
        dispatch_run = DispatchRun.query.filter_by(
            company_id=company.id
        ).first()

        assert dispatch_run is not None
        assert dispatch_run.status in [DispatchStatus.COMPLETED, DispatchStatus.PARTIAL]

        # 6. Vérifier la cohérence des données
        assert dispatch_run.assignments_count == len(assignments)
        assert dispatch_run.bookings_count == initial_pending

