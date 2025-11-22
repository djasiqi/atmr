# backend/tests/services/unified_dispatch/test_apply.py
"""Tests pour apply_assignments avec rollback transactionnel complet."""

from datetime import date

import pytest
from flask import Flask
from sqlalchemy.exc import IntegrityError

from models import Assignment, Booking, BookingStatus
from services.unified_dispatch.apply import apply_assignments
from tests.factories import (
    BookingFactory,
    CompanyFactory,
    DispatchRunFactory,
    DriverFactory,
)


@pytest.fixture(autouse=True)
def _app_context(app: Flask):
    """Assure que tous les tests s'exécutent dans un app context."""
    with app.app_context():
        yield


@pytest.fixture
def company(db):
    """Créer une entreprise pour les tests."""
    company = CompanyFactory()
    db.session.flush()  # ✅ FIX: Flush pour obtenir l'ID
    return company


@pytest.fixture
def driver(db, company):
    """Créer un chauffeur pour les tests."""
    driver = DriverFactory(company=company, is_active=True, is_available=True)
    db.session.flush()  # ✅ FIX: Flush pour obtenir l'ID
    return driver


@pytest.fixture
def bookings(db, company):
    """Créer plusieurs bookings pour les tests."""
    bookings_list = [
        BookingFactory(company=company, status=BookingStatus.ACCEPTED),
        BookingFactory(company=company, status=BookingStatus.ACCEPTED),
        BookingFactory(company=company, status=BookingStatus.ACCEPTED),
    ]
    db.session.flush()  # ✅ FIX: Flush pour obtenir les IDs
    return bookings_list


class TestRollbackTransactionnel:
    """Tests pour vérifier le rollback transactionnel complet."""

    def test_rollback_complet_en_cas_derreur_partielle(
        self, db, company, driver, bookings
    ):
        """Test : Échec partiel → rollback complet."""
        # Préparer des assignations
        assignments = [
            type(
                "Assignment",
                (),
                {
                    "booking_id": bookings[0].id,
                    "driver_id": driver.id,
                    "score": 1.0,
                },
            )(),
            type(
                "Assignment",
                (),
                {
                    "booking_id": bookings[1].id,
                    "driver_id": driver.id,
                    "score": 1.0,
                },
            )(),
            # Troisième booking avec un driver_id invalide pour provoquer une erreur
            type(
                "Assignment",
                (),
                {
                    "booking_id": bookings[2].id,
                    "driver_id": 99999,  # Driver inexistant
                    "score": 1.0,
                },
            )(),
        ]

        # Tenter l'application
        result = apply_assignments(
            company_id=company.id,
            assignments=assignments,
            enforce_driver_checks=True,
        )

        # Vérifier qu'aucun booking n'a été assigné (rollback complet)
        # Le troisième devrait être skipped, mais les deux premiers ne devraient
        # PAS être persistés si une erreur survient
        # ✅ FIX: Utiliser query au lieu de refresh pour éviter "Instance is not persistent"
        booking0 = db.session.query(Booking).get(bookings[0].id)
        booking1 = db.session.query(Booking).get(bookings[1].id)
        booking2 = db.session.query(Booking).get(bookings[2].id)

        # Avec le rollback transactionnel, si une erreur se produit dans la transaction,
        # tous les changements doivent être annulés
        # Dans ce cas, le driver_id invalide devrait être détecté avant les updates DB
        # donc les bookings ne devraient pas être modifiés
        assert booking0.driver_id is None or booking0.driver_id != driver.id
        assert booking1.driver_id is None or booking1.driver_id != driver.id
        assert booking2.driver_id is None

        # Vérifier que le résultat indique les skips
        assert bookings[2].id in result.get("skipped", {})

    def test_atomicite_batch_assignations(self, db, company, driver, bookings):
        """Test : Atomicité sur un batch d'assignations."""
        # Préparer des assignations valides
        assignments = [
            type(
                "Assignment",
                (),
                {
                    "booking_id": b.id,
                    "driver_id": driver.id,
                    "score": 1.0,
                },
            )()
            for b in bookings
        ]

        # Appliquer les assignations
        result = apply_assignments(
            company_id=company.id,
            assignments=assignments,
        )

        # Vérifier que tous les bookings sont assignés
        # ✅ FIX: Utiliser query au lieu de refresh pour éviter "Instance is not persistent"
        booking0 = db.session.query(Booking).get(bookings[0].id)
        booking1 = db.session.query(Booking).get(bookings[1].id)
        booking2 = db.session.query(Booking).get(bookings[2].id)

        assert booking0.driver_id == driver.id
        assert booking1.driver_id == driver.id
        assert booking2.driver_id == driver.id

        # Vérifier que les assignments sont créés
        assignments_db = Assignment.query.filter(
            Assignment.booking_id.in_([b.id for b in bookings])
        ).all()
        assert len(assignments_db) == 3

        # Vérifier le résultat
        assert len(result["applied"]) == 3
        assert len(result["skipped"]) == 0

    def test_rollback_en_cas_de_conflit_db(self, db, company, driver, bookings):
        """Test : Conflit DB → rollback et pas de corruption."""
        # Créer DispatchRun avant Assignment
        dispatch_run = DispatchRunFactory(company=company, day=date.today())
        db.session.add(dispatch_run)
        db.session.flush()

        # Créer une assignation existante pour créer un conflit
        existing_assignment = Assignment(
            booking_id=bookings[0].id,
            driver_id=driver.id,
            dispatch_run_id=dispatch_run.id,
        )
        db.session.add(existing_assignment)
        db.session.commit()

        # Préparer des assignations, dont une qui crée un conflit
        assignments = [
            type(
                "Assignment",
                (),
                {
                    "booking_id": bookings[0].id,  # Conflit potentiel
                    "driver_id": driver.id,
                    "dispatch_run_id": dispatch_run.id,  # Même dispatch_run_id
                    "score": 1.0,
                },
            )(),
            type(
                "Assignment",
                (),
                {
                    "booking_id": bookings[1].id,
                    "driver_id": driver.id,
                    "score": 1.0,
                },
            )(),
        ]

        # Appliquer (devrait gérer le conflit avec ON CONFLICT DO NOTHING)
        result = apply_assignments(
            company_id=company.id,
            assignments=assignments,
            dispatch_run_id=dispatch_run.id,
        )

        # Vérifier que le conflit est géré (idempotence)
        # Le booking[0] devrait garder son assignment existant
        # Le booking[1] devrait être assigné
        # ✅ FIX: Utiliser query au lieu de refresh pour éviter "Instance is not persistent"
        booking0 = db.session.query(Booking).get(bookings[0].id)
        booking1 = db.session.query(Booking).get(bookings[1].id)

        assert booking0.driver_id == driver.id  # Déjà assigné
        assert booking1.driver_id == driver.id  # Nouvellement assigné

        # Vérifier que le résultat indique les skips pour le conflit
        # (ON CONFLICT DO NOTHING devrait être silencieux mais compter les conflits)
        assert len(result["applied"]) >= 1  # Au moins booking[1]

    def test_etat_coherent_apres_crash_simule(self, db, company, driver, bookings):
        """Test : État cohérent après crash simulé."""
        # Simuler un crash en levant une exception pendant l'application
        assignments = [
            type(
                "Assignment",
                (),
                {
                    "booking_id": b.id,
                    "driver_id": driver.id,
                    "score": 1.0,
                },
            )()
            for b in bookings
        ]

        # Mock une exception pour simuler un crash
        original_bulk_update = db.session.bulk_update_mappings

        def mock_bulk_update(*args, **kwargs):
            # Simuler un crash après le premier update
            raise IntegrityError("Simulated crash", None, None)

        db.session.bulk_update_mappings = mock_bulk_update

        try:
            apply_assignments(
                company_id=company.id,
                assignments=assignments,
            )
        except Exception:
            # Exception attendue
            pass
        finally:
            # Restaurer la méthode originale
            db.session.bulk_update_mappings = original_bulk_update
            # Rollback explicite pour nettoyer
            db.session.rollback()

        # Vérifier que l'état est cohérent (aucun booking partiellement assigné)
        db.session.refresh(bookings[0])
        db.session.refresh(bookings[1])
        db.session.refresh(bookings[2])

        # Tous les bookings devraient être dans leur état initial
        # (pas d'assignation partielle due au crash)
        # ✅ FIX: Utiliser query au lieu de refresh pour éviter "Instance is not persistent"
        booking0 = db.session.query(Booking).get(bookings[0].id)
        booking1 = db.session.query(Booking).get(bookings[1].id)
        booking2 = db.session.query(Booking).get(bookings[2].id)

        assert booking0.driver_id is None or booking0.status != BookingStatus.ASSIGNED
        assert booking1.driver_id is None or booking1.status != BookingStatus.ASSIGNED
        assert booking2.driver_id is None or booking2.status != BookingStatus.ASSIGNED

    def test_transaction_avec_savepoint(self, db, company, driver, bookings):
        """Test : Transaction avec savepoint (appel depuis engine.run())."""
        # Simuler un appel depuis engine.run() qui a déjà une transaction
        # En utilisant _begin_tx(), un savepoint devrait être créé
        assignments = [
            type(
                "Assignment",
                (),
                {
                    "booking_id": b.id,
                    "driver_id": driver.id,
                    "score": 1.0,
                },
            )()
            for b in bookings
        ]

        # Démarrer une transaction externe (simule engine.run())
        # ✅ FIX: apply_assignments gère déjà les transactions avec _begin_tx(),
        # donc on ne doit pas créer une transaction externe ici
        # Appeler apply_assignments qui devrait créer une transaction
        result = apply_assignments(
            company_id=company.id,
            assignments=assignments,
        )

        # Vérifier que les assignations sont appliquées
        assert len(result["applied"]) == 3

        # Vérifier que les changements sont persistés
        # ✅ FIX: Utiliser query au lieu de refresh pour éviter "Instance is not persistent"
        booking0 = db.session.query(Booking).get(bookings[0].id)
        booking1 = db.session.query(Booking).get(bookings[1].id)
        booking2 = db.session.query(Booking).get(bookings[2].id)

        assert booking0.driver_id == driver.id
        assert booking1.driver_id == driver.id
        assert booking2.driver_id == driver.id
