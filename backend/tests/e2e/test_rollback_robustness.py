# backend/tests/e2e/test_rollback_robustness.py
"""✅ Tests de non-régression : Robustesse des rollbacks.

Ces tests vérifient systématiquement que les rollbacks SQLAlchemy restaurent
correctement les valeurs en DB dans différents scénarios.
"""

from datetime import date

from models import Assignment, Booking, BookingStatus
from services.unified_dispatch import engine
from tests.factories import BookingFactory, DriverFactory
from tests.helpers.rollback_verification import (
    capture_original_values,
    verify_multiple_rollbacks,
    verify_rollback_restores_values,
)


class TestRollbackRobustness:
    """Tests de robustesse des rollbacks SQLAlchemy."""

    def test_rollback_restores_single_field(self, db, company):
        """✅ Test : Rollback restaure un champ unique modifié.

        Ce test vérifie que le rollback restaure correctement un seul champ modifié.
        """
        booking = BookingFactory(
            company=company, driver_id=None, status=BookingStatus.ACCEPTED
        )
        db.session.commit()

        # Capturer les valeurs originales
        original_values = capture_original_values(booking, ["driver_id", "status"])

        # Modifier le booking
        driver = DriverFactory(company=company)
        db.session.commit()
        booking.driver_id = driver.id
        booking.status = BookingStatus.ASSIGNED
        db.session.flush()

        # Rollback
        db.session.rollback()

        # Vérifier que les valeurs sont restaurées
        verify_rollback_restores_values(
            db.session,
            Booking,
            booking.id,
            original_values,
        )

    def test_rollback_restores_multiple_fields(self, db, company):
        """✅ Test : Rollback restaure plusieurs champs modifiés.

        Ce test vérifie que le rollback restaure correctement plusieurs champs modifiés.
        """
        booking = BookingFactory(
            company=company,
            driver_id=None,
            status=BookingStatus.ACCEPTED,
        )
        db.session.commit()

        # Capturer les valeurs originales
        original_values = capture_original_values(booking, ["driver_id", "status"])

        # Modifier plusieurs champs
        driver = DriverFactory(company=company)
        db.session.commit()
        booking.driver_id = driver.id
        booking.status = BookingStatus.ASSIGNED
        db.session.flush()

        # Rollback
        db.session.rollback()

        # Vérifier que toutes les valeurs sont restaurées
        verify_rollback_restores_values(
            db.session,
            Booking,
            booking.id,
            original_values,
        )

    def test_rollback_restores_multiple_objects(self, db, company):
        """✅ Test : Rollback restaure plusieurs objets modifiés.

        Ce test vérifie que le rollback restaure correctement plusieurs objets modifiés.
        """
        booking1 = BookingFactory(company=company, driver_id=None)
        booking2 = BookingFactory(company=company, driver_id=None)
        db.session.commit()

        # Capturer les valeurs originales
        original_values1 = capture_original_values(booking1, ["driver_id"])
        original_values2 = capture_original_values(booking2, ["driver_id"])

        # Modifier les deux bookings
        driver = DriverFactory(company=company)
        db.session.commit()
        booking1.driver_id = driver.id
        booking2.driver_id = driver.id
        db.session.flush()

        # Rollback
        db.session.rollback()

        # Vérifier que les deux objets sont restaurés
        verify_multiple_rollbacks(
            db.session,
            [
                {
                    "model_class": Booking,
                    "object_id": booking1.id,
                    "original_values": original_values1,
                },
                {
                    "model_class": Booking,
                    "object_id": booking2.id,
                    "original_values": original_values2,
                },
            ],
        )

    def test_rollback_restores_after_flush(self, db, company):
        """✅ Test : Rollback restaure même après flush.

        Ce test vérifie que le rollback restaure correctement les valeurs même
        si un flush() a été appelé (qui assigne les IDs mais ne commit pas).
        """
        booking = BookingFactory(company=company, driver_id=None)
        db.session.add(booking)
        db.session.flush()  # Flush pour obtenir l'ID

        # Capturer les valeurs originales après flush
        original_values = capture_original_values(booking, ["driver_id"])

        # Modifier le booking
        driver = DriverFactory(company=company)
        db.session.add(driver)
        db.session.flush()
        booking.driver_id = driver.id
        db.session.flush()

        # Rollback
        db.session.rollback()

        # Vérifier que les valeurs sont restaurées
        verify_rollback_restores_values(
            db.session,
            Booking,
            booking.id,
            original_values,
        )

    def test_rollback_restores_after_partial_commit(self, db, company):
        """✅ Test : Rollback restaure après commit partiel.

        Ce test vérifie que le rollback restaure correctement les valeurs
        même si certains objets ont été commités et d'autres non.
        """
        booking1 = BookingFactory(company=company, driver_id=None)
        booking2 = BookingFactory(company=company, driver_id=None)
        db.session.commit()  # Commit initial

        # Capturer les valeurs originales
        # Note: original_values1 n'est pas utilisé car booking1 est déjà commité
        _ = capture_original_values(booking1, ["driver_id"])  # Capturé mais non utilisé
        original_values2 = capture_original_values(booking2, ["driver_id"])

        # Modifier booking1 et committer
        driver = DriverFactory(company=company)
        db.session.commit()
        booking1.driver_id = driver.id
        db.session.commit()  # Commit booking1

        # Modifier booking2 mais ne pas committer
        booking2.driver_id = driver.id
        db.session.flush()

        # Rollback (devrait restaurer booking2 mais pas booking1)
        db.session.rollback()

        # Vérifier que booking2 est restauré
        verify_rollback_restores_values(
            db.session,
            Booking,
            booking2.id,
            original_values2,
        )

        # Vérifier que booking1 n'est PAS restauré (déjà commité)
        booking1_reloaded = db.session.query(Booking).get(booking1.id)
        assert booking1_reloaded.driver_id == driver.id, (
            "booking1 should NOT be restored (already committed)"
        )

    def test_rollback_restores_after_engine_run_rollback_defensive(self, db, company):
        """✅ Test : Rollback restaure après rollback défensif de engine.run().

        Ce test vérifie que le rollback défensif de engine.run() n'affecte pas
        les objets commités, mais restaure correctement les objets non commités.
        """
        # Créer et committer un booking
        booking = BookingFactory(company=company, driver_id=None)
        db.session.commit()

        # Capturer les valeurs originales
        original_values = capture_original_values(booking, ["driver_id"])

        # Modifier le booking mais ne pas committer
        driver = DriverFactory(company=company)
        db.session.commit()
        booking.driver_id = driver.id
        db.session.flush()

        # Appeler engine.run() qui fait un rollback défensif
        result = engine.run(company_id=company.id, for_date=date.today().isoformat())

        # Vérifier que le booking est restauré (rollback défensif)
        verify_rollback_restores_values(
            db.session,
            Booking,
            booking.id,
            original_values,
        )

        # Vérifier que engine.run() a quand même créé un DispatchRun
        dispatch_run_id = result.get("dispatch_run_id") or result.get("meta", {}).get(
            "dispatch_run_id"
        )
        assert dispatch_run_id is not None, (
            "DispatchRun should be created despite rollback defensive"
        )

    def test_rollback_restores_assignment_after_dispatch_failure(
        self, db, company, drivers
    ):
        """✅ Test : Rollback restaure les assignments après échec de dispatch.

        Ce test vérifie que les assignments créés mais non commités sont
        correctement restaurés après un rollback.
        """
        booking = BookingFactory(company=company, driver_id=None)
        db.session.commit()

        # Capturer les valeurs originales
        original_values = capture_original_values(booking, ["driver_id", "status"])

        # Créer un assignment (mais ne pas committer)
        assignment = Assignment()
        assignment.booking_id = booking.id
        assignment.driver_id = drivers[0].id
        assignment.status = BookingStatus.ASSIGNED
        db.session.add(assignment)
        db.session.flush()

        # Modifier le booking
        booking.driver_id = drivers[0].id
        booking.status = BookingStatus.ASSIGNED
        db.session.flush()

        # Rollback
        db.session.rollback()

        # Vérifier que le booking est restauré
        verify_rollback_restores_values(
            db.session,
            Booking,
            booking.id,
            original_values,
        )

        # Vérifier que l'assignment n'existe pas (n'a jamais été commité)
        assignment_reloaded = (
            db.session.query(Assignment).filter_by(booking_id=booking.id).first()
        )
        assert assignment_reloaded is None, (
            "Assignment should not exist (never committed)"
        )
