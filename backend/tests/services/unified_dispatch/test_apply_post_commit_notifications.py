# backend/tests/services/unified_dispatch/test_apply_post_commit_notifications.py
"""Tests pour validation P0: Isolation effets externes (notifications post-commit)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from sqlalchemy.exc import IntegrityError

from models import Booking, BookingStatus
from services.unified_dispatch.optimization.assignment_applier import apply_assignments
from tests.factories import BookingFactory, CompanyFactory, DriverFactory


@pytest.fixture(autouse=True)
def _app_context(app: Flask):
    """Assure que tous les tests s'exécutent dans un app context."""
    with app.app_context():
        yield


@pytest.fixture
def company(db):
    """Créer une entreprise pour les tests."""
    company = CompanyFactory()
    db.session.flush()
    return company


@pytest.fixture
def driver(db, company):
    """Créer un chauffeur pour les tests."""
    driver = DriverFactory(company=company, is_active=True, is_available=True)
    db.session.flush()
    return driver


@pytest.fixture
def bookings(db, company):
    """Créer plusieurs bookings pour les tests."""
    bookings_list = [
        BookingFactory(company=company, status=BookingStatus.ACCEPTED),
        BookingFactory(company=company, status=BookingStatus.ACCEPTED),
    ]
    db.session.flush()
    # ✅ Commit pour s'assurer que les bookings sont persistés
    db.session.commit()
    return bookings_list


class TestPostCommitNotifications:
    """Tests pour vérifier que les notifications sont émises APRÈS commit uniquement."""

    @patch("services.unified_dispatch.optimization.assignment_applier.publish_event")
    @patch("services.unified_dispatch.optimization.assignment_applier._in_tx", return_value=False)
    def test_notifications_emises_apres_commit_reussi(
        self, mock_in_tx, mock_publish_event, db, company, driver, bookings
    ):
        """Test 1: Notifications envoyées après commit réussi (had_existing_tx=False)."""
        # ✅ S'assurer que les bookings ne sont pas déjà assignés
        for booking in bookings:
            booking.driver_id = None
            booking.status = BookingStatus.ACCEPTED
        db.session.commit()

        # ✅ Debug: Vérifier les bookings avant apply_assignments
        print("\n=== DEBUG TEST 1 - BEFORE ===")
        print(f"company.id = {company.id}")
        print(f"driver.id = {driver.id}")
        print(f"driver.is_active = {driver.is_active}")
        print(f"driver.is_available = {driver.is_available}")
        for i, booking in enumerate(bookings):
            print(f"booking[{i}].id = {booking.id}")
            print(f"booking[{i}].company_id = {booking.company_id}")
            print(f"booking[{i}].status = {booking.status}")
            print(f"booking[{i}].driver_id = {booking.driver_id}")
            print(f"booking[{i}].user_id = {getattr(booking, 'user_id', 'N/A')}")
            print(f"booking[{i}].client_id = {getattr(booking, 'client_id', 'N/A')}")
            # Vérifier si le booking existe en DB
            db_booking = db.session.query(Booking).filter_by(id=booking.id).first()
            print(f"booking[{i}] exists in DB = {db_booking is not None}")
            if db_booking:
                print(f"booking[{i}] DB company_id = {db_booking.company_id}")
                print(f"booking[{i}] DB status = {db_booking.status}")
        print("=============================\n")

        # Préparer des assignations valides
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
        ]

        print("=== DEBUG TEST 1 - ASSIGNMENTS ===")
        for i, a in enumerate(assignments):
            print(f"assignment[{i}].booking_id = {a.booking_id}")
            print(f"assignment[{i}].driver_id = {a.driver_id}")
            print(f"assignment[{i}].score = {a.score}")
        print("==================================\n")

        # ✅ S'assurer que les bookings sont bien en DB avant apply_assignments
        db.session.expire_all()
        for i, booking in enumerate(bookings):
            db_booking = db.session.query(Booking).filter_by(id=booking.id).first()
            if db_booking is None:
                print(f"ERROR: booking[{i}] with id={booking.id} not found in DB!")
            else:
                print(
                    f"OK: booking[{i}] with id={booking.id} found in DB, company_id={db_booking.company_id}, status={db_booking.status}"
                )

        # Appliquer les assignations
        result = apply_assignments(
            company_id=company.id,
            assignments=assignments,
            return_pairs=True,  # Important: retourner applied_pairs pour notifications
            respect_existing=False,  # Ne pas respecter les assignations existantes
            enforce_driver_checks=False,  # Ne pas vérifier la disponibilité du driver
        )

        # ✅ Debug: Vérifier le contenu de result
        print("\n=== DEBUG TEST 1 - AFTER ===")
        print(f"result keys = {list(result.keys())}")
        print(f"applied_pairs in result = {'applied_pairs' in result}")
        if "applied_pairs" in result:
            print(f"applied_pairs = {result['applied_pairs']}")
            print(f"len(applied_pairs) = {len(result['applied_pairs'])}")
        print(f"applied = {result.get('applied', [])}")
        print(f"skipped = {result.get('skipped', {})}")
        print(f"conflicts = {result.get('conflicts', [])}")
        print(f"mock_publish_event.called = {mock_publish_event.called}")
        print(f"mock_publish_event.call_count = {mock_publish_event.call_count}")
        print("=============================\n")

        # ✅ Vérifier que publish_event a été appelé (après commit réussi)
        assert mock_publish_event.called, (
            "publish_event should be called after successful commit"
        )

        # ✅ Vérifier le nombre d'appels (2 bookings = 2 notifications)
        call_count = mock_publish_event.call_count
        assert call_count == 2, f"Expected 2 notifications, got {call_count}"

        # ✅ Vérifier que les arguments sont corrects
        calls = mock_publish_event.call_args_list
        for i, call in enumerate(calls):
            event = call[0][0]  # Premier argument de publish_event
            assert event.booking_id == bookings[i].id
            assert event.driver_id == driver.id
            assert event.company_id == company.id

        # ✅ Vérifier que les bookings sont bien assignés
        db.session.expire_all()
        booking0 = db.session.query(Booking).filter_by(id=bookings[0].id).first()
        booking1 = db.session.query(Booking).filter_by(id=bookings[1].id).first()
        assert booking0.driver_id == driver.id
        assert booking1.driver_id == driver.id

    @patch("services.unified_dispatch.optimization.assignment_applier.publish_event")
    @patch("services.unified_dispatch.optimization.assignment_applier._in_tx", return_value=False)
    def test_aucune_notification_si_rollback(
        self, mock_in_tx, mock_publish_event, db, company, driver, bookings
    ):
        """Test 2: Aucune notification envoyée si transaction rollback."""
        # ✅ S'assurer que les bookings ne sont pas déjà assignés
        for booking in bookings:
            booking.driver_id = None
            booking.status = BookingStatus.ACCEPTED
        db.session.commit()

        # Préparer des assignations valides
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
        ]

        # ✅ Forcer une erreur DB pour provoquer un rollback
        # Mock db.session.commit() pour lever une exception
        original_commit = db.session.commit

        def mock_commit():
            raise IntegrityError("Simulated DB error", None, None)

        db.session.commit = mock_commit

        try:
            # Tenter l'application (devrait échouer avec rollback)
            apply_assignments(
                company_id=company.id,
                assignments=assignments,
                return_pairs=True,
                respect_existing=False,
                enforce_driver_checks=False,
            )
        except IntegrityError:
            # Exception attendue
            pass
        finally:
            # Restaurer la méthode originale
            db.session.commit = original_commit

        # ✅ Vérifier que publish_event N'A PAS été appelé (rollback = pas de commit)
        assert not mock_publish_event.called, (
            "publish_event should NOT be called if transaction rollback"
        )

        # ✅ Vérifier que les bookings ne sont PAS assignés (rollback)
        db.session.expire_all()
        booking0 = db.session.query(Booking).filter_by(id=bookings[0].id).first()
        assert booking0.driver_id is None, (
            "Booking should not be assigned after rollback"
        )

    @patch("services.unified_dispatch.optimization.assignment_applier.publish_event")
    def test_notifications_deferred_si_transaction_externe(
        self, mock_publish_event, db, company, driver, bookings
    ):
        """Test 3: Pas d'émission si had_existing_tx=True, deferred_notifications dans result."""
        # Préparer des assignations valides
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
        ]

        # ✅ Commit les bookings avant apply_assignments
        db.session.commit()

        # ✅ Simuler une transaction externe (had_existing_tx=True)
        # En démarrant une transaction avant apply_assignments
        # Note: On utilise db.session.begin() pour créer une transaction externe
        # qui sera détectée par _in_tx() dans apply_assignments
        with db.session.begin():  # Transaction externe
            result = apply_assignments(
                company_id=company.id,
                assignments=assignments,
                return_pairs=True,
                respect_existing=False,
            )

        # ✅ Vérifier que publish_event N'A PAS été appelé (transaction externe)
        assert not mock_publish_event.called, (
            "publish_event should NOT be called if had_existing_tx=True"
        )

        # ✅ Vérifier que deferred_notifications est présent dans result
        assert "deferred_notifications" in result, (
            "deferred_notifications should be in result when had_existing_tx=True"
        )
        deferred = result["deferred_notifications"]
        assert deferred["company_id"] == company.id
        assert len(deferred["applied_pairs"]) == 1
        assert deferred["applied_pairs"][0][0] == bookings[0].id
        assert deferred["applied_pairs"][0][1] == driver.id

        # ✅ Vérifier que les bookings sont assignés (transaction externe commitée automatiquement par le context manager)
        db.session.expire_all()
        booking0 = db.session.query(Booking).filter_by(id=bookings[0].id).first()
        assert booking0.driver_id == driver.id

    @patch("services.unified_dispatch.optimization.assignment_applier.publish_event")
    @patch("services.unified_dispatch.optimization.assignment_applier._in_tx", return_value=False)
    def test_idempotence_driver_changed(
        self, mock_in_tx, mock_publish_event, db, company, driver, bookings
    ):
        """Test 4: Notification skip si driver_id a changé entre commit et émission."""
        # ✅ S'assurer que les bookings ne sont pas déjà assignés
        for booking in bookings:
            booking.driver_id = None
            booking.status = BookingStatus.ACCEPTED
        db.session.commit()

        # Créer un deuxième driver
        driver2 = DriverFactory(company=company, is_active=True, is_available=True)
        db.session.flush()

        # Préparer une assignation
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
        ]

        # Appliquer l'assignation (première fois - devrait émettre)
        result = apply_assignments(
            company_id=company.id,
            assignments=assignments,
            return_pairs=True,
            respect_existing=False,
            enforce_driver_checks=False,
        )

        # ✅ Vérifier que publish_event a été appelé la première fois
        initial_call_count = mock_publish_event.call_count
        assert initial_call_count == 1, (
            f"Expected 1 call after apply, got {initial_call_count}"
        )

        # ✅ Modifier le driver_id du booking APRÈS commit (simule changement entre commit et émission)
        db.session.expire_all()
        booking0 = db.session.query(Booking).filter_by(id=bookings[0].id).first()
        booking0.driver_id = driver2.id  # Changer le driver
        db.session.commit()

        # ✅ Réémettre les notifications (simule le comportement post-commit)
        from services.unified_dispatch.optimization.assignment_applier import (
            _emit_notifications_after_commit,
        )

        applied_pairs = result.get("applied_pairs", [])
        _emit_notifications_after_commit(applied_pairs, company.id)

        # ✅ Vérifier que publish_event N'A PAS été appelé une deuxième fois
        # (driver_id changé = skip de la notification)
        final_call_count = mock_publish_event.call_count
        assert final_call_count == initial_call_count, (
            f"Expected no additional calls after driver change, "
            f"got {final_call_count} calls (was {initial_call_count})"
        )

    @patch("services.unified_dispatch.optimization.assignment_applier.publish_event")
    @patch("services.unified_dispatch.optimization.assignment_applier._in_tx", return_value=False)
    def test_notifications_idempotentes_pas_de_duplication(
        self, mock_in_tx, mock_publish_event, db, company, driver, bookings
    ):
        """Test 5: Notifications idempotentes (pas de duplication)."""
        # ✅ S'assurer que les bookings ne sont pas déjà assignés
        for booking in bookings:
            booking.driver_id = None
            booking.status = BookingStatus.ACCEPTED
        db.session.commit()

        # Préparer une assignation
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
        ]

        # Appliquer l'assignation (première fois)
        _ = apply_assignments(  # noqa: F841
            company_id=company.id,
            assignments=assignments,
            return_pairs=True,
            respect_existing=False,
            enforce_driver_checks=False,
        )

        # ✅ Réappliquer la même assignation (idempotence)
        # Note: La deuxième fois, le booking est déjà assigné, donc skipé
        # On s'attend à 1 notification (première fois seulement)
        _ = apply_assignments(  # noqa: F841
            company_id=company.id,
            assignments=assignments,
            return_pairs=True,
            respect_existing=True,  # Respecter l'assignation existante
        )

        # ✅ Vérifier que publish_event a été appelé exactement 1 fois
        # (seulement la première fois, la deuxième fois le booking est skipé)
        assert mock_publish_event.call_count == 1, (
            f"Expected 1 call (first apply only, second is skipped), got {mock_publish_event.call_count}"
        )

        # ✅ Vérifier que l'appel a les bons arguments
        calls = mock_publish_event.call_args_list
        assert len(calls) == 1
        event = calls[0][0][0]
        assert event.booking_id == bookings[0].id
        assert event.driver_id == driver.id
        assert event.company_id == company.id

    @patch("services.unified_dispatch.optimization.assignment_applier.publish_event")
    @patch("services.unified_dispatch.optimization.assignment_applier._in_tx", return_value=False)
    def test_metriques_prometheus_si_disponible(
        self, mock_in_tx, mock_publish_event, db, company, driver, bookings
    ):
        """Test 6: Métriques Prometheus enregistrées si disponibles."""
        # ✅ S'assurer que les bookings ne sont pas déjà assignés
        for booking in bookings:
            booking.driver_id = None
            booking.status = BookingStatus.ACCEPTED
        db.session.commit()

        # Préparer une assignation
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
        ]

        # ✅ Vérifier que les métriques sont utilisées si Prometheus disponible
        from services.unified_dispatch.optimization.assignment_applier import (
            NOTIF_EMITTED,
            NOTIF_LATENCY,
        )

        # Si Prometheus non disponible, skip le test
        if NOTIF_EMITTED is None or NOTIF_LATENCY is None:
            pytest.skip(
                "Prometheus metrics not available (prometheus_client not installed)"
            )

        # Mock des méthodes labels() et inc()/observe() pour vérifier les appels
        original_emitted_labels = NOTIF_EMITTED.labels
        original_latency_labels = NOTIF_LATENCY.labels

        mock_emitted_labels = MagicMock()
        mock_emitted_inc = MagicMock()
        mock_emitted_labels.return_value.inc = mock_emitted_inc
        NOTIF_EMITTED.labels = mock_emitted_labels

        mock_latency_labels = MagicMock()
        mock_latency_observe = MagicMock()
        mock_latency_labels.return_value.observe = mock_latency_observe
        NOTIF_LATENCY.labels = mock_latency_labels

        try:
            # Appliquer l'assignation
            _ = apply_assignments(  # noqa: F841
                company_id=company.id,
                assignments=assignments,
                return_pairs=True,
                respect_existing=False,
            )

            # ✅ Vérifier que les métriques sont utilisées
            assert mock_emitted_labels.called, "NOTIF_EMITTED.labels should be called"
            assert mock_emitted_inc.called, "NOTIF_EMITTED.inc should be called"
            assert mock_latency_labels.called, "NOTIF_LATENCY.labels should be called"
            assert mock_latency_observe.called, "NOTIF_LATENCY.observe should be called"
        finally:
            # Restaurer les méthodes originales
            NOTIF_EMITTED.labels = original_emitted_labels
            NOTIF_LATENCY.labels = original_latency_labels

    @patch("services.unified_dispatch.optimization.assignment_applier.publish_event")
    @patch("services.unified_dispatch.optimization.assignment_applier._in_tx", return_value=True)
    def test_logs_ameliores_contexte_supplementaire(
        self, mock_in_tx, mock_publish_event, db, company, driver, bookings, caplog
    ):
        """Test 7: Logs améliorés avec contexte supplémentaire (P1)."""
        # ✅ S'assurer que les bookings ne sont pas déjà assignés
        for booking in bookings:
            booking.driver_id = None
            booking.status = BookingStatus.ACCEPTED
        db.session.commit()

        # Préparer une assignation
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
        ]

        # ✅ Forcer une erreur dans _apply_assignments_inner pour vérifier les logs améliorés
        # On mocke _apply_assignments_inner pour lever une exception
        def mock_apply_inner(*args, **kwargs):
            raise IntegrityError("Simulated DB error", None, None)

        with patch(
            "services.unified_dispatch.optimization.assignment_applier._apply_assignments_inner",
            side_effect=mock_apply_inner,
        ):
            try:
                with caplog.at_level("ERROR"):
                    apply_assignments(
                        company_id=company.id,
                        assignments=assignments,
                        dispatch_run_id=123,  # Pour vérifier dans les logs
                        respect_existing=False,
                        enforce_driver_checks=False,
                    )
            except IntegrityError:
                pass

        # ✅ Vérifier que les logs contiennent le contexte supplémentaire
        error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_logs) > 0, "Should have error logs"

        # ✅ Vérifier que les logs contiennent les informations de contexte
        log_message = error_logs[0].message
        assert (
            "Assignments count" in log_message
            or "assignments count" in log_message.lower()
        ), "Log should contain 'Assignments count'"
        assert (
            "had_existing_tx" in log_message or "had_existing_tx" in log_message.lower()
        ), "Log should contain 'had_existing_tx'"
        assert (
            "dispatch_run_id" in log_message or "dispatch_run_id" in log_message.lower()
        ), "Log should contain 'dispatch_run_id'"
