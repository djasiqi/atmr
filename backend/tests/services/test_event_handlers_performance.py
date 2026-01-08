"""Tests de performance pour les handlers d'événements.

Vérifie que la latence ajoutée par les événements reste acceptable
(< 10ms par événement).
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from application.events.event_bus import publish_event, set_event_bus
from domain.events.events import (
    BookingCancelledEvent,
    BookingUpdatedEvent,
    DriverNewBookingEvent,
)
from infrastructure.events.in_memory_event_bus import InMemoryEventBus
from services import event_handlers_registry as registry


@pytest.fixture(autouse=True)
def reset_registry_and_bus():
    """Reset le registry et le bus avant chaque test."""
    registry._HANDLERS.clear()
    set_event_bus(InMemoryEventBus())
    yield
    registry._HANDLERS.clear()


class TestEventPerformance:
    """Tests de performance pour les événements."""

    def test_booking_updated_event_latency(self, db):
        """Test que BookingUpdatedEvent a une latence < 10ms."""
        from tests.factories import BookingFactory, CompanyFactory, DriverFactory

        # Créer un booking et un driver via factories
        company = CompanyFactory()
        driver = DriverFactory(company=company)
        booking = BookingFactory(
            company=company,
            driver=driver,
            customer_name="Test Customer",
        )
        db.session.flush()  # Flush pour obtenir les IDs

        # Mock notify_booking_update pour éviter les appels réels
        with patch("services.notification_service.notify_booking_update"):
            # Mesurer le temps d'exécution
            start_time = time.perf_counter()
            publish_event(
                BookingUpdatedEvent(
                    booking_id=booking.id,
                    driver_id=driver.id,
                    company_id=1,
                )
            )
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            assert latency_ms < 10, (
                f"Latence trop élevée: {latency_ms:.2f}ms (attendu < 10ms)"
            )

    def test_booking_cancelled_event_latency(self):
        """Test que BookingCancelledEvent a une latence < 10ms."""
        # Mock notify_booking_cancelled pour éviter les appels réels
        with patch("services.notification_service.notify_booking_cancelled"):
            # Mesurer le temps d'exécution
            start_time = time.perf_counter()
            publish_event(
                BookingCancelledEvent(
                    booking_id=123,
                    driver_id=456,
                    company_id=1,
                )
            )
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            assert latency_ms < 10, (
                f"Latence trop élevée: {latency_ms:.2f}ms (attendu < 10ms)"
            )

    def test_driver_new_booking_event_latency(self, db):
        """Test que DriverNewBookingEvent a une latence < 10ms."""
        import uuid

        from ext import bcrypt
        from models import Booking, Client, Driver, User

        # Créer un booking et un driver
        unique_id = str(uuid.uuid4()).replace("-", "")[:16]
        driver_user = User(
            email=f"driver_{unique_id}@test.com",
            username=f"driver_test_{unique_id}",
            role="driver",
        )
        driver_user.password = bcrypt.generate_password_hash("password123").decode(
            "utf-8"
        )
        db.session.add(driver_user)
        db.session.commit()

        driver = Driver(user_id=driver_user.id, company_id=1)
        db.session.add(driver)
        db.session.commit()

        # Créer un utilisateur client pour le booking
        client_user = User(
            email=f"client_{unique_id}@test.com",
            username=f"client_test_{unique_id}",
            role="client",
        )
        client_user.password = bcrypt.generate_password_hash("password123").decode(
            "utf-8"
        )
        db.session.add(client_user)
        db.session.commit()

        # Créer un objet Client associé à l'utilisateur client
        client = Client(
            user_id=client_user.id,
            company_id=1,
            billing_address="Test Billing Address",
        )
        db.session.add(client)
        db.session.commit()

        booking = Booking(
            company_id=1,
            driver_id=driver.id,
            user_id=client_user.id,
            client_id=client.id,
            pickup_location="Test Pickup",
            dropoff_location="Test Dropoff",
            customer_name="Test Customer",
            amount=50.0,
        )
        db.session.add(booking)
        db.session.commit()

        # Mock notify_driver_new_booking pour éviter les appels réels
        with patch("services.notification_service.notify_driver_new_booking"):
            # Mesurer le temps d'exécution
            start_time = time.perf_counter()
            publish_event(
                DriverNewBookingEvent(
                    booking_id=booking.id,
                    driver_id=driver.id,
                    company_id=1,
                )
            )
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            assert latency_ms < 10, (
                f"Latence trop élevée: {latency_ms:.2f}ms (attendu < 10ms)"
            )

    def test_multiple_events_throughput(self, db):
        """Test le débit de traitement de plusieurs événements."""
        import uuid

        from ext import bcrypt
        from models import Booking, Client, Driver, User

        # Créer un booking et un driver
        unique_id = str(uuid.uuid4()).replace("-", "")[:16]
        driver_user = User(
            email=f"driver_{unique_id}@test.com",
            username=f"driver_test_{unique_id}",
            role="driver",
        )
        driver_user.password = bcrypt.generate_password_hash("password123").decode(
            "utf-8"
        )
        db.session.add(driver_user)
        db.session.commit()

        driver = Driver(user_id=driver_user.id, company_id=1)
        db.session.add(driver)
        db.session.commit()

        # Créer un utilisateur client pour le booking
        client_user = User(
            email=f"client_{unique_id}@test.com",
            username=f"client_test_{unique_id}",
            role="client",
        )
        client_user.password = bcrypt.generate_password_hash("password123").decode(
            "utf-8"
        )
        db.session.add(client_user)
        db.session.commit()

        # Créer un objet Client associé à l'utilisateur client
        client = Client(
            user_id=client_user.id,
            company_id=1,
            billing_address="Test Billing Address",
        )
        db.session.add(client)
        db.session.commit()

        booking = Booking(
            company_id=1,
            driver_id=driver.id,
            user_id=client_user.id,
            client_id=client.id,
            pickup_location="Test Pickup",
            dropoff_location="Test Dropoff",
            customer_name="Test Customer",
            amount=50.0,
        )
        db.session.add(booking)
        db.session.commit()

        # Mock tous les handlers pour éviter les appels réels
        with (
            patch("services.notification_service.notify_booking_update"),
            patch("services.notification_service.notify_booking_cancelled"),
            patch("services.notification_service.notify_driver_new_booking"),
        ):
            # Publier 100 événements et mesurer le temps total
            num_events = 100
            start_time = time.perf_counter()

            for i in range(num_events):
                if i % 3 == 0:
                    publish_event(
                        BookingUpdatedEvent(
                            booking_id=booking.id,
                            driver_id=driver.id,
                            company_id=1,
                        )
                    )
                elif i % 3 == 1:
                    publish_event(
                        BookingCancelledEvent(
                            booking_id=booking.id,
                            driver_id=driver.id,
                            company_id=1,
                        )
                    )
                else:
                    publish_event(
                        DriverNewBookingEvent(
                            booking_id=booking.id,
                            driver_id=driver.id,
                            company_id=1,
                        )
                    )

            end_time = time.perf_counter()

            total_time_ms = (end_time - start_time) * 1000
            avg_latency_ms = total_time_ms / num_events

            # Vérifier que la latence moyenne est < 10ms
            assert avg_latency_ms < 10, (
                f"Latence moyenne trop élevée: {avg_latency_ms:.2f}ms (attendu < 10ms)"
            )

            # Vérifier que le débit est acceptable (> 10 événements/seconde)
            throughput = num_events / (total_time_ms / 1000)
            assert throughput > 10, (
                f"Débit trop faible: {throughput:.2f} événements/s (attendu > 10/s)"
            )

    def test_metrics_handler_performance(self):
        """Test que le metrics_handler n'ajoute pas de latence significative."""
        # Mesurer le temps d'exécution avec et sans metrics_handler
        event = BookingCancelledEvent(booking_id=123, driver_id=456, company_id=1)

        # Sans metrics_handler (désactiver temporairement)
        registry._HANDLERS.clear()
        start_time = time.perf_counter()
        publish_event(event)
        end_time = time.perf_counter()
        time_without_metrics = (end_time - start_time) * 1000

        # Réactiver les handlers (incluant metrics_handler)
        from services.events.handlers.booking_handlers import handle_booking_cancelled
        from services.events.handlers.metrics_handler import handle_event_metrics

        registry.register("BookingCancelledEvent", handle_booking_cancelled)
        registry.register("BookingCancelledEvent", handle_event_metrics)

        start_time = time.perf_counter()
        publish_event(event)
        end_time = time.perf_counter()
        time_with_metrics = (end_time - start_time) * 1000

        # La différence doit être < 5ms (overhead acceptable)
        overhead = time_with_metrics - time_without_metrics
        assert overhead < 5, (
            f"Overhead metrics_handler trop élevé: {overhead:.2f}ms (attendu < 5ms)"
        )
