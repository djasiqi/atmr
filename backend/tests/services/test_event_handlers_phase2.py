"""Tests unitaires pour les handlers d'événements Phase 2.

Tests pour :
- BookingUpdatedEvent handler
- BookingCancelledEvent handler
- DriverNewBookingEvent handler
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


class TestBookingUpdatedHandler:
    """Tests pour handle_booking_updated."""

    def test_handle_booking_updated_success(self, db):
        """Test que handle_booking_updated notifie correctement le driver."""
        import uuid

        from ext import bcrypt
        from models import Booking, Client, Driver, User
        from services.event_handlers.booking_handlers import handle_booking_updated

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

        # Mock notify_booking_update
        with patch(
            "services.notification_service.notify_booking_update"
        ) as mock_notify:
            event = {
                "event_type": "BookingUpdatedEvent",
                "booking_id": booking.id,
                "driver_id": driver.id,
                "company_id": 1,
            }
            handle_booking_updated(event)

            # Vérifier que notify_booking_update a été appelé
            mock_notify.assert_called_once()
            call_args = mock_notify.call_args[0]
            assert call_args[0] == driver.id
            assert call_args[1].id == booking.id

    def test_handle_booking_updated_missing_fields(self):
        """Test que handle_booking_updated gère les champs manquants."""
        from services.event_handlers.booking_handlers import handle_booking_updated

        # Event sans booking_id
        event = {"event_type": "BookingUpdatedEvent", "driver_id": 1}
        with patch("services.event_handlers.booking_handlers.logger") as mock_logger:
            handle_booking_updated(event)
            mock_logger.warning.assert_called_once()

    def test_handle_booking_updated_booking_not_found(self, db):
        """Test que handle_booking_updated gère le cas où le booking n'existe pas."""
        from services.event_handlers.booking_handlers import handle_booking_updated

        event = {
            "event_type": "BookingUpdatedEvent",
            "booking_id": 99999,  # ID inexistant
            "driver_id": 1,
            "company_id": 1,
        }

        with patch(
            "services.notification_service.notify_booking_update"
        ) as mock_notify:
            handle_booking_updated(event)
            # Ne doit pas appeler notify si le booking n'existe pas
            mock_notify.assert_not_called()

    def test_handle_booking_updated_exception_handling(self, db):
        """Test que handle_booking_updated gère les exceptions."""
        import uuid

        from ext import bcrypt
        from models import Booking, Client, Driver, User
        from services.event_handlers.booking_handlers import handle_booking_updated

        # Créer un booking
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

        # Mock notify_booking_update pour lever une exception
        with (
            patch(
                "services.notification_service.notify_booking_update",
                side_effect=Exception("Notification failed"),
            ),
            patch("services.event_handlers.booking_handlers.logger") as mock_logger,
        ):
            event = {
                "event_type": "BookingUpdatedEvent",
                "booking_id": booking.id,
                "driver_id": driver.id,
                "company_id": 1,
            }
            handle_booking_updated(event)

            # Vérifier que l'exception est loggée mais n'est pas propagée
            # Le handler utilise logger.exception() pour les exceptions génériques
            mock_logger.exception.assert_called_once()
            assert "Failed to notify driver" in str(mock_logger.exception.call_args)


class TestBookingCancelledHandler:
    """Tests pour handle_booking_cancelled."""

    def test_handle_booking_cancelled_success(self):
        """Test que handle_booking_cancelled notifie correctement le driver."""
        from services.event_handlers.booking_handlers import handle_booking_cancelled

        event = {
            "event_type": "BookingCancelledEvent",
            "booking_id": 123,
            "driver_id": 456,
            "company_id": 1,
        }

        with patch(
            "services.notification_service.notify_booking_cancelled"
        ) as mock_notify:
            handle_booking_cancelled(event)

            # Vérifier que notify_booking_cancelled a été appelé
            mock_notify.assert_called_once_with(456, 123)

    def test_handle_booking_cancelled_missing_fields(self):
        """Test que handle_booking_cancelled gère les champs manquants."""
        from services.event_handlers.booking_handlers import handle_booking_cancelled

        # Event sans booking_id
        event = {"event_type": "BookingCancelledEvent", "driver_id": 1}
        with patch("services.event_handlers.booking_handlers.logger") as mock_logger:
            handle_booking_cancelled(event)
            mock_logger.warning.assert_called_once()

    def test_handle_booking_cancelled_exception_handling(self):
        """Test que handle_booking_cancelled gère les exceptions."""
        from services.event_handlers.booking_handlers import handle_booking_cancelled

        event = {
            "event_type": "BookingCancelledEvent",
            "booking_id": 123,
            "driver_id": 456,
            "company_id": 1,
        }

        # Mock notify_booking_cancelled pour lever une exception
        with (
            patch(
                "services.notification_service.notify_booking_cancelled",
                side_effect=Exception("Notification failed"),
            ),
            patch("services.event_handlers.booking_handlers.logger") as mock_logger,
        ):
            handle_booking_cancelled(event)

            # Vérifier que l'exception est loggée mais n'est pas propagée
            # Le handler utilise logger.exception() pour les exceptions génériques
            mock_logger.exception.assert_called_once()
            assert "Failed to notify driver" in str(mock_logger.exception.call_args)


class TestDriverNewBookingHandler:
    """Tests pour handle_driver_new_booking."""

    def test_handle_driver_new_booking_success(self, db):
        """Test que handle_driver_new_booking notifie correctement le driver."""
        import uuid

        from ext import bcrypt
        from models import Booking, Client, Driver, User
        from services.event_handlers.driver_handlers import handle_driver_new_booking

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

        # Mock notify_driver_new_booking
        with patch(
            "services.notification_service.notify_driver_new_booking"
        ) as mock_notify:
            event = {
                "event_type": "DriverNewBookingEvent",
                "booking_id": booking.id,
                "driver_id": driver.id,
                "company_id": 1,
            }
            handle_driver_new_booking(event)

            # Vérifier que notify_driver_new_booking a été appelé
            mock_notify.assert_called_once()
            call_args = mock_notify.call_args[0]
            assert call_args[0] == driver.id
            assert call_args[1].id == booking.id

    def test_handle_driver_new_booking_missing_fields(self):
        """Test que handle_driver_new_booking gère les champs manquants."""
        from services.event_handlers.driver_handlers import handle_driver_new_booking

        # Event sans booking_id
        event = {"event_type": "DriverNewBookingEvent", "driver_id": 1}
        with patch("services.event_handlers.driver_handlers.logger") as mock_logger:
            handle_driver_new_booking(event)
            mock_logger.warning.assert_called_once()

    def test_handle_driver_new_booking_booking_not_found(self, db):
        """Test que handle_driver_new_booking gère le cas où le booking n'existe pas."""
        from services.event_handlers.driver_handlers import handle_driver_new_booking

        event = {
            "event_type": "DriverNewBookingEvent",
            "booking_id": 99999,  # ID inexistant
            "driver_id": 1,
            "company_id": 1,
        }

        with patch(
            "services.notification_service.notify_driver_new_booking"
        ) as mock_notify:
            handle_driver_new_booking(event)
            # Ne doit pas appeler notify si le booking n'existe pas
            mock_notify.assert_not_called()

    def test_handle_driver_new_booking_exception_handling(self, db):
        """Test que handle_driver_new_booking gère les exceptions."""
        import uuid

        from ext import bcrypt
        from models import Booking, Client, Driver, User
        from services.event_handlers.driver_handlers import handle_driver_new_booking

        # Créer un booking
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

        # Mock notify_driver_new_booking pour lever une exception
        with (
            patch(
                "services.notification_service.notify_driver_new_booking",
                side_effect=Exception("Notification failed"),
            ),
            patch("services.event_handlers.driver_handlers.logger") as mock_logger,
        ):
            event = {
                "event_type": "DriverNewBookingEvent",
                "booking_id": booking.id,
                "driver_id": driver.id,
                "company_id": 1,
            }
            handle_driver_new_booking(event)

            # Vérifier que l'exception est loggée mais n'est pas propagée
            # Le handler utilise logger.exception() pour les exceptions génériques
            mock_logger.exception.assert_called_once()
            assert "Failed to notify driver" in str(mock_logger.exception.call_args)


class TestEventIntegration:
    """Tests d'intégration pour vérifier le flux complet événements → handlers."""

    def test_booking_updated_event_flow(self, db):
        """Test le flux complet : publication → handler → notification."""
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

        # Réenregistrer les handlers après le clear du registry
        from services.event_handlers.booking_handlers import handle_booking_updated

        registry.register("BookingUpdatedEvent", handle_booking_updated)

        # Mock notify_booking_update
        with patch(
            "services.notification_service.notify_booking_update"
        ) as mock_notify:
            # Publier l'événement
            publish_event(
                BookingUpdatedEvent(
                    booking_id=booking.id,
                    driver_id=driver.id,
                    company_id=1,
                )
            )

            # Vérifier que le handler a été appelé
            mock_notify.assert_called_once()

    def test_booking_cancelled_event_flow(self):
        """Test le flux complet : publication → handler → notification."""
        # Réenregistrer les handlers après le clear du registry
        from services.event_handlers.booking_handlers import handle_booking_cancelled

        registry.register("BookingCancelledEvent", handle_booking_cancelled)

        # Mock notify_booking_cancelled
        with patch(
            "services.notification_service.notify_booking_cancelled"
        ) as mock_notify:
            # Publier l'événement
            publish_event(
                BookingCancelledEvent(
                    booking_id=123,
                    driver_id=456,
                    company_id=1,
                )
            )

            # Vérifier que le handler a été appelé
            mock_notify.assert_called_once_with(456, 123)

    def test_driver_new_booking_event_flow(self, db):
        """Test le flux complet : publication → handler → notification."""
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

        # Réenregistrer les handlers après le clear du registry
        from services.event_handlers.driver_handlers import handle_driver_new_booking

        registry.register("DriverNewBookingEvent", handle_driver_new_booking)

        # Mock notify_driver_new_booking
        with patch(
            "services.notification_service.notify_driver_new_booking"
        ) as mock_notify:
            # Publier l'événement
            publish_event(
                DriverNewBookingEvent(
                    booking_id=booking.id,
                    driver_id=driver.id,
                    company_id=1,
                )
            )

            # Vérifier que le handler a été appelé
            mock_notify.assert_called_once()
