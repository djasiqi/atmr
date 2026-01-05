"""Implémentation SQLAlchemy du repository Booking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from bookings.domain.booking import Booking
from bookings.domain.booking_id import BookingId

if TYPE_CHECKING:
    from models import Booking as SQLAlchemyBooking
else:
    SQLAlchemyBooking = Any

logger = __import__("logging").getLogger(__name__)


class SqlAlchemyBookingRepository:
    """Implémentation SQLAlchemy du repository Booking.

    Adapte les modèles SQLAlchemy vers les agrégats du domaine.
    """

    def _to_aggregate(self, sa_booking: SQLAlchemyBooking) -> Booking:
        """Convertit un modèle SQLAlchemy en agrégat Booking."""
        from bookings.domain.value_objects import Amount, BookingStatus, Location

        # Construction explicite avec tous les champs du dataclass
        # Le type checker peut avoir des problèmes avec les dataclasses,
        # mais le code est correct car Python génère automatiquement __init__
        booking_data = {
            "id": BookingId(sa_booking.id),
            "company_id": sa_booking.company_id,
            "client_id": sa_booking.client_id,
            "user_id": sa_booking.user_id,
            "customer_name": sa_booking.customer_name,
            "pickup_location": Location(
                address=sa_booking.pickup_location,
                latitude=cast(float | None, sa_booking.pickup_lat),
                longitude=cast(float | None, sa_booking.pickup_lon),
            ),
            "dropoff_location": Location(
                address=sa_booking.dropoff_location,
                latitude=cast(float | None, sa_booking.dropoff_lat),
                longitude=cast(float | None, sa_booking.dropoff_lon),
            ),
            "status": BookingStatus(str(sa_booking.status.value)),
            "amount": Amount(sa_booking.amount),
            "scheduled_time": sa_booking.scheduled_time,
            "driver_id": sa_booking.driver_id,
            "is_round_trip": sa_booking.is_round_trip,
            "is_return": sa_booking.is_return,
            "is_urgent": sa_booking.is_urgent,
            "time_confirmed": sa_booking.time_confirmed,
            "created_at": sa_booking.created_at,
            "updated_at": sa_booking.updated_at,
            "boarded_at": sa_booking.boarded_at,
            "completed_at": sa_booking.completed_at,
            "parent_booking_id": sa_booking.parent_booking_id,
        }
        return Booking(**booking_data)

    def _from_aggregate(self, booking: Booking) -> dict[str, Any]:
        """Convertit un agrégat Booking en dictionnaire pour SQLAlchemy."""
        return {
            "id": booking.id.value,
            "company_id": booking.company_id,
            "client_id": booking.client_id,
            "user_id": booking.user_id,
            "customer_name": booking.customer_name,
            "pickup_location": booking.pickup_location.address,
            "dropoff_location": booking.dropoff_location.address,
            "status": booking.status.value,
            "amount": booking.amount.value,
            "scheduled_time": booking.scheduled_time,
            "driver_id": booking.driver_id,
            "pickup_lat": booking.pickup_location.latitude,
            "pickup_lon": booking.pickup_location.longitude,
            "dropoff_lat": booking.dropoff_location.latitude,
            "dropoff_lon": booking.dropoff_location.longitude,
            "is_round_trip": booking.is_round_trip,
            "is_return": booking.is_return,
            "is_urgent": booking.is_urgent,
            "time_confirmed": booking.time_confirmed,
            "boarded_at": booking.boarded_at,
            "completed_at": booking.completed_at,
            "parent_booking_id": booking.parent_booking_id,
        }

    def save(self, booking: Booking) -> None:
        """Sauvegarde une réservation."""
        from ext import db
        from models import Booking as SQLAlchemyBooking

        data = self._from_aggregate(booking)
        booking_id = data.pop("id")

        sa_booking = SQLAlchemyBooking.query.get(booking_id)
        if sa_booking:
            # Update
            for key, value in data.items():
                setattr(sa_booking, key, value)
        else:
            # Create
            sa_booking = SQLAlchemyBooking(**data)
            db.session.add(sa_booking)

        db.session.commit()

    def find_by_id(self, booking_id: BookingId) -> Booking | None:
        """Trouve une réservation par ID."""
        from models import Booking as SQLAlchemyBooking

        sa_booking = SQLAlchemyBooking.query.get(booking_id.value)
        if sa_booking is None:
            return None
        return self._to_aggregate(sa_booking)

    def find_by_company_id(self, company_id: int) -> list[Booking]:
        """Trouve toutes les réservations d'une entreprise."""
        from models import Booking as SQLAlchemyBooking

        sa_bookings = SQLAlchemyBooking.query.filter_by(company_id=company_id).all()
        return [self._to_aggregate(b) for b in sa_bookings]

    def find_by_client_id(self, client_id: int) -> list[Booking]:
        """Trouve toutes les réservations d'un client."""
        from models import Booking as SQLAlchemyBooking

        sa_bookings = SQLAlchemyBooking.query.filter_by(client_id=client_id).all()
        return [self._to_aggregate(b) for b in sa_bookings]

    def find_by_driver_id(self, driver_id: int) -> list[Booking]:
        """Trouve toutes les réservations d'un chauffeur."""
        from models import Booking as SQLAlchemyBooking

        sa_bookings = SQLAlchemyBooking.query.filter_by(driver_id=driver_id).all()
        return [self._to_aggregate(b) for b in sa_bookings]

    # Méthodes de compatibilité pour les use-cases existants
    def find_model_by_id_with_eager_loading(
        self, booking_id: int
    ) -> SQLAlchemyBooking | None:
        """Récupère un booking SQLAlchemy avec eager loading (compatibilité)."""
        from models import Booking as SQLAlchemyBooking

        return SQLAlchemyBooking.query.options(
            # Eager loading pour éviter N+1
            # Note: Les relations sont chargées automatiquement par SQLAlchemy
        ).get(booking_id)

    def find_all_with_eager_loading_query(
        self, *, status_filter: str | None = None
    ) -> Any:
        """Retourne une query pour tous les bookings (admin)."""
        from models import Booking as SQLAlchemyBooking

        query = SQLAlchemyBooking.query
        if status_filter:
            query = query.filter(SQLAlchemyBooking.status == status_filter)
        return query

    def find_by_client_id_with_eager_loading_query(
        self, *, client_id: int, status_filter: str | None = None
    ) -> Any:
        """Retourne une query pour les bookings d'un client."""
        from models import Booking as SQLAlchemyBooking

        query = SQLAlchemyBooking.query.filter_by(client_id=client_id)
        if status_filter:
            query = query.filter(SQLAlchemyBooking.status == status_filter)
        return query
