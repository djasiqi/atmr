from __future__ import annotations

from typing import Any, Protocol, cast

from ext import db
from models import Booking, BookingStatus


class BookingWriterPort(Protocol):
    """Port: persistance d'un booking (Infrastructure).

    Ce port existe pour permettre à la couche Application d'orchestrer la création
    d'un booking sans dépendre de SQLAlchemy / `ext.db` / `models`.
    """

    def create_and_commit(
        self,
        *,
        user_id: int,
        client_id: int,
        company_id: int,
        customer_name: str,
        pickup_location: str,
        dropoff_location: str,
        scheduled_time: Any,
        amount: float,
        medical_facility: str,
        doctor_name: str,
        duration_seconds: int,
        distance_meters: int,
        pickup_lat: float,
        pickup_lon: float,
        dropoff_lat: float,
        dropoff_lon: float,
        is_round_trip: bool,
    ) -> Booking:
        """Crée et commit le booking (et son retour si round-trip)."""
        ...


class SqlAlchemyBookingWriter:
    """Implémentation SQLAlchemy du port de persistance booking."""

    def create_and_commit(
        self,
        *,
        user_id: int,
        client_id: int,
        company_id: int,
        customer_name: str,
        pickup_location: str,
        dropoff_location: str,
        scheduled_time: Any,
        amount: float,
        medical_facility: str,
        doctor_name: str,
        duration_seconds: int,
        distance_meters: int,
        pickup_lat: float,
        pickup_lon: float,
        dropoff_lat: float,
        dropoff_lon: float,
        is_round_trip: bool,
    ) -> Booking:
        new_booking = cast("Any", Booking)(
            customer_name=customer_name,
            pickup_location=pickup_location,
            dropoff_location=dropoff_location,
            scheduled_time=scheduled_time,
            amount=float(amount),
            status=BookingStatus.PENDING,
            user_id=user_id,
            client_id=client_id,
            company_id=company_id,
            medical_facility=medical_facility,
            doctor_name=doctor_name,
            duration_seconds=duration_seconds,
            distance_meters=distance_meters,
            is_return=False,
            pickup_lat=pickup_lat,
            pickup_lon=pickup_lon,
            dropoff_lat=dropoff_lat,
            dropoff_lon=dropoff_lon,
        )

        try:
            db.session.add(new_booking)
            db.session.flush()  # attribue new_booking.id

            if is_round_trip:
                return_booking = self._create_return_booking(
                    outbound_booking=new_booking,
                    user_id=user_id,
                    client_id=client_id,
                    duration_seconds=duration_seconds,
                    distance_meters=distance_meters,
                )
                db.session.add(return_booking)

            db.session.commit()
            return new_booking
        except Exception as e:
            with db.session.no_autoflush:
                db.session.rollback()
            raise e

        raise RuntimeError("Unreachable")

    def _create_return_booking(
        self,
        *,
        outbound_booking: Booking,
        user_id: int,
        client_id: int,
        duration_seconds: int,
        distance_meters: int,
    ) -> Booking:
        return_booking = cast("Any", Booking)(
            customer_name=outbound_booking.customer_name,
            pickup_location=outbound_booking.dropoff_location,
            dropoff_location=outbound_booking.pickup_location,
            scheduled_time=None,
            amount=0,
            status=BookingStatus.PENDING,
            is_return=True,
            parent_booking_id=outbound_booking.id,
            user_id=user_id,
            client_id=client_id,
            company_id=outbound_booking.company_id,
            duration_seconds=duration_seconds,
            distance_meters=distance_meters,
        )

        # Best effort coords
        try:
            return_booking.pickup_lat = outbound_booking.dropoff_lat
            return_booking.pickup_lon = outbound_booking.dropoff_lon
            return_booking.dropoff_lat = outbound_booking.pickup_lat
            return_booking.dropoff_lon = outbound_booking.pickup_lon
        except Exception:
            pass

        return return_booking
