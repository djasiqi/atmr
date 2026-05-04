from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, cast

from ext import db
from models import Booking, BookingStatus


def _split_round_trip_total_amount(total: float) -> tuple[float, float]:
    """Répartit le tarif aller-retour 50/50 avec arrondi CHF et somme exacte.

    Si le total est insuffisant pour deux segments au minimum légal (0,50 CHF),
    tout reste sur l'aller et le retour est à 0,00 CHF (comportement historique).
    """
    total_r = round(float(total), 2)
    min_leg = 0.5
    if total_r < 2 * min_leg - 1e-9:
        return total_r, 0.0
    first = round(total_r / 2.0, 2)
    second = round(total_r - first, 2)
    return first, second


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
        hospital_service: str,
        duration_seconds: int,
        distance_meters: int,
        pickup_lat: float,
        pickup_lon: float,
        dropoff_lat: float,
        dropoff_lon: float,
        is_round_trip: bool,
        pickup_admin_token: str | None,
        pickup_canton_code: str | None,
        pickup_admin_source: str | None,
        pickup_admin_confidence: str | None,
        pickup_admin_label: str | None,
        pickup_admin_resolved_at: datetime | None,
        dropoff_admin_token: str | None,
        dropoff_canton_code: str | None,
        dropoff_admin_source: str | None,
        dropoff_admin_confidence: str | None,
        dropoff_admin_label: str | None,
        dropoff_admin_resolved_at: datetime | None,
        pickup_geo_unit_id: int | None,
        dropoff_geo_unit_id: int | None,
        pricing_profile_id: int | None,
        pricing_profile_version_id: int | None,
        price_amount: float | None,
        price_breakdown_json: dict[str, Any] | None,
        notes_medical: str | None = None,
        return_scheduled_time: Any | None = None,
        return_time_exact: bool = False,
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
        hospital_service: str,
        duration_seconds: int,
        distance_meters: int,
        pickup_lat: float,
        pickup_lon: float,
        dropoff_lat: float,
        dropoff_lon: float,
        is_round_trip: bool,
        pickup_admin_token: str | None,
        pickup_canton_code: str | None,
        pickup_admin_source: str | None,
        pickup_admin_confidence: str | None,
        pickup_admin_label: str | None,
        pickup_admin_resolved_at: datetime | None,
        dropoff_admin_token: str | None,
        dropoff_canton_code: str | None,
        dropoff_admin_source: str | None,
        dropoff_admin_confidence: str | None,
        dropoff_admin_label: str | None,
        dropoff_admin_resolved_at: datetime | None,
        pickup_geo_unit_id: int | None,
        dropoff_geo_unit_id: int | None,
        pricing_profile_id: int | None,
        pricing_profile_version_id: int | None,
        price_amount: float | None,
        price_breakdown_json: dict[str, Any] | None,
        notes_medical: str | None = None,
        return_scheduled_time: Any | None = None,
        return_time_exact: bool = False,
    ) -> Booking:
        outbound_amount = float(amount)
        return_amount = 0.0
        outbound_price_amount = price_amount
        return_price_amount: float | None = None
        if is_round_trip:
            outbound_amount, return_amount = _split_round_trip_total_amount(
                float(amount)
            )
            if price_amount is not None:
                outbound_price_amount, return_price_amount = (
                    _split_round_trip_total_amount(float(price_amount))
                )

        new_booking = cast("Any", Booking)(
            customer_name=customer_name,
            pickup_location=pickup_location,
            dropoff_location=dropoff_location,
            scheduled_time=scheduled_time,
            amount=outbound_amount,
            status=BookingStatus.PENDING,
            user_id=user_id,
            client_id=client_id,
            company_id=company_id,
            medical_facility=medical_facility,
            doctor_name=doctor_name,
            hospital_service=hospital_service or None,
            notes_medical=notes_medical,
            duration_seconds=duration_seconds,
            distance_meters=distance_meters,
            is_return=False,
            is_round_trip=bool(is_round_trip),
            pickup_lat=pickup_lat,
            pickup_lon=pickup_lon,
            dropoff_lat=dropoff_lat,
            dropoff_lon=dropoff_lon,
            pickup_admin_token=pickup_admin_token,
            pickup_canton_code=pickup_canton_code,
            pickup_admin_source=pickup_admin_source,
            pickup_admin_confidence=pickup_admin_confidence,
            pickup_admin_label=pickup_admin_label,
            pickup_admin_resolved_at=pickup_admin_resolved_at,
            dropoff_admin_token=dropoff_admin_token,
            dropoff_canton_code=dropoff_canton_code,
            dropoff_admin_source=dropoff_admin_source,
            dropoff_admin_confidence=dropoff_admin_confidence,
            dropoff_admin_label=dropoff_admin_label,
            dropoff_admin_resolved_at=dropoff_admin_resolved_at,
            pickup_geo_unit_id=pickup_geo_unit_id,
            dropoff_geo_unit_id=dropoff_geo_unit_id,
            pricing_profile_id=pricing_profile_id,
            pricing_profile_version_id=pricing_profile_version_id,
            price_amount=outbound_price_amount,
            price_breakdown_json=price_breakdown_json,
        )

        def _write_core() -> None:
            db.session.add(new_booking)
            db.session.flush()  # attribue new_booking.id

            if is_round_trip:
                return_booking = self._create_return_booking(
                    outbound_booking=new_booking,
                    user_id=user_id,
                    client_id=client_id,
                    duration_seconds=duration_seconds,
                    distance_meters=distance_meters,
                    return_scheduled_time=return_scheduled_time,
                    return_time_exact=return_time_exact,
                    return_leg_amount=return_amount,
                    return_price_amount=return_price_amount,
                )
                db.session.add(return_booking)

        get_transaction = getattr(db.session, "get_transaction", None)
        owns_transaction = not bool(
            get_transaction() if callable(get_transaction) else False
        )
        if owns_transaction:
            with db.session.begin():
                _write_core()
        else:
            _write_core()
        return new_booking

    def _create_return_booking(
        self,
        *,
        outbound_booking: Booking,
        user_id: int,
        client_id: int,
        duration_seconds: int,
        distance_meters: int,
        return_scheduled_time: Any | None = None,
        return_time_exact: bool = False,
        return_leg_amount: float = 0.0,
        return_price_amount: float | None = None,
    ) -> Booking:
        # Ordre des kwargs : is_return / parent_booking_id / time_confirmed AVANT scheduled_time,
        # sinon @validates("scheduled_time") s'exécute avec is_return encore à False → ValueError.
        return_booking = cast("Any", Booking)(
            customer_name=outbound_booking.customer_name,
            pickup_location=outbound_booking.dropoff_location,
            dropoff_location=outbound_booking.pickup_location,
            is_return=True,
            parent_booking_id=outbound_booking.id,
            time_confirmed=bool(return_time_exact),
            scheduled_time=return_scheduled_time,
            amount=float(return_leg_amount),
            status=BookingStatus.PENDING,
            user_id=user_id,
            client_id=client_id,
            company_id=outbound_booking.company_id,
            duration_seconds=duration_seconds,
            distance_meters=distance_meters,
            price_amount=return_price_amount,
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
