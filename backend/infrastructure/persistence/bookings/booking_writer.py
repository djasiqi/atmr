from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Protocol, cast

from ext import db
from models.booking import Booking
from models.enums import BookingCreatedVia, BookingStatus


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
        wheelchair_client_has: bool = False,
        wheelchair_need: bool = False,
        needs_assistance: bool = False,
        assistance_detail: str | None = None,
        is_urgent: bool = False,
        time_confirmed: bool = True,
        pickup_access_notes: str | None = None,
        dropoff_access_notes: str | None = None,
        requester_name: str | None = None,
        requester_phone: str | None = None,
        extra_route_stops: list[dict[str, Any]] | None = None,
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
        wheelchair_client_has: bool = False,
        wheelchair_need: bool = False,
        needs_assistance: bool = False,
        assistance_detail: str | None = None,
        is_urgent: bool = False,
        time_confirmed: bool = True,
        pickup_access_notes: str | None = None,
        dropoff_access_notes: str | None = None,
        requester_name: str | None = None,
        requester_phone: str | None = None,
        extra_route_stops: list[dict[str, Any]] | None = None,
    ) -> Booking:
        if bool(wheelchair_client_has) and bool(wheelchair_need):
            raise ValueError(
                "wheelchair_client_has et wheelchair_need ne peuvent pas être vrais ensemble."
            )
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
            is_return=False,
            time_confirmed=bool(time_confirmed),
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
            pickup_access_notes=_blank_to_none(pickup_access_notes),
            dropoff_access_notes=_blank_to_none(dropoff_access_notes),
            wheelchair_client_has=bool(wheelchair_client_has),
            wheelchair_need=bool(wheelchair_need),
            needs_assistance=bool(needs_assistance),
            assistance_detail=_blank_to_none(assistance_detail)
            if needs_assistance
            else None,
            is_urgent=bool(is_urgent),
            requester_name=_blank_to_none(requester_name),
            requester_phone=_blank_to_none(requester_phone),
            duration_seconds=duration_seconds,
            distance_meters=distance_meters,
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
            created_via=BookingCreatedVia.CLIENT_APP,
        )
        stops = [stop for stop in (extra_route_stops or []) if isinstance(stop, dict)]
        route_group_id = str(uuid.uuid4()) if stops else None
        if route_group_id:
            new_booking.route_group_id = route_group_id
            new_booking.route_sequence_number = 1
        from services.platform_billing.billing_origin import apply_origin_on_booking

        apply_origin_on_booking(new_booking, created_via=BookingCreatedVia.CLIENT_APP)

        def _write_core() -> None:
            db.session.add(new_booking)
            db.session.flush()  # attribue new_booking.id

            last_leg = new_booking
            for index, stop in enumerate(stops, start=2):
                last_leg = self._create_route_leg(
                    previous=last_leg,
                    stop=stop,
                    sequence=index,
                    route_group_id=route_group_id or "",
                    user_id=user_id,
                    client_id=client_id,
                )
                db.session.add(last_leg)
                db.session.flush()

            if is_round_trip:
                return_booking = self._create_return_booking(
                    outbound_booking=last_leg,
                    home_booking=new_booking if stops else None,
                    user_id=user_id,
                    client_id=client_id,
                    duration_seconds=duration_seconds,
                    distance_meters=distance_meters,
                    return_scheduled_time=return_scheduled_time,
                    return_time_exact=return_time_exact,
                    return_leg_amount=return_amount,
                    return_price_amount=return_price_amount,
                    route_group_id=route_group_id,
                    route_sequence_number=(len(stops) + 2) if stops else None,
                )
                db.session.add(return_booking)

        session = db.session()
        in_transaction = getattr(session, "in_transaction", None)
        get_transaction = getattr(session, "get_transaction", None)
        owns_transaction = not (
            bool(in_transaction() if callable(in_transaction) else False)
            or bool(get_transaction() if callable(get_transaction) else False)
        )
        if owns_transaction:
            with session.begin():
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
        home_booking: Booking | None = None,
        route_group_id: str | None = None,
        route_sequence_number: int | None = None,
    ) -> Booking:
        # Heure absente : rester à None. Ne jamais substituer 00:00.
        if not return_time_exact:
            return_scheduled_time = None
        # Ordre des kwargs : is_return / parent_booking_id / time_confirmed AVANT scheduled_time,
        # sinon @validates("scheduled_time") s'exécute avec is_return encore à False → ValueError.
        home = home_booking or outbound_booking
        pickup_location = outbound_booking.dropoff_location
        dropoff_location = home.pickup_location
        pickup_access = outbound_booking.dropoff_access_notes
        dropoff_access = home.pickup_access_notes
        # Retour vers le domicile : la destination n'est plus l'établissement de la dernière étape.
        clear_medical = (
            home_booking is not None and home_booking is not outbound_booking
        )
        return_booking = cast("Any", Booking)(
            customer_name=outbound_booking.customer_name,
            pickup_location=pickup_location,
            dropoff_location=dropoff_location,
            is_return=True,
            parent_booking_id=outbound_booking.id,
            time_confirmed=bool(return_time_exact),
            scheduled_time=return_scheduled_time,
            amount=float(return_leg_amount),
            status=BookingStatus.PENDING,
            user_id=user_id,
            client_id=client_id,
            company_id=outbound_booking.company_id,
            medical_facility="" if clear_medical else outbound_booking.medical_facility,
            doctor_name="" if clear_medical else outbound_booking.doctor_name,
            hospital_service=None
            if clear_medical
            else outbound_booking.hospital_service,
            notes_medical=outbound_booking.notes_medical,
            pickup_access_notes=pickup_access,
            dropoff_access_notes=dropoff_access,
            wheelchair_client_has=bool(outbound_booking.wheelchair_client_has),
            wheelchair_need=bool(outbound_booking.wheelchair_need),
            needs_assistance=bool(outbound_booking.needs_assistance),
            assistance_detail=outbound_booking.assistance_detail,
            requester_name=outbound_booking.requester_name,
            requester_phone=outbound_booking.requester_phone,
            duration_seconds=duration_seconds,
            distance_meters=distance_meters,
            price_amount=return_price_amount,
            route_group_id=route_group_id,
            route_sequence_number=route_sequence_number,
            created_via=BookingCreatedVia.CLIENT_APP,
        )
        from services.platform_billing.billing_origin import apply_origin_on_booking

        apply_origin_on_booking(
            return_booking, created_via=BookingCreatedVia.CLIENT_APP
        )
        # Hérite l'origine commerciale de l'aller si définie
        if getattr(outbound_booking, "billing_origin", None):
            return_booking.billing_origin = outbound_booking.billing_origin
            return_booking.billing_origin_source = (
                outbound_booking.billing_origin_source
            )
            return_booking.billing_origin_reason = "RETURN_LEG_SAME_ORIGIN"

        # Best effort coords
        try:
            return_booking.pickup_lat = outbound_booking.dropoff_lat
            return_booking.pickup_lon = outbound_booking.dropoff_lon
            return_booking.dropoff_lat = home.pickup_lat
            return_booking.dropoff_lon = home.pickup_lon
        except Exception:
            pass

        return return_booking

    def _create_route_leg(
        self,
        *,
        previous: Booking,
        stop: dict[str, Any],
        sequence: int,
        route_group_id: str,
        user_id: int,
        client_id: int,
    ) -> Booking:
        """Étape suivante du même parcours. L'heure absente reste None."""
        leg = cast("Any", Booking)(
            customer_name=previous.customer_name,
            pickup_location=previous.dropoff_location,
            dropoff_location=str(stop.get("dropoff_location") or ""),
            is_return=False,
            time_confirmed=bool(stop.get("time_confirmed")),
            scheduled_time=stop.get("scheduled_time"),
            # 0 est refusé par le modèle. L'estimation client reste sur la première course.
            amount=0.5,
            status=BookingStatus.PENDING,
            user_id=user_id,
            client_id=client_id,
            company_id=previous.company_id,
            medical_facility=str(stop.get("medical_facility") or ""),
            doctor_name=str(stop.get("doctor_name") or ""),
            hospital_service=stop.get("hospital_service") or None,
            pickup_access_notes=previous.dropoff_access_notes,
            dropoff_access_notes=_blank_to_none(stop.get("dropoff_access_notes")),
            wheelchair_client_has=bool(previous.wheelchair_client_has),
            wheelchair_need=bool(previous.wheelchair_need),
            needs_assistance=bool(previous.needs_assistance),
            assistance_detail=previous.assistance_detail,
            is_urgent=bool(stop.get("is_urgent")),
            requester_name=previous.requester_name,
            requester_phone=previous.requester_phone,
            is_round_trip=False,
            route_group_id=route_group_id,
            route_sequence_number=sequence,
            created_via=BookingCreatedVia.CLIENT_APP,
        )
        from services.platform_billing.billing_origin import apply_origin_on_booking

        apply_origin_on_booking(leg, created_via=BookingCreatedVia.CLIENT_APP)
        if getattr(previous, "billing_origin", None):
            leg.billing_origin = previous.billing_origin
            leg.billing_origin_source = previous.billing_origin_source
            leg.billing_origin_reason = "ROUTE_LEG_SAME_ORIGIN"
        return leg
