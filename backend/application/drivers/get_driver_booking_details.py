from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


class _BookingLike(Protocol):
    id: int
    customer_name: str | None
    customer_full_name: (
        str | None
    )  # Compat: certains modèles exposent un nom complet client différent
    pickup_location: str | None
    dropoff_location: str | None
    scheduled_time: datetime | None
    amount: float | int | None
    status: object
    medical_facility: str | None
    doctor_name: str | None
    hospital_service: str | None
    notes_medical: str | None
    wheelchair_client_has: bool | None
    wheelchair_need: bool | None


class _BookingRepo(Protocol):
    def find_model_by_id_and_driver(
        self, booking_id: int, driver_id: int
    ) -> _BookingLike | None: ...

    def find_model_by_id_and_company(
        self, booking_id: int, company_id: int
    ) -> _BookingLike | None: ...


@dataclass(frozen=True, slots=True)
class BookingDetailsResponse:
    payload: dict[str, object]


class GetDriverBookingDetailsUseCase:
    """Use-case Application: récupérer les détails d'un booking pour un chauffeur."""

    def __init__(self, *, booking_repo: _BookingRepo) -> None:
        super().__init__()
        self._booking_repo = booking_repo

    def execute(
        self,
        *,
        booking_id: int,
        driver_id: int,
        driver_company_id: int | None = None,
    ) -> BookingDetailsResponse | None:
        booking = self._booking_repo.find_model_by_id_and_driver(
            booking_id=booking_id,
            driver_id=driver_id,
        )
        if booking is None and driver_company_id is not None:
            booking = self._booking_repo.find_model_by_id_and_company(
                booking_id=booking_id,
                company_id=driver_company_id,
            )
        if booking is None:
            return None

        status_obj = booking.status
        try:
            # status peut être un Enum avec un attribut value
            status_str = str(getattr(status_obj, "value", status_obj))
        except Exception:
            status_str = str(status_obj)

        # Compat: certains modèles exposent un nom complet client différent
        customer_full_name: str | None = None
        try:
            customer_full_name = booking.customer_full_name
        except Exception:
            customer_full_name = None

        customer_name = booking.customer_name or customer_full_name

        return BookingDetailsResponse(
            payload={
                "id": booking.id,
                "customer_name": customer_name,
                "client_name": customer_name,
                "pickup_location": booking.pickup_location,
                "dropoff_location": booking.dropoff_location,
                "scheduled_time": booking.scheduled_time.isoformat()
                if booking.scheduled_time
                else None,
                "amount": booking.amount,
                "status": status_str,
                # 🏥 Informations médicales
                "medical_facility": booking.medical_facility,
                "doctor_name": booking.doctor_name,
                "hospital_service": booking.hospital_service,
                "notes_medical": booking.notes_medical,
                "wheelchair_client_has": booking.wheelchair_client_has,
                "wheelchair_need": booking.wheelchair_need,
            }
        )
