# models/booking.py

# Constantes pour éviter les valeurs magiques
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Optional, cast

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    event,
    func,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from ext import db
from shared.time_utils import (
    iso_utc_z,
    now_local,
    parse_local_naive,
    split_date_time_local,
    to_geneva_local,
    to_utc_from_db,
)

from .base import _as_bool, _as_dt, _as_float, _as_int, _as_str
from .enums import BookingStatus

USER_ID_ZERO = 0
AMOUNT_ZERO = 0
VALUE_ZERO = 0
COMPANY_ID_ZERO = 0
CUSTOMER_NAME_MAX_LENGTH = 100
LOCATION_MAX_LENGTH = 200

"""Model Booking - Gestion des réservations de transport.
Extrait depuis models.py (lignes ~1294-1642).
"""


# Import des helpers timezone


class Booking(db.Model):
    __tablename__ = "booking"
    __table_args__ = (
        CheckConstraint(
            "pickup_lat IS NULL OR (pickup_lat BETWEEN -90 AND 90)",
            name="chk_booking_pickup_lat",
        ),
        CheckConstraint(
            "pickup_lon IS NULL OR (pickup_lon BETWEEN -180 AND 180)",
            name="chk_booking_pickup_lon",
        ),
        CheckConstraint(
            "dropoff_lat IS NULL OR (dropoff_lat BETWEEN -90 AND 90)",
            name="chk_booking_drop_lat",
        ),
        CheckConstraint(
            "dropoff_lon IS NULL OR (dropoff_lon BETWEEN -180 AND 180)",
            name="chk_booking_drop_lon",
        ),
        Index("ix_booking_company_scheduled", "company_id", "scheduled_time"),
        Index("ix_booking_status_scheduled", "status", "scheduled_time"),
        Index("ix_booking_driver_status", "driver_id", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_name: Mapped[str] = mapped_column(String(100), nullable=False)
    pickup_location: Mapped[str] = mapped_column(String(200), nullable=False)
    dropoff_location: Mapped[str] = mapped_column(String(200), nullable=False)
    booking_type = Column(String(200), nullable=False, server_default="standard")

    scheduled_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=False), nullable=True
    )

    amount: Mapped[float] = mapped_column(Float, nullable=False)
    status = Column(
        SAEnum(BookingStatus, name="booking_status"),
        index=True,
        nullable=False,
        default=BookingStatus.PENDING,
        server_default=BookingStatus.PENDING.value,
    )

    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    rejected_by = Column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )

    duration_seconds = Column(Integer)
    distance_meters = Column(Integer)

    client_id = Column(
        Integer, ForeignKey("client.id", ondelete="CASCADE"), nullable=False, index=True
    )
    company_id = Column(
        Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=True, index=True
    )
    driver_id = Column(
        Integer, ForeignKey("driver.id", ondelete="SET NULL"), nullable=True, index=True
    )

    is_round_trip = Column(Boolean, nullable=False, server_default=text("false"))
    is_return = Column(Boolean, nullable=False, server_default=text("false"))

    boarded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    parent_booking_id = Column(
        Integer, ForeignKey("booking.id", ondelete="SET NULL"), nullable=True
    )

    medical_facility = Column(String(200))
    doctor_name = Column(String(200))
    hospital_service = Column(String(100))
    notes_medical = Column(Text)
    is_urgent = Column(Boolean, nullable=False, server_default=text("false"))
    time_confirmed = Column(Boolean, nullable=False, server_default=text("true"))

    wheelchair_client_has = Column(
        Boolean, nullable=False, server_default=text("false")
    )
    wheelchair_need = Column(Boolean, nullable=False, server_default=text("false"))

    pickup_lat = Column(Float)
    pickup_lon = Column(Float)
    dropoff_lat = Column(Float)
    dropoff_lon = Column(Float)

    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    billed_to_type = Column(String(50), nullable=False, server_default="patient")
    billed_to_company_id = Column(
        Integer, ForeignKey("company.id", ondelete="SET NULL"), nullable=True
    )
    billed_to_contact = Column(String(120))

    invoice_line_id = Column(
        Integer,
        ForeignKey(
            "invoice_lines.id", ondelete="SET NULL", name="fk_booking_invoice_line"
        ),
        nullable=True,
        index=True,
    )

    # Relations
    client = relationship("Client", back_populates="bookings", passive_deletes=True)
    company = relationship(
        "Company",
        back_populates="bookings",
        foreign_keys=[company_id],
        passive_deletes=True,
    )
    driver = relationship("Driver", back_populates="bookings", passive_deletes=True)
    payments = relationship(
        "Payment", back_populates="booking", passive_deletes=True, lazy=True
    )

    billed_to_company = relationship("Company", foreign_keys=[billed_to_company_id])
    invoice_line = relationship(
        "InvoiceLine",
        foreign_keys=[invoice_line_id],
        primaryjoin="Booking.invoice_line_id == InvoiceLine.id",
        uselist=False,
        backref="billed_booking",
    )

    return_trip = relationship(
        "Booking",
        backref="original_booking",
        remote_side=[id],
        foreign_keys=[parent_booking_id],
        uselist=False,
    )

    # Propriétés
    @property
    def customer_full_name(self) -> str:
        cust = _as_str(self.customer_name)
        if cust:
            return cust
        if self.client and self.client.user:
            u = self.client.user
            if u.first_name or u.last_name:
                return f"{u.first_name or ''} {u.last_name or ''}".strip()
            return u.username
        return "Non spécifié"

    def get_effective_payer(self) -> dict[str, Any]:
        btype = (_as_str(self.billed_to_type) or "patient").lower()
        if btype != "patient" and getattr(self, "billed_to_company", None):
            comp = self.billed_to_company
            return {
                "type": btype,
                "name": comp.name,
                "address": getattr(comp, "address", None),
                "email": getattr(comp, "contact_email", None),
                "phone": getattr(comp, "contact_phone", None),
                "company_id": comp.id,
            }

        cli = getattr(self, "client", None)
        if cli and getattr(cli, "user", None):
            u = cli.user
            full = (
                f"{(u.first_name or '').strip()} {(u.last_name or '').strip()}".strip()
            )
            return {
                "type": "patient",
                "name": full or (u.username or "Client"),
                "address": getattr(cli, "billing_address", None),
                "email": getattr(cli, "contact_email", None)
                or getattr(u, "email", None),
                "phone": getattr(cli, "contact_phone", None)
                or getattr(u, "phone", None),
                "client_id": cli.id,
            }
        return {"type": "patient", "name": self.customer_full_name}

    @property
    def serialize(self):
        scheduled_dt = _as_dt(self.scheduled_time)
        created_dt = _as_dt(self.created_at)
        updated_dt = _as_dt(self.updated_at)
        boarded_dt = _as_dt(self.boarded_at)
        completed_dt = _as_dt(self.completed_at)

        date_local, time_local = (
            split_date_time_local(scheduled_dt) if scheduled_dt else (None, None)
        )
        created_loc = (
            to_geneva_local(created_dt) if isinstance(created_dt, datetime) else None
        )
        updated_loc = (
            to_geneva_local(updated_dt) if isinstance(updated_dt, datetime) else None
        )

        amt = _as_float(self.amount)
        status_val = cast("BookingStatus", self.status)

        cli = self.client
        cli_user = getattr(cli, "user", None)

        return {
            "id": self.id,
            "customer_name": self.customer_full_name,
            "client_name": self.customer_full_name,
            "pickup_location": self.pickup_location,
            "dropoff_location": self.dropoff_location,
            "amount": round(amt, 2),
            "scheduled_time": scheduled_dt.isoformat() if scheduled_dt else None,
            "date_formatted": date_local or "Non spécifié",
            "time_formatted": time_local or "Non spécifié",
            "status": getattr(status_val, "value", "unknown").lower(),
            "client": {
                "id": getattr(cli, "id", None),
                "first_name": getattr(cli_user, "first_name", "") if cli_user else "",
                "last_name": getattr(cli_user, "last_name", "") if cli_user else "",
                "email": getattr(cli_user, "email", "") if cli_user else "",
                "full_name": self.customer_full_name,
            },
            "company": self.company.name if self.company else "Non assignée",
            "driver": {
                "id": self.driver.id,
                "username": self.driver.user.username if self.driver.user else None,
                "first_name": self.driver.user.first_name if self.driver.user else None,
                "last_name": self.driver.user.last_name if self.driver.user else None,
                "full_name": f"{self.driver.user.first_name or ''} {self.driver.user.last_name or ''}".strip()
                or self.driver.user.username
                if self.driver.user
                else None,
            }
            if self.driver
            else None,
            "driver_id": self.driver_id,
            "duration_seconds": self.duration_seconds,
            "distance_meters": self.distance_meters,
            "medical_facility": self.medical_facility or "Non spécifié",
            "doctor_name": self.doctor_name or "Non spécifié",
            "hospital_service": self.hospital_service or "Non spécifié",
            "notes_medical": self.notes_medical or "Aucune note",
            "wheelchair_client_has": _as_bool(self.wheelchair_client_has),
            "wheelchair_need": _as_bool(self.wheelchair_need),
            "created_at": created_loc.strftime("%d/%m/%Y %H:%M")
            if created_loc
            else "Non spécifié",
            "updated_at": updated_loc.strftime("%d/%m/%Y %H:%M")
            if updated_loc
            else "Non spécifié",
            "rejected_by": self.rejected_by,
            "is_round_trip": _as_bool(self.is_round_trip),
            "is_return": _as_bool(self.is_return),
            "parent_booking_id": self.parent_booking_id,
            "time_confirmed": _as_bool(self.time_confirmed),
            "has_return": self.return_trip is not None,
            "boarded_at": iso_utc_z(to_utc_from_db(boarded_dt)) if boarded_dt else None,
            "completed_at": iso_utc_z(to_utc_from_db(completed_dt))
            if completed_dt
            else None,
            "duree_minutes": (
                int((completed_dt - boarded_dt).total_seconds() // 60)
                if (completed_dt and boarded_dt)
                else None
            ),
            "duration_in_minutes": self.duration_in_minutes,
            "billing": {
                "billed_to_type": (_as_str(self.billed_to_type) or "patient"),
                "billed_to_company": self.billed_to_company.serialize
                if self.billed_to_company
                else None,
                "billed_to_contact": self.billed_to_contact,
            },
            "patient_name": _as_str(self.customer_name),
        }

    # Validations
    @validates("user_id")
    def validate_user_id(self, _key, user_id):
        if not isinstance(user_id, int) or user_id <= USER_ID_ZERO:
            msg = "L'ID utilisateur doit être un entier positif."
            raise ValueError(msg)
        return user_id

    @validates("is_return")
    def validate_is_return(self, _key, val):
        return bool(val)

    @validates("amount")
    def validate_amount(self, _key, amount):
        if amount is None:
            return None
        if amount <= AMOUNT_ZERO:
            msg = "Le montant doit être supérieur à 0"
            raise ValueError(msg)
        return round(amount, 2)

    @validates("scheduled_time")
    def validate_scheduled_time(self, _key, scheduled_time):
        st = parse_local_naive(scheduled_time)
        # Validation désactivée si time_confirmed=False (pour import
        # historique)
        time_confirmed = getattr(self, "time_confirmed", True)
        if st and st < now_local() and time_confirmed:
            msg = "Heure prévue dans le passé."
            raise ValueError(msg)
        return st

    @validates("customer_name")
    def validate_customer_name(self, _key, name):
        if not name or len(name.strip()) == 0:
            msg = "Le nom du client ne peut pas être vide"
            raise ValueError(msg)
        if len(name) > CUSTOMER_NAME_MAX_LENGTH:
            msg = f"Le nom du client ne peut pas dépasser {CUSTOMER_NAME_MAX_LENGTH} caractères"
            raise ValueError(msg)
        return name

    @validates("pickup_location", "dropoff_location")
    def validate_location(self, key, location):
        if not location or len(location.strip()) == 0:
            msg = f"{key} ne peut pas être vide"
            raise ValueError(msg)
        if len(location) > LOCATION_MAX_LENGTH:
            msg = f"{key} ne peut pas dépasser {LOCATION_MAX_LENGTH} caractères"
            raise ValueError(msg)
        return location

    @validates("status")
    def validate_status(self, _key, status):
        if isinstance(status, str):
            status = status.upper()
            try:
                status = BookingStatus[status]
            except KeyError:
                msg = f"Statut invalide : {status}. Doit être l'un de {list(BookingStatus.__members__.keys())}"
                raise ValueError(msg) from None
        if not isinstance(status, BookingStatus):
            msg = f"Statut invalide : {status}. Doit être un BookingStatus valide."
            raise ValueError(msg)
        return status

    @validates("driver_id")
    def validate_driver_id(self, _key, value):
        if value is not None and (not isinstance(value, int) or value < VALUE_ZERO):
            msg = "driver_id doit être un entier positif ou null"
            raise ValueError(msg)
        return value

    @validates("billed_to_type")
    def _v_billed_to_type(self, _key, value):
        v = (value or "patient").lower().strip()
        if v not in ("patient", "clinic", "insurance"):
            msg = "billed_to_type invalide (patient|clinic|insurance)"
            raise ValueError(msg)
        return v

    @validates("billed_to_company_id")
    def _v_billed_to_company_id(self, _key, value):
        if value is not None and (not isinstance(value, int) or value <= VALUE_ZERO):
            msg = "billed_to_company_id doit être un entier positif ou NULL"
            raise ValueError(msg)
        current_type = _as_str(getattr(self, "billed_to_type", None)) or "patient"
        if current_type.strip().lower() == "patient":
            return None
        return value

    # Méthodes métier
    def is_future(self) -> bool:
        st = _as_dt(self.scheduled_time)
        return bool(st and st > now_local())

    def update_status(self, new_status):
        if not isinstance(new_status, BookingStatus):
            msg = "Statut invalide."
            raise ValueError(msg)
        self.status = new_status

    @property
    def duration_in_minutes(self) -> int | None:
        b = _as_dt(self.boarded_at)
        c = _as_dt(self.completed_at)
        if b and c:
            return int((c - b).total_seconds() // 60)
        return None

    def to_dict(self):
        return self.serialize

    def is_assignable(self) -> bool:
        st = _as_dt(self.scheduled_time)
        status_val = cast("BookingStatus", self.status)
        return (status_val in (BookingStatus.PENDING, BookingStatus.ACCEPTED)) and bool(
            st and st > now_local()
        )

    def assign_driver(self, driver_id: int):
        if not self.is_assignable():
            msg = "La réservation ne peut pas être attribuée actuellement."
            raise ValueError(msg)
        current_driver_id = _as_int(getattr(self, "driver_id", None))
        target_driver_id = _as_int(driver_id)
        if current_driver_id == target_driver_id:
            return
        self.driver_id = driver_id
        self.status = BookingStatus.ASSIGNED
        self.updated_at = datetime.now(UTC)

    def cancel_booking(self):
        if self.status not in [BookingStatus.ASSIGNED, BookingStatus.ACCEPTED]:
            msg = "Seules les réservations en cours peuvent être annulées."
            raise ValueError(msg)
        self.status = BookingStatus.CANCELED
        self.updated_at = datetime.now(UTC)

    @staticmethod
    def _enforce_billing_exclusive(_mapper, _connection, target: Booking) -> None:
        btype = (
            (_as_str(getattr(target, "billed_to_type", None)) or "patient")
            .strip()
            .lower()
        )
        if btype == "patient":
            target.billed_to_company_id = None
            return
        company_id = _as_int(getattr(target, "billed_to_company_id", None))
        if company_id <= COMPANY_ID_ZERO:
            msg = "billed_to_company_id est obligatoire si billed_to_type n'est pas 'patient'"
            raise ValueError(msg)


# Hooks ORM
event.listen(Booking, "before_insert", Booking._enforce_billing_exclusive)
event.listen(Booking, "before_update", Booking._enforce_billing_exclusive)
