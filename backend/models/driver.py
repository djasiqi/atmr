# models/driver.py

# Constantes pour éviter les valeurs magiques
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Time,
    UniqueConstraint,
    event,
    func,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from typing_extensions import override

from ext import db

from .base import _as_bool, _as_int
from .enums import (
    BreakType,
    DriverType,
    ShiftStatus,
    ShiftType,
    UnavailabilityReason,
    VacationType,
)

WEEKDAY_ZERO = 0
WEEKDAY_THRESHOLD = 6
V_THRESHOLD = 1440
V_ZERO = 0

"""Models Driver et tous ses modèles liés (shift, unavailability, vacation, etc.).
Extrait depuis models.py.
"""


# ========== MODEL PRINCIPAL ==========


class Driver(db.Model):
    __tablename__ = "driver"
    __table_args__ = (
        Index("ix_driver_company_active", "company_id", "is_active", "is_available"),
        Index("ix_driver_geo", "company_id", "latitude", "longitude"),
        CheckConstraint("(latitude IS NULL OR (latitude BETWEEN -90 AND 90))", name="chk_driver_lat"),
        CheckConstraint("(longitude IS NULL OR (longitude BETWEEN -180 AND 180))", name="chk_driver_lon"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    company_id = Column(Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True)

    # Véhicule
    vehicle_assigned: Mapped[str] = mapped_column(String(100), nullable=True)
    brand: Mapped[str] = mapped_column(String(100), nullable=True)
    license_plate: Mapped[str] = mapped_column(String(50), nullable=True)

    # États
    is_active = Column(Boolean, nullable=False, server_default="true")
    is_available = Column(Boolean, nullable=False, server_default="true")
    driver_type = Column(SAEnum(DriverType, name="driver_type"), nullable=False, server_default="REGULAR")

    # Localisation
    latitude: Mapped[float] = mapped_column(Float, nullable=True)
    longitude: Mapped[float] = mapped_column(Float, nullable=True)
    last_position_update: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Média & notifications
    driver_photo: Mapped[str] = mapped_column(String(500), nullable=True)
    push_token: Mapped[str] = mapped_column(String(255), nullable=True, index=True)

    # HR / Contrats & Qualifications
    contract_type = Column(String(20), nullable=False, server_default="CDI")
    weekly_hours: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    hourly_rate_cents: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    employment_start_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    employment_end_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    license_categories = Column(JSON, nullable=False, server_default=text("'[]'"))
    license_valid_until: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    trainings = Column(JSON, nullable=False, server_default=text("'[]'"))
    medical_valid_until: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    # Relations
    user = relationship("User", back_populates="driver", passive_deletes=True)
    company = relationship("Company", back_populates="drivers", passive_deletes=True)
    vacations = relationship(
        "DriverVacation", back_populates="driver", cascade="all, delete-orphan", passive_deletes=True
    )
    bookings = relationship("Booking", back_populates="driver", passive_deletes=True)

    @property
    def serialize(self):
        try:
            user = getattr(self, "user", None)
            last_pos = getattr(self, "last_position_update", None)
            emp_start = getattr(self, "employment_start_date", None)
            emp_end = getattr(self, "employment_end_date", None)
            license_valid = getattr(self, "license_valid_until", None)
            medical_valid = getattr(self, "medical_valid_until", None)
            user_payload = None
            username = None
            first_name = None
            last_name = None
            email = None
            full_name = None
            if user is not None:
                try:
                    user_payload = user.serialize
                except Exception:
                    user_payload = {
                        "id": getattr(user, "id", None),
                        "public_id": getattr(user, "public_id", None),
                        "username": getattr(user, "username", None),
                        "email": getattr(user, "email", None),
                        "first_name": getattr(user, "first_name", None),
                        "last_name": getattr(user, "last_name", None),
                    }
                username = getattr(user, "username", None)
                first_name = getattr(user, "first_name", None)
                last_name = getattr(user, "last_name", None)
                email = getattr(user, "email", None)
                fn = (first_name or "").strip()
                ln = (last_name or "").strip()
                full_name = (f"{fn} {ln}".strip()) or username

            return {
                "id": self.id,
                "user_id": self.user_id,
                "company_id": self.company_id,
                "user": user_payload,
                "username": username,
                "first_name": first_name,
                "last_name": last_name,
                "full_name": full_name,
                "email": email,
                "is_active": _as_bool(getattr(self, "is_active", False)),
                "is_available": _as_bool(getattr(self, "is_available", False)),
                "driver_type": getattr(getattr(self, "driver_type", None), "value", getattr(self, "driver_type", None)),
                "vehicle_assigned": getattr(self, "vehicle_assigned", None),
                "brand": getattr(self, "brand", None),
                "latitude": getattr(self, "latitude", None),
                "longitude": getattr(self, "longitude", None),
                "last_position_update": (last_pos.isoformat() if last_pos is not None else None),
                "driver_photo": getattr(self, "driver_photo", None),
                "photo": getattr(self, "driver_photo", None),
                "push_token": getattr(self, "push_token", None),
                "contract_type": getattr(self, "contract_type", None),
                "weekly_hours": getattr(self, "weekly_hours", None),
                "hourly_rate_cents": getattr(self, "hourly_rate_cents", None),
                "employment_start_date": emp_start.isoformat() if emp_start else None,
                "employment_end_date": emp_end.isoformat() if emp_end else None,
                "license_categories": getattr(self, "license_categories", []),
                "license_valid_until": license_valid.isoformat() if license_valid else None,
                "trainings": getattr(self, "trainings", []),
                "medical_valid_until": medical_valid.isoformat() if medical_valid else None,
            }
        except Exception as e:
            # Log l'erreur mais retourne un profil minimal pour éviter une erreur 500
            import logging

            logger = logging.getLogger("driver_model")
            logger.exception("Erreur lors de la sérialisation du driver %s: %s", self.id, e)
            # Retourner un profil minimal en cas d'erreur
            return {
                "id": self.id,
                "user_id": getattr(self, "user_id", None),
                "company_id": getattr(self, "company_id", None),
                "user": None,
                "username": None,
                "first_name": None,
                "last_name": None,
                "full_name": None,
                "email": None,
                "is_active": False,
                "is_available": False,
                "driver_type": None,
                "vehicle_assigned": None,
                "brand": None,
                "latitude": None,
                "longitude": None,
                "last_position_update": None,
                "driver_photo": None,
                "photo": None,
                "push_token": None,
                "contract_type": None,
                "weekly_hours": None,
                "hourly_rate_cents": None,
                "employment_start_date": None,
                "employment_end_date": None,
                "license_categories": [],
                "license_valid_until": None,
                "trainings": [],
                "medical_valid_until": None,
            }

    def to_dict(self):
        return self.serialize


# ========== MODELS LIÉS (PLANNING) ==========


class DriverShift(db.Model):
    __tablename__ = "driver_shift"
    __table_args__ = (
        Index("ix_shift_company_driver_start", "company_id", "driver_id", "start_local"),
        CheckConstraint("end_local > start_local", name="ck_shift_time_order"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True)
    driver_id = Column(Integer, ForeignKey("driver.id", ondelete="CASCADE"), nullable=False, index=True)

    start_local = Column(DateTime(timezone=False), nullable=False, index=True)
    end_local = Column(DateTime(timezone=False), nullable=False, index=True)

    timezone = Column(String(64), nullable=False, server_default="Europe/Zurich")
    type = Column(
        SAEnum(ShiftType, name="shift_type", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        server_default=ShiftType.REGULAR.value,
    )
    status = Column(
        SAEnum(ShiftStatus, name="shift_status", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        server_default=ShiftStatus.PLANNED.value,
    )

    site: Mapped[str] = mapped_column(String(120), nullable=True)
    zone: Mapped[str] = mapped_column(String(120), nullable=True)
    client_ref: Mapped[str] = mapped_column(String(120), nullable=True)

    pay_code: Mapped[str] = mapped_column(String(50), nullable=True)
    vehicle_id = Column(Integer, ForeignKey("vehicle.id", ondelete="SET NULL"), nullable=True, index=True)

    notes_internal: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes_employee: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_by_user_id = Column(Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    updated_by_user_id = Column(Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")

    compliance_flags = Column(JSON, nullable=False, server_default=text("'[]'"))

    company = relationship("Company")
    driver = relationship("Driver")
    vehicle = relationship("Vehicle", foreign_keys=[vehicle_id])


class DriverUnavailability(db.Model):
    __tablename__ = "driver_unavailability"
    __table_args__ = (
        Index("ix_unav_company_driver_start", "company_id", "driver_id", "start_local"),
        CheckConstraint("end_local > start_local", name="ck_unav_time_order"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True)
    driver_id = Column(Integer, ForeignKey("driver.id", ondelete="CASCADE"), nullable=False, index=True)

    start_local = Column(DateTime(timezone=False), nullable=False, index=True)
    end_local = Column(DateTime(timezone=False), nullable=False, index=True)

    reason = Column(
        SAEnum(UnavailabilityReason, name="unavailability_reason", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        server_default=UnavailabilityReason.OTHER.value,
    )
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    company = relationship("Company")
    driver = relationship("Driver")


class DriverWeeklyTemplate(db.Model):
    __tablename__ = "driver_weekly_template"
    __table_args__ = (
        Index("ix_tpl_company_driver_weekday", "company_id", "driver_id", "weekday"),
        CheckConstraint("weekday >= 0 AND weekday <= WEEKDAY_THRESHOLD", name="ck_tpl_weekday_range"),
        CheckConstraint("end_time > start_time", name="ck_tpl_time_order"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True)
    driver_id = Column(Integer, ForeignKey("driver.id", ondelete="CASCADE"), nullable=False, index=True)

    weekday: Mapped[int] = mapped_column(Integer, nullable=False)
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)

    effective_from: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    effective_to: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    company = relationship("Company")
    driver = relationship("Driver")


class DriverBreak(db.Model):
    __tablename__ = "driver_break"
    __table_args__ = (CheckConstraint("end_local > start_local", name="ck_break_time_order"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    shift_id = Column(Integer, ForeignKey("driver_shift.id", ondelete="CASCADE"), nullable=False, index=True)

    start_local = Column(DateTime(timezone=False), nullable=False, index=True)
    end_local = Column(DateTime(timezone=False), nullable=False, index=True)
    type = Column(
        SAEnum(BreakType, name="break_type", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        server_default=BreakType.MANDATORY.value,
    )

    shift = relationship("DriverShift", backref="breaks")


class DriverPreference(db.Model):
    __tablename__ = "driver_preference"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True)
    driver_id = Column(Integer, ForeignKey("driver.id", ondelete="CASCADE"), nullable=False, index=True)

    mornings_pref = Column(Boolean, nullable=False, server_default="false")
    evenings_pref = Column(Boolean, nullable=False, server_default="false")
    forbidden_windows = Column(JSON, nullable=False, server_default=text("'[]'"))
    weekend_rotation_weight = Column(Integer, nullable=False, server_default="0")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    company = relationship("Company")
    driver = relationship("Driver")


class CompanyPlanningSettings(db.Model):
    __tablename__ = "company_planning_settings"
    company_id = Column(Integer, ForeignKey("company.id", ondelete="CASCADE"), primary_key=True)
    settings = Column(JSON, nullable=False, server_default=text("'{}'"))

    company = relationship("Company")


class DriverWorkingConfig(db.Model):
    """Config horaire d'un chauffeur (minutes depuis minuit)."""

    __tablename__ = "driver_working_config"
    __table_args__ = (
        CheckConstraint("earliest_start BETWEEN 0 AND 1440", name="chk_dwc_earliest"),
        CheckConstraint("latest_start BETWEEN 0 AND 1440", name="chk_dwc_latest"),
        CheckConstraint("total_working_minutes BETWEEN 0 AND 1440", name="chk_dwc_total"),
        CheckConstraint("break_duration BETWEEN 0 AND 1440", name="chk_dwc_break_dur"),
        CheckConstraint("break_earliest BETWEEN 0 AND 1440", name="chk_dwc_break_earliest"),
        CheckConstraint("break_latest BETWEEN 0 AND 1440", name="chk_dwc_break_latest"),
        CheckConstraint("earliest_start < latest_start", name="chk_dwc_start_window"),
        CheckConstraint("break_earliest < break_latest", name="chk_dwc_break_window"),
        CheckConstraint("break_duration <= total_working_minutes", name="chk_dwc_break_vs_total"),
        UniqueConstraint("driver_id", name="uq_dwc_driver"),
        Index("ix_dwc_driver", "driver_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    driver_id = Column(Integer, ForeignKey("driver.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)

    earliest_start: Mapped[int] = mapped_column(Integer, nullable=False, server_default="360")
    latest_start: Mapped[int] = mapped_column(Integer, nullable=False, server_default="600")
    total_working_minutes = Column(Integer, nullable=False, server_default="510")
    break_duration: Mapped[int] = mapped_column(Integer, nullable=False, server_default="60")
    break_earliest: Mapped[int] = mapped_column(Integer, nullable=False, server_default="600")
    break_latest: Mapped[int] = mapped_column(Integer, nullable=False, server_default="900")

    @property
    def serialize(self):
        return {
            "id": self.id,
            "driver_id": self.driver_id,
            "earliest_start": self.earliest_start,
            "latest_start": self.latest_start,
            "total_working_minutes": self.total_working_minutes,
            "break_duration": self.break_duration,
            "break_earliest": self.break_earliest,
            "break_latest": self.break_latest,
        }

    @override
    def __repr__(self):
        return (
            f"<DriverWorkingConfig id={self.id} driver_id={self.driver_id} "
            f"start={self.earliest_start}-{self.latest_start} "
            f"maxWork={self.total_working_minutes} "
            f"break={self.break_duration} ({self.break_earliest}-{self.break_latest})>"
        )

    @validates(
        "earliest_start",
        "latest_start",
        "total_working_minutes",
        "break_duration",
        "break_earliest",
        "break_latest",
    )
    def validate_working_times(self, key, value):
        if value is None:
            msg = f"'{key}' ne peut pas être NULL"
            raise ValueError(msg)
        v = int(value)
        if v < V_ZERO or v > V_THRESHOLD:
            msg = f"'{key}' doit être entre 0 et 1440"
            raise ValueError(msg)
        return v

    def validate_config(self):
        earliest = _as_int(self.earliest_start)
        latest = _as_int(self.latest_start)
        if earliest >= latest:
            msg = "earliest_start doit être avant latest_start"
            raise ValueError(msg)

        be = _as_int(self.break_earliest)
        bl = _as_int(self.break_latest)
        if be >= bl:
            msg = "break_earliest doit être avant break_latest"
            raise ValueError(msg)

        if _as_int(self.break_duration) > _as_int(self.total_working_minutes):
            msg = "La pause ne peut pas excéder le temps de travail total"
            raise ValueError(msg)

    @staticmethod
    def _enforce_config(_mapper, _connection, target: DriverWorkingConfig) -> None:
        target.validate_config()


event.listen(DriverWorkingConfig, "before_insert", DriverWorkingConfig._enforce_config)
event.listen(DriverWorkingConfig, "before_update", DriverWorkingConfig._enforce_config)


class DriverVacation(db.Model):
    """Période d'absence / vacances d'un chauffeur."""

    __tablename__ = "driver_vacations"
    __table_args__ = (
        CheckConstraint("start_date <= end_date", name="chk_vacation_dates_order"),
        Index("ix_vacation_driver_start", "driver_id", "start_date"),
        Index("ix_vacation_driver_end", "driver_id", "end_date"),
        UniqueConstraint("driver_id", "start_date", "end_date", "vacation_type", name="uq_vacation_exact_period"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    driver_id = Column(Integer, ForeignKey("driver.id", ondelete="CASCADE"), nullable=False, index=True)
    start_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    end_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    vacation_type = Column(
        SAEnum(VacationType, name="vacation_type"), nullable=False, server_default=VacationType.VACANCES.value
    )

    driver = relationship("Driver", back_populates="vacations", passive_deletes=True)

    @override
    def __repr__(self) -> str:
        vt = getattr(self.vacation_type, "value", self.vacation_type)
        sd = getattr(self, "start_date", None)
        ed = getattr(self, "end_date", None)
        sd_str = sd.isoformat() if isinstance(sd, date) else "?"
        ed_str = ed.isoformat() if isinstance(ed, date) else "?"
        return f"<DriverVacation id={self.id}, driver_id={self.driver_id}, {sd_str} → {ed_str}, type={vt}>"

    @validates("start_date", "end_date")
    def validate_dates(self, key: str, value: date | None) -> date | None:
        if value is None:
            msg = f"{key} doit être une date valide."
            raise ValueError(msg)
        return value

    def validate_logic(self) -> None:
        sd = getattr(self, "start_date", None)
        ed = getattr(self, "end_date", None)
        if isinstance(sd, date) and isinstance(ed, date) and sd > ed:
            msg = "La date de début ne peut pas être après la date de fin."
            raise ValueError(msg)

    @staticmethod
    def _enforce_logic(_mapper, _connection, target: DriverVacation) -> None:
        target.validate_logic()

    def overlaps(self, other_start: date, other_end: date) -> bool:
        """Retourne True si les fenêtres se chevauchent (bornes incluses)."""
        sd = getattr(self, "start_date", None)
        ed = getattr(self, "end_date", None)
        if not (isinstance(sd, date) and isinstance(ed, date)):
            return False
        return (sd <= other_end) and (ed >= other_start)

    @property
    def serialize(self) -> dict[str, Any]:
        sd = getattr(self, "start_date", None)
        ed = getattr(self, "end_date", None)
        vt = getattr(self, "vacation_type", None)
        return {
            "id": self.id,
            "driver_id": self.driver_id,
            "start_date": (sd.isoformat() if isinstance(sd, date) else None),
            "end_date": (ed.isoformat() if isinstance(ed, date) else None),
            "vacation_type": getattr(vt, "value", vt),
        }


event.listen(DriverVacation, "before_insert", DriverVacation._enforce_logic)
event.listen(DriverVacation, "before_update", DriverVacation._enforce_logic)
