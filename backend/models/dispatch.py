# models/dispatch.py
"""
Models liés au dispatch et au temps réel.
Extrait depuis models.py (lignes ~2395-3050).
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, Dict, List

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    UniqueConstraint,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from ext import db

from .base import _as_dt, _as_int, _as_str, _coerce_enum, _iso
from .enums import AssignmentStatus, DispatchStatus, DriverState, RealtimeEntityType, RealtimeEventType

if TYPE_CHECKING:
    from .company import Company


class DispatchRun(db.Model):
    __tablename__ = "dispatch_run"
    __table_args__ = (
        UniqueConstraint('company_id', 'day', name='uq_dispatch_run_company_day'),
        Index('ix_dispatch_run_company_day', 'company_id', 'day'),
        Index('ix_dispatch_run_company_status_day', 'company_id', 'status', 'day'),
    )
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('company.id', ondelete="CASCADE"), index=True)
    day: Mapped[date] = mapped_column(Date, nullable=False)

    status: Mapped[DispatchStatus] = mapped_column(
        SAEnum(DispatchStatus, name="dispatch_status", native_enum=False),
        default=DispatchStatus.PENDING,
        nullable=False,
    )

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )

    config: Mapped[Dict[str, Any] | None] = mapped_column(MutableDict.as_mutable(JSONB()))
    metrics: Mapped[Dict[str, Any] | None] = mapped_column(MutableDict.as_mutable(JSONB()))

    # Relations
    company: Mapped[Company] = relationship(back_populates='dispatch_runs', passive_deletes=True)
    assignments: Mapped[List[Assignment]] = relationship(
        back_populates='dispatch_run',
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    metrics_record: Mapped[DispatchMetrics | None] = relationship(
        "DispatchMetrics",
        back_populates="dispatch_run",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    # Méthodes métier
    def mark_started(self) -> None:
        self.status = DispatchStatus.RUNNING
        self.started_at = datetime.now(UTC)

    def mark_completed(self, metrics: dict | None = None) -> None:
        self.status = DispatchStatus.COMPLETED
        self.completed_at = datetime.now(UTC)
        if metrics:
            self.metrics = {**(self.metrics or {}), **dict(metrics)}

    def mark_failed(self, reason: str | None = None) -> None:
        self.status = DispatchStatus.FAILED
        self.completed_at = datetime.now(UTC)
        if reason:
            self.metrics = {**(self.metrics or {}), "error": reason}

    # Validations
    @validates('company_id')
    def _v_company_id(self, _k: str, v: Any) -> int:
        if not isinstance(v, int) or v <= 0:
            raise ValueError("company_id invalide")
        return v

    @validates('day')
    def _v_day(self, _k: str, v: Any) -> date:
        if not isinstance(v, date):
            raise ValueError("'day' doit être un objet date")
        return v

    @validates('status')
    def _v_status(self, _k: str, v: Any) -> DispatchStatus:
        if isinstance(v, DispatchStatus):
            return v
        if isinstance(v, str):
            s = v.strip()
            try:
                return DispatchStatus[s.upper()]
            except KeyError:
                pass
            try:
                return DispatchStatus(s.upper())
            except Exception:
                pass
        raise ValueError(f"status invalide: {v}")

    @validates('config', 'metrics')
    def _v_jsonb(self, _k: str, v: Any) -> Dict[str, Any] | None:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("config/metrics doivent être des objets JSON (dict)")
        return v

    @property
    def serialize(self) -> Dict[str, Any]:
        status_str = self.status.value if isinstance(self.status, DispatchStatus) else str(self.status) or "pending"
        return {
            "id": self.id,
            "company_id": self.company_id,
            "day": self.day.isoformat(),
            "status": status_str,
            "started_at": _iso(_as_dt(self.started_at)),
            "completed_at": _iso(_as_dt(self.completed_at)),
            "created_at": _iso(_as_dt(self.created_at)),
            "metrics": self.metrics or {},
        }

    def __repr__(self) -> str:
        return f"<DispatchRun id={self.id} company={self.company_id} day={self.day} status={self.status}>"


class Assignment(db.Model):
    __tablename__ = "assignment"
    __table_args__ = (
        UniqueConstraint('dispatch_run_id', 'booking_id', name='uq_assignment_run_booking'),
        Index('ix_assignment_driver_status', 'driver_id', 'status'),
        CheckConstraint('delay_seconds >= 0', name='ck_assignment_delay_nonneg'),
    )

    id = Column(Integer, primary_key=True)

    dispatch_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("dispatch_run.id", ondelete="SET NULL"), index=True, nullable=True
    )
    booking_id = Column(Integer, ForeignKey('booking.id', ondelete="CASCADE"), nullable=False, index=True)
    driver_id = Column(Integer, ForeignKey('driver.id', ondelete="SET NULL"), nullable=True, index=True)

    status = Column(SAEnum(AssignmentStatus, name="assignment_status"), nullable=False, default=AssignmentStatus.SCHEDULED)

    planned_pickup_at = Column(DateTime(timezone=True), nullable=True)
    planned_dropoff_at = Column(DateTime(timezone=True), nullable=True)
    actual_pickup_at = Column(DateTime(timezone=True), nullable=True)
    actual_dropoff_at = Column(DateTime(timezone=True), nullable=True)

    eta_pickup_at = Column(DateTime(timezone=True), nullable=True)
    eta_dropoff_at = Column(DateTime(timezone=True), nullable=True)
    delay_seconds = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(UTC))

    # Relations
    dispatch_run: Mapped[DispatchRun | None] = relationship("DispatchRun", back_populates="assignments")
    booking = relationship('Booking', backref='assignments', passive_deletes=True)
    driver = relationship('Driver', backref='assignments', passive_deletes=True)

    @property
    def serialize(self):
        status_val = getattr(self.status, "value", self.status)
        delay: int = _as_int(getattr(self, "delay_seconds", 0), 0)
        return {
            "id": self.id,
            "dispatch_run_id": self.dispatch_run_id,
            "booking_id": self.booking_id,
            "driver_id": self.driver_id,
            "status": status_val,
            "planned_pickup_at": _iso(_as_dt(self.planned_pickup_at)),
            "planned_dropoff_at": _iso(_as_dt(self.planned_dropoff_at)),
            "actual_pickup_at": _iso(_as_dt(self.actual_pickup_at)),
            "actual_dropoff_at": _iso(_as_dt(self.actual_dropoff_at)),
            "eta_pickup_at": _iso(_as_dt(self.eta_pickup_at)),
            "eta_dropoff_at": _iso(_as_dt(self.eta_dropoff_at)),
            "delay_seconds": delay,
            "created_at": _iso(_as_dt(self.created_at)),
            "updated_at": _iso(_as_dt(self.updated_at)),
        }

    def __repr__(self):
        status_str = getattr(self.status, "value", self.status)
        return f"<Assignment id={self.id} booking={self.booking_id} driver={self.driver_id} status={status_str}>"

    # Validations
    @validates('dispatch_run_id')
    def _v_dispatch_run_id(self, _k, v):
        if v is None:
            return None
        if not isinstance(v, int) or v <= 0:
            raise ValueError("dispatch_run_id doit être NULL ou un entier positif.")
        return v

    @validates('booking_id')
    def _v_booking_id(self, _k, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("booking_id doit être un entier positif.")
        return v

    @validates('driver_id')
    def _v_driver_id(self, _k, v):
        if v is None:
            return None
        if not isinstance(v, int) or v <= 0:
            raise ValueError("driver_id doit être NULL ou un entier positif.")
        return v

    @validates('status')
    def _v_status(self, _k, value):
        coerced = _coerce_enum(value, AssignmentStatus)
        if coerced is None and isinstance(value, str):
            try:
                coerced = AssignmentStatus[value.upper()]
            except KeyError:
                pass
        if coerced is None:
            allowed = ", ".join(AssignmentStatus.__members__.keys())
            raise ValueError(f"Statut invalide : {value}. Doit être l'un de {allowed}")
        return coerced

    @validates('delay_seconds')
    def _v_delay(self, _k: str, v: Any) -> int:
        val = _as_int(v, 0)
        if val < 0:
            raise ValueError("delay_seconds ne peut pas être négatif.")
        return val

    def validate_chronology(self):
        pp = _as_dt(getattr(self, "planned_pickup_at", None))
        pd = _as_dt(getattr(self, "planned_dropoff_at", None))
        ap = _as_dt(getattr(self, "actual_pickup_at", None))
        ad = _as_dt(getattr(self, "actual_dropoff_at", None))

        if pp and pd and pd < pp:
            raise ValueError("planned_dropoff_at < planned_pickup_at")
        if ap and ad and ad < ap:
            raise ValueError("actual_dropoff_at < actual_pickup_at")


class DriverStatus(db.Model):
    __tablename__ = "driver_status"
    __table_args__ = (
        Index('ix_driver_status_state_nextfree', 'state', 'next_free_at'),
        CheckConstraint("latitude IS NULL OR (latitude BETWEEN -90 AND 90)", name="ck_driver_status_lat"),
        CheckConstraint("longitude IS NULL OR (longitude BETWEEN -180 AND 180)", name="ck_driver_status_lon"),
        CheckConstraint("heading IS NULL OR (heading >= 0 AND heading <= 360)", name="ck_driver_status_heading"),
        CheckConstraint("speed IS NULL OR speed >= 0", name="ck_driver_status_speed"),
    )

    id = Column(Integer, primary_key=True)
    driver_id = Column(Integer, ForeignKey('driver.id', ondelete="CASCADE"), nullable=False, unique=True, index=True)

    state = Column(
        SAEnum(DriverState, name="driver_state"),
        nullable=False,
        default=DriverState.AVAILABLE,
        server_default=DriverState.AVAILABLE.value,
    )

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    heading = Column(Float, nullable=True)
    speed = Column(Float, nullable=True)
    next_free_at = Column(DateTime(timezone=True), nullable=True)

    current_assignment_id = Column(Integer, ForeignKey('assignment.id', ondelete="SET NULL"), nullable=True)

    last_update = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))

    # Relations
    driver = relationship('Driver', backref='status', uselist=False, passive_deletes=True)
    current_assignment = relationship('Assignment', passive_deletes=True)

    @property
    def serialize(self):
        st_any = getattr(self, "state", None)
        state_str = st_any.value if isinstance(st_any, DriverState) else _as_str(st_any)

        return {
            "id": self.id,
            "driver_id": self.driver_id,
            "state": state_str,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "heading": self.heading,
            "speed": self.speed,
            "next_free_at": _iso(self.next_free_at),
            "current_assignment_id": self.current_assignment_id,
            "last_update": _iso(self.last_update),
        }

    def __repr__(self):
        st_any = getattr(self, "state", None)
        state_str = st_any.value if isinstance(st_any, DriverState) else _as_str(st_any)
        return f"<DriverStatus driver={self.driver_id} state={state_str} next_free_at={_iso(self.next_free_at)}>"

    # Validateurs
    @validates('driver_id')
    def _v_driver_id(self, _k, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("driver_id doit être un entier positif.")
        return v

    @validates('current_assignment_id')
    def _v_current_assignment_id(self, _k, v):
        if v is None:
            return None
        if not isinstance(v, int) or v <= 0:
            raise ValueError("current_assignment_id doit être NULL ou un entier positif.")
        return v

    @validates('state')
    def _v_state(self, _k, state):
        if isinstance(state, str):
            try:
                state = DriverState[state.upper()]
            except KeyError:
                raise ValueError(f"state invalide. Valeurs autorisées : {[s.value for s in DriverState]}")
        if not isinstance(state, DriverState):
            raise ValueError("state invalide (Enum attendu).")
        return state

    @validates('latitude')
    def _v_lat(self, _k, v):
        if v is None:
            return None
        v = float(v)
        if v < -90 or v > 90:
            raise ValueError("latitude hors bornes [-90, 90].")
        return v

    @validates('longitude')
    def _v_lon(self, _k, v):
        if v is None:
            return None
        v = float(v)
        if v < -180 or v > 180:
            raise ValueError("longitude hors bornes [-180, 180].")
        return v

    @validates('heading')
    def _v_heading(self, _k, v):
        if v is None:
            return None
        v = float(v)
        if v < 0 or v > 360:
            raise ValueError("heading doit être entre 0 et 360 degrés.")
        return v

    @validates('speed')
    def _v_speed(self, _k, v):
        if v is None:
            return None
        v = float(v)
        if v < 0:
            raise ValueError("speed ne peut pas être négative.")
        return v

    # Helpers métier
    def mark_available(self, when: datetime | None = None):
        self.state = DriverState.AVAILABLE
        self.next_free_at = when
        self.last_update = datetime.now(UTC)

    def mark_busy(self, next_free_at: datetime | None = None):
        self.state = DriverState.BUSY
        self.next_free_at = next_free_at
        self.last_update = datetime.now(UTC)

    def touch_location(self, lat: float, lon: float, heading: float | None = None, speed: float | None = None):
        self.latitude = lat
        self.longitude = lon
        if heading is not None:
            self.heading = heading
        if speed is not None:
            self.speed = speed
        self.last_update = datetime.now(UTC)


class RealtimeEvent(db.Model):
    __tablename__ = "realtime_event"
    __table_args__ = (
        Index('idx_realtime_event_company_type_time', 'company_id', 'event_type', 'timestamp'),
        Index('idx_realtime_event_entity_time', 'entity_type', 'entity_id', 'timestamp'),
        CheckConstraint("entity_id > 0", name="ck_realtime_entity_id_positive"),
        Index('ix_realtime_event_data_gin', 'data', postgresql_using='gin'),
    )

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('company.id', ondelete="CASCADE"), nullable=False, index=True)

    event_type = Column(SAEnum(RealtimeEventType, name="realtime_event_type"), nullable=False)
    entity_type = Column(SAEnum(RealtimeEntityType, name="realtime_entity_type"), nullable=False)

    entity_id = Column(Integer, nullable=False)
    data = Column(JSONB, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    company = relationship('Company', passive_deletes=True)

    def __repr__(self):
        et_any = getattr(self, "event_type", None)
        et_str = et_any.value if isinstance(et_any, RealtimeEventType) else _as_str(et_any) or "?"
        en_any = getattr(self, "entity_type", None)
        en_str = en_any.value if isinstance(en_any, RealtimeEntityType) else _as_str(en_any) or "?"
        return f"<RealtimeEvent id={self.id} company={self.company_id} type={et_str} entity={en_str}:{self.entity_id}>"

    @property
    def serialize(self):
        et_any = getattr(self, "event_type", None)
        et_str = et_any.value if isinstance(et_any, RealtimeEventType) else _as_str(et_any)

        en_any = getattr(self, "entity_type", None)
        en_str = en_any.value if isinstance(en_any, RealtimeEntityType) else _as_str(en_any)

        return {
            "id": self.id,
            "company_id": self.company_id,
            "event_type": et_str,
            "entity_type": en_str,
            "entity_id": self.entity_id,
            "data": self.data,
            "timestamp": _iso(self.timestamp),
        }

    # Validateurs
    @validates('entity_id')
    def _v_entity_id(self, _k, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("entity_id doit être un entier positif.")
        return v

    @validates('event_type')
    def _v_event_type(self, _k, v):
        if isinstance(v, str):
            try:
                v = RealtimeEventType[v.upper()]
            except KeyError:
                raise ValueError(f"event_type invalide. Valeurs autorisées : {[e.value for e in RealtimeEventType]}")
        if not isinstance(v, RealtimeEventType):
            raise ValueError("event_type invalide (Enum attendu).")
        return v

    @validates('entity_type')
    def _v_entity_type(self, _k, v):
        if isinstance(v, str):
            try:
                v = RealtimeEntityType[v.upper()]
            except KeyError:
                raise ValueError(f"entity_type invalide. Valeurs autorisées : {[e.value for e in RealtimeEntityType]}")
        if not isinstance(v, RealtimeEntityType):
            raise ValueError("entity_type invalide (Enum attendu).")
        return v


class DispatchMetrics(db.Model):
    """Métriques de dispatch pour analyse historique et performance"""
    __tablename__ = "dispatch_metrics"
    __table_args__ = (
        Index('ix_dispatch_metrics_company_date', 'company_id', 'date'),
        Index('ix_dispatch_metrics_dispatch_run', 'dispatch_run_id'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('company.id', ondelete="CASCADE"), index=True)
    dispatch_run_id: Mapped[int] = mapped_column(ForeignKey('dispatch_run.id', ondelete="CASCADE"))

    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False
    )

    # Métriques de performance
    total_bookings: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    on_time_bookings: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    delayed_bookings: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cancelled_bookings: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Métriques de retard
    average_delay_minutes: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    max_delay_minutes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_delay_minutes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Métriques des chauffeurs
    total_drivers: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    active_drivers: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    avg_bookings_per_driver: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Métriques d'optimisation
    total_distance_km: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_distance_per_booking: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    suggestions_generated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    suggestions_applied: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    quality_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    extra_data: Mapped[Dict[str, Any] | None] = mapped_column(MutableDict.as_mutable(JSONB()), default=dict)

    # Relations
    company: Mapped[Company] = relationship(back_populates='dispatch_metrics')
    dispatch_run: Mapped[DispatchRun] = relationship(back_populates='metrics_record')

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'company_id': self.company_id,
            'dispatch_run_id': self.dispatch_run_id,
            'date': self.date.isoformat(),
            'created_at': _iso(self.created_at),
            'total_bookings': self.total_bookings,
            'on_time_bookings': self.on_time_bookings,
            'delayed_bookings': self.delayed_bookings,
            'cancelled_bookings': self.cancelled_bookings,
            'average_delay_minutes': round(self.average_delay_minutes, 2),
            'max_delay_minutes': self.max_delay_minutes,
            'total_delay_minutes': self.total_delay_minutes,
            'total_drivers': self.total_drivers,
            'active_drivers': self.active_drivers,
            'avg_bookings_per_driver': round(self.avg_bookings_per_driver, 2),
            'total_distance_km': round(self.total_distance_km, 2),
            'avg_distance_per_booking': round(self.avg_distance_per_booking, 2),
            'suggestions_generated': self.suggestions_generated,
            'suggestions_applied': self.suggestions_applied,
            'quality_score': round(self.quality_score, 2),
            'extra_data': self.extra_data or {}
        }


class DailyStats(db.Model):
    """Statistiques agrégées par jour (pré-calculées pour performance)"""
    __tablename__ = "daily_stats"
    __table_args__ = (
        UniqueConstraint('company_id', 'date', name='uq_daily_stats_company_date'),
        Index('ix_daily_stats_company_date', 'company_id', 'date'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('company.id', ondelete="CASCADE"), index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    total_bookings: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    on_time_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_delay: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    bookings_trend: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    delay_trend: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=lambda: datetime.now(UTC)
    )

    # Relations
    company: Mapped[Company] = relationship(back_populates='daily_stats')

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'company_id': self.company_id,
            'date': self.date.isoformat(),
            'total_bookings': self.total_bookings,
            'on_time_rate': round(self.on_time_rate, 2),
            'avg_delay': round(self.avg_delay, 2),
            'quality_score': round(self.quality_score, 2),
            'bookings_trend': round(self.bookings_trend, 2),
            'delay_trend': round(self.delay_trend, 2),
            'created_at': _iso(self.created_at),
            'updated_at': _iso(self.updated_at)
        }

