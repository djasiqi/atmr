# models/booking.py

# Constantes pour éviter les valeurs magiques
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
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
from shared.driver_display import format_driver_display_name
from shared.serialize_compat import as_serialize_result
from shared.time_utils import (
    api_scheduled_iso_to_naive_geneva,
    iso_utc_z,
    now_local,
    split_date_time_local,
    to_geneva_local,
    to_utc_from_db,
)

from .base import _as_bool, _as_dt, _as_float, _as_int, _as_str
from .enums import (
    BillingReviewStatus,
    BillingSource,
    BookingCreatedVia,
    BookingStatus,
)

logger = logging.getLogger(__name__)


def _created_via_value(booking: Booking) -> str:
    raw = getattr(booking, "created_via", None)
    if raw is None:
        return BookingCreatedVia.LEGACY.value
    if isinstance(raw, BookingCreatedVia):
        return raw.value
    return str(raw).lower()


USER_ID_ZERO = 0
AMOUNT_ZERO = 0
VALUE_ZERO = 0
COMPANY_ID_ZERO = 0
CUSTOMER_NAME_MAX_LENGTH = 200
LOCATION_MAX_LENGTH = 500

# Constantes pour règles d'arrondi métier
AMOUNT_MINIMUM = 0.5
AMOUNT_ROUNDING_THRESHOLD_1 = 0.6
AMOUNT_ROUNDING_THRESHOLD_2 = 0.75
AMOUNT_ROUNDING_THRESHOLD_3 = 0.8
AMOUNT_ROUNDING_THRESHOLD_4 = 39.98
AMOUNT_ROUNDING_TARGET_1 = 0.5
AMOUNT_ROUNDING_TARGET_2 = 0.8
AMOUNT_ROUNDING_TARGET_3 = 40.0

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
        # ✅ Contrainte : status='ASSIGNED' implique driver_id IS NOT NULL
        CheckConstraint(
            "status != 'ASSIGNED' OR driver_id IS NOT NULL",
            name="chk_booking_assigned_requires_driver",
        ),
        # Aligné DB (migration institution) : description requise pour livraison matériel
        CheckConstraint(
            "mission_type::text <> 'material_delivery'::text OR delivery_description IS NOT NULL",
            name="ck_booking_material_delivery_description",
        ),
        Index("ix_booking_company_scheduled", "company_id", "scheduled_time"),
        Index("ix_booking_status_scheduled", "status", "scheduled_time"),
        Index("ix_booking_driver_status", "driver_id", "status"),
        # ✅ P1: Index composite optimisé pour requêtes dispatch (company_id, status, scheduled_time)
        # Améliore les performances de get_bookings_for_dispatch()
        Index(
            "ix_booking_company_status_scheduled",
            "company_id",
            "status",
            "scheduled_time",
        ),
        # ✅ P1: Index composite pour recherches par client avec tri temporel
        # Optimise les requêtes qui filtrent par client_id et trient par scheduled_time
        # Note: DESC sera géré par la migration (SQLAlchemy Index ne supporte pas directement DESC)
        Index("ix_booking_client_time", "client_id", "scheduled_time"),
        # Agrégations / filtres admin (tendances par mois sur created_at)
        Index("ix_booking_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_name: Mapped[str] = mapped_column(String(200), nullable=False)
    pickup_location: Mapped[str] = mapped_column(String(500), nullable=False)
    dropoff_location: Mapped[str] = mapped_column(String(500), nullable=False)
    booking_type = Column(String(200), nullable=False, server_default="standard")

    # ✅ Livraison matériel : type de mission (transport patient vs livraison)
    mission_type = Column(
        String(50), nullable=False, server_default="patient_transport"
    )  # "patient_transport" | "material_delivery"
    delivery_description = Column(
        Text, nullable=True
    )  # Requis si mission_type == "material_delivery"

    scheduled_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=False), nullable=True
    )

    amount: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[BookingStatus] = mapped_column(
        SAEnum(BookingStatus, name="booking_status"),
        index=True,
        nullable=False,
        default=BookingStatus.PENDING,
        server_default=BookingStatus.PENDING.value,
    )

    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=True)
    rejected_by = Column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )

    duration_seconds = Column(Integer)
    distance_meters = Column(Integer)

    client_id = Column(
        Integer, ForeignKey("client.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    company_id = Column(
        Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=True, index=True
    )
    executing_company_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )  # Entreprise qui exécute réellement (si transférée à un partenaire)
    driver_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("driver.id", ondelete="SET NULL"), nullable=True, index=True
    )

    is_round_trip = Column(Boolean, nullable=False, server_default=text("false"))
    is_return = Column(Boolean, nullable=False, server_default=text("false"))

    boarded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    parent_booking_id = Column(
        Integer, ForeignKey("booking.id", ondelete="SET NULL"), nullable=True
    )

    # ✅ NOUVEAU: Groupe de courses (pour courses groupées)
    booking_group_id = Column(
        Integer,
        ForeignKey("booking.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    medical_facility = Column(String(200))
    doctor_name = Column(String(200))
    hospital_service = Column(String(100))
    notes_medical = Column(Text)
    pickup_access_notes = Column(Text, nullable=True)
    dropoff_access_notes = Column(Text, nullable=True)
    pickup_floor = Column(String(50), nullable=True)
    pickup_door_code = Column(String(50), nullable=True)
    dropoff_floor = Column(String(50), nullable=True)
    dropoff_door_code = Column(String(50), nullable=True)
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
    pickup_geo_unit_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("geo_unit.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    dropoff_geo_unit_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("geo_unit.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    pickup_zip = Column(String(16), nullable=True)
    dropoff_zip = Column(String(16), nullable=True)
    pickup_admin_token = Column(String(64), nullable=True, index=True)
    pickup_canton_code = Column(String(8), nullable=True, index=True)
    pickup_admin_source = Column(String(24), nullable=True)
    pickup_admin_confidence = Column(String(24), nullable=True)
    pickup_admin_label = Column(String(160), nullable=True)
    pickup_admin_resolved_at = Column(DateTime(timezone=True), nullable=True)
    dropoff_admin_token = Column(String(64), nullable=True, index=True)
    dropoff_canton_code = Column(String(8), nullable=True, index=True)
    dropoff_admin_source = Column(String(24), nullable=True)
    dropoff_admin_confidence = Column(String(24), nullable=True)
    dropoff_admin_label = Column(String(160), nullable=True)
    dropoff_admin_resolved_at = Column(DateTime(timezone=True), nullable=True)
    pricing_profile_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("pricing_profile.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    pricing_profile_version_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("pricing_profile_version.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    price_amount = Column(Numeric(10, 2), nullable=True)
    price_breakdown_json = Column(JSONB, nullable=True)

    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    edit_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("1"),
        default=1,
    )

    active_change_request_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("booking_change_requests.id", ondelete="SET NULL"),
        nullable=True,
    )
    route_group_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    route_sequence_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    billed_to_type = Column(String(50), nullable=False, server_default="patient")
    billed_to_company_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="SET NULL"), nullable=True
    )
    billed_to_contact = Column(String(120))

    # ✅ Workflow de contrôle facturation (V1)
    billing_review_status: Mapped[BillingReviewStatus] = mapped_column(
        SAEnum(
            BillingReviewStatus,
            name="billing_review_status",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
        server_default=BillingReviewStatus.DRAFT.value,
    )
    billing_locked_at = Column(DateTime(timezone=True), nullable=True)
    billing_locked_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True, index=True
    )
    billing_override_reason = Column(Text, nullable=True)

    # Nouveau lien vers un tiers payeur “unifié” (optionnel en V1).
    billing_party_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("billing_parties.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Patient institutionnel transporté (copie de TransportRequest.patient_id à l'acceptation).
    # Distinct du client_id technique porteur (souvent le compte institution).
    institution_patient_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("institution_patients.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ✅ Source de la décision de facturation (traçabilité)
    billing_source: Mapped[BillingSource | None] = mapped_column(
        SAEnum(
            BillingSource,
            name="billing_source",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=True,
    )
    billing_source_ref: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # ex "voucher#12345", "stay#456", "import_clinic_csv_2026-01"

    invoice_line_id = Column(
        Integer,
        ForeignKey(
            "invoice_lines.id", ondelete="SET NULL", name="fk_booking_invoice_line"
        ),
        nullable=True,
        index=True,
    )

    # ✅ Annulation standardisée (motif obligatoire, facturation déterministe)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_by_role = Column(
        String(20), nullable=True
    )  # company | driver | admin | system
    cancellation_reason_code = Column(String(50), nullable=True)
    cancellation_reason_text = Column(Text, nullable=True)
    is_cancellation_billable = Column(Boolean, nullable=True)
    cancellation_display_label = Column(String(120), nullable=True)
    cancellation_fee_amount = Column(Numeric(10, 2), nullable=True)
    cancellation_fee_percent = Column(Integer, nullable=True)
    cancellation_fee_tier_id = Column(String(50), nullable=True)

    created_via: Mapped[BookingCreatedVia] = mapped_column(
        SAEnum(
            BookingCreatedVia,
            name="booking_created_via",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
        default=BookingCreatedVia.LEGACY,
        server_default=BookingCreatedVia.LEGACY.value,
    )

    # Origine commerciale facturable (distinct de created_via technique)
    billing_origin: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        index=True,
        comment="OWN_PORTFOLIO | LIRIE_MARKETPLACE | IMPORTED | ADMIN_CREATED | UNKNOWN",
    )
    billing_origin_source: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="EXPLICIT_AT_CREATION | BACKFILL_* | ADMIN_CORRECTION",
    )
    billing_origin_reason: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
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
    executing_company = relationship("Company", foreign_keys=[executing_company_id])
    invoice_line = relationship(
        "InvoiceLine",
        foreign_keys=[invoice_line_id],
        primaryjoin="Booking.invoice_line_id == InvoiceLine.id",
        uselist=False,
        backref="billed_booking",
    )

    billing_locked_by_user = relationship(
        "User", foreign_keys=[billing_locked_by_user_id]
    )
    billing_party = relationship("BillingParty", foreign_keys=[billing_party_id])
    institution_patient = relationship(
        "InstitutionPatient", foreign_keys=[institution_patient_id]
    )
    pickup_geo_unit = relationship("GeoUnit", foreign_keys=[pickup_geo_unit_id])
    dropoff_geo_unit = relationship("GeoUnit", foreign_keys=[dropoff_geo_unit_id])
    pricing_profile = relationship("PricingProfile", foreign_keys=[pricing_profile_id])
    pricing_profile_version = relationship(
        "PricingProfileVersion", foreign_keys=[pricing_profile_version_id]
    )

    return_trip = relationship(
        "Booking",
        backref="original_booking",
        remote_side=[id],
        foreign_keys=[parent_booking_id],
        uselist=False,
    )
    transport_vouchers = relationship(
        "TransportVoucher",
        back_populates="booking",
        passive_deletes=True,
    )
    change_requests = relationship(
        "BookingChangeRequest",
        back_populates="booking",
        foreign_keys="BookingChangeRequest.booking_id",
    )
    active_change_request = relationship(
        "BookingChangeRequest",
        foreign_keys=[active_change_request_id],
        uselist=False,
    )

    # Propriétés
    @property
    def customer_full_name(self) -> str:
        cust = _as_str(self.customer_name)
        if cust:
            return str(cust)
        if self.client and self.client.user:
            u = self.client.user
            if u.first_name or u.last_name:
                return f"{u.first_name or ''} {u.last_name or ''}".strip()
            return str(u.username)
        return "Non spécifié"

    @property
    def driver_display_name(self) -> str | None:
        """Nom affichable du chauffeur assigné (sans « None None »)."""
        return format_driver_display_name(getattr(self, "driver", None))

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
        billing_locked_at_dt = _as_dt(getattr(self, "billing_locked_at", None))
        cancelled_dt = cast("datetime | None", _as_dt(self.cancelled_at))
        pickup_admin_resolved_dt = cast(
            "datetime | None", _as_dt(getattr(self, "pickup_admin_resolved_at", None))
        )
        dropoff_admin_resolved_dt = cast(
            "datetime | None", _as_dt(getattr(self, "dropoff_admin_resolved_at", None))
        )

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
        status_val = self.status

        cli = self.client
        cli_user = getattr(cli, "user", None)

        return as_serialize_result(
            {
                "id": self.id,
                # ✅ P1-4 Phase 1.1: Supprimer customer_name, garder uniquement client_name
                "client_name": self.customer_full_name,
                "pickup_location": self.pickup_location,
                "dropoff_location": self.dropoff_location,
                # ✅ P1-4 Phase 1.3: Ajouter coordonnées GPS
                "pickup_lat": _as_float(self.pickup_lat),
                "pickup_lon": _as_float(self.pickup_lon),
                "dropoff_lat": _as_float(self.dropoff_lat),
                "dropoff_lon": _as_float(self.dropoff_lon),
                "pickup_geo_unit_id": self.pickup_geo_unit_id,
                "dropoff_geo_unit_id": self.dropoff_geo_unit_id,
                "pickup_zip": self.pickup_zip,
                "dropoff_zip": self.dropoff_zip,
                "pickup_admin_token": self.pickup_admin_token,
                "pickup_canton_code": self.pickup_canton_code,
                "pickup_admin_source": self.pickup_admin_source,
                "pickup_admin_confidence": self.pickup_admin_confidence,
                "pickup_admin_label": self.pickup_admin_label,
                "pickup_admin_resolved_at": (
                    pickup_admin_resolved_dt.isoformat()
                    if pickup_admin_resolved_dt
                    else None
                ),
                "dropoff_admin_token": self.dropoff_admin_token,
                "dropoff_canton_code": self.dropoff_canton_code,
                "dropoff_admin_source": self.dropoff_admin_source,
                "dropoff_admin_confidence": self.dropoff_admin_confidence,
                "dropoff_admin_label": self.dropoff_admin_label,
                "dropoff_admin_resolved_at": (
                    dropoff_admin_resolved_dt.isoformat()
                    if dropoff_admin_resolved_dt
                    else None
                ),
                "amount": round(amt, 2),
                "price_amount": _as_float(self.price_amount),
                "price_breakdown_json": self.price_breakdown_json,
                "pricing_profile_id": self.pricing_profile_id,
                "pricing_profile_version_id": self.pricing_profile_version_id,
                "scheduled_time": (
                    iso_utc_z(to_utc_from_db(scheduled_dt)) if scheduled_dt else None
                ),
                "date_formatted": date_local or "Non spécifié",
                "time_formatted": time_local or "Non spécifié",
                "status": getattr(status_val, "value", "unknown").lower(),
                "created_via": _created_via_value(self),
                "booking_type": _as_str(getattr(self, "booking_type", None))
                or "standard",
                "client": {
                    "id": getattr(cli, "id", None),
                    "first_name": getattr(cli_user, "first_name", "")
                    if cli_user
                    else "",
                    "last_name": getattr(cli_user, "last_name", "") if cli_user else "",
                    "email": getattr(cli_user, "email", "") if cli_user else "",
                    "full_name": self.customer_full_name,
                    "client_type": (
                        cli.client_type.value
                        if cli and getattr(cli, "client_type", None) is not None
                        else None
                    ),
                    "linked_institution_id": (
                        getattr(cli, "linked_institution_id", None) if cli else None
                    ),
                    "birth_date": (
                        cli_user.birth_date.strftime("%Y-%m-%d")
                        if cli_user and cli_user.birth_date
                        else None
                    ),
                    "gender": (
                        cli_user.gender.value
                        if cli_user and getattr(cli_user, "gender", None)
                        else None
                    ),
                    "contact_phone": getattr(cli, "contact_phone", None)
                    if cli
                    else None,
                    "phone": getattr(cli_user, "phone", None) if cli_user else None,
                    "gp_phone": getattr(cli, "gp_phone", None) if cli else None,
                    "door_code": getattr(cli, "door_code", None) if cli else None,
                    "floor": getattr(cli, "floor", None) if cli else None,
                    "access_notes": getattr(cli, "access_notes", None) if cli else None,
                    "is_institution": bool(getattr(cli, "is_institution", False))
                    if cli
                    else False,
                    "institution_name": getattr(cli, "institution_name", None)
                    if cli
                    else None,
                },
                # ✅ P1-4 Phase 1.2: Remplacer company (string) par company_id + company_name
                "company_id": self.company_id,
                "company_name": self.company.name if self.company else None,
                "company_contact_phone": (
                    getattr(self.company, "contact_phone", None)
                    if self.company
                    else None
                ),
                # ✅ Informations de transfert partenaire
                "executing_company_id": self.executing_company_id,
                "executing_company_name": (
                    self.executing_company.name if self.executing_company else None
                ),
                # ✅ FIX: is_transferred doit être True s'il y a un transfert ACCEPTED/COMPLETED
                # car après acceptation, company_id est mis à jour mais on doit garder la trace
                # que la course a été transférée (owner_company_id != company_id dans le transfert)
                "is_transferred": self._is_transferred(),
                "driver": {
                    "id": self.driver.id,
                    "username": self.driver.user.username if self.driver.user else None,
                    "first_name": self.driver.user.first_name
                    if self.driver.user
                    else None,
                    "last_name": self.driver.user.last_name
                    if self.driver.user
                    else None,
                    "full_name": self.driver_display_name,
                }
                if self.driver
                else None,
                "driver_id": self.driver_id,
                "driver_name": self.driver_display_name,
                "duration_seconds": self.duration_seconds,
                "distance_meters": self.distance_meters,
                "medical_facility": self.medical_facility or "Non spécifié",
                "doctor_name": self.doctor_name or "Non spécifié",
                "hospital_service": self.hospital_service or "Non spécifié",
                "notes_medical": self.notes_medical or "Aucune note",
                "pickup_access_notes": getattr(self, "pickup_access_notes", None)
                or None,
                "dropoff_access_notes": getattr(self, "dropoff_access_notes", None)
                or None,
                "pickup_floor": getattr(self, "pickup_floor", None) or None,
                "pickup_door_code": getattr(self, "pickup_door_code", None) or None,
                "dropoff_floor": getattr(self, "dropoff_floor", None) or None,
                "dropoff_door_code": getattr(self, "dropoff_door_code", None) or None,
                "wheelchair_client_has": _as_bool(self.wheelchair_client_has),
                "wheelchair_need": _as_bool(self.wheelchair_need),
                # ✅ P1-4 Phase 1.4: Standardiser timestamps en ISO 8601
                "created_at": (
                    iso_utc_z(to_utc_from_db(created_dt))
                    if isinstance(created_dt, datetime)
                    else None
                ),
                "edit_version": int(self.edit_version or 1),
                "updated_at": (
                    iso_utc_z(to_utc_from_db(updated_dt))
                    if isinstance(updated_dt, datetime)
                    else None
                ),
                # ✅ P1-4 Phase 1.4: Ajouter versions formatées pour compatibilité (optionnel)
                "created_at_formatted": created_loc.strftime("%d/%m/%Y %H:%M")
                if created_loc
                else "Non spécifié",
                "updated_at_formatted": updated_loc.strftime("%d/%m/%Y %H:%M")
                if updated_loc
                else "Non spécifié",
                "rejected_by": self.rejected_by,
                "is_round_trip": _as_bool(self.is_round_trip),
                "is_return": _as_bool(self.is_return),
                "parent_booking_id": self.parent_booking_id,
                "time_confirmed": _as_bool(self.time_confirmed),
                "has_return": self.return_trip is not None,
                "boarded_at": iso_utc_z(to_utc_from_db(boarded_dt))
                if boarded_dt
                else None,
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
                "billed_to_type": (_as_str(self.billed_to_type) or "patient"),
                "billed_to_company_id": self.billed_to_company_id,
                "billing_locked_at": (
                    iso_utc_z(to_utc_from_db(billing_locked_at_dt))
                    if billing_locked_at_dt
                    else None
                ),
                "invoice_line_id": self.invoice_line_id,
                # ✅ Traçabilité de la décision de facturation
                "billing_source": (
                    self.billing_source.value if self.billing_source else None
                ),
                "billing_source_ref": self.billing_source_ref,
                # ✅ P1-4 Phase 1.1: patient_name reste pour compatibilité (utilise customer_name de la DB)
                "patient_name": _as_str(self.customer_name),
                # ✅ Informations du transfert actif (si existe)
                "active_transfer": self._get_active_transfer_info(),
                # ✅ Livraison matériel
                "mission_type": getattr(self, "mission_type", None)
                or "patient_transport",
                "delivery_description": getattr(self, "delivery_description", None)
                or None,
                # ✅ Annulation
                "cancelled_at": iso_utc_z(to_utc_from_db(cancelled_dt))
                if cancelled_dt
                else None,
                "cancelled_by_role": self.cancelled_by_role,
                "cancellation_reason_code": self.cancellation_reason_code,
                "cancellation_reason_text": self.cancellation_reason_text,
                "is_cancellation_billable": self.is_cancellation_billable,
                "cancellation_display_label": self.cancellation_display_label,
                "cancellation_fee_amount": float(self.cancellation_fee_amount)
                if getattr(self, "cancellation_fee_amount", None) is not None
                else None,  # type: ignore[arg-type]
                "cancellation_fee_percent": self.cancellation_fee_percent,
                "cancellation_fee_tier_id": self.cancellation_fee_tier_id,
                # ✅ Timeline institution (si booking issu d'une demande institution)
                "institution_timeline": self._get_institution_timeline(),
                # ✅ Historique opérationnel consolidé (tous les legs + retours)
                "route_journey": self._get_route_journey(),
                "online_payment": self._online_client_payment_brief(),
                "active_change_request_id": self.active_change_request_id,
                "active_change_request": (
                    self.active_change_request.serialize()
                    if getattr(self, "active_change_request", None)
                    else None
                ),
                "route_group_id": getattr(self, "route_group_id", None),
                "route_sequence_number": getattr(self, "route_sequence_number", None),
                "passenger": self._get_institution_passenger_brief(),
                "institution_leg": self._get_institution_leg_clinical_brief(),
                **self._canonical_display_payload(),
            }
        )

    @property
    def serialize_dashboard(self):
        """Projection légère pour le dashboard entreprise (KPI, table, dispatch)."""
        scheduled_dt = _as_dt(self.scheduled_time)
        status_val = self.status
        cli = self.client
        transfer_cache = getattr(self, "_transfer_cache", None)
        if transfer_cache is not None:
            is_transferred = bool(transfer_cache.get("is_transferred"))
            active_transfer = transfer_cache.get("active_transfer")
        else:
            is_transferred = (
                self.executing_company_id is not None
                and self.executing_company_id != self.company_id
            )
            active_transfer = None

        date_local, time_local = (
            split_date_time_local(scheduled_dt) if scheduled_dt else (None, None)
        )

        return as_serialize_result(
            {
                "id": self.id,
                "client_name": self.customer_full_name,
                "pickup_location": self.pickup_location,
                "dropoff_location": self.dropoff_location,
                "pickup_lat": _as_float(self.pickup_lat),
                "pickup_lon": _as_float(self.pickup_lon),
                "dropoff_lat": _as_float(self.dropoff_lat),
                "dropoff_lon": _as_float(self.dropoff_lon),
                "scheduled_time": (
                    iso_utc_z(to_utc_from_db(scheduled_dt)) if scheduled_dt else None
                ),
                "date_formatted": date_local or "Non spécifié",
                "time_formatted": time_local or "Non spécifié",
                "status": getattr(status_val, "value", "unknown").lower(),
                "company_id": self.company_id,
                "company_name": self.company.name if self.company else None,
                "executing_company_id": self.executing_company_id,
                "executing_company_name": (
                    self.executing_company.name if self.executing_company else None
                ),
                "is_transferred": is_transferred,
                "active_transfer": active_transfer,
                "driver_id": self.driver_id,
                "driver_name": self.driver_display_name,
                "driver": {
                    "id": self.driver.id,
                    "full_name": self.driver_display_name,
                }
                if self.driver
                else None,
                "client": {
                    "id": getattr(cli, "id", None),
                    "full_name": self.customer_full_name,
                    "is_institution": bool(getattr(cli, "is_institution", False))
                    if cli
                    else False,
                    "institution_name": getattr(cli, "institution_name", None)
                    if cli
                    else None,
                    "client_type": (
                        cli.client_type.value
                        if cli and getattr(cli, "client_type", None) is not None
                        else None
                    ),
                    "linked_institution_id": (
                        getattr(cli, "linked_institution_id", None) if cli else None
                    ),
                },
                "is_return": _as_bool(self.is_return),
                "is_round_trip": _as_bool(self.is_round_trip),
                "parent_booking_id": self.parent_booking_id,
                "has_return": self.return_trip is not None,
                "time_confirmed": _as_bool(self.time_confirmed),
                "created_via": _created_via_value(self),
                "booking_type": _as_str(getattr(self, "booking_type", None))
                or "standard",
                "mission_type": getattr(self, "mission_type", None)
                or "patient_transport",
                "wheelchair_need": _as_bool(self.wheelchair_need),
                "amount": round(_as_float(self.amount), 2),
                "billed_to_type": (_as_str(self.billed_to_type) or "patient"),
                "billed_to_company_id": self.billed_to_company_id,
                "route_group_id": getattr(self, "route_group_id", None),
                "route_sequence_number": getattr(self, "route_sequence_number", None),
                "institution_timeline": self._get_institution_timeline(),
                "route_journey": self._get_route_journey(),
                "active_change_request_id": self.active_change_request_id,
                "active_change_request": (
                    self.active_change_request.serialize()
                    if getattr(self, "active_change_request", None)
                    else None
                ),
                **self._canonical_display_payload(),
            }
        )

    def _online_client_payment_brief(self):
        """Résumé du dernier paiement en ligne (Saferpay ou ancien Worldline) pour le portail client."""
        try:
            payments = list(getattr(self, "payments", []) or [])
            candidate_rows = [
                p
                for p in payments
                if str(getattr(p, "payment_provider", "")).strip().lower()
                in {"saferpay", "worldline"}
            ]
            row = max(
                candidate_rows, key=lambda p: int(getattr(p, "id", 0)), default=None
            )
            if not row:
                return None
            st = row.status.value if hasattr(row.status, "value") else str(row.status)
            prov = (row.payment_provider or "").strip().lower()
            pending_session = False
            if st.lower() == "pending":
                if prov == "worldline":
                    pending_session = bool(row.worldline_hosted_checkout_id)
                elif prov == "saferpay":
                    pending_session = bool(row.saferpay_token) or bool(
                        getattr(row, "saferpay_transaction_id", None)
                    )
            return {
                "payment_id": row.id,
                "status": st.lower(),
                "provider": prov or None,
                "has_pending_session": pending_session,
            }
        except Exception:
            logger.warning(
                "online_payment brief indisponible booking_id=%s",
                getattr(self, "id", None),
                exc_info=True,
            )
            return None

    def _canonical_display_payload(self) -> dict[str, Any]:
        """Blocs BookingDisplayModel v1 (identity, scheduling, trip_flags, search_index)."""
        from services.companies.booking_display import build_booking_display_blocks

        viewer_id = getattr(self, "_serialize_viewer_company_id", None)
        return build_booking_display_blocks(self, viewer_company_id=viewer_id)

    def _resolve_source_transport_request(self):
        """Demande institution source (directe, parent A/R ou route_group_id)."""
        try:
            reqs = getattr(self, "source_request", None)
            if (
                not reqs
                and getattr(self, "is_return", False)
                and self.parent_booking_id
            ):  # type: ignore[truthy-bool]
                parent = getattr(self, "return_trip", None)
                if parent:
                    reqs = getattr(parent, "source_request", None)
            if not reqs and getattr(self, "route_group_id", None):
                from models.transport_request import TransportRequest

                return TransportRequest.query.filter_by(
                    route_group_id=self.route_group_id
                ).first()
            if not reqs:
                return None
            return reqs[0] if isinstance(reqs, list) else reqs
        except Exception:
            return None

    def _get_institution_passenger_brief(self) -> dict[str, Any] | None:
        """Snapshot patient institution (DOB, ref. externe) pour l'UI transporteur."""
        try:
            req = self._resolve_source_transport_request()
            if req is None:
                return None
            patient = getattr(req, "patient", None)
            if patient is None:
                return None
            dob = getattr(patient, "dob", None)
            gender_raw = getattr(patient, "gender", None)
            gender_val = (
                gender_raw.value
                if gender_raw is not None and hasattr(gender_raw, "value")
                else (str(gender_raw).strip() if gender_raw else None)
            )
            return {
                "institution_patient_id": getattr(patient, "id", None),
                "first_name": getattr(patient, "first_name", None),
                "last_name": getattr(patient, "last_name", None),
                "birth_date": dob.isoformat() if dob is not None else None,
                "gender": gender_val,
                "external_reference": getattr(patient, "external_reference", None),
                "phone": getattr(patient, "phone", None),
            }
        except Exception:
            return None

    def _get_institution_leg_clinical_brief(self) -> dict[str, Any] | None:
        """Détails clinique du leg source (établissement, RDV) pour l'UI transporteur."""
        try:
            req = self._resolve_source_transport_request()
            if req is None:
                return None
            legs = sorted(
                getattr(req, "legs", None) or [],
                key=lambda item: getattr(item, "sequence_index", 0),
            )
            if not legs:
                return None
            seq = getattr(self, "route_sequence_number", None)
            leg = None
            if seq is not None:
                leg = next(
                    (
                        item
                        for item in legs
                        if getattr(item, "route_sequence_number", None) == seq
                    ),
                    None,
                )
            if leg is None:
                leg = legs[0]
            from shared.time_utils import iso_utc_z, to_utc_from_db

            appt_dt = getattr(leg, "scheduled_time", None)
            appt_iso = (
                iso_utc_z(to_utc_from_db(appt_dt))
                if appt_dt is not None and getattr(leg, "time_confirmed", False)
                else None
            )
            return {
                "establishment": getattr(leg, "dropoff_establishment", None),
                "service": getattr(leg, "dropoff_service", None),
                "doctor": getattr(leg, "dropoff_doctor", None),
                "appointment_time": appt_iso,
                "time_confirmed": bool(getattr(leg, "time_confirmed", False)),
            }
        except Exception:
            return None

    def _get_institution_timeline(self):
        """Retourne les dates cles de la demande institution source, si elle existe.

        Pour un booking retour (is_return=True), remonte au parent pour
        trouver la TransportRequest source.
        """
        try:
            req = self._resolve_source_transport_request()
            if req is None:
                return None
            inst = getattr(req, "institution", None)
            inst_name = getattr(inst, "name", None) if inst else None
            company = getattr(req, "accepted_by_company", None)
            company_name = getattr(company, "name", None) if company else None
            creator_name = None
            creator = getattr(req, "created_by", None)
            if creator:
                first = getattr(creator, "first_name", "") or ""
                last = getattr(creator, "last_name", "") or ""
                creator_name = f"{first} {last}".strip() or None

            return {
                "institution_name": inst_name,
                "created_at": req.created_at.isoformat() if req.created_at else None,
                "created_by_name": creator_name,
                "sent_at": req.sent_at.isoformat() if req.sent_at else None,
                "accepted_at": req.accepted_at.isoformat() if req.accepted_at else None,
                "accepted_by_company_name": company_name,
                "converted_at": req.converted_at.isoformat()
                if req.converted_at
                else None,
                "cancelled_at": req.cancelled_at.isoformat()
                if req.cancelled_at
                else None,
            }
        except Exception:
            return None

    def _collect_journey_legs(self):
        """Construit la liste ordonnée des trajets composant un même transport.

        Un transport partage un seul historique :
        - aller simple : 1 trajet ;
        - aller/retour : aller + retour ;
        - multi-étapes : tous les legs du ``route_group_id`` + leurs retours.

        L'ancrage se fait toujours sur l'aller (si l'on consulte un retour, on
        remonte au parent), puis on agrège les legs du groupe et, pour chaque
        leg, ses retours (enfants liés par ``parent_booking_id``).

        Retourne une liste de tuples ``(booking, is_return)``.
        """
        from models.booking import Booking

        # Ancrer sur l'aller si l'on consulte un retour.
        base = self
        if getattr(self, "is_return", False) and getattr(
            self, "parent_booking_id", None
        ):
            parent = Booking.query.get(self.parent_booking_id)
            if parent is not None:
                base = parent

        # Legs du parcours (multi-étapes) ou aller unique.
        group_id = getattr(base, "route_group_id", None)
        if group_id:
            legs = (
                Booking.query.filter_by(route_group_id=group_id)
                .order_by(Booking.route_sequence_number.asc(), Booking.id.asc())
                .all()
            )
        else:
            legs = [base]

        pairs: list[tuple[Any, bool]] = []
        for leg in legs:
            pairs.append((leg, False))
            returns = (
                Booking.query.filter_by(parent_booking_id=leg.id)
                .order_by(Booking.id.asc())
                .all()
            )
            for rt in returns:
                pairs.append((rt, True))
        return pairs

    def _get_route_journey(self):
        """Historique opérationnel consolidé et partagé d'un transport.

        Couvre la prise en charge (``boarded_at``) et la dépose / fin de course
        (``completed_at``) de chaque trajet du transport (legs multi-étapes +
        retours), afin d'afficher un historique unique quel que soit le trajet
        consulté.

        Retourne une liste d'événements ``{event, date, type, leg_index, is_return, is_final_leg, leg_count}``
        triée par date, ou ``None`` si aucun événement opérationnel.
        """
        try:
            # Chemin rapide : course simple sans groupe ni retour → pas de requête.
            if (
                not getattr(self, "route_group_id", None)
                and not getattr(self, "parent_booking_id", None)
                and not getattr(self, "is_round_trip", False)
            ):
                pairs: list[tuple[Any, bool]] = [(self, False)]
            else:
                pairs = self._collect_journey_legs()

            non_return_legs = [b for b, is_ret in pairs if not is_ret]
            multi = len(non_return_legs) > 1
            leg_count = len(non_return_legs)
            events: list[dict[str, Any]] = []

            def _add_leg(leg, base_label, *, is_return, leg_index, is_final_leg):
                if base_label and is_return:
                    suffix = f" — {base_label} (retour)"
                elif base_label:
                    suffix = f" — {base_label}"
                elif is_return:
                    suffix = " — Retour"
                else:
                    suffix = ""
                boarded = _as_dt(getattr(leg, "boarded_at", None))
                if boarded:
                    events.append(
                        {
                            "event": f"Prise en charge{suffix}",
                            "date": iso_utc_z(to_utc_from_db(boarded)),
                            "type": "pickup",
                            "leg_index": leg_index,
                            "is_return": is_return,
                            "is_final_leg": False,
                            "leg_count": leg_count,
                        }
                    )
                completed = _as_dt(getattr(leg, "completed_at", None))
                if completed:
                    events.append(
                        {
                            "event": f"Dépose / course terminée{suffix}",
                            "date": iso_utc_z(to_utc_from_db(completed)),
                            "type": "dropoff",
                            "leg_index": leg_index,
                            "is_return": is_return,
                            "is_final_leg": is_final_leg and not is_return,
                            "leg_count": leg_count,
                        }
                    )

            for leg, is_return in pairs:
                seq = getattr(leg, "route_sequence_number", None)
                base_label = f"Trajet {seq}" if (multi and seq) else None
                leg_index = int(seq) if (multi and seq) else None
                is_final = (
                    not is_return and leg_index is not None and leg_index == leg_count
                )
                _add_leg(
                    leg,
                    base_label,
                    is_return=is_return,
                    leg_index=leg_index,
                    is_final_leg=is_final,
                )

            events.sort(key=lambda e: e.get("date") or "")
            return events or None
        except Exception:
            return None

    def _is_transferred(self) -> bool:
        """Détermine si la course a été transférée (ACCEPTED ou COMPLETED).

        Après acceptation d'un transfert, company_id est mis à jour pour pointer
        vers l'entreprise receveuse. Pour garder la trace que la course a été
        transférée, on vérifie s'il existe un transfert ACCEPTED ou COMPLETED
        où owner_company_id != company_id actuel.
        """
        transfer_cache = getattr(self, "_transfer_cache", None)
        if transfer_cache is not None:
            return bool(transfer_cache.get("is_transferred"))
        try:
            from models.booking_transfer import BookingTransfer
            from models.enums import TransferStatus

            # Chercher un transfert ACCEPTED ou COMPLETED
            # (PENDING ne compte pas car pas encore accepté)
            active_transfer = (
                BookingTransfer.query.filter_by(booking_id=self.id)
                .filter(
                    BookingTransfer.status.in_(
                        [
                            TransferStatus.ACCEPTED,
                            TransferStatus.COMPLETED,
                        ]
                    )
                )
                .first()
            )

            if active_transfer:
                # Si un transfert ACCEPTED/COMPLETED existe, la course a été transférée
                # On vérifie aussi que owner_company_id != company_id actuel pour s'assurer
                # que la course a bien été transférée (et non créée par l'entreprise actuelle)
                return active_transfer.owner_company_id != self.company_id

            return False
        except Exception:
            # En cas d'erreur, fallback sur l'ancienne logique
            return (
                self.executing_company_id is not None
                and self.executing_company_id != self.company_id
            )

    def _get_active_transfer_info(self):
        """Récupère les informations du transfert actif pour cette réservation."""
        transfer_cache = getattr(self, "_transfer_cache", None)
        if transfer_cache is not None:
            return transfer_cache.get("active_transfer")
        try:
            from models.booking_transfer import BookingTransfer
            from models.enums import TransferStatus

            # Chercher un transfert actif (PENDING, ACCEPTED ou COMPLETED)
            # ✅ Inclure COMPLETED pour afficher le badge même après la fin de la course
            active_transfer = (
                BookingTransfer.query.filter_by(booking_id=self.id)
                .filter(
                    BookingTransfer.status.in_(
                        [
                            TransferStatus.PENDING,
                            TransferStatus.ACCEPTED,
                            TransferStatus.COMPLETED,
                        ]
                    )
                )
                .first()
            )

            if active_transfer:
                return active_transfer.to_dict()
            return None
        except Exception:
            # En cas d'erreur, retourner None pour ne pas bloquer la sérialisation
            return None

    # Validations
    @validates("user_id")
    def validate_user_id(self, _key, user_id):
        if user_id is None:
            return None
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
        amt = float(amount)
        # Aller-retour : le segment retour est créé avec amount=0 (tarif porté par l'aller).
        # Placer is_return avant amount dans le constructeur pour que ce cas soit reconnu.
        if getattr(self, "is_return", False) and amt == 0:
            return 0.0
        if amt < AMOUNT_MINIMUM:
            msg = f"Le montant minimum accepté est {AMOUNT_MINIMUM}"
            raise ValueError(msg)
        # Règles d'arrondi métier
        # - 0.5-0.6 → 0.5
        # - 0.75-0.8 → 0.8
        # - 39.98-40.0 → 40.0
        if (
            AMOUNT_MINIMUM <= amt < AMOUNT_ROUNDING_THRESHOLD_1
            or AMOUNT_ROUNDING_THRESHOLD_1 <= amt < AMOUNT_ROUNDING_THRESHOLD_2
        ):
            return AMOUNT_ROUNDING_TARGET_1
        if AMOUNT_ROUNDING_THRESHOLD_2 <= amt < AMOUNT_ROUNDING_THRESHOLD_3:
            return AMOUNT_ROUNDING_TARGET_2
        if AMOUNT_ROUNDING_THRESHOLD_4 <= amt < AMOUNT_ROUNDING_TARGET_3:
            return AMOUNT_ROUNDING_TARGET_3
        # Arrondi standard à 2 décimales pour les autres cas
        return round(amt, 2)

    @validates("scheduled_time")
    def validate_scheduled_time(self, _key, scheduled_time):
        # scheduled_time est obligatoire pour le dispatch. Les retours et courses
        # « heure à définir » (time_confirmed=False) peuvent rester sans horaire.
        # Note: time_confirmed peut être None pendant la construction Python
        # (server_default DB non encore appliqué) — ne pas le forcer à True ici,
        # sinon les fixtures historiques (dates passées) cassent en masse.
        is_return = getattr(self, "is_return", False)
        time_confirmed = getattr(self, "time_confirmed", True)
        if scheduled_time is None:
            if is_return or not time_confirmed:
                return None
            msg = "scheduled_time est obligatoire. Le dispatch nécessite cette valeur."
            raise ValueError(msg)
        st = api_scheduled_iso_to_naive_geneva(scheduled_time)
        if st is None:
            raise ValueError("Format de scheduled_time invalide.")
        # ⚠️ IMPORTANT : Validation désactivée si time_confirmed=False
        # Cela permet d'importer des bookings historiques avec des dates passées.
        # Lorsque time_confirmed=False :
        #   - Les dates passées sont acceptées (pour import historique)
        #   - Le booking est exclu du dispatch automatique (voir get_bookings_for_dispatch)
        #   - Utile pour les retours avec heure à confirmer ou imports de données anciennes
        # ✅ Sentinelle 00:00:00 = "heure à définir" : ne pas valider "dans le passé"
        is_sentinel_midnight = st.hour == 0 and st.minute == 0 and st.second == 0
        if not is_sentinel_midnight and st < now_local() and time_confirmed:
            msg = "Heure prévue dans le passé."
            raise ValueError(msg)
        return st

    @validates("customer_name")
    def validate_customer_name(self, _key, name):
        if not name or len(name.strip()) == 0:
            msg = "Le nom du client ne peut pas être vide"
            raise ValueError(msg)
        if len(name) > CUSTOMER_NAME_MAX_LENGTH:
            msg = (
                f"Le nom du client ne peut pas dépasser "
                f"{CUSTOMER_NAME_MAX_LENGTH} caractères"
            )
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
                msg = (
                    f"Statut invalide : {status}. "
                    f"Doit être l'un de {list(BookingStatus.__members__.keys())}"
                )
                raise ValueError(msg) from None
        if not isinstance(status, BookingStatus):
            msg = f"Statut invalide : {status}. Doit être un BookingStatus valide."
            raise ValueError(msg)

        # ✅ Validation : status=ASSIGNED implique driver_id IS NOT NULL
        if status == BookingStatus.ASSIGNED:
            driver_id = getattr(self, "driver_id", None)
            if driver_id is None:
                msg = (
                    "status=ASSIGNED nécessite un driver_id. "
                    "Un booking assigné doit avoir un chauffeur."
                )
                raise ValueError(msg)

        return status

    @validates("driver_id")
    def validate_driver_id(self, _key, value):
        if value is not None and (not isinstance(value, int) or value < VALUE_ZERO):
            msg = "driver_id doit être un entier positif ou null"
            raise ValueError(msg)

        # ✅ Validation : si driver_id est None et status=ASSIGNED, c'est invalide
        # (vérification après validation du type)
        if value is None:
            current_status = getattr(self, "status", None)
            if current_status == BookingStatus.ASSIGNED:
                msg = (
                    "driver_id ne peut pas être NULL si status=ASSIGNED. "
                    "Un booking assigné doit avoir un chauffeur."
                )
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

    # ✅ P1-2: State machine courses avec validation
    def validate_status_transition(
        self, new_status: BookingStatus
    ) -> tuple[bool, str | None]:
        """Valide si une transition de statut est autorisée selon la state machine.

        Returns:
            (is_valid, error_message): Tuple indiquant si la transition est valide
                et un message d'erreur si elle ne l'est pas.
        """
        current_status = self.status

        # Définir les transitions autorisées
        ALLOWED_TRANSITIONS: dict[BookingStatus, list[BookingStatus]] = {
            BookingStatus.AWAITING_CLIENT_PAYMENT: [
                BookingStatus.PENDING,
                BookingStatus.CANCELED,
            ],
            BookingStatus.PENDING: [
                BookingStatus.ACCEPTED,
                BookingStatus.ASSIGNED,
                BookingStatus.CANCELED,
            ],
            BookingStatus.ACCEPTED: [
                BookingStatus.ASSIGNED,
                BookingStatus.CANCELED,
            ],
            BookingStatus.ASSIGNED: [
                BookingStatus.EN_ROUTE,
                BookingStatus.CANCELED,
            ],
            BookingStatus.EN_ROUTE: [
                BookingStatus.IN_PROGRESS,
                BookingStatus.CANCELED,
            ],
            BookingStatus.IN_PROGRESS: [
                BookingStatus.COMPLETED,
                BookingStatus.RETURN_COMPLETED,
                BookingStatus.CANCELED,
            ],
            BookingStatus.COMPLETED: [
                # COMPLETED est un état terminal, pas de transition autorisée
            ],
            BookingStatus.RETURN_COMPLETED: [
                # RETURN_COMPLETED est un état terminal, pas de transition autorisée
            ],
            BookingStatus.CANCELED: [
                # CANCELED est un état terminal, pas de transition autorisée
            ],
        }

        # Vérifier si la transition est autorisée
        allowed_next_statuses = ALLOWED_TRANSITIONS.get(current_status, [])
        if new_status not in allowed_next_statuses:
            allowed_values = [s.value for s in allowed_next_statuses]
            return (
                False,
                (
                    f"Transition invalide : {current_status.value} → {new_status.value}. "
                    f"Transitions autorisées depuis {current_status.value} : {allowed_values}"
                ),
            )

        # Validations supplémentaires selon le nouveau statut
        if new_status == BookingStatus.ASSIGNED and not self.driver_id:
            return (
                False,
                (
                    "Impossible d'assigner une course sans driver_id. "
                    "Un chauffeur doit être assigné avant de passer au statut ASSIGNED."
                ),
            )

        if new_status == BookingStatus.EN_ROUTE and not self.driver_id:
            return (
                False,
                "Impossible de passer en EN_ROUTE sans driver_id.",
            )

        if new_status == BookingStatus.IN_PROGRESS and not self.driver_id:
            return (
                False,
                "Impossible de passer en IN_PROGRESS sans driver_id.",
            )

        if (
            new_status in (BookingStatus.COMPLETED, BookingStatus.RETURN_COMPLETED)
            and not self.driver_id
        ):
            return (
                False,
                "Impossible de compléter une course sans driver_id.",
            )

        return (True, None)

    def update_status(self, new_status: BookingStatus) -> None:
        """Met à jour le statut avec validation de transition.

        Raises:
            ValueError: Si la transition n'est pas autorisée ou si le statut est invalide.
        """
        # ✅ P1-2: Valider la transition avant de l'appliquer
        is_valid, error_message = self.validate_status_transition(new_status)
        if not is_valid:
            raise ValueError(error_message or "Transition de statut invalide")

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
        status_val = self.status
        return (status_val in (BookingStatus.PENDING, BookingStatus.ACCEPTED)) and bool(
            st and st > now_local()
        )

    def assign_driver(self, driver_id: int):
        """Pose driver_id + ASSIGNED sur le Booking uniquement.

        ARRIVED-SOT-1B : ne crée **pas** d'Assignment. Les callers produit
        doivent utiliser `AssignDriverToReservationUseCase` (ou au minimum
        `ensure_booking_assignment` juste après) pour respecter l'invariant
        ``driver_id + statut actif ⇒ Assignment``.
        """
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
            msg = (
                "billed_to_company_id est obligatoire "
                "si billed_to_type n'est pas 'patient'"
            )
            raise ValueError(msg)


# Hooks ORM
event.listen(Booking, "before_insert", Booking._enforce_billing_exclusive)
event.listen(Booking, "before_update", Booking._enforce_billing_exclusive)
