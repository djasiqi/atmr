# models/booking.py

# Constantes pour éviter les valeurs magiques
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from .enums import BillingReviewStatus, BillingSource, BookingStatus

USER_ID_ZERO = 0
AMOUNT_ZERO = 0
VALUE_ZERO = 0
COMPANY_ID_ZERO = 0
CUSTOMER_NAME_MAX_LENGTH = 100
LOCATION_MAX_LENGTH = 200

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
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_name: Mapped[str] = mapped_column(String(100), nullable=False)
    pickup_location: Mapped[str] = mapped_column(String(200), nullable=False)
    dropoff_location: Mapped[str] = mapped_column(String(200), nullable=False)
    booking_type = Column(String(200), nullable=False, server_default="standard")

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
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

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
        # #region agent log
        try:
            import json

            with Path(r"c:\Users\jasiq\atmr\.cursor\debug.log").open(
                "a", encoding="utf-8"
            ) as f:
                status_raw = getattr(self, "status", None)
                f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "B",
                            "location": "booking.py:293",
                            "message": "Booking.serialize entry",
                            "data": {
                                "booking_id": self.id,
                                "status_type": str(type(status_raw)),
                                "status_value": str(status_raw),
                                "has_value_attr": hasattr(status_raw, "value")
                                if status_raw is not None
                                else False,
                            },
                            "timestamp": int(__import__("time").time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        status_val = self.status

        cli = self.client
        cli_user = getattr(cli, "user", None)

        return {
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
                "birth_date": (
                    cli_user.birth_date.strftime("%Y-%m-%d")
                    if cli_user and cli_user.birth_date
                    else None
                ),
                "contact_phone": getattr(cli, "contact_phone", None) if cli else None,
                "door_code": getattr(cli, "door_code", None) if cli else None,
                "floor": getattr(cli, "floor", None) if cli else None,
                "access_notes": getattr(cli, "access_notes", None) if cli else None,
            },
            # ✅ P1-4 Phase 1.2: Remplacer company (string) par company_id + company_name
            "company_id": self.company_id,
            "company_name": self.company.name if self.company else None,
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
                "first_name": self.driver.user.first_name if self.driver.user else None,
                "last_name": self.driver.user.last_name if self.driver.user else None,
                "full_name": (
                    (
                        f"{self.driver.user.first_name or ''} "
                        + f"{self.driver.user.last_name or ''}"
                    ).strip()
                    or self.driver.user.username
                    if self.driver.user
                    else None
                ),
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
            # ✅ P1-4 Phase 1.4: Standardiser timestamps en ISO 8601
            "created_at": (
                iso_utc_z(to_utc_from_db(created_dt))
                if isinstance(created_dt, datetime)
                else None
            ),
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
            # ✅ Traçabilité de la décision de facturation
            "billing_source": (
                self.billing_source.value if self.billing_source else None
            ),
            "billing_source_ref": self.billing_source_ref,
            # ✅ P1-4 Phase 1.1: patient_name reste pour compatibilité (utilise customer_name de la DB)
            "patient_name": _as_str(self.customer_name),
            # ✅ Informations du transfert actif (si existe)
            "active_transfer": self._get_active_transfer_info(),
        }

    def _is_transferred(self) -> bool:
        """Détermine si la course a été transférée (ACCEPTED ou COMPLETED).

        Après acceptation d'un transfert, company_id est mis à jour pour pointer
        vers l'entreprise receveuse. Pour garder la trace que la course a été
        transférée, on vérifie s'il existe un transfert ACCEPTED ou COMPLETED
        où owner_company_id != company_id actuel.
        """
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
        if amount < AMOUNT_MINIMUM:
            msg = f"Le montant minimum accepté est {AMOUNT_MINIMUM}"
            raise ValueError(msg)
        # Règles d'arrondi métier
        # - 0.5-0.6 → 0.5
        # - 0.75-0.8 → 0.8
        # - 39.98-40.0 → 40.0
        if (
            AMOUNT_MINIMUM <= amount < AMOUNT_ROUNDING_THRESHOLD_1
            or AMOUNT_ROUNDING_THRESHOLD_1 <= amount < AMOUNT_ROUNDING_THRESHOLD_2
        ):
            return AMOUNT_ROUNDING_TARGET_1
        if AMOUNT_ROUNDING_THRESHOLD_2 <= amount < AMOUNT_ROUNDING_THRESHOLD_3:
            return AMOUNT_ROUNDING_TARGET_2
        if AMOUNT_ROUNDING_THRESHOLD_4 <= amount < AMOUNT_ROUNDING_TARGET_3:
            return AMOUNT_ROUNDING_TARGET_3
        # Arrondi standard à 2 décimales pour les autres cas
        return round(amount, 2)

    @validates("scheduled_time")
    def validate_scheduled_time(self, _key, scheduled_time):
        # ✅ scheduled_time est obligatoire pour le dispatch
        # SAUF pour les courses retour (is_return=True) qui peuvent avoir scheduled_time=None
        # Le dispatch gère déjà ce cas (voir get_bookings_for_dispatch dans booking_repository)
        is_return = getattr(self, "is_return", False)
        if scheduled_time is None:
            if is_return:
                # ✅ Permettre scheduled_time=None pour les courses retour
                # L'heure pourra être définie plus tard via l'endpoint /schedule
                return None
            msg = "scheduled_time est obligatoire. Le dispatch nécessite cette valeur."
            raise ValueError(msg)
        st = parse_local_naive(scheduled_time)
        # ⚠️ IMPORTANT : Validation désactivée si time_confirmed=False
        # Cela permet d'importer des bookings historiques avec des dates passées.
        # Lorsque time_confirmed=False :
        #   - Les dates passées sont acceptées (pour import historique)
        #   - Le booking est exclu du dispatch automatique (voir get_bookings_for_dispatch)
        #   - Utile pour les retours avec heure à confirmer ou imports de données anciennes
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
