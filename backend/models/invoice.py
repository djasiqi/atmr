# models/invoice.py
# pyright: reportRedeclaration=false
# Le linter détecte un conflit entre Column(name="iban"/"qr_iban") et @hybrid_property iban/qr_iban,
# mais c'est un faux positif : Column avec name ne crée pas d'attribut Python.

"""Models Invoice et tous ses modèles liés (lignes, paiements, rappels, etc.).
Extrait depuis models.py (lignes ~1763-3258).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from typing_extensions import override

from ext import db
from security.crypto import get_encryption_service

from .base import _as_bool, _iso
from .enums import InvoiceBillingStrategy, InvoiceLineType, InvoiceStatus, PaymentMethod

logger = logging.getLogger(__name__)


def _ordered_unique_booking_ids_from_lines(
    lines: list[Any] | tuple[Any, ...],
) -> list[int]:
    """IDs courses pour registre mobile / ``list_view`` (plusieurs trajets, A/R fusionné).

    Une ligne fusionnée n’a qu’un ``reservation_id`` (segment principal) ; les autres IDs
    sont souvent dans ``line_meta.booking_ids`` ou ``round_trip_secondary_reservation_*``.
    """
    ordered: list[int] = []
    seen: set[int] = set()

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                add(item)
            return
        try:
            i = int(value)
        except (TypeError, ValueError):
            return
        if i in seen:
            return
        seen.add(i)
        ordered.append(i)

    for ln in lines:
        add(getattr(ln, "reservation_id", None))
        meta = getattr(ln, "line_meta", None)
        if isinstance(meta, dict):
            add(meta.get("booking_ids"))
            add(meta.get("reservation_ids"))
            add(meta.get("round_trip_secondary_reservation_ids"))
            add(meta.get("round_trip_secondary_reservation_id"))
            add(meta.get("round_trip_merge_partner_reservation_id"))
            add(meta.get("round_trip_merge_primary_reservation_id"))

    return ordered


class Invoice(db.Model):
    """Modèle principal pour les factures."""

    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("company.id"), nullable=False, index=True
    )
    client_id: Mapped[int] = mapped_column(
        ForeignKey("client.id"), nullable=False, index=True
    )

    # Facturation tierce (legacy) — préférer ``billing_party_id`` pour le nouveau code (V2+).
    bill_to_client_id: Mapped[int | None] = mapped_column(
        ForeignKey("client.id"), nullable=True, index=True
    )

    # Payeur unifié (canon cible) — quand renseigné, il prime sur ``bill_to_client_id`` / clinique héritée.
    billing_party_id: Mapped[int | None] = mapped_column(
        ForeignKey("billing_parties.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # ✅ Stratégie de facturation (S1/S2/...) + destinataire "company" (ex: clinique payeur en S2)
    billing_strategy: Mapped[InvoiceBillingStrategy] = mapped_column(
        SAEnum(
            InvoiceBillingStrategy,
            name="invoice_billing_strategy",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
        server_default=InvoiceBillingStrategy.S1_PATIENT.value,
    )
    # Clinique payeuse (S2, legacy) — V2 : dériver depuis BillingParty quand possible.
    billed_to_company_id: Mapped[int | None] = mapped_column(
        ForeignKey("company.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Période de facturation
    period_month: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-12
    period_year: Mapped[int] = mapped_column(Integer, nullable=False)

    # Numéro de facture unique par entreprise
    invoice_number: Mapped[str] = mapped_column(String(50), nullable=False)
    currency = Column(String(3), default="CHF", nullable=False)

    # Montants
    subtotal_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    late_fee_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    reminder_fee_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    vat_total_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    vat_breakdown = Column(JSONB, nullable=True)
    total_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    amount_paid: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    balance_due: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )

    # Dates clés
    issued_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    due_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    paid_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Statut
    status: Mapped[InvoiceStatus] = mapped_column(
        SAEnum(InvoiceStatus, name="invoice_status"),
        nullable=False,
        default=InvoiceStatus.DRAFT,
    )

    # Rappels
    reminder_level = Column(Integer, nullable=False, default=0)  # 0 = aucun, 1, 2, 3
    last_reminder_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Artifacts
    pdf_url: Mapped[str] = mapped_column(String(500), nullable=True)
    qr_reference: Mapped[str] = mapped_column(String(50), nullable=True)
    meta = Column(JSONB, nullable=True)

    # Relations
    # NOTE: invoices a 2 FK vers Company (company_id + billed_to_company_id),
    # donc il faut expliciter la FK pour éviter AmbiguousForeignKeysError.
    company = relationship("Company", foreign_keys=[company_id], backref="invoices")
    client = relationship(
        "Client", foreign_keys=[client_id], backref="service_invoices"
    )
    bill_to_client = relationship(
        "Client", foreign_keys=[bill_to_client_id], backref="billing_invoices"
    )
    billing_party = relationship("BillingParty", foreign_keys=[billing_party_id])
    billed_to_company = relationship("Company", foreign_keys=[billed_to_company_id])
    lines = relationship(
        "InvoiceLine", back_populates="invoice", cascade="all, delete-orphan"
    )
    payments = relationship(
        "InvoicePayment", back_populates="invoice", cascade="all, delete-orphan"
    )
    reminders = relationship(
        "InvoiceReminder", back_populates="invoice", cascade="all, delete-orphan"
    )

    # Index et contraintes
    __table_args__ = (
        UniqueConstraint(
            "company_id", "invoice_number", name="uq_company_invoice_number"
        ),
        Index("ix_invoice_company_period", "company_id", "period_year", "period_month"),
        Index("ix_invoice_status", "company_id", "status"),
        Index("ix_invoice_due_date", "due_date"),
        CheckConstraint("total_amount >= 0", name="chk_invoice_amount_positive"),
        CheckConstraint("balance_due >= 0", name="chk_invoice_balance_nonneg"),
        CheckConstraint("amount_paid >= 0", name="chk_invoice_paid_nonneg"),
    )

    @validates("bill_to_client_id", "billing_party_id", "billed_to_company_id")
    def _v_payer_fields(self, key, value):
        """Garde-fous P1: éviter les combinaisons incohérentes entre legacy et BillingParty.

        Note: `@validates` est appelé AVANT l'affectation, donc on valide avec des
        valeurs "prospectives" (value + autres champs existants).
        """
        bill_to_client_id = (
            value if key == "bill_to_client_id" else self.bill_to_client_id
        )
        billing_party_id = value if key == "billing_party_id" else self.billing_party_id
        if key == "billed_to_company_id":
            _ = value  # valeur acceptée (validation métier ailleurs)

        # Cas clinique: si billed_to_company_id est défini mais billing_party_id est NULL,
        # on autorise l'état en V1 (P1.4) à condition que les courses soient mises en NEEDS_REVIEW
        # et que l'UI permette de corriger le mapping. L'interdiction stricte pourra être réactivée
        # plus tard quand la configuration sera complète en prod.

        # Cas legacy: on autorise bill_to_client_id + billing_party_id uniquement si
        # le BillingParty correspond au lien legacy (external_ref=legacy_client:<id>).
        if bill_to_client_id is not None and billing_party_id is not None:
            try:
                from models.billing_party import BillingParty

                bp = BillingParty.query.filter_by(id=int(billing_party_id)).first()
                expected = f"legacy_client:{int(bill_to_client_id)}"
                if not bp or (bp.external_ref or "") != expected:
                    raise ValueError(
                        "bill_to_client_id et billing_party_id sont incohérents (BillingParty ne correspond pas au payeur legacy)."
                    )
            except ValueError:
                raise
            except Exception as e:
                # Ne pas casser le flush sur un problème de lecture ponctuel;
                # on log et on laisse passer (fallback: validations use case).
                logger.warning(
                    "[Invoice] Impossible de valider la cohérence du payeur: %s",
                    str(e),
                )

        return value

    @override
    def __repr__(self):
        return f"<Invoice {self.invoice_number} - {self.status.value}>"

    @property
    def is_overdue(self):
        """Vérifie si la facture est en retard."""
        return (
            self.balance_due > 0
            and self.due_date is not None
            and datetime.now(UTC) > self.due_date
        )

    def update_balance(self):
        """Met à jour le solde et le statut basé sur les paiements."""
        self.amount_paid = sum(payment.amount for payment in self.payments)
        self.balance_due = self.total_amount - self.amount_paid

        # Mise à jour du statut
        if self.balance_due <= 0:
            self.status = InvoiceStatus.PAID
            self.paid_at = datetime.now(UTC)
        elif self.amount_paid > 0:
            self.status = InvoiceStatus.PARTIALLY_PAID
        elif self.is_overdue:
            self.status = InvoiceStatus.OVERDUE

    def mark_as_paid(self):
        """Marque la facture comme payée."""
        self.status = InvoiceStatus.PAID
        self.paid_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def mark_as_sent(self):
        """Marque la facture comme envoyée."""
        self.status = InvoiceStatus.SENT
        self.sent_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def cancel(self):
        """Annule la facture."""
        self.status = InvoiceStatus.CANCELLED
        self.cancelled_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def _serialize_client(self) -> dict[str, Any] | None:
        """Sérialise le client, avec override patient pour institution + S1_PATIENT."""
        client = self.client
        if not client and self.client_id:
            from sqlalchemy.orm import joinedload as _jl

            from models.client import Client as ClientModel

            client = (
                ClientModel.query.options(_jl(ClientModel.user))
                .filter_by(id=self.client_id)
                .first()
            )
        if not client:
            return None
        first_name = (
            getattr(client.user, "first_name", "")
            if hasattr(client, "user") and client.user
            else ""
        )
        last_name = (
            getattr(client.user, "last_name", "")
            if hasattr(client, "user") and client.user
            else ""
        )
        username = (
            getattr(client.user, "username", "")
            if hasattr(client, "user") and client.user
            else ""
        )
        is_institution = _as_bool(client.is_institution) if client else False
        institution_name = client.institution_name if client else None
        patient_display_name = None

        # Pour les clients institution avec facturation patient (S1_PATIENT),
        # retrouver le nom du patient depuis le premier booking de la facture
        if (
            is_institution
            and self.billing_strategy == InvoiceBillingStrategy.S1_PATIENT
        ):
            try:
                if hasattr(self, "lines") and self.lines:
                    from models.booking import Booking

                    for line in self.lines:
                        if line.reservation_id:
                            bk = Booking.query.get(line.reservation_id)
                            if bk and bk.customer_name:
                                patient_display_name = bk.customer_name
                                break
            except Exception:
                pass

        return {
            "id": client.id,
            "first_name": first_name,
            "last_name": last_name,
            "username": username,
            "is_institution": is_institution,
            "institution_name": institution_name,
            "patient_display_name": patient_display_name,
        }

    def to_dict(self, *, include_reminder_rows: bool = True, list_view: bool = False, company_id: int | None = None):
        """Sérialise la facture en dictionnaire.

        Args:
            include_reminder_rows: Si False, ne sérialise pas les lignes ``reminders``
                (``reminder_level`` / ``last_reminder_at`` restent sur la facture).
            list_view: Réponse liste / tableau : pas de lignes, paiements, méta lourde,
                ni ventilation TVA détaillée. Les rappels sont inclus en version allégée
                (accès PDF depuis le registre des factures).
        """
        resolved_company_id = company_id or self.company_id

        reminder_payload: list[dict[str, Any]]
        if list_view:
            reminder_payload = (
                [
                    reminder.to_dict_list_view(company_id=resolved_company_id)
                    for reminder in self.reminders
                ]
                if hasattr(self, "reminders") and self.reminders
                else []
            )
        elif include_reminder_rows:
            reminder_payload = (
                [
                    reminder.to_dict(company_id=resolved_company_id)
                    for reminder in self.reminders
                ]
                if hasattr(self, "reminders")
                else []
            )
        else:
            reminder_payload = []

        lines_out: list[dict[str, Any]]
        if list_view:
            lines_out = []
        elif hasattr(self, "lines"):
            lines_out = [line.to_dict() for line in self.lines]
            enrich_invoice_line_payloads_for_api(list(self.lines), lines_out)
        else:
            lines_out = []

        payments_out: list[dict[str, Any]]
        if list_view:
            payments_out = []
        elif hasattr(self, "payments"):
            payments_out = [payment.to_dict() for payment in self.payments]
        else:
            payments_out = []

        booking_ids_ordered: list[int] = []
        if hasattr(self, "lines") and self.lines:
            booking_ids_ordered = _ordered_unique_booking_ids_from_lines(
                list(self.lines)
            )

        primary_booking_id: int | None = (
            booking_ids_ordered[0] if booking_ids_ordered else None
        )

        from shared.invoice_due_dates import resolve_effective_due_date

        effective_due = resolve_effective_due_date(
            self, company_id=resolved_company_id
        )

        return {
            "id": self.id,
            "company_id": self.company_id,
            "booking_id": primary_booking_id,
            "booking_ids": booking_ids_ordered,
            "client_id": self.client_id,
            "bill_to_client_id": self.bill_to_client_id,
            "billing_party_id": self.billing_party_id,
            "billing_strategy": (
                self.billing_strategy.value
                if hasattr(self.billing_strategy, "value")
                else str(self.billing_strategy)
            ),
            "billed_to_company_id": self.billed_to_company_id,
            "period_month": self.period_month,
            "period_year": self.period_year,
            "invoice_number": self.invoice_number,
            "currency": self.currency,
            "subtotal_amount": float(self.subtotal_amount),
            "late_fee_amount": float(self.late_fee_amount),
            "reminder_fee_amount": float(self.reminder_fee_amount),
            "vat_total_amount": float(self.vat_total_amount),
            "vat_breakdown": None if list_view else self.vat_breakdown,
            "total_amount": float(self.total_amount),
            "amount_paid": float(self.amount_paid),
            "balance_due": float(self.balance_due),
            "issued_at": _iso(self.issued_at),
            "due_date": _iso(self.due_date),
            "effective_due_date": _iso(effective_due),
            "sent_at": _iso(self.sent_at),
            "paid_at": _iso(self.paid_at),
            "cancelled_at": _iso(self.cancelled_at),
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "status": (
                self.status.value if hasattr(self.status, "value") else str(self.status)
            ),
            "reminder_level": self.reminder_level,
            "last_reminder_at": _iso(self.last_reminder_at),
            "pdf_url": self.pdf_url,
            "qr_reference": self.qr_reference,
            "meta": None if list_view else self.meta,
            "client": self._serialize_client(),
            "bill_to_client": {
                "id": self.bill_to_client.id,
                "first_name": getattr(self.bill_to_client.user, "first_name", "")
                if hasattr(self.bill_to_client, "user") and self.bill_to_client.user
                else "",
                "last_name": getattr(self.bill_to_client.user, "last_name", "")
                if hasattr(self.bill_to_client, "user") and self.bill_to_client.user
                else "",
                "username": getattr(self.bill_to_client.user, "username", "")
                if hasattr(self.bill_to_client, "user") and self.bill_to_client.user
                else "",
                "is_institution": _as_bool(self.bill_to_client.is_institution),
                "institution_name": self.bill_to_client.institution_name,
                "billing_address": self.bill_to_client.billing_address,
                "contact_email": self.bill_to_client.contact_email,
            }
            if self.bill_to_client
            else None,
            "billing_party": {
                "id": self.billing_party.id,
                "display_name": self.billing_party.display_name,
                "contact_email": self.billing_party.contact_email,
                "type": (
                    self.billing_party.type.value
                    if hasattr(self.billing_party.type, "value")
                    else str(self.billing_party.type)
                ),
            }
            if self.billing_party
            else None,
            "billed_to_company": {
                "id": self.billed_to_company.id,
                "name": self.billed_to_company.name,
                "billing_email": self.billed_to_company.billing_email,
                "contact_email": self.billed_to_company.contact_email,
            }
            if self.billed_to_company
            else None,
            "payer_resolution": {
                "primary": (
                    "billing_party" if self.billing_party_id is not None else "legacy"
                ),
                "billing_party_id": self.billing_party_id,
                "legacy_bill_to_client_id": self.bill_to_client_id,
                "legacy_billed_to_company_id": self.billed_to_company_id,
            },
            "lines": lines_out,
            "payments": payments_out,
            "reminders": reminder_payload,
        }


class InvoiceLine(db.Model):
    """Lignes de facture."""

    __tablename__ = "invoice_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), nullable=False)

    type: Mapped[InvoiceLineType] = mapped_column(
        SAEnum(
            InvoiceLineType,
            name="invoice_line_type",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    qty: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False, default=1)
    unit_price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    line_total: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    vat_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    vat_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    total_with_vat: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    adjustment_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Optionnel : tracer la source (réservation)
    reservation_id = Column(
        Integer,
        ForeignKey("booking.id", name="fk_invoice_line_reservation"),
        nullable=True,
    )
    # Prestations CUSTOM : mode facturation (temps/quantité, unité) pour PDF / affichage
    line_meta = Column(JSONB, nullable=True)

    # Relations
    invoice = relationship("Invoice", back_populates="lines")
    reservation = relationship(
        "Booking",
        foreign_keys=[reservation_id],
        backref="invoice_lines_for_reservation",
    )

    @override
    def __repr__(self):
        return f"<InvoiceLine {self.description} - {self.line_total} CHF>"

    def to_dict(self):
        """Sérialise la ligne de facture en dictionnaire."""
        desc_out = self.description
        if self.type in (InvoiceLineType.RIDE, InvoiceLineType.MATERIAL_DELIVERY):
            from shared.utils.transport_description_normalize import (
                normalize_transport_line_description,
            )

            if self.type == InvoiceLineType.RIDE:
                desc_out = normalize_transport_line_description(desc_out, kind="ride")
            else:
                desc_out = normalize_transport_line_description(
                    desc_out, kind="material_delivery"
                )
        d = {
            "id": self.id,
            "invoice_id": self.invoice_id,
            "type": self.type.value,
            "description": desc_out,
            "qty": float(self.qty),
            "unit_price": float(self.unit_price),
            "line_total": float(self.line_total),
            "vat_rate": float(self.vat_rate) if self.vat_rate is not None else None,
            "vat_amount": float(self.vat_amount),
            "total_with_vat": float(self.total_with_vat),
            "adjustment_note": self.adjustment_note,
            "reservation_id": self.reservation_id,
        }
        line_meta_val = getattr(self, "line_meta", None)
        if line_meta_val is not None:
            d["line_meta"] = line_meta_val
        return d


def _enrich_invoice_line_payloads_booking_dates(
    invoice_lines: list[Any],
    line_dicts: list[dict[str, Any]],
) -> None:
    """Ajoute ``line_meta.service_date`` depuis la réservation (trajets sans méta persistée)."""
    from models.booking import Booking

    targets: list[tuple[int, dict[str, Any]]] = []
    for ln, d in zip(invoice_lines, line_dicts, strict=True):
        if (
            ln.type
            not in (
                InvoiceLineType.RIDE,
                InvoiceLineType.MATERIAL_DELIVERY,
            )
            or not ln.reservation_id
        ):
            continue
        meta = d.get("line_meta")
        if isinstance(meta, dict) and meta.get("service_date"):
            continue
        targets.append((ln.reservation_id, d))

    if not targets:
        return

    rids = list({rid for rid, _ in targets})
    bookings = Booking.query.filter(Booking.id.in_(rids)).all()
    by_id = {b.id: b for b in bookings}

    for rid, d in targets:
        bk = by_id.get(rid)
        if bk is None:
            continue
        st = getattr(bk, "scheduled_time", None)
        if st is None:
            continue
        try:
            ds = st.date().isoformat() if hasattr(st, "date") else None
        except Exception:
            ds = None
        if not ds:
            continue
        meta = dict(d.get("line_meta") or {})
        if meta.get("service_date"):
            continue
        meta["service_date"] = ds
        d["line_meta"] = meta


def _enrich_invoice_line_payloads_round_trip_merge(
    invoice_lines: list[Any],
    line_dicts: list[dict[str, Any]],
) -> None:
    """Marque les paires aller-retour (RIDE) pour l'aperçu HTML : une ligne visible [A/R], l'autre masquée.

    Réutilise ``find_round_trip_merge_booking_pairs`` (même famille que PDF / aperçu période).
    Les montants affichés sont agrégés côté client à partir des deux lignes facture.
    """
    from application.invoices.round_trip_booking_pairs import (
        find_round_trip_merge_booking_pairs,
    )
    from models.booking import Booking

    _MIN_RIDE_ROWS_ROUND_TRIP = 2
    ride_rows: list[tuple[Any, dict[str, Any]]] = []
    for ln, d in zip(invoice_lines, line_dicts, strict=True):
        if ln.type != InvoiceLineType.RIDE or not ln.reservation_id:
            continue
        ride_rows.append((ln, d))

    if len(ride_rows) < _MIN_RIDE_ROWS_ROUND_TRIP:
        return

    rids = list({int(ln.reservation_id) for ln, _ in ride_rows})
    bookings = Booking.query.filter(Booking.id.in_(rids)).all()
    if len(bookings) < _MIN_RIDE_ROWS_ROUND_TRIP:
        return

    by_rid: dict[int, dict[str, Any]] = {}
    for ln, d in ride_rows:
        by_rid[int(ln.reservation_id)] = d

    def amount_ht_fn(b: Any) -> Decimal:
        dd = by_rid.get(int(b.id))
        if not dd:
            return Decimal("0")
        try:
            return Decimal(str(dd.get("line_total") or 0))
        except Exception:
            return Decimal("0")

    pairs = find_round_trip_merge_booking_pairs(bookings, amount_ht_fn=amount_ht_fn)

    for pri_id, sec_id in pairs:
        d_pri = by_rid.get(int(pri_id))
        d_sec = by_rid.get(int(sec_id))
        if not d_pri or not d_sec:
            continue
        meta_pri = dict(d_pri.get("line_meta") or {})
        meta_sec = dict(d_sec.get("line_meta") or {})
        meta_pri["is_round_trip_leg"] = True
        meta_pri["transport_type"] = "A/R"
        meta_pri["round_trip_merge_partner_reservation_id"] = int(sec_id)
        meta_sec["preview_hide_merged_round_trip"] = True
        meta_sec["round_trip_merge_primary_reservation_id"] = int(pri_id)
        d_pri["line_meta"] = meta_pri
        d_sec["line_meta"] = meta_sec


def _enrich_invoice_line_payloads_single_round_trip(
    invoice_lines: list[Any],
    line_dicts: list[dict[str, Any]],
    *,
    bookings_by_id: dict[int, Any] | None = None,
) -> None:
    """Marque les lignes A/R facturées en une seule entrée (réservation ``is_round_trip``)."""
    rids = [
        int(ln.reservation_id)
        for ln in invoice_lines
        if ln.type == InvoiceLineType.RIDE and ln.reservation_id
    ]
    if not rids:
        return

    if bookings_by_id is None:
        from models.booking import Booking

        bookings = Booking.query.filter(Booking.id.in_(list(set(rids)))).all()
        bookings_by_id = {int(b.id): b for b in bookings}

    invoice_rid_set = set(rids)

    for ln, d in zip(invoice_lines, line_dicts, strict=True):
        if ln.type != InvoiceLineType.RIDE or not ln.reservation_id:
            continue
        meta = dict(d.get("line_meta") or {})
        if meta.get("round_trip_merge_partner_reservation_id"):
            continue
        if meta.get("preview_hide_merged_round_trip"):
            continue
        if meta.get("billing_unit") == "round_trip":
            if meta.get("transport_type") != "A/R":
                meta["transport_type"] = "A/R"
                d["line_meta"] = meta
            continue

        b = bookings_by_id.get(int(ln.reservation_id))
        if not b:
            continue
        if getattr(b, "is_return", False):
            continue

        is_rt = bool(getattr(b, "is_round_trip", False))
        if not is_rt:
            bid = int(b.id)
            for other in bookings_by_id.values():
                opid = getattr(other, "parent_booking_id", None)
                if opid is not None and int(opid) == bid and int(other.id) in invoice_rid_set:
                    is_rt = True
                    break

        if not is_rt:
            continue

        meta["billing_unit"] = "round_trip"
        meta["transport_type"] = "A/R"
        d["line_meta"] = meta


def enrich_invoice_line_payloads_for_api(
    invoice_lines: list[Any],
    line_dicts: list[dict[str, Any]],
    *,
    bookings_by_id: dict[int, Any] | None = None,
) -> None:
    """Enrichit les payloads lignes facture (dates, paires A/R, A/R mono-ligne) pour l'API."""
    _enrich_invoice_line_payloads_booking_dates(invoice_lines, line_dicts)
    _enrich_invoice_line_payloads_round_trip_merge(invoice_lines, line_dicts)
    _enrich_invoice_line_payloads_single_round_trip(
        invoice_lines,
        line_dicts,
        bookings_by_id=bookings_by_id,
    )


class InvoicePayment(db.Model):
    """Paiements des factures."""

    __tablename__ = "invoice_payments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), nullable=False)

    amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    paid_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    method: Mapped[PaymentMethod] = mapped_column(
        SAEnum(
            PaymentMethod,
            name="payment_method",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
    )
    reference: Mapped[str] = mapped_column(String(100), nullable=True)

    # ✅ NOUVEAU : Lien vers le rappel (pour ventilation automatique)
    reminder_id: Mapped[int | None] = mapped_column(
        ForeignKey("invoice_reminders.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Relations
    invoice = relationship("Invoice", back_populates="payments")
    reminder = relationship("InvoiceReminder", foreign_keys=[reminder_id])

    @override
    def __repr__(self):
        return f"<InvoicePayment {self.amount} CHF - {self.method.value}>"

    def to_dict(self):
        """Sérialise le paiement en dictionnaire."""
        return {
            "id": self.id,
            "invoice_id": self.invoice_id,
            "amount": float(self.amount),
            "paid_at": _iso(self.paid_at),
            "method": self.method.value,
            "reference": self.reference,
            "reminder_id": self.reminder_id,
        }


class InvoiceReminder(db.Model):
    """Rappels de facture consolidés.

    Modèle pour les rappels consolidés : le client voit UN document et paie UN montant.
    En interne, on garde la traçabilité fine (montant principal + frais de rappel).
    """

    __tablename__ = "invoice_reminders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), nullable=False)

    level: Mapped[int] = mapped_column(Integer, nullable=False)  # 1, 2, 3
    added_fee: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )

    # ✅ NOUVEAU : Montants consolidés pour le rappel
    principal_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )  # Montant de la facture initiale (sans frais)
    reminder_fee_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )  # Frais de rappel
    total_due: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )  # Total à payer (principal + frais)

    # ✅ NOUVEAU : QR-bill pour le rappel consolidé
    qr_reference: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # ✅ NOUVEAU : Statut du rappel (OPEN/PAID)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="OPEN", server_default="OPEN"
    )  # OPEN, PAID

    generated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    due_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    paid_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    pdf_url: Mapped[str] = mapped_column(String(500), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relations
    invoice = relationship("Invoice", back_populates="reminders")

    @override
    def __repr__(self):
        return f"<InvoiceReminder Level {self.level} - {self.added_fee} CHF>"

    def to_dict_list_view(
        self, *, company_id: int | None = None
    ) -> dict[str, Any]:
        """Sérialisation allégée pour la liste des factures (accès PDF rappel)."""
        from shared.invoice_due_dates import resolve_reminder_due_date

        due = resolve_reminder_due_date(self, company_id=company_id)
        return {
            "id": self.id,
            "level": self.level,
            "pdf_url": self.pdf_url,
            "status": self.status,
            "generated_at": _iso(self.generated_at),
            "due_date": _iso(due),
            "reminder_fee_amount": float(self.reminder_fee_amount),
        }

    def to_dict(self, *, company_id: int | None = None):
        """Sérialise le rappel en dictionnaire."""
        from shared.invoice_due_dates import resolve_reminder_due_date

        due = resolve_reminder_due_date(self, company_id=company_id)
        return {
            "id": self.id,
            "invoice_id": self.invoice_id,
            "level": self.level,
            "added_fee": float(self.added_fee),
            "principal_amount": float(self.principal_amount),
            "reminder_fee_amount": float(self.reminder_fee_amount),
            "total_due": float(self.total_due),
            "qr_reference": self.qr_reference,
            "status": self.status,
            "generated_at": _iso(self.generated_at),
            "due_date": _iso(due),
            "sent_at": _iso(self.sent_at),
            "paid_at": _iso(self.paid_at),
            "pdf_url": self.pdf_url,
            "note": self.note,
        }


class CompanyBillingSettings(db.Model):
    """Paramètres de facturation par entreprise."""

    __tablename__ = "company_billing_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("company.id"), nullable=False, unique=True
    )

    # Délais et frais
    payment_terms_days: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=10
    )
    overdue_fee: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 2), nullable=True, default=15
    )
    reminder1_fee: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 2), nullable=True, default=0
    )
    reminder2_fee: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 2), nullable=True, default=40
    )
    reminder3_fee: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 2), nullable=True, default=0
    )
    # ✅ Prix fixe livraison matériel (CHF) - configuré par entreprise
    material_delivery_price_fixed: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 2), nullable=True, default=None
    )
    cancellation_policy = Column(
        JSONB,
        nullable=True,
        server_default=None,
        comment="Policy d'annulation parametrable: tiers, min/max, overrides",
    )

    vat_applicable = Column(Boolean, nullable=False, default=True)
    vat_rate: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2), nullable=True, default=Decimal("7.7")
    )
    vat_label: Mapped[str | None] = mapped_column(String(50), nullable=True)
    vat_number: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Planning des rappels (en jours)
    reminder_schedule_days = Column(
        JSON,
        nullable=False,
        default={
            "1": 10,  # 1er rappel 10j après échéance
            "2": 5,  # 2e rappel 5j après le 1er
            "3": 5,  # 3e rappel 5j après le 2e
        },
    )

    # Configuration
    auto_reminders_enabled = Column(Boolean, nullable=False, default=True)
    email_sender: Mapped[str] = mapped_column(String(200), nullable=True)

    # Configuration SMTP par entreprise (multi-tenant)
    # Si non configuré, utilise la config globale du .env
    smtp_server: Mapped[str | None] = mapped_column(String(200), nullable=True)
    smtp_port: Mapped[int | None] = mapped_column(Integer, nullable=True, default=587)
    smtp_use_tls: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    smtp_use_ssl: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    smtp_username: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # ✅ Mot de passe SMTP chiffré (même système que IBAN)
    _smtp_password_raw = Column(
        String(200), nullable=True, name="smtp_password"
    )  # Stocke le texte chiffré

    smtp_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )  # Active/désactive le SMTP custom

    # Configuration Brevo (email transactionnel)
    # Remplace progressivement la config SMTP complexe
    from_name: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )  # Nom d'expéditeur (ex: "Lirie Transports")
    domain_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )  # Domaine validé dans Brevo (SPF/DKIM configurés)
    domain_dns_records = Column(
        JSON, nullable=True
    )  # {"spf": "v=spf1...", "dkim": "k=rsa..."}

    # Format de numérotation
    invoice_number_format = Column(
        String(50), nullable=False, default="{PREFIX}-{YYYY}-{MM}-{SEQ4}"
    )
    invoice_prefix = Column(String(10), nullable=False, default="EM")

    # Informations bancaires
    # ✅ S2: IBAN et QR-IBAN chiffrés en base de données (conformité RGPD)
    # Les champs _iban_raw et _qr_iban_raw stockent le texte chiffré (base64)
    # Utilisation de Column avec name pour garder les noms de colonnes en base
    _iban_raw = Column(
        String(200), nullable=True, name="iban"
    )  # Augmenté à 200 pour stocker le texte chiffré
    _qr_iban_raw = Column(
        String(200), nullable=True, name="qr_iban"
    )  # Augmenté à 200 pour stocker le texte chiffré
    esr_ref_base: Mapped[str] = mapped_column(String(50), nullable=True)

    # Templates de messages
    invoice_message_template: Mapped[str | None] = mapped_column(Text, nullable=True)
    reminder1_template: Mapped[str | None] = mapped_column(Text, nullable=True)
    reminder2_template: Mapped[str | None] = mapped_column(Text, nullable=True)
    reminder3_template: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Signature email personnalisée (injectée dans tous les emails de facturation/rappel)
    email_signature_mode: Mapped[str] = mapped_column(
        String(10), nullable=False, default="form", server_default="form"
    )  # "text", "form" ou "html"
    email_signature_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Mode "form": champs normalisés (génération auto du HTML)
    signature_name: Mapped[str | None] = mapped_column(
        String(200), nullable=True
    )  # Nom complet
    signature_title: Mapped[str | None] = mapped_column(
        String(200), nullable=True
    )  # Titre (ex: "Associé gérant")
    signature_company: Mapped[str | None] = mapped_column(
        String(200), nullable=True
    )  # Société
    signature_phone_main: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # Téléphone principal
    signature_phone_mobile: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # Téléphone mobile
    signature_email: Mapped[str | None] = mapped_column(
        String(200), nullable=True
    )  # Email
    signature_website: Mapped[str | None] = mapped_column(
        String(200), nullable=True
    )  # Site web
    signature_address_line: Mapped[str | None] = mapped_column(
        String(200), nullable=True
    )  # Ligne adresse
    signature_zip: Mapped[str | None] = mapped_column(
        String(10), nullable=True
    )  # Code postal
    signature_city: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )  # Ville
    # Note: signature_logo_url supprimé - on utilise maintenant company.logo_url automatiquement
    # Mode "html": template personnalisé
    email_signature_html_template: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    # Pied de page légal
    legal_footer: Mapped[str | None] = mapped_column(Text, nullable=True)
    pdf_template_variant = Column(String(20), nullable=False, default="default")

    # Relations
    company = relationship("Company", backref="billing_settings")

    @override
    def __repr__(self):
        return f"<CompanyBillingSettings {self.company_id}>"

    def to_dict(self):
        """Convertit l'objet en dictionnaire."""
        return {
            "id": self.id,
            "company_id": self.company_id,
            "payment_terms_days": self.payment_terms_days,
            "overdue_fee": float(self.overdue_fee)
            if self.overdue_fee is not None
            else None,
            "reminder1_fee": float(self.reminder1_fee)
            if self.reminder1_fee is not None
            else None,
            "reminder2_fee": float(self.reminder2_fee)
            if self.reminder2_fee is not None
            else None,
            "reminder3_fee": float(self.reminder3_fee)
            if self.reminder3_fee is not None
            else None,
            "material_delivery_price_fixed": float(self.material_delivery_price_fixed)
            if self.material_delivery_price_fixed is not None
            else None,
            "reminder_schedule_days": self.reminder_schedule_days,
            "auto_reminders_enabled": self.auto_reminders_enabled,
            "email_sender": self.email_sender,
            "invoice_number_format": self.invoice_number_format,
            "invoice_prefix": self.invoice_prefix,
            "iban": self.iban,
            "qr_iban": self.qr_iban,
            "esr_ref_base": self.esr_ref_base,
            "invoice_message_template": self.invoice_message_template,
            "reminder1_template": self.reminder1_template,
            "reminder2_template": self.reminder2_template,
            "reminder3_template": self.reminder3_template,
            "email_signature_mode": self.email_signature_mode,
            "email_signature_text": self.email_signature_text,
            "signature_name": self.signature_name,
            "signature_title": self.signature_title,
            "signature_company": self.signature_company,
            "signature_phone_main": self.signature_phone_main,
            "signature_phone_mobile": self.signature_phone_mobile,
            "signature_email": self.signature_email,
            "signature_website": self.signature_website,
            "signature_address_line": self.signature_address_line,
            "signature_zip": self.signature_zip,
            "signature_city": self.signature_city,
            # Note: signature_logo_url supprimé - on utilise maintenant company.logo_url automatiquement
            "email_signature_html_template": self.email_signature_html_template,
            "legal_footer": self.legal_footer,
            "pdf_template_variant": self.pdf_template_variant,
            "vat_applicable": self.vat_applicable,
            "vat_rate": float(self.vat_rate) if self.vat_rate is not None else None,
            "vat_label": self.vat_label,
            "vat_number": self.vat_number,
            # Configuration SMTP (ne pas exposer le mot de passe)
            "smtp_enabled": self.smtp_enabled,
            "smtp_server": self.smtp_server,
            "smtp_port": self.smtp_port,
            "smtp_use_tls": self.smtp_use_tls,
            "smtp_use_ssl": self.smtp_use_ssl,
            "smtp_username": self.smtp_username,
            "smtp_password_configured": bool(self.smtp_password),  # Juste un booléen
            "cancellation_policy": self.cancellation_policy,
        }

    @hybrid_property  # Le linter détecte un conflit mais c'est intentionnel : _iban_raw mappe la colonne "iban" et iban est la propriété Python
    def iban(self) -> str | None:
        """✅ S2: Propriété hybride pour déchiffrer automatiquement l'IBAN.

        Returns:
            IBAN en clair ou None si vide
        """
        if not bool(getattr(self, "_iban_raw", None)):
            return None
        try:
            encryption_service = get_encryption_service()
            return encryption_service.decrypt_field(
                str(getattr(self, "_iban_raw", None))
            )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                "[CompanyBillingSettings] Erreur déchiffrement IBAN pour company_id=%s: %s",
                self.company_id,
                e,
            )
            return None

    @iban.setter
    def iban(self, value: str | None) -> None:
        """✅ S2: Setter pour chiffrer automatiquement l'IBAN avant stockage.

        Args:
            value: IBAN en clair ou None
        """
        if not value:
            self._iban_raw = None
            return

        # Chiffrer l'IBAN
        try:
            encryption_service = get_encryption_service()
            self._iban_raw = encryption_service.encrypt_field(value.strip().upper())
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                "[CompanyBillingSettings] Erreur chiffrement IBAN pour company_id=%s: %s",
                self.company_id,
                e,
            )
            raise

    @hybrid_property
    def smtp_password(self) -> str | None:
        """✅ Propriété hybride pour déchiffrer automatiquement le mot de passe SMTP.

        Returns:
            Mot de passe SMTP en clair ou None si vide
        """
        if not bool(getattr(self, "_smtp_password_raw", None)):
            return None
        try:
            encryption_service = get_encryption_service()
            return encryption_service.decrypt_field(
                str(getattr(self, "_smtp_password_raw", None))
            )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                (
                    "[CompanyBillingSettings] Erreur déchiffrement SMTP password "
                    "pour company_id=%s: %s"
                ),
                self.company_id,
                e,
            )
            return None

    @smtp_password.setter
    def smtp_password(self, value: str | None) -> None:
        """✅ Setter pour chiffrer automatiquement le mot de passe SMTP avant stockage.

        Args:
            value: Mot de passe en clair ou None
        """
        if not value:
            self._smtp_password_raw = None
            return

        # Chiffrer le mot de passe
        try:
            encryption_service = get_encryption_service()
            self._smtp_password_raw = encryption_service.encrypt_field(value.strip())
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                (
                    "[CompanyBillingSettings] Erreur chiffrement SMTP password "
                    "pour company_id=%s: %s"
                ),
                self.company_id,
                e,
            )
            raise

    @hybrid_property
    def qr_iban(self) -> str | None:
        """✅ S2: Propriété hybride pour déchiffrer automatiquement le QR-IBAN.

        Returns:
            QR-IBAN en clair ou None si vide
        """
        if not bool(getattr(self, "_qr_iban_raw", None)):
            return None
        try:
            encryption_service = get_encryption_service()
            return encryption_service.decrypt_field(
                str(getattr(self, "_qr_iban_raw", None))
            )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                "[CompanyBillingSettings] Erreur déchiffrement QR-IBAN pour company_id=%s: %s",
                self.company_id,
                e,
            )
            return None

    @qr_iban.setter
    def qr_iban(self, value: str | None) -> None:
        """✅ S2: Setter pour chiffrer automatiquement le QR-IBAN avant stockage.

        Args:
            value: QR-IBAN en clair ou None
        """
        if not value:
            self._qr_iban_raw = None
            return

        # Chiffrer le QR-IBAN
        try:
            encryption_service = get_encryption_service()
            self._qr_iban_raw = encryption_service.encrypt_field(value.strip().upper())
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                "[CompanyBillingSettings] Erreur chiffrement QR-IBAN pour company_id=%s: %s",
                self.company_id,
                e,
            )
            raise


class InvoiceSequence(db.Model):
    """Séquence de numérotation des factures par entreprise et mois."""

    __tablename__ = "invoice_sequences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id"), nullable=False
    )
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    month: Mapped[int] = mapped_column(Integer, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Relations
    company = relationship("Company", backref="invoice_sequences")

    # Contrainte d'unicité
    __table_args__ = (
        UniqueConstraint("company_id", "year", "month", name="uq_company_year_month"),
    )

    @override
    def __repr__(self):
        return (
            f"<InvoiceSequence {self.company_id}-{self.year}-{self.month}: "
            f"{self.sequence}>"
        )
