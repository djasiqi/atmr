"""Prévisualisation facture période (V1) — alignée sur l’éligibilité génération."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import aliased

from infrastructure.invoices.invoice_calculator import round_to_5_cents
from models import Booking, ClientStay, Company
from models.enums import BookingStatus
from repositories.booking_repository import BookingRepository
from repositories.client_repository import ClientRepository


def _booking_money_ht(booking: Any) -> Decimal:
    """Montant HT de référence (amount ou estimated) — aligné génération."""
    raw = getattr(booking, "amount", None) or getattr(booking, "estimated_amount", None)
    if raw is None:
        return Decimal("0.00")
    try:
        return Decimal(str(raw))
    except Exception:
        return Decimal("0.00")


@dataclass(frozen=True, slots=True)
class PeriodPreviewResult:
    mode: str
    transports_count: int
    estimated_total: float
    currency: str
    warnings: list[str]


def build_period_invoice_preview(
    *,
    company_id: int,
    period_year: int,
    period_month: int,
    client_id: int | None = None,
    clinic_company_id: int | None = None,
) -> PeriodPreviewResult:
    """Aperçu read-only : exactement un de client_id (patient direct) ou clinic_company_id (S2)."""
    if bool(client_id) == bool(clinic_company_id):
        raise ValueError("Fournir exactement un des paramètres: client_id ou clinic_company_id")

    warnings: list[str] = []

    if client_id is not None:
        crepo = ClientRepository()
        if not crepo.find_model_by_id_and_company(int(client_id), company_id):
            raise ValueError("Client introuvable pour cette entreprise")

        repo = BookingRepository()
        bookings = repo.find_models_unbilled_by_company_and_client(
            company_id,
            int(client_id),
            period_year,
            period_month,
            billed_to_type="patient",
        )
        gross = Decimal("0.00")
        for b in bookings:
            m = _booking_money_ht(b)
            gross += m
            if m == 0:
                warnings.append(
                    f"Transport #{getattr(b, 'id', '?')} sans montant (un estimé peut s’appliquer à la génération)"
                )

        if len(bookings) == 0:
            warnings.insert(
                0,
                "Aucun transport à facturer pour ce patient sur cette période "
                "(vérifiez le mois, le payeur facturé « patient » et les courses déjà facturées).",
            )

        total = round_to_5_cents(gross)
        return PeriodPreviewResult(
            mode="standard",
            transports_count=len(bookings),
            estimated_total=float(total),
            currency="CHF",
            warnings=warnings[:25],
        )

    # --- S2 : même filtre que ClinicMonthlyTotals (éligibles non facturés) ---
    ccid = int(clinic_company_id)
    if not Company.query.filter_by(id=ccid).first():
        raise ValueError("Clinique (entreprise) introuvable")

    PERIOD_MONTH_THRESHOLD = 12
    start_date = datetime(period_year, period_month, 1)
    if period_month == PERIOD_MONTH_THRESHOLD:
        end_date = datetime(period_year + 1, 1, 1)
    else:
        end_date = datetime(period_year, period_month + 1, 1)

    target_statuses = [
        BookingStatus.COMPLETED.value,
        BookingStatus.RETURN_COMPLETED.value,
    ]
    from sqlalchemy import exists

    stay_overlaps_booking = exists().where(
        ClientStay.client_id == Booking.client_id,
        ClientStay.company_id == ccid,
        ClientStay.status == "active",
        ClientStay.start_date <= Booking.scheduled_time,
        or_(
            ClientStay.end_date.is_(None),
            ClientStay.end_date >= Booking.scheduled_time,
        ),
    )
    canceled_eligible = (
        (Booking.status == BookingStatus.CANCELED.value)
        & (Booking.amount > 0)
        & (
            (Booking.is_cancellation_billable == True)  # noqa: E712
            | (
                Booking.billing_override_reason.isnot(None)
                & (Booking.billing_override_reason != "")
            )
        )
        & stay_overlaps_booking
        & (Booking.is_return == False)  # noqa: E712
    )
    eligible_query = Booking.query.filter(
        Booking.company_id == company_id,
        Booking.billed_to_company_id == ccid,
        Booking.billed_to_type == "clinic",
        or_(
            Booking.status.in_(target_statuses),
            canceled_eligible,
        ),
        Booking.invoice_line_id.is_(None),
        Booking.scheduled_time >= start_date,
        Booking.scheduled_time < end_date,
    )
    ParentB = aliased(Booking)
    eligible_query = eligible_query.outerjoin(
        ParentB, ParentB.id == Booking.parent_booking_id
    ).filter(
        or_(
            Booking.is_return == False,  # noqa: E712
            ParentB.id.is_(None),
            ParentB.status != BookingStatus.CANCELED.value,
        )
    )
    eligible_bookings = eligible_query.all()
    gross_s2 = sum(_booking_money_ht(b) for b in eligible_bookings)
    for b in eligible_bookings:
        if _booking_money_ht(b) == 0:
            warnings.append(
                f"Transport #{getattr(b, 'id', '?')} sans montant (un estimé peut s’appliquer à la génération)"
            )

    if len(eligible_bookings) == 0:
        warnings.insert(
            0,
            "Aucun transport clinique à facturer sur cette période "
            "(transports `billed_to_type=clinic` pour cette clinique, non encore facturés).",
        )

    total = round_to_5_cents(gross_s2)
    return PeriodPreviewResult(
        mode="clinic_monthly",
        transports_count=len(eligible_bookings),
        estimated_total=float(total),
        currency="CHF",
        warnings=warnings[:25],
    )


def preview_to_dict(p: PeriodPreviewResult) -> dict[str, Any]:
    return {
        "mode": p.mode,
        "transports_count": p.transports_count,
        "estimated_total": p.estimated_total,
        "currency": p.currency,
        "warnings": p.warnings,
    }
