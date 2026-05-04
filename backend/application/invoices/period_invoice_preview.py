"""Prévisualisation facture période — alignée sur l'éligibilité et les montants de génération."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import aliased

from application.invoices.invoice_line_description import (
    booking_source_type_for_preview,
    build_invoice_line_description,
    build_invoice_line_description_clinic_monthly,
    resolve_patient_name_for_invoice,
)
from application.invoices.round_trip_billing_lock import (
    filter_bookings_open_for_new_invoice_line,
    round_trip_component_id_sets,
)
from infrastructure.invoices.invoice_calculator import (
    InvoiceCalculator,
    round_to_5_cents,
)
from models import Booking, ClientStay, Company
from models.enums import BookingStatus
from repositories.booking_repository import BookingRepository
from repositories.client_repository import ClientRepository
from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsRepository,
)

_MIN_ROUND_TRIP_COMPONENT_SIZE = 2


def _preview_base_amount_ht(booking: Any, billing_settings_dto: Any) -> Decimal:
    """HT ligne pour preview : même logique de base que generate_invoice / clinic (hors overrides)."""
    two_places = Decimal("0.01")
    mission_type = getattr(booking, "mission_type", None) or "patient_transport"
    if mission_type == "material_delivery":
        fp = getattr(billing_settings_dto, "material_delivery_price_fixed", None)
        if fp is None or fp <= 0:
            return Decimal("0.00")
        return Decimal(str(fp)).quantize(two_places)
    base_amount = Decimal(str(getattr(booking, "amount", None) or 0)).quantize(two_places)
    if (
        str(getattr(booking, "status", "") or "").upper() == "CANCELED"
        and getattr(booking, "cancellation_fee_amount", None) is not None
    ):
        base_amount = Decimal(str(booking.cancellation_fee_amount)).quantize(two_places)
    return round_to_5_cents(base_amount)


def _scheduled_iso(booking: Any) -> str | None:
    st = getattr(booking, "scheduled_time", None)
    if st is None:
        return None
    if hasattr(st, "isoformat"):
        return st.isoformat()
    return str(st)


def _round_trip_leg_by_booking_id(bookings: list[Any]) -> dict[int, bool]:
    """True si le booking est l'aller ou le retour d'un A/R avec l'autre segment dans la même liste."""
    if not bookings:
        return {}
    ids = {int(b.id) for b in bookings}
    parents_with_return_in_list: set[int] = set()
    for b in bookings:
        pid = getattr(b, "parent_booking_id", None)
        if pid is not None and int(pid) in ids:
            parents_with_return_in_list.add(int(pid))
    out: dict[int, bool] = {}
    for b in bookings:
        bid = int(b.id)
        is_ret = bool(getattr(b, "is_return", False))
        pid = getattr(b, "parent_booking_id", None)
        linked_to_parent = pid is not None and int(pid) in ids
        has_child_return = bid in parents_with_return_in_list
        out[bid] = is_ret or linked_to_parent or has_child_return
    return out


def _consolidate_period_preview_round_trip_rows(
    preview_lines: list[PeriodPreviewLine],
    bookings: list[Any],
    _billing_settings_dto: Any,
) -> list[PeriodPreviewLine]:
    """Une ligne par A/R (HT cumulé, description du segment principal)."""
    preview_ht_by_bid = {
        pl.booking_id: Decimal(str(pl.amount_ht)).quantize(Decimal("0.01"))
        for pl in preview_lines
    }

    def _amount_for_booking(b: Any) -> Decimal:
        return preview_ht_by_bid.get(int(b.id), Decimal("0"))

    comps = [
        c
        for c in round_trip_component_id_sets(bookings, amount_ht_fn=_amount_for_booking)
        if len(c) >= _MIN_ROUND_TRIP_COMPONENT_SIZE
    ]
    if not comps:
        return preview_lines
    by_id = {pl.booking_id: pl for pl in preview_lines}
    hidden_ids: set[int] = set()
    merged_rows: dict[int, PeriodPreviewLine] = {}
    for comp in comps:
        pls = [by_id[bid] for bid in comp if bid in by_id]
        if len(pls) < _MIN_ROUND_TRIP_COMPONENT_SIZE:
            continue
        primary_pl = min(
            pls,
            key=lambda pl: ((pl.scheduled_at or ""), pl.booking_id),
        )
        pri = primary_pl.booking_id
        sum_ht = round(sum(float(p.amount_ht) for p in pls), 2)
        sum_origin = round(sum(float(p.origin_amount_ht) for p in pls), 2)
        _sched_candidates = [p.scheduled_at for p in pls if p.scheduled_at]
        sched = min(_sched_candidates) if _sched_candidates else None
        merged_rows[pri] = PeriodPreviewLine(
            booking_id=pri,
            scheduled_at=sched,
            amount_ht=sum_ht,
            origin_amount_ht=sum_origin,
            description=primary_pl.description,
            source_type=primary_pl.source_type,
            is_locked=any(p.is_locked for p in pls),
            already_invoiced=any(p.already_invoiced for p in pls),
            is_round_trip_leg=True,
        )
        for p in pls:
            if p.booking_id != pri:
                hidden_ids.add(p.booking_id)
    out: list[PeriodPreviewLine] = []
    emitted: set[int] = set()
    for pl in preview_lines:
        if pl.booking_id in hidden_ids:
            continue
        if pl.booking_id in merged_rows:
            if pl.booking_id not in emitted:
                out.append(merged_rows[pl.booking_id])
                emitted.add(pl.booking_id)
            continue
        out.append(pl)
    return out


@dataclass(frozen=True, slots=True)
class PeriodPreviewLine:
    booking_id: int
    scheduled_at: str | None
    amount_ht: float
    origin_amount_ht: float
    description: str
    source_type: str
    is_locked: bool
    already_invoiced: bool
    is_round_trip_leg: bool = False


@dataclass(frozen=True, slots=True)
class PeriodPreviewResult:
    mode: str
    transports_count: int
    estimated_total: float
    currency: str
    warnings: list[str]
    preview_lines: tuple[PeriodPreviewLine, ...] = field(default_factory=tuple)
    estimated_subtotal_ht: float = 0.0
    estimated_vat_total: float = 0.0
    estimated_total_with_vat: float = 0.0


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
    settings_repo = CompanyBillingSettingsRepository()
    billing_settings_dto = settings_repo.find_or_create(company_id)
    calc = InvoiceCalculator()

    vat_rate_setting = billing_settings_dto.vat_rate
    vat_applicable = bool(billing_settings_dto.vat_applicable) and vat_rate_setting is not None
    default_vat_rate = Decimal("0")
    if vat_applicable:
        try:
            default_vat_rate = Decimal(str(vat_rate_setting)).quantize(Decimal("0.01"))
            if default_vat_rate <= 0:
                vat_applicable = False
        except Exception:
            vat_applicable = False
            default_vat_rate = Decimal("0")

    preview_lines_out: list[PeriodPreviewLine] = []

    if client_id is not None:
        crepo = ClientRepository()
        if not crepo.find_model_by_id_and_company(int(client_id), company_id):
            raise ValueError("Client introuvable pour cette entreprise")

        repo = BookingRepository()
        eligible_b = repo.find_models_eligible_for_billing_period_by_company_and_client(
            company_id,
            int(client_id),
            period_year,
            period_month,
            billed_to_type="patient",
        )
        bookings = filter_bookings_open_for_new_invoice_line(
            eligible_b,
            amount_ht_fn=lambda b: _preview_base_amount_ht(b, billing_settings_dto),
        )
        client = crepo.find_model_by_id_with_user(int(client_id), company_id)
        patient_name = resolve_patient_name_for_invoice(client, bookings)
        rt_map = _round_trip_leg_by_booking_id(bookings)

        gross = Decimal("0.00")
        vat_sum = Decimal("0.00")
        tw_sum = Decimal("0.00")

        for b in bookings:
            amt = _preview_base_amount_ht(b, billing_settings_dto)
            gross += amt
            if amt == 0:
                warnings.append(
                    f"Transport #{getattr(b, 'id', '?')} sans montant (un estimé peut s'appliquer à la génération)"
                )

            va, tw = calc.calculate_vat(amt, default_vat_rate if vat_applicable else Decimal("0"))
            vat_sum += va
            tw_sum += tw

            desc = build_invoice_line_description(
                b,
                patient_name=patient_name,
                bill_to_client_id=None,
                clinic_company_id=None,
                billing_party_id=None,
                booking_for_cancellation=b,
                description_builder=None,
            )
            locked = bool(getattr(b, "invoice_line_id", None))
            preview_lines_out.append(
                PeriodPreviewLine(
                    booking_id=int(b.id),
                    scheduled_at=_scheduled_iso(b),
                    amount_ht=float(amt),
                    origin_amount_ht=float(amt),
                    description=desc,
                    source_type=booking_source_type_for_preview(b),
                    is_locked=locked,
                    already_invoiced=bool(getattr(b, "invoice_line_id", None)),
                    is_round_trip_leg=bool(rt_map.get(int(b.id), False)),
                )
            )

        preview_lines_out = _consolidate_period_preview_round_trip_rows(
            preview_lines_out, bookings, billing_settings_dto
        )

        if len(bookings) == 0:
            warnings.insert(
                0,
                (
                    "Aucun transport à facturer pour ce patient sur cette période "
                    + "(vérifiez le mois, le payeur facturé « patient » et les courses déjà facturées)."
                ),
            )

        total = round_to_5_cents(gross)
        total_tw = round_to_5_cents(tw_sum)
        total_vat = round_to_5_cents(vat_sum)

        return PeriodPreviewResult(
            mode="standard",
            transports_count=len(bookings),
            estimated_total=float(total),
            currency="CHF",
            warnings=warnings[:25],
            preview_lines=tuple(preview_lines_out),
            estimated_subtotal_ht=float(total),
            estimated_vat_total=float(total_vat),
            estimated_total_with_vat=float(total_tw),
        )

    # --- S2 : même filtre que preview clinique / génération ---
    if clinic_company_id is None:
        raise ValueError("clinic_company_id requis pour l'aperçu clinique")
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
    eligible_bookings = eligible_query.order_by(Booking.scheduled_time.asc()).all()
    rt_map_s2 = _round_trip_leg_by_booking_id(eligible_bookings)

    gross_s2 = Decimal("0.00")
    vat_sum_s2 = Decimal("0.00")
    tw_sum_s2 = Decimal("0.00")

    for b in eligible_bookings:
        amt = _preview_base_amount_ht(b, billing_settings_dto)
        gross_s2 += amt
        if amt == 0:
            warnings.append(
                f"Transport #{getattr(b, 'id', '?')} sans montant (un estimé peut s'appliquer à la génération)"
            )

        va, tw = calc.calculate_vat(amt, default_vat_rate if vat_applicable else Decimal("0"))
        vat_sum_s2 += va
        tw_sum_s2 += tw

        desc = build_invoice_line_description_clinic_monthly(b, description_builder=None)

        locked = bool(getattr(b, "invoice_line_id", None))
        preview_lines_out.append(
            PeriodPreviewLine(
                booking_id=int(b.id),
                scheduled_at=_scheduled_iso(b),
                amount_ht=float(amt),
                origin_amount_ht=float(amt),
                description=desc,
                source_type=booking_source_type_for_preview(b),
                is_locked=locked,
                already_invoiced=bool(getattr(b, "invoice_line_id", None)),
                is_round_trip_leg=bool(rt_map_s2.get(int(b.id), False)),
            )
        )

    preview_lines_out = _consolidate_period_preview_round_trip_rows(
        preview_lines_out, eligible_bookings, billing_settings_dto
    )

    if len(eligible_bookings) == 0:
        warnings.insert(
            0,
            (
                "Aucun transport clinique à facturer sur cette période "
                + "(transports `billed_to_type=clinic` pour cette clinique, non encore facturés)."
            ),
        )

    total = round_to_5_cents(gross_s2)
    total_tw = round_to_5_cents(tw_sum_s2)
    total_vat = round_to_5_cents(vat_sum_s2)

    return PeriodPreviewResult(
        mode="clinic_monthly",
        transports_count=len(eligible_bookings),
        estimated_total=float(total),
        currency="CHF",
        warnings=warnings[:25],
        preview_lines=tuple(preview_lines_out),
        estimated_subtotal_ht=float(total),
        estimated_vat_total=float(total_vat),
        estimated_total_with_vat=float(total_tw),
    )


def preview_line_to_dict(pl: PeriodPreviewLine) -> dict[str, Any]:
    return {
        "booking_id": pl.booking_id,
        "scheduled_at": pl.scheduled_at,
        "amount_ht": pl.amount_ht,
        "origin_amount_ht": pl.origin_amount_ht,
        "description": pl.description,
        "source_type": pl.source_type,
        "is_locked": pl.is_locked,
        "already_invoiced": pl.already_invoiced,
        "is_round_trip_leg": pl.is_round_trip_leg,
    }


def preview_to_dict(p: PeriodPreviewResult) -> dict[str, Any]:
    return {
        "mode": p.mode,
        "transports_count": p.transports_count,
        "estimated_total": p.estimated_total,
        "currency": p.currency,
        "warnings": p.warnings,
        "preview_lines": [preview_line_to_dict(x) for x in p.preview_lines],
        "estimated_subtotal_ht": p.estimated_subtotal_ht,
        "estimated_vat_total": p.estimated_vat_total,
        "estimated_total_with_vat": p.estimated_total_with_vat,
    }
