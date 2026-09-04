"""Plan de facturation institution : buckets détectés (clinique / patients).

Le type de payeur n'est pas choisi à l'avance : on résout le payeur de
chaque jambe éligible, puis on propose les factures.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any

from application.invoices.billable_amount import calculate_billable_booking_amount
from application.invoices.institution_invoice_eligibility import (
    EligibilitySummary,
    InstitutionInvoicePlan,
    PayerBucket,
    attach_invoice_request_ids,
    build_eligibility_summary,
    filter_institution_invoice_eligible,
    resolve_invoice_payer_type,
)
from application.invoices.institution_invoice_reconciliation import (
    build_reconciliation_ledger,
)
from application.invoices.invoice_line_description import (
    resolve_patient_name_for_invoice,
)
from application.invoices.period_invoice_preview import build_period_invoice_preview
from models import Booking, Client, Company, InstitutionPatient
from repositories.client_repository import ClientRepository
from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsRepository,
)


def _institution_id_from_clinic_client(clinic_client_id: int | None) -> int | None:
    if clinic_client_id is None:
        return None
    from ext import db

    client = db.session.get(Client, int(clinic_client_id))
    if client is None:
        return None
    linked = getattr(client, "linked_institution_id", None)
    if linked:
        return int(linked)
    return None


def _collect_institution_patient_bookings(
    *,
    company_id: int,
    period_year: int,
    period_month: int,
    clinic_company_id: int,
    institution_id: int | None,
    clinic_client_id: int | None,
    open_only: bool = True,
) -> list[Booking]:
    """Courses patient liées à l'institution (Market LIRIE + portefeuille)."""
    _ = clinic_company_id
    from sqlalchemy import or_

    from application.institutions.billing_control.resolve import (
        list_institution_control_booking_ids,
    )
    from application.invoices.billing_period_eligibility import period_bounds

    start, end = period_bounds(period_year, period_month)
    ids: set[int] = set()
    if institution_id:
        ids.update(list_institution_control_booking_ids(int(institution_id)))

    query = Booking.query.filter(
        Booking.company_id == company_id,
        Booking.billed_to_type == "patient",
        Booking.scheduled_time >= start,
        Booking.scheduled_time < end,
    )
    if open_only:
        query = query.filter(Booking.invoice_line_id.is_(None))
    extras: list[Any] = []
    if ids:
        extras.append(Booking.id.in_(ids))
    if institution_id:
        extras.append(
            Booking.institution_patient_id.in_(
                InstitutionPatient.query.with_entities(InstitutionPatient.id).filter(
                    InstitutionPatient.institution_id == int(institution_id)
                )
            )
        )
    if extras:
        query = query.filter(or_(*extras))
    elif clinic_client_id:
        query = query.filter(Booking.client_id == int(clinic_client_id))
    return query.all()


def _collect_institution_clinic_universe(
    *,
    company_id: int,
    period_year: int,
    period_month: int,
    clinic_company_id: int,
) -> list[Booking]:
    """Toutes les courses clinique de la période (y compris déjà facturées / bloquées)."""
    from sqlalchemy import or_
    from sqlalchemy.orm import aliased

    from application.invoices.billing_period_eligibility import period_bounds
    from models.enums import BookingStatus
    from services.billing.clinic_s2_eligibility import (
        clinic_s2_billed_to_company_predicate,
    )

    start, end = period_bounds(period_year, period_month)
    target_statuses = [
        BookingStatus.COMPLETED.value,
        BookingStatus.RETURN_COMPLETED.value,
    ]
    query = Booking.query.filter(
        Booking.company_id == company_id,
        clinic_s2_billed_to_company_predicate(int(clinic_company_id), company_id),
        Booking.billed_to_type == "clinic",
        Booking.status.in_(target_statuses),
        Booking.scheduled_time >= start,
        Booking.scheduled_time < end,
    )
    ParentB = aliased(Booking)
    query = query.outerjoin(ParentB, ParentB.id == Booking.parent_booking_id).filter(
        or_(
            Booking.is_return == False,  # noqa: E712
            ParentB.id.is_(None),
            ParentB.status != BookingStatus.CANCELED.value,
        )
    )
    return query.all()


def build_institution_invoice_plan(
    *,
    company_id: int,
    period_year: int,
    period_month: int,
    clinic_company_id: int,
    clinic_client_id: int | None = None,
    now: datetime | None = None,
) -> InstitutionInvoicePlan:
    clinic_preview = build_period_invoice_preview(
        company_id=company_id,
        period_year=period_year,
        period_month=period_month,
        clinic_company_id=clinic_company_id,
        include_line_details=False,
        now=now,
    )

    institution_id = _institution_id_from_clinic_client(clinic_client_id)
    patient_raw = _collect_institution_patient_bookings(
        company_id=company_id,
        period_year=period_year,
        period_month=period_month,
        clinic_company_id=clinic_company_id,
        institution_id=institution_id,
        clinic_client_id=clinic_client_id,
    )
    attach_invoice_request_ids(patient_raw)
    patient_eligible = [
        b
        for b in filter_institution_invoice_eligible(patient_raw, now=now)
        if resolve_invoice_payer_type(b) == "patient"
    ]

    settings = CompanyBillingSettingsRepository().find_or_create(company_id)

    def _amt(b: Booking) -> Decimal:
        return calculate_billable_booking_amount(b, billing_settings=settings).amount_ht

    clinic_co = None
    from ext import db

    clinic_co = db.session.get(Company, int(clinic_company_id))
    clinic_name = (
        str(getattr(clinic_co, "name", None) or "").strip()
        or f"Clinique #{clinic_company_id}"
    )
    clinic_bucket = None
    if clinic_preview.transports_count > 0:
        clinic_bucket = PayerBucket(
            payer_type="clinic",
            key=f"clinic:{clinic_company_id}",
            display_name=clinic_name,
            transports_count=clinic_preview.transports_count,
            estimated_total=float(clinic_preview.estimated_total),
            clinic_company_id=int(clinic_company_id),
        )

    grouped: dict[tuple[Any, ...], list[Booking]] = defaultdict(list)
    for b in patient_eligible:
        ipid = getattr(b, "institution_patient_id", None)
        bpid = getattr(b, "billing_party_id", None)
        cid = getattr(b, "client_id", None)
        grouped[(ipid, bpid, cid)].append(b)

    crepo = ClientRepository()
    client_ids = {
        int(b.client_id) for b in patient_eligible if getattr(b, "client_id", None)
    }
    clients_by_id = (
        crepo.find_models_by_ids_and_company_with_user(client_ids, company_id)
        if client_ids
        else {}
    )

    patients: list[PayerBucket] = []
    for (ipid, bpid, cid), segs in grouped.items():
        if not segs:
            continue
        total = float(sum((_amt(b) for b in segs), Decimal("0")))
        sample = segs[0]
        client = clients_by_id.get(int(cid)) if cid is not None else None
        name = resolve_patient_name_for_invoice(client, segs) or (
            getattr(sample, "customer_name", None) or "Patient"
        )
        booking_ids: list[int] = []
        for booking in segs:
            try:
                booking_ids.append(int(booking.id))
            except (TypeError, ValueError):
                continue
        patients.append(
            PayerBucket(
                payer_type="patient",
                key=f"patient:{ipid or cid}:{bpid or 0}",
                display_name=str(name).strip() or "Patient",
                transports_count=len(segs),
                estimated_total=round(total, 2),
                client_id=int(cid) if cid is not None else None,
                institution_patient_id=int(ipid) if ipid is not None else None,
                billing_party_id=int(bpid) if bpid is not None else None,
                booking_ids=booking_ids,
            )
        )
    patients.sort(key=lambda p: p.display_name.lower())

    considered = list(patient_raw)
    eligibility_patient = build_eligibility_summary(
        considered,
        patient_eligible,
        now=now,
        amount_ht_fn=_amt,
    )
    clinic_el = clinic_preview.eligibility or {}
    origin = clinic_el.get("origin") or {}
    market = clinic_el.get("market_lirie") or {}
    eligibility = EligibilitySummary(
        eligible_count=clinic_preview.transports_count + len(patient_eligible),
        eligible_amount_ht=round(
            float(clinic_preview.estimated_total)
            + eligibility_patient.eligible_amount_ht,
            2,
        ),
        origin_own_portfolio=int(origin.get("own_portfolio") or 0)
        + eligibility_patient.origin_own_portfolio,
        origin_market_lirie=int(origin.get("market_lirie") or 0)
        + eligibility_patient.origin_market_lirie,
        market_validated=int(market.get("validated") or 0)
        + eligibility_patient.market_validated,
        market_auto_released=int(market.get("auto_released") or 0)
        + eligibility_patient.market_auto_released,
        market_pending=int(market.get("pending") or 0)
        + eligibility_patient.market_pending,
        market_disputed=int(market.get("disputed") or 0)
        + eligibility_patient.market_disputed,
        excluded_count=int(clinic_el.get("excluded_count") or 0)
        + eligibility_patient.excluded_count,
    )

    clinic_universe = _collect_institution_clinic_universe(
        company_id=company_id,
        period_year=period_year,
        period_month=period_month,
        clinic_company_id=clinic_company_id,
    )
    patient_universe = _collect_institution_patient_bookings(
        company_id=company_id,
        period_year=period_year,
        period_month=period_month,
        clinic_company_id=clinic_company_id,
        institution_id=institution_id,
        clinic_client_id=clinic_client_id,
        open_only=False,
    )
    universe_by_id: dict[int, Booking] = {}
    for b in list(clinic_universe) + list(patient_universe):
        try:
            universe_by_id[int(b.id)] = b
        except (TypeError, ValueError):
            continue
    universe = list(universe_by_id.values())
    attach_invoice_request_ids(universe)
    ledger = build_reconciliation_ledger(
        universe,
        period_year=period_year,
        period_month=period_month,
        now=now,
        amount_ht_fn=_amt,
    )

    return InstitutionInvoicePlan(
        period_year=period_year,
        period_month=period_month,
        clinic_company_id=int(clinic_company_id),
        eligibility=eligibility,
        clinic=clinic_bucket,
        patients=patients,
        reconciliation=ledger.to_dict(),
    )
