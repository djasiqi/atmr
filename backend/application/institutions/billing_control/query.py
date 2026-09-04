"""Requête liste contrôle facturation — filtres, pagination, summary (couche présentation)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import joinedload

from application.institutions.billing_control.presentation import (
    serialize_billing_control_booking,
)
from application.institutions.billing_control.resolve import (
    list_institution_control_booking_ids,
)
from application.institutions.billing_control.status import effective_control_status
from application.invoices.billing_period_eligibility import (
    booking_matches_period_preview_eligibility,
)
from ext import db
from models import Booking, TransportRequest

_PERIOD_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{1,2})$")
_DEFAULT_PAGE_SIZE = 50
_MAX_PAGE_SIZE = 200


@dataclass(frozen=True, slots=True)
class BillingControlQueryParams:
    period_year: int | None = None
    period_month: int | None = None
    control_status: str | None = None
    payer_type: str | None = None
    transport_company_id: int | None = None
    patient_id: int | None = None
    page: int = 1
    page_size: int = _DEFAULT_PAGE_SIZE


@dataclass(frozen=True, slots=True)
class BillingControlSummary:
    total: int
    pending_review: int
    validated: int
    anomaly: int
    payer_patient: int
    payer_clinic: int
    locked_or_invoiced: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "pending_review": self.pending_review,
            "validated": self.validated,
            "anomaly": self.anomaly,
            "payer_patient": self.payer_patient,
            "payer_clinic": self.payer_clinic,
            "locked_or_invoiced": self.locked_or_invoiced,
        }


@dataclass(frozen=True, slots=True)
class BillingControlListResult:
    items: list[dict[str, Any]]
    summary: BillingControlSummary
    page: int
    page_size: int
    total: int
    total_pages: int


def parse_billing_control_query(
    *,
    period: str | None = None,
    period_year: int | None = None,
    period_month: int | None = None,
    control_status: str | None = None,
    payer_type: str | None = None,
    transport_company: int | None = None,
    patient: int | None = None,
    page: int | None = None,
    page_size: int | None = None,
) -> BillingControlQueryParams | tuple[str, int]:
    """Parse les query params route ; retourne (error, 400) si invalide."""
    py, pm = period_year, period_month
    if period:
        m = _PERIOD_RE.match(period.strip())
        if not m:
            return ("Paramètre period invalide (attendu YYYY-MM).", 400)
        py = int(m.group("year"))
        pm = int(m.group("month"))
    if py is not None and pm is not None and not (1 <= pm <= 12):
        return ("period_month invalide (1-12).", 400)

    pg = max(1, int(page or 1))
    ps = int(page_size or _DEFAULT_PAGE_SIZE)
    ps = min(max(1, ps), _MAX_PAGE_SIZE)

    payer_norm = None
    if payer_type:
        payer_norm = payer_type.lower().strip()
        if payer_norm in ("institution",):
            payer_norm = "clinic"
        if payer_norm not in ("patient", "clinic"):
            return ("payer_type invalide (patient ou clinic).", 400)

    status_norm = control_status.lower().strip() if control_status else None
    if status_norm and status_norm not in (
        "pending_review",
        "validated",
        "anomaly",
    ):
        return ("control_status invalide.", 400)

    return BillingControlQueryParams(
        period_year=py,
        period_month=pm,
        control_status=status_norm,
        payer_type=payer_norm,
        transport_company_id=transport_company,
        patient_id=patient,
        page=pg,
        page_size=ps,
    )


def _load_institution_bookings(
    institution_id: int,
) -> tuple[list[Booking], dict[int, TransportRequest]]:
    booking_ids = list_institution_control_booking_ids(institution_id)
    if not booking_ids:
        return [], {}

    bookings = (
        db.session.query(Booking)
        .options(
            joinedload(Booking.company),
            joinedload(Booking.institution_patient),
            joinedload(Booking.billing_party),
        )
        .filter(Booking.id.in_(booking_ids))
        .all()
    )
    by_id = {int(b.id): b for b in bookings}

    transport_rows = (
        db.session.query(TransportRequest)
        .filter(
            TransportRequest.institution_id == institution_id,
        )
        .all()
    )
    tr_by_booking: dict[int, TransportRequest] = {}
    tr_by_route_group: dict[str, TransportRequest] = {}
    for tr in transport_rows:
        if tr.booking_id is not None:
            tr_by_booking[int(tr.booking_id)] = tr
        if tr.route_group_id:
            tr_by_route_group[str(tr.route_group_id)] = tr

    tr_map: dict[int, TransportRequest] = {}
    for bid in booking_ids:
        booking = by_id.get(bid)
        if booking is None:
            continue
        tr = tr_by_booking.get(bid)
        if tr is None:
            rg = getattr(booking, "route_group_id", None)
            if rg:
                tr = tr_by_route_group.get(str(rg))
        if tr is None:
            parent_id = getattr(booking, "parent_booking_id", None)
            if parent_id is not None:
                tr = tr_by_booking.get(int(parent_id))
        if tr is not None:
            tr_map[bid] = tr

    ordered = [by_id[bid] for bid in booking_ids if bid in by_id]
    return ordered, tr_map


def _passes_period_eligibility(
    booking: Booking,
    *,
    period_year: int,
    period_month: int,
    parent_by_id: dict[int, Booking],
) -> bool:
    company_id = getattr(booking, "company_id", None)
    if company_id is None:
        return False
    btype = str(getattr(booking, "billed_to_type", None) or "patient")
    return booking_matches_period_preview_eligibility(
        booking,
        company_id=int(company_id),
        period_year=period_year,
        period_month=period_month,
        billed_to_type=btype,
        parent_by_id=parent_by_id,
    )


def _apply_filters(
    bookings: list[Booking],
    params: BillingControlQueryParams,
) -> list[Booking]:
    parent_by_id: dict[int, Booking] = {int(b.id): b for b in bookings}

    filtered: list[Booking] = []
    for booking in bookings:
        if (
            params.period_year is not None
            and params.period_month is not None
            and not _passes_period_eligibility(
                booking,
                period_year=params.period_year,
                period_month=params.period_month,
                parent_by_id=parent_by_id,
            )
        ):
            continue

        effective = effective_control_status(booking)
        if params.control_status and effective != params.control_status:
            continue

        billed = (getattr(booking, "billed_to_type", None) or "").lower()
        if params.payer_type and billed != params.payer_type:
            continue

        if params.transport_company_id is not None and int(
            getattr(booking, "company_id", 0) or 0
        ) != int(params.transport_company_id):
            continue

        if params.patient_id is not None:
            ipid = getattr(booking, "institution_patient_id", None)
            if ipid is None or int(ipid) != int(params.patient_id):
                continue

        filtered.append(booking)

    filtered.sort(
        key=lambda b: (
            getattr(b, "scheduled_time", None) or "",
            int(b.id),
        )
    )
    return filtered


def _compute_summary(bookings: list[Booking]) -> BillingControlSummary:
    pending = validated = anomaly = payer_patient = payer_clinic = locked = 0
    for booking in bookings:
        eff = effective_control_status(booking)
        if eff == "pending_review":
            pending += 1
        elif eff == "validated":
            validated += 1
        elif eff == "anomaly":
            anomaly += 1

        billed = (getattr(booking, "billed_to_type", None) or "").lower()
        if billed == "patient":
            payer_patient += 1
        elif billed == "clinic":
            payer_clinic += 1

        from application.companies.reservations.billing_adjustment import (
            booking_billing_is_locked,
        )

        is_locked, _ = booking_billing_is_locked(booking)
        invoiced = bool(getattr(booking, "invoice_line_id", None))
        if is_locked or invoiced:
            locked += 1

    return BillingControlSummary(
        total=len(bookings),
        pending_review=pending,
        validated=validated,
        anomaly=anomaly,
        payer_patient=payer_patient,
        payer_clinic=payer_clinic,
        locked_or_invoiced=locked,
    )


def query_billing_control_bookings(
    institution_id: int,
    params: BillingControlQueryParams,
) -> BillingControlListResult:
    """Liste paginée + summary sur la même population filtrée."""
    all_bookings, tr_map = _load_institution_bookings(institution_id)
    filtered = _apply_filters(all_bookings, params)
    summary = _compute_summary(filtered)

    total = len(filtered)
    total_pages = max(1, math.ceil(total / params.page_size)) if total else 1
    start = (params.page - 1) * params.page_size
    end = start + params.page_size
    page_bookings = filtered[start:end]

    institution_by_id = {int(b.id): b for b in all_bookings}
    from application.invoices.booking_dispute.service import latest_dispute_summaries

    dispute_map = latest_dispute_summaries([int(b.id) for b in page_bookings])
    items = [
        serialize_billing_control_booking(
            b,
            transport_request=tr_map.get(int(b.id)),
            institution_bookings_by_id=institution_by_id,
            dispute_summary=dispute_map.get(int(b.id)),
        )
        for b in page_bookings
    ]

    return BillingControlListResult(
        items=items,
        summary=summary,
        page=params.page,
        page_size=params.page_size,
        total=total,
        total_pages=total_pages,
    )


def booking_control_detail_payload(
    booking: Booking,
    *,
    institution_id: int,
) -> dict[str, Any]:
    """Détail enrichi pour GET /control/bookings/{id}."""
    all_bookings, tr_map = _load_institution_bookings(institution_id)
    institution_by_id = {int(b.id): b for b in all_bookings}
    tr = tr_map.get(int(booking.id))
    if tr is None:
        tr = TransportRequest.query.filter_by(
            booking_id=booking.id,
            institution_id=institution_id,
        ).first()
    return serialize_billing_control_booking(
        booking,
        transport_request=tr,
        institution_bookings_by_id=institution_by_id,
    )
