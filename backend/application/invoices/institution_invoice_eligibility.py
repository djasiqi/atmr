"""Éligibilité facture institution : origine × gate Market LIRIE × payeur.

Source de vérité pour « Nouvelle facture » institutionnelle.

1. Origine (PORTFOLIO_PROPRE / MARKET_LIRIE)
2. Gate Market LIRIE (PENDING / VALIDATED / DISPUTED / AUTO_RELEASED)
3. Payeur de chaque jambe (clinic / patient / partner)
4. Buckets de facturation — jamais « la clinique paie parce qu'elle a créé »

AUTO_RELEASED est un statut *effectif* : on ne l'écrit jamais comme
``validated``. Le persisté reste ``pending_review`` — l'audit reste exact.
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from application.institutions.billing_control.status import persisted_control_status
from models.enums import (
    BookingBillingOrigin,
    BookingCreatedVia,
    InstitutionBillingControlStatus,
)

ZURICH = ZoneInfo("Europe/Zurich")

ORIGIN_OWN_PORTFOLIO = "OWN_PORTFOLIO"
ORIGIN_MARKET_LIRIE = "LIRIE_MARKETPLACE"

InvoiceGate = Literal[
    "not_required",
    "pending",
    "validated",
    "validated_after_dispute",
    "disputed",
    "not_billable",
    "auto_released",
]
InvoicePayer = Literal["clinic", "patient", "partner", "other"]

_ELIGIBLE_GATES = frozenset(
    {"not_required", "validated", "validated_after_dispute", "auto_released"}
)
_MARKET_ORIGINS = frozenset(
    {
        BookingBillingOrigin.LIRIE_MARKETPLACE.value,
        "MARKET_LIRIE",
        "LIRIE_MARKETPLACE",
    }
)
_PORTFOLIO_ORIGINS = frozenset(
    {
        BookingBillingOrigin.OWN_PORTFOLIO.value,
        "PORTFOLIO_PROPRE",
        "OWN_PORTFOLIO",
    }
)


def _as_zurich(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ZURICH)
    return dt.astimezone(ZURICH)


def _now_zurich(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(ZURICH)
    return _as_zurich(now)


def service_datetime(booking: Any) -> datetime | None:
    st = getattr(booking, "scheduled_time", None)
    if isinstance(st, datetime):
        return st
    return None


def market_lirie_deadline(service_dt: datetime) -> datetime:
    """Dernier instant du mois calendaire de la prestation (Europe/Zurich)."""
    local = _as_zurich(service_dt)
    last_day = monthrange(local.year, local.month)[1]
    return datetime(
        local.year, local.month, last_day, 23, 59, 59, 999999, tzinfo=ZURICH
    )


def market_lirie_release_instant(service_dt: datetime) -> datetime:
    """Premier instant du mois suivant — libération automatique."""
    local = _as_zurich(service_dt)
    if local.month == 12:
        return datetime(local.year + 1, 1, 1, tzinfo=ZURICH)
    return datetime(local.year, local.month + 1, 1, tzinfo=ZURICH)


def is_market_lirie_deadline_passed(
    service_dt: datetime, *, now: datetime | None = None
) -> bool:
    return _now_zurich(now) >= market_lirie_release_instant(service_dt)


def _origin_raw(booking: Any) -> str:
    raw = getattr(booking, "billing_origin", None)
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw) or "").strip().upper()


def _created_via_raw(booking: Any) -> str:
    raw = getattr(booking, "created_via", None)
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw) or "").strip().lower()


def _has_institution_source_request(booking: Any) -> bool:
    reqs = getattr(booking, "source_request", None)
    if reqs:
        first = reqs[0] if isinstance(reqs, (list, tuple)) else reqs
        if getattr(first, "institution_id", None):
            return True
        if getattr(first, "id", None):
            return True
    resolver = getattr(booking, "_resolve_source_transport_request", None)
    if callable(resolver):
        try:
            req = resolver()
        except Exception:
            req = None
        if req is not None and (
            getattr(req, "institution_id", None) or getattr(req, "id", None)
        ):
            return True
    cached = getattr(booking, "_invoice_request_id", None)
    return cached is not None


def resolve_commercial_origin(booking: Any) -> str:
    """PORTFOLIO_PROPRE ou MARKET_LIRIE — jamais déduit du payeur."""
    raw = _origin_raw(booking)
    if raw in _MARKET_ORIGINS:
        return ORIGIN_MARKET_LIRIE
    if raw in _PORTFOLIO_ORIGINS:
        return ORIGIN_OWN_PORTFOLIO
    via = _created_via_raw(booking)
    if via == BookingCreatedVia.INSTITUTION_PORTAL.value:
        return ORIGIN_MARKET_LIRIE
    if _has_institution_source_request(booking):
        return ORIGIN_MARKET_LIRIE
    return ORIGIN_OWN_PORTFOLIO


def is_market_lirie_booking(booking: Any) -> bool:
    return resolve_commercial_origin(booking) == ORIGIN_MARKET_LIRIE


def invoice_gate_status(booking: Any, *, now: datetime | None = None) -> InvoiceGate:
    """Statut d'éligibilité facture (effectif, sans write)."""
    billing = str(getattr(booking, "invoice_billing_status", None) or "").strip()
    if billing == "not_billable":
        return "not_billable"

    if not is_market_lirie_booking(booking):
        return "not_required"

    persisted = persisted_control_status(booking)
    if persisted == InstitutionBillingControlStatus.ANOMALY.value:
        return "disputed"
    if persisted == InstitutionBillingControlStatus.VALIDATED.value:
        if billing == "billable":
            return "validated_after_dispute"
        return "validated"
    if persisted == "auto_released":
        return "auto_released"

    service_dt = service_datetime(booking)
    if service_dt is not None and is_market_lirie_deadline_passed(service_dt, now=now):
        return "auto_released"
    return "pending"


def is_institution_invoice_eligible(
    booking: Any, *, now: datetime | None = None
) -> bool:
    return invoice_gate_status(booking, now=now) in _ELIGIBLE_GATES


def resolve_invoice_payer_type(booking: Any) -> InvoicePayer:
    """Payeur de la jambe — source de vérité financière, pas l'origine."""
    btype = str(getattr(booking, "billed_to_type", None) or "").lower().strip()
    if btype == "clinic":
        return "clinic"
    if btype == "patient":
        return "patient"
    if btype in ("partner", "company"):
        return "partner"
    return "other"


def filter_institution_invoice_eligible(
    bookings: list[Any], *, now: datetime | None = None
) -> list[Any]:
    return [b for b in bookings if is_institution_invoice_eligible(b, now=now)]


def attach_invoice_request_ids(bookings: list[Any]) -> None:
    """Attache ``_invoice_request_id`` en batch (évite N+1)."""
    if not bookings:
        return
    already = all(
        getattr(b, "_invoice_request_id", None) is not None
        or getattr(b, "_invoice_request_resolved", False)
        for b in bookings
    )
    if already:
        return

    from ext import db
    from models import TransportRequest

    booking_ids: list[int] = []
    route_groups: set[str] = set()
    parent_ids: set[int] = set()
    for b in bookings:
        try:
            booking_ids.append(int(b.id))
        except (TypeError, ValueError):
            continue
        rg = getattr(b, "route_group_id", None)
        if rg:
            route_groups.add(str(rg))
        pid = getattr(b, "parent_booking_id", None)
        if pid is not None:
            try:
                parent_ids.add(int(pid))
            except (TypeError, ValueError):
                continue

    by_booking: dict[int, int] = {}
    by_rg: dict[str, int] = {}
    lookup_ids = set(booking_ids) | parent_ids
    if lookup_ids:
        rows = (
            db.session.query(TransportRequest.id, TransportRequest.booking_id)
            .filter(TransportRequest.booking_id.in_(lookup_ids))
            .all()
        )
        for rid, bid in rows:
            if bid is not None:
                by_booking[int(bid)] = int(rid)
    if route_groups:
        rg_rows = (
            db.session.query(TransportRequest.id, TransportRequest.route_group_id)
            .filter(TransportRequest.route_group_id.in_(route_groups))
            .all()
        )
        for rid, rg in rg_rows:
            if rg:
                by_rg.setdefault(str(rg), int(rid))

    for b in bookings:
        rid = None
        try:
            bid = int(b.id)
        except (TypeError, ValueError):
            bid = None
        if bid is not None:
            rid = by_booking.get(bid)
        if rid is None:
            pid = getattr(b, "parent_booking_id", None)
            if pid is not None:
                try:
                    rid = by_booking.get(int(pid))
                except (TypeError, ValueError):
                    rid = None
        if rid is None:
            rg = getattr(b, "route_group_id", None)
            if rg:
                rid = by_rg.get(str(rg))
        b._invoice_request_id = rid
        b._invoice_request_resolved = True


def reopen_market_lirie_validation_after_financial_change(booking: Any) -> bool:
    """Réouvre le contrôle si une course Market LIRIE validée change financièrement.

    Ne touche pas ANOMALY (contestation conservée). Retourne True si réouvert.
    """
    if not is_market_lirie_booking(booking):
        return False
    persisted = persisted_control_status(booking)
    if persisted != InstitutionBillingControlStatus.VALIDATED.value:
        return False
    from application.institutions.billing_control.status import (
        reset_control_after_payer_correction,
    )

    reset_control_after_payer_correction(booking)
    return True


@dataclass(frozen=True, slots=True)
class EligibilitySummary:
    eligible_count: int
    eligible_amount_ht: float
    origin_own_portfolio: int
    origin_market_lirie: int
    market_validated: int
    market_auto_released: int
    market_pending: int
    market_disputed: int
    excluded_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_count": self.eligible_count,
            "eligible_amount_ht": self.eligible_amount_ht,
            "origin": {
                "own_portfolio": self.origin_own_portfolio,
                "market_lirie": self.origin_market_lirie,
            },
            "market_lirie": {
                "validated": self.market_validated,
                "auto_released": self.market_auto_released,
                "pending": self.market_pending,
                "disputed": self.market_disputed,
            },
            "excluded_count": self.excluded_count,
        }


def build_eligibility_summary(
    considered: list[Any],
    eligible: list[Any],
    *,
    now: datetime | None = None,
    amount_ht_fn: Any | None = None,
) -> EligibilitySummary:
    own = market = 0
    validated = auto_rel = pending = disputed = 0
    for b in considered:
        origin = resolve_commercial_origin(b)
        if origin == ORIGIN_MARKET_LIRIE:
            market += 1
            gate = invoice_gate_status(b, now=now)
            if gate == "validated":
                validated += 1
            elif gate == "auto_released":
                auto_rel += 1
            elif gate == "pending":
                pending += 1
            elif gate == "disputed":
                disputed += 1
        else:
            own += 1

    amount = 0.0
    if amount_ht_fn is not None:
        for b in eligible:
            try:
                amount += float(amount_ht_fn(b))
            except (TypeError, ValueError):
                continue
        amount = round(amount, 2)

    return EligibilitySummary(
        eligible_count=len(eligible),
        eligible_amount_ht=amount,
        origin_own_portfolio=own,
        origin_market_lirie=market,
        market_validated=validated,
        market_auto_released=auto_rel,
        market_pending=pending,
        market_disputed=disputed,
        excluded_count=max(0, len(considered) - len(eligible)),
    )


@dataclass
class PayerBucket:
    payer_type: str
    key: str
    display_name: str
    transports_count: int
    estimated_total: float
    client_id: int | None = None
    institution_patient_id: int | None = None
    billing_party_id: int | None = None
    clinic_company_id: int | None = None
    booking_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "payer_type": self.payer_type,
            "key": self.key,
            "display_name": self.display_name,
            "transports_count": self.transports_count,
            "estimated_total": self.estimated_total,
            "client_id": self.client_id,
            "institution_patient_id": self.institution_patient_id,
            "billing_party_id": self.billing_party_id,
            "clinic_company_id": self.clinic_company_id,
            "booking_ids": list(self.booking_ids),
        }


@dataclass
class InstitutionInvoicePlan:
    period_year: int
    period_month: int
    clinic_company_id: int | None
    eligibility: EligibilitySummary
    clinic: PayerBucket | None
    patients: list[PayerBucket] = field(default_factory=list)
    partners: list[PayerBucket] = field(default_factory=list)
    reconciliation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_year": self.period_year,
            "period_month": self.period_month,
            "clinic_company_id": self.clinic_company_id,
            "eligibility": self.eligibility.to_dict(),
            "clinic": self.clinic.to_dict() if self.clinic else None,
            "patients": [p.to_dict() for p in self.patients],
            "partners": [p.to_dict() for p in self.partners],
            "reconciliation": self.reconciliation,
        }
