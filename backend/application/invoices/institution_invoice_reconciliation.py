"""Réconciliation comptable du plan de facturation institution.

Chaque prestation de la période appartient à **exactement un** seau :

- facturable clinique / patient / partenaire
- PENDING bloquée
- DISPUTED bloquée
- exclusion explicite (déjà facturée, claim actif, autre)

Aucune disparition, aucun doublon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from application.invoices.institution_invoice_eligibility import (
    invoice_gate_status,
    is_institution_invoice_eligible,
    resolve_commercial_origin,
    resolve_invoice_payer_type,
)

BucketKind = Literal[
    "clinic_billable",
    "patient_billable",
    "partner_billable",
    "pending_blocked",
    "disputed_blocked",
    "already_invoiced",
    "other_excluded",
]


def grouping_relation(booking: Any) -> tuple[str | None, str | None]:
    """Relation métier A/R (jamais patient+date)."""
    pid = getattr(booking, "parent_booking_id", None)
    if pid is not None:
        return f"parent_booking_id:{int(pid)}", "parent_booking_id"
    rid = getattr(booking, "_invoice_request_id", None)
    if rid is not None:
        return f"request_id:{int(rid)}", "request_id"
    rg = getattr(booking, "route_group_id", None)
    if rg:
        return f"route_group_id:{rg}", "route_group_id"
    return None, None


def _exclusion_reason(
    *,
    gate: str,
    eligible: bool,
    invoice_line_id: int | None,
) -> str | None:
    if invoice_line_id is not None:
        return "already_invoiced"
    if gate == "not_billable":
        return "resolved_institution_not_billable"
    if gate == "disputed":
        return "market_disputed"
    if gate == "pending":
        return "market_pending_before_deadline"
    if eligible:
        return None
    if gate == "auto_released":
        return "not_eligible_after_auto_release"
    return f"not_eligible:{gate}"


def classify_booking_bucket(
    booking: Any,
    *,
    now: datetime | None = None,
) -> tuple[BucketKind, str | None]:
    """Seau unique + raison d'exclusion éventuelle."""
    try:
        line_id = (
            int(booking.invoice_line_id)
            if getattr(booking, "invoice_line_id", None) is not None
            else None
        )
    except (TypeError, ValueError):
        line_id = None
    gate = invoice_gate_status(booking, now=now)
    eligible = is_institution_invoice_eligible(booking, now=now)
    if line_id is not None:
        return "already_invoiced", "already_invoiced"
    if gate == "not_billable":
        return "other_excluded", "resolved_institution_not_billable"
    if gate == "disputed":
        return "disputed_blocked", "market_disputed"
    if gate == "pending":
        return "pending_blocked", "market_pending_before_deadline"
    if eligible:
        payer = resolve_invoice_payer_type(booking)
        if payer == "clinic":
            return "clinic_billable", None
        if payer == "patient":
            return "patient_billable", None
        if payer == "partner":
            return "partner_billable", None
        return "other_excluded", "payer_unresolved"
    return "other_excluded", _exclusion_reason(
        gate=gate, eligible=False, invoice_line_id=line_id
    )


@dataclass(frozen=True, slots=True)
class BookingInvoiceAudit:
    booking_id: int
    origin: str
    validation_status: str
    persisted_control_status: str | None
    payer: str
    eligible: bool
    invoice_bucket: str
    group_id: str | None
    grouping_relation: str | None
    invoice_line_id: int | None
    amount_ht: float
    exclusion_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "booking_id": self.booking_id,
            "origin": self.origin,
            "validation_status": self.validation_status,
            "persisted_control_status": self.persisted_control_status,
            "payer": self.payer,
            "eligible": self.eligible,
            "invoice_bucket": self.invoice_bucket,
            "group_id": self.group_id,
            "grouping_relation": self.grouping_relation,
            "invoice_line_id": self.invoice_line_id,
            "amount_ht": self.amount_ht,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass
class BucketTotals:
    count: int = 0
    amount_ht: float = 0.0
    booking_ids: list[int] = field(default_factory=list)

    def add(self, booking_id: int, amount: float) -> None:
        self.count += 1
        self.amount_ht = round(self.amount_ht + amount, 2)
        self.booking_ids.append(int(booking_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "amount_ht": self.amount_ht,
            "booking_ids": list(self.booking_ids),
        }


@dataclass
class ReconciliationLedger:
    period_year: int
    period_month: int
    as_of: str
    considered_count: int
    considered_amount_ht: float
    buckets: dict[str, BucketTotals]
    bookings: list[BookingInvoiceAudit]
    conservation_ok: bool
    duplicate_booking_ids: list[int]
    missing_from_buckets: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_year": self.period_year,
            "period_month": self.period_month,
            "as_of": self.as_of,
            "considered_count": self.considered_count,
            "considered_amount_ht": self.considered_amount_ht,
            "buckets": {k: v.to_dict() for k, v in self.buckets.items()},
            "bookings": [a.to_dict() for a in self.bookings],
            "conservation_ok": self.conservation_ok,
            "duplicate_booking_ids": self.duplicate_booking_ids,
            "missing_from_buckets": self.missing_from_buckets,
        }


_BUCKET_KEYS: tuple[BucketKind, ...] = (
    "clinic_billable",
    "patient_billable",
    "partner_billable",
    "pending_blocked",
    "disputed_blocked",
    "already_invoiced",
    "other_excluded",
)


def build_reconciliation_ledger(
    bookings: list[Any],
    *,
    period_year: int,
    period_month: int,
    now: datetime | None = None,
    amount_ht_fn: Any | None = None,
) -> ReconciliationLedger:
    """Classe chaque booking et vérifie la conservation."""
    from application.institutions.billing_control.status import persisted_control_status
    from application.invoices.institution_invoice_eligibility import ZURICH

    clock = now
    if clock is None:
        clock = datetime.now(ZURICH)
    elif clock.tzinfo is None:
        clock = clock.replace(tzinfo=ZURICH)
    else:
        clock = clock.astimezone(ZURICH)
    as_of = clock.isoformat()
    buckets: dict[str, BucketTotals] = {k: BucketTotals() for k in _BUCKET_KEYS}
    audits: list[BookingInvoiceAudit] = []
    seen: set[int] = set()
    duplicates: list[int] = []
    considered_amount = 0.0

    for b in bookings:
        try:
            bid = int(b.id)
        except (TypeError, ValueError):
            continue
        if bid in seen:
            duplicates.append(bid)
            continue
        seen.add(bid)

        if amount_ht_fn is not None:
            try:
                amount = round(float(amount_ht_fn(b)), 2)
            except (TypeError, ValueError):
                amount = 0.0
        else:
            try:
                amount = round(float(getattr(b, "amount", 0) or 0), 2)
            except (TypeError, ValueError):
                amount = 0.0
        considered_amount = round(considered_amount + amount, 2)

        gate = invoice_gate_status(b, now=now)
        eligible = is_institution_invoice_eligible(b, now=now)
        bucket, reason = classify_booking_bucket(b, now=now)
        group_id, rel = grouping_relation(b)
        try:
            line_id = (
                int(b.invoice_line_id)
                if getattr(b, "invoice_line_id", None) is not None
                else None
            )
        except (TypeError, ValueError):
            line_id = None
        persisted = persisted_control_status(b)
        audits.append(
            BookingInvoiceAudit(
                booking_id=bid,
                origin=resolve_commercial_origin(b),
                validation_status=gate,
                persisted_control_status=persisted,
                payer=resolve_invoice_payer_type(b),
                eligible=eligible,
                invoice_bucket=bucket,
                group_id=group_id,
                grouping_relation=rel,
                invoice_line_id=line_id,
                amount_ht=amount,
                exclusion_reason=reason,
            )
        )
        buckets[bucket].add(bid, amount)

    bucket_ids = [i for tot in buckets.values() for i in tot.booking_ids]
    missing = sorted(seen - set(bucket_ids))
    bucket_sum = round(sum(t.amount_ht for t in buckets.values()), 2)
    conservation_ok = (
        not duplicates
        and not missing
        and len(bucket_ids) == len(seen)
        and bucket_sum == considered_amount
        and len(audits) == len(seen)
    )
    audits.sort(key=lambda a: a.booking_id)
    return ReconciliationLedger(
        period_year=period_year,
        period_month=period_month,
        as_of=as_of,
        considered_count=len(seen),
        considered_amount_ht=considered_amount,
        buckets=buckets,
        bookings=audits,
        conservation_ok=conservation_ok,
        duplicate_booking_ids=duplicates,
        missing_from_buckets=missing,
    )
