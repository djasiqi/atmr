"""Revendications actives InvoiceLine → Booking (défense BUG B).

Un booking est déjà couvert si une InvoiceLine d'une facture bloquante
le revendique explicitement via ``covered_booking_ids(line)``, même lorsque
``Booking.invoice_line_id`` est NULL (FK corrompue / partielle).

Une jambe volontairement libérée (absente de covered) reste éligible.
Aucune heuristique d'adresse / DSU / subject_identity ici.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Integer, bindparam, cast, or_, text

from application.invoices.invoice_line_booking_integrity import covered_booking_ids
from ext import db
from models import Booking, Invoice, InvoiceLine
from models.enums import InvoiceLineType, InvoiceStatus

# Statuts qui officialisent une claim (partagés avec le filet FK propre).
BLOCKING_INVOICE_STATUSES_FOR_CLAIM: frozenset[InvoiceStatus] = frozenset(
    {
        InvoiceStatus.DRAFT,
        InvoiceStatus.SENT,
        InvoiceStatus.PARTIALLY_PAID,
        InvoiceStatus.PAID,
        InvoiceStatus.OVERDUE,
    }
)
_BLOCKING_STATUS_VALUES: frozenset[str] = frozenset(
    s.value.upper() for s in BLOCKING_INVOICE_STATUSES_FOR_CLAIM
)


@dataclass(frozen=True, slots=True)
class BlockingInvoiceClaim:
    """Claim bloquante d'une InvoiceLine active sur un booking."""

    invoice_line_id: int
    invoice_id: int
    invoice_status: str
    claim_source: str
    claim_count: int = 1


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status) or "")


def _is_blocking_status(status: Any) -> bool:
    if status in BLOCKING_INVOICE_STATUSES_FOR_CLAIM:
        return True
    return _status_value(status).upper() in _BLOCKING_STATUS_VALUES


def _meta_contains_id(raw: Any, booking_id: int) -> bool:
    if raw is None:
        return False
    if isinstance(raw, (list, tuple, set, frozenset)):
        for item in raw:
            if _meta_contains_id(item, booking_id):
                return True
        return False
    try:
        return int(raw) == int(booking_id)
    except (TypeError, ValueError):
        return False


def claim_source_for_booking(line: Any, booking_id: int) -> str:
    """Identifie la clé explicite qui inclut ``booking_id`` dans covered.

    Diagnostic uniquement — la décision d'éligibilité reste ``covered_booking_ids``.
    """
    bid = int(booking_id)
    rid = getattr(line, "reservation_id", None)
    try:
        if rid is not None and int(rid) == bid:
            return "reservation_id"
    except (TypeError, ValueError):
        pass
    meta = getattr(line, "line_meta", None)
    if not isinstance(meta, dict):
        return "unknown"
    for key in (
        "booking_ids",
        "reservation_ids",
        "round_trip_secondary_reservation_ids",
        "round_trip_secondary_reservation_id",
        "round_trip_merge_partner_reservation_id",
        "round_trip_merge_primary_reservation_id",
    ):
        if key in meta and _meta_contains_id(meta.get(key), bid):
            return key
    return "unknown"


def _company_ids_for_candidates(
    candidate_ids: set[int],
    context_bookings: list[Any] | None,
) -> set[int]:
    """Entreprises des candidats (jamais cross-tenant)."""
    company_ids: set[int] = set()
    if context_bookings:
        for b in context_bookings:
            cid = getattr(b, "company_id", None)
            if cid is not None:
                try:
                    company_ids.add(int(cid))
                except (TypeError, ValueError):
                    continue
    missing = candidate_ids
    if context_bookings:
        known = set()
        for b in context_bookings:
            try:
                known.add(int(b.id))
            except (TypeError, ValueError):
                continue
        missing = candidate_ids - known
    if not company_ids or missing:
        seed_ids = sorted(candidate_ids if not company_ids else missing)
        if seed_ids:
            for row in (
                Booking.query.filter(Booking.id.in_(seed_ids))
                .with_entities(Booking.company_id)
                .distinct()
                .all()
            ):
                if row.company_id is not None:
                    company_ids.add(int(row.company_id))
    return company_ids


def expand_explicit_claim_context_ids(
    candidate_booking_ids: Iterable[int],
    *,
    company_ids: set[int] | None = None,
) -> set[int]:
    """Contexte explicite pour retrouver des InvoiceLine (pas une unité facturable)."""
    candidates = {int(x) for x in candidate_booking_ids}
    if not candidates:
        return set()

    context = set(candidates)
    seed_q = Booking.query.filter(Booking.id.in_(sorted(candidates)))
    if company_ids:
        seed_q = seed_q.filter(Booking.company_id.in_(sorted(company_ids)))
    seed = seed_q.with_entities(
        Booking.id,
        Booking.parent_booking_id,
        Booking.route_group_id,
        Booking.company_id,
    ).all()

    parent_ids: set[int] = set()
    route_groups: set[str] = set()
    for row in seed:
        pid = row.parent_booking_id
        if pid is not None:
            parent_ids.add(int(pid))
            context.add(int(pid))
        rg = row.route_group_id
        if rg:
            route_groups.add(str(rg))

    if parent_ids:
        pq = Booking.query.filter(Booking.id.in_(sorted(parent_ids)))
        if company_ids:
            pq = pq.filter(Booking.company_id.in_(sorted(company_ids)))
        for prow in pq.with_entities(Booking.route_group_id).all():
            if prow.route_group_id:
                route_groups.add(str(prow.route_group_id))

    # Enfants des candidats / parents (même tenant)
    cq = Booking.query.filter(Booking.parent_booking_id.in_(sorted(context)))
    if company_ids:
        cq = cq.filter(Booking.company_id.in_(sorted(company_ids)))
    for crow in cq.with_entities(Booking.id, Booking.route_group_id).all():
        context.add(int(crow.id))
        if crow.route_group_id:
            route_groups.add(str(crow.route_group_id))

    if route_groups:
        rq = Booking.query.filter(Booking.route_group_id.in_(sorted(route_groups)))
        if company_ids:
            rq = rq.filter(Booking.company_id.in_(sorted(company_ids)))
        for peer in rq.with_entities(Booking.id).all():
            context.add(int(peer.id))

    return context


def _load_candidate_claim_lines(
    context_ids: set[int],
    *,
    company_ids: set[int],
) -> list[tuple[InvoiceLine, Invoice]]:
    """Charge les lignes RIDE sur factures bloquantes du même tenant."""
    if not context_ids or not company_ids:
        # Sans company_id : fail-closed (aucun claim DB) pour éviter le cross-tenant.
        return []
    ids = sorted(context_ids)
    meta = InvoiceLine.line_meta
    partner = cast(meta["round_trip_merge_partner_reservation_id"].astext, Integer)
    primary = cast(meta["round_trip_merge_primary_reservation_id"].astext, Integer)
    secondary = cast(meta["round_trip_secondary_reservation_id"].astext, Integer)

    # Arrays JSON : overlap via SQL (batch, pas N+1).
    array_overlap = text(
        "("
        "EXISTS ("
        "  SELECT 1 FROM jsonb_array_elements_text("
        "    COALESCE(invoice_lines.line_meta->'booking_ids', '[]'::jsonb)"
        "  ) AS e(val) WHERE NULLIF(e.val, '')::int IN :ctx_ids"
        ") OR EXISTS ("
        "  SELECT 1 FROM jsonb_array_elements_text("
        "    COALESCE(invoice_lines.line_meta->'reservation_ids', '[]'::jsonb)"
        "  ) AS e(val) WHERE NULLIF(e.val, '')::int IN :ctx_ids"
        ") OR EXISTS ("
        "  SELECT 1 FROM jsonb_array_elements_text("
        "    COALESCE("
        "      invoice_lines.line_meta->'round_trip_secondary_reservation_ids',"
        "      '[]'::jsonb"
        "    )"
        "  ) AS e(val) WHERE NULLIF(e.val, '')::int IN :ctx_ids"
        ")"
        ")"
    ).bindparams(bindparam("ctx_ids", expanding=True))

    rows = (
        db.session.query(InvoiceLine, Invoice)
        .join(Invoice, Invoice.id == InvoiceLine.invoice_id)
        .filter(
            Invoice.company_id.in_(sorted(company_ids)),
            Invoice.status.in_(tuple(BLOCKING_INVOICE_STATUSES_FOR_CLAIM)),
            InvoiceLine.type == InvoiceLineType.RIDE,
            or_(
                InvoiceLine.reservation_id.in_(ids),
                partner.in_(ids),
                primary.in_(ids),
                secondary.in_(ids),
                array_overlap.bindparams(ctx_ids=ids),
            ),
        )
        .all()
    )
    return list(rows)


def _invoice_company_id(invoice: Any) -> int | None:
    cid = getattr(invoice, "company_id", None)
    if cid is None:
        return None
    try:
        return int(cid)
    except (TypeError, ValueError):
        return None


def _claim_from_pair(line: Any, invoice: Any, booking_id: int) -> BlockingInvoiceClaim | None:
    try:
        lid = int(getattr(line, "id"))
        iid = int(getattr(invoice, "id"))
    except (TypeError, ValueError):
        return None
    return BlockingInvoiceClaim(
        invoice_line_id=lid,
        invoice_id=iid,
        invoice_status=_status_value(getattr(invoice, "status", None)),
        claim_source=claim_source_for_booking(line, booking_id),
        claim_count=1,
    )


def find_all_blocking_invoice_claims(
    candidate_booking_ids: Iterable[int],
    *,
    context_bookings: list[Any] | None = None,
    invoice_lines_with_invoices: list[tuple[Any, Any]] | None = None,
) -> dict[int, tuple[BlockingInvoiceClaim, ...]]:
    """Map ``booking_id → toutes les claims actives`` (diagnostic multi-claim)."""
    candidates = {int(x) for x in candidate_booking_ids}
    if not candidates:
        return {}

    if invoice_lines_with_invoices is not None:
        # Injection tests / caller : pas de lookup DB company.
        company_ids: set[int] = set()
        if context_bookings:
            for b in context_bookings:
                cid = getattr(b, "company_id", None)
                if cid is not None:
                    try:
                        company_ids.add(int(cid))
                    except (TypeError, ValueError):
                        continue
        pairs = invoice_lines_with_invoices
    else:
        company_ids = _company_ids_for_candidates(candidates, context_bookings)
        context_ids = set(candidates)
        if context_bookings:
            for b in context_bookings:
                try:
                    context_ids.add(int(b.id))
                except (TypeError, ValueError):
                    continue
                pid = getattr(b, "parent_booking_id", None)
                if pid is not None:
                    try:
                        context_ids.add(int(pid))
                    except (TypeError, ValueError):
                        pass
        context_ids |= expand_explicit_claim_context_ids(
            context_ids, company_ids=company_ids or None
        )
        pairs = _load_candidate_claim_lines(context_ids, company_ids=company_ids)

    by_booking: dict[int, list[BlockingInvoiceClaim]] = defaultdict(list)
    for line, invoice in pairs:
        status = getattr(invoice, "status", None)
        if not _is_blocking_status(status):
            continue
        inv_cid = _invoice_company_id(invoice)
        if company_ids:
            if inv_cid is None or inv_cid not in company_ids:
                continue
        covered = covered_booking_ids(line)
        for bid in covered & candidates:
            claim = _claim_from_pair(line, invoice, bid)
            if claim is None:
                continue
            # Éviter doublons exacts même ligne
            existing_lids = {c.invoice_line_id for c in by_booking[bid]}
            if claim.invoice_line_id in existing_lids:
                continue
            by_booking[bid].append(claim)

    return {bid: tuple(claims) for bid, claims in by_booking.items()}


def find_blocking_invoice_claims(
    candidate_booking_ids: Iterable[int],
    *,
    context_bookings: list[Any] | None = None,
    invoice_lines_with_invoices: list[tuple[Any, Any]] | None = None,
) -> dict[int, BlockingInvoiceClaim]:
    """Map ``booking_id → claim`` (représentative) ; ``claim_count`` si multi-claim.

    Un booking avec ≥1 claim active reste toujours bloqué (jamais ouvert par
    sélection arbitraire d'une ligne).
    """
    all_claims = find_all_blocking_invoice_claims(
        candidate_booking_ids,
        context_bookings=context_bookings,
        invoice_lines_with_invoices=invoice_lines_with_invoices,
    )
    out: dict[int, BlockingInvoiceClaim] = {}
    for bid, claims in all_claims.items():
        if not claims:
            continue
        first = claims[0]
        out[bid] = BlockingInvoiceClaim(
            invoice_line_id=first.invoice_line_id,
            invoice_id=first.invoice_id,
            invoice_status=first.invoice_status,
            claim_source=first.claim_source,
            claim_count=len(claims),
        )
    return out


def booking_has_active_invoice_claim(
    booking_id: int,
    *,
    context_bookings: list[Any] | None = None,
    invoice_lines_with_invoices: list[tuple[Any, Any]] | None = None,
) -> bool:
    """True si une facture bloquante revendique explicitement ce booking."""
    return int(booking_id) in find_blocking_invoice_claims(
        {int(booking_id)},
        context_bookings=context_bookings,
        invoice_lines_with_invoices=invoice_lines_with_invoices,
    )


def filter_bookings_without_active_invoice_claim(
    bookings: list[Any],
    *,
    invoice_lines_with_invoices: list[tuple[Any, Any]] | None = None,
) -> list[Any]:
    """Retire les bookings explicitement revendiqués par une facture active."""
    if not bookings:
        return []
    candidate_ids = []
    for b in bookings:
        try:
            candidate_ids.append(int(b.id))
        except (TypeError, ValueError):
            continue
    claims = find_blocking_invoice_claims(
        candidate_ids,
        context_bookings=bookings,
        invoice_lines_with_invoices=invoice_lines_with_invoices,
    )
    if not claims:
        return list(bookings)
    out: list[Any] = []
    for b in bookings:
        try:
            bid = int(b.id)
        except (TypeError, ValueError):
            continue
        if bid in claims:
            continue
        out.append(b)
    return out


def filter_bookings_open_and_unclaimed(
    bookings: list[Any],
    *,
    invoice_lines_with_invoices: list[tuple[Any, Any]] | None = None,
) -> list[Any]:
    """Ouverts sur leur propre FK + non revendiqués par une claim active."""
    if not bookings:
        return []
    from application.invoices.round_trip_billing_lock import (
        booking_open_for_new_invoice_line,
    )

    own_open = [b for b in bookings if booking_open_for_new_invoice_line(b)]
    return filter_bookings_without_active_invoice_claim(
        own_open,
        invoice_lines_with_invoices=invoice_lines_with_invoices,
    )
