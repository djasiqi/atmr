"""Intégrité des rattachements InvoiceLine ↔ Booking (A/R inclus).

Invariant BILLING-INTEGRITY-G2 :
une InvoiceLine RIDE ne peut pas revendiquer un booking
sans que ``Booking.invoice_line_id`` pointe vers cette même ligne.

Formats supportés (explicites uniquement, pas d'heuristique adresse) :
- ``reservation_id``
- ``line_meta.booking_ids`` / ``reservation_ids``
- ``round_trip_secondary_reservation_id(s)``
- ``round_trip_merge_partner_reservation_id``
- ``round_trip_merge_primary_reservation_id``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models.enums import InvoiceLineType

ERROR_CODE_INVOICE_LINE_LINK_INCOMPLETE = "BILLING_INVOICE_LINE_LINK_INCOMPLETE"

_META_BOOKING_ID_KEYS: tuple[str, ...] = (
    "booking_ids",
    "reservation_ids",
    "round_trip_secondary_reservation_ids",
    "round_trip_secondary_reservation_id",
    "round_trip_merge_partner_reservation_id",
    "round_trip_merge_primary_reservation_id",
)


def _as_int_ids(value: Any, *, into: set[int]) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _as_int_ids(item, into=into)
        return
    try:
        into.add(int(value))
    except (TypeError, ValueError):
        return


def covered_booking_ids(line: Any) -> set[int]:
    """Bookings explicitement revendiqués par une ligne de facture.

    Interprète uniquement reservation_id + clés meta A/R connues.
    """
    out: set[int] = set()
    _as_int_ids(getattr(line, "reservation_id", None), into=out)
    meta = getattr(line, "line_meta", None)
    if isinstance(meta, dict):
        for key in _META_BOOKING_ID_KEYS:
            if key in meta:
                _as_int_ids(meta.get(key), into=out)
    return out


@dataclass(frozen=True, slots=True)
class InvoiceLineBookingLinkIssue:
    """Écart d'intégrité pour une ligne RIDE."""

    invoice_line_id: int | None
    reservation_id: int | None
    covered_booking_ids: tuple[int, ...]
    linked_correctly: tuple[int, ...]
    null_link_booking_ids: tuple[int, ...]
    wrong_line_booking_ids: tuple[int, ...]
    missing_booking_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class InvoiceBookingLinkIntegrityResult:
    """Résultat du checker read-only."""

    ok: bool
    invoice_id: int | None = None
    line_issues: tuple[InvoiceLineBookingLinkIssue, ...] = field(default_factory=tuple)

    def to_error_details(self) -> dict[str, Any]:
        first = self.line_issues[0] if self.line_issues else None
        incorrect: list[int] = []
        expected: list[int] = []
        if first is not None:
            incorrect = sorted(
                set(first.null_link_booking_ids)
                | set(first.wrong_line_booking_ids)
                | set(first.missing_booking_ids)
            )
            expected = list(first.covered_booking_ids)
        return {
            "error_code": ERROR_CODE_INVOICE_LINE_LINK_INCOMPLETE,
            "invoice_id": self.invoice_id,
            "invoice_line_id": first.invoice_line_id if first else None,
            "expected_booking_ids": expected,
            "incorrect_booking_ids": incorrect,
            "line_issues": [
                {
                    "invoice_line_id": issue.invoice_line_id,
                    "reservation_id": issue.reservation_id,
                    "covered_booking_ids": list(issue.covered_booking_ids),
                    "linked_correctly": list(issue.linked_correctly),
                    "null_link_booking_ids": list(issue.null_link_booking_ids),
                    "wrong_line_booking_ids": list(issue.wrong_line_booking_ids),
                    "missing_booking_ids": list(issue.missing_booking_ids),
                }
                for issue in self.line_issues
            ],
        }


class InvoiceBookingLinkIncompleteError(Exception):
    """Fail-closed : ligne A/R (ou multi-booking) partiellement rattachée."""

    def __init__(self, result: InvoiceBookingLinkIntegrityResult) -> None:
        self.result = result
        super().__init__(
            "Liaisons InvoiceLine/Booking incomplètes "
            f"(invoice_id={result.invoice_id}, issues={len(result.line_issues)})"
        )

    @property
    def error_code(self) -> str:
        return ERROR_CODE_INVOICE_LINE_LINK_INCOMPLETE

    def to_error_payload(self) -> dict[str, Any]:
        details = self.result.to_error_details()
        return {
            "error": (
                "Impossible de créer ou d'émettre la facture : "
                "une ligne revendique des transports non rattachés."
            ),
            "error_code": self.error_code,
            "details": details,
            "invoice_id": details.get("invoice_id"),
            "invoice_line_id": details.get("invoice_line_id"),
            "expected_booking_ids": details.get("expected_booking_ids"),
            "incorrect_booking_ids": details.get("incorrect_booking_ids"),
        }


def _line_type_value(line: Any) -> str | None:
    raw = getattr(line, "type", None)
    if raw is None:
        return None
    if hasattr(raw, "value"):
        return str(raw.value)
    return str(raw)


def _is_ride_line(line: Any) -> bool:
    return _line_type_value(line) == InvoiceLineType.RIDE.value


def _inspect_line_against_bookings(
    line: Any,
    bookings_by_id: dict[int, Any],
) -> InvoiceLineBookingLinkIssue | None:
    if not _is_ride_line(line):
        return None
    covered = covered_booking_ids(line)
    if not covered:
        return None

    line_id = getattr(line, "id", None)
    try:
        line_id_int = int(line_id) if line_id is not None else None
    except (TypeError, ValueError):
        line_id_int = None

    linked_ok: list[int] = []
    null_link: list[int] = []
    wrong_line: list[int] = []
    missing: list[int] = []

    for bid in sorted(covered):
        booking = bookings_by_id.get(bid)
        if booking is None:
            missing.append(bid)
            continue
        linked = getattr(booking, "invoice_line_id", None)
        if linked is None:
            null_link.append(bid)
            continue
        try:
            linked_int = int(linked)
        except (TypeError, ValueError):
            wrong_line.append(bid)
            continue
        if line_id_int is not None and linked_int == line_id_int:
            linked_ok.append(bid)
        else:
            wrong_line.append(bid)

    if not null_link and not wrong_line and not missing:
        return None

    rid = getattr(line, "reservation_id", None)
    try:
        rid_int = int(rid) if rid is not None else None
    except (TypeError, ValueError):
        rid_int = None

    return InvoiceLineBookingLinkIssue(
        invoice_line_id=line_id_int,
        reservation_id=rid_int,
        covered_booking_ids=tuple(sorted(covered)),
        linked_correctly=tuple(linked_ok),
        null_link_booking_ids=tuple(null_link),
        wrong_line_booking_ids=tuple(wrong_line),
        missing_booking_ids=tuple(missing),
    )


def check_invoice_lines_booking_link_integrity(
    lines: list[Any] | tuple[Any, ...],
    *,
    bookings_by_id: dict[int, Any],
    invoice_id: int | None = None,
) -> InvoiceBookingLinkIntegrityResult:
    """Checker read-only sur une liste de lignes + bookings déjà chargés."""
    issues: list[InvoiceLineBookingLinkIssue] = []
    for line in lines:
        issue = _inspect_line_against_bookings(line, bookings_by_id)
        if issue is not None:
            issues.append(issue)
    return InvoiceBookingLinkIntegrityResult(
        ok=not issues,
        invoice_id=invoice_id,
        line_issues=tuple(issues),
    )


def check_invoice_booking_link_integrity(
    invoice: Any,
    *,
    bookings_by_id: dict[int, Any] | None = None,
) -> InvoiceBookingLinkIntegrityResult:
    """Checker read-only pour une facture (charge les bookings si besoin)."""
    lines = list(getattr(invoice, "lines", None) or [])
    invoice_id = getattr(invoice, "id", None)
    try:
        invoice_id_int = int(invoice_id) if invoice_id is not None else None
    except (TypeError, ValueError):
        invoice_id_int = None

    if bookings_by_id is None:
        covered_all: set[int] = set()
        for line in lines:
            if _is_ride_line(line):
                covered_all |= covered_booking_ids(line)
        bookings_by_id = {}
        if covered_all:
            from models.booking import Booking

            for booking in Booking.query.filter(
                Booking.id.in_(sorted(covered_all))
            ).all():
                bookings_by_id[int(booking.id)] = booking

    return check_invoice_lines_booking_link_integrity(
        lines,
        bookings_by_id=bookings_by_id,
        invoice_id=invoice_id_int,
    )


def assert_invoice_booking_link_integrity(
    invoice: Any,
    *,
    bookings_by_id: dict[int, Any] | None = None,
) -> None:
    """Fail-closed : lève si une ligne revendique un booking non rattaché."""
    result = check_invoice_booking_link_integrity(
        invoice, bookings_by_id=bookings_by_id
    )
    if not result.ok:
        raise InvoiceBookingLinkIncompleteError(result)


def assert_invoice_lines_booking_link_integrity(
    lines: list[Any] | tuple[Any, ...],
    *,
    bookings_by_id: dict[int, Any],
    invoice_id: int | None = None,
) -> None:
    """Fail-closed sur lignes + bookings déjà en mémoire (génération)."""
    result = check_invoice_lines_booking_link_integrity(
        lines,
        bookings_by_id=bookings_by_id,
        invoice_id=invoice_id,
    )
    if not result.ok:
        raise InvoiceBookingLinkIncompleteError(result)
