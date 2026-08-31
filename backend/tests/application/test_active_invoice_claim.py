"""Tests BILLING-ELIGIBILITY-RT-01 : claim active vs BUG B."""

from __future__ import annotations

from types import SimpleNamespace

from application.invoices.active_invoice_claim import (
    booking_has_active_invoice_claim,
    filter_bookings_without_active_invoice_claim,
    find_blocking_invoice_claims,
)
from application.invoices.invoice_line_booking_integrity import covered_booking_ids
from models.enums import InvoiceLineType, InvoiceStatus


def _booking(
    bid: int,
    *,
    invoice_line_id: int | None = None,
    parent_booking_id: int | None = None,
    route_group_id: str | None = None,
    is_return: bool = False,
    company_id: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=bid,
        invoice_line_id=invoice_line_id,
        parent_booking_id=parent_booking_id,
        route_group_id=route_group_id,
        is_return=is_return,
        company_id=company_id,
    )


def _line(
    lid: int,
    *,
    reservation_id: int | None,
    meta: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=lid,
        type=InvoiceLineType.RIDE,
        reservation_id=reservation_id,
        line_meta=meta,
    )


def _inv(
    iid: int,
    status: InvoiceStatus | str,
    *,
    company_id: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(id=iid, status=status, company_id=company_id)


def _pairs(*items: tuple[SimpleNamespace, SimpleNamespace]):
    return list(items)


# ---------------------------------------------------------------------------
# Covered / claim helper
# ---------------------------------------------------------------------------


def test_t1_rca_historical_merge_partner_paid_blocks_orphan():
    """A=line, B=NULL, merge_partner=B, PAID → claim(B)=True."""
    a, b = 37127, 37128
    line = _line(
        4648,
        reservation_id=a,
        meta={"round_trip_merge_partner_reservation_id": b, "is_round_trip_leg": True},
    )
    inv = _inv(1773, InvoiceStatus.PAID)
    claims = find_blocking_invoice_claims(
        {a, b},
        invoice_lines_with_invoices=_pairs((line, inv)),
    )
    assert a in claims
    assert b in claims
    assert claims[b].claim_source == "round_trip_merge_partner_reservation_id"
    assert booking_has_active_invoice_claim(
        b, invoice_lines_with_invoices=_pairs((line, inv))
    )
    bookings = [
        _booking(a, invoice_line_id=4648),
        _booking(b, invoice_line_id=None, parent_booking_id=a),
    ]
    open_b = filter_bookings_without_active_invoice_claim(
        bookings, invoice_lines_with_invoices=_pairs((line, inv))
    )
    assert [x.id for x in open_b] == []


def test_t2_sent_blocks():
    line = _line(1, reservation_id=10, meta={"round_trip_merge_partner_reservation_id": 11})
    inv = _inv(2, InvoiceStatus.SENT)
    assert 11 in find_blocking_invoice_claims(
        {11}, invoice_lines_with_invoices=_pairs((line, inv))
    )


def test_t3_draft_blocks():
    line = _line(1, reservation_id=10, meta={"round_trip_merge_partner_reservation_id": 11})
    inv = _inv(2, InvoiceStatus.DRAFT)
    assert 11 in find_blocking_invoice_claims(
        {11}, invoice_lines_with_invoices=_pairs((line, inv))
    )


def test_t4_cancelled_does_not_block():
    line = _line(1, reservation_id=10, meta={"round_trip_merge_partner_reservation_id": 11})
    inv = _inv(2, InvoiceStatus.CANCELLED)
    claims = find_blocking_invoice_claims(
        {10, 11}, invoice_lines_with_invoices=_pairs((line, inv))
    )
    assert claims == {}


def test_t5_intentional_single_leg_keeps_return_open():
    line = _line(
        10,
        reservation_id=101,
        meta={
            "intentional_single_leg": True,
            "released_round_trip_booking_ids": [102],
            "booking_ids": [101],
            "billing_unit": "single",
        },
    )
    inv = _inv(1, InvoiceStatus.PAID)
    assert covered_booking_ids(line) == {101}
    claims = find_blocking_invoice_claims(
        {101, 102}, invoice_lines_with_invoices=_pairs((line, inv))
    )
    assert 101 in claims
    assert 102 not in claims
    open_b = filter_bookings_without_active_invoice_claim(
        [_booking(102, invoice_line_id=None)],
        invoice_lines_with_invoices=_pairs((line, inv)),
    )
    assert [x.id for x in open_b] == [102]


def test_t6_new_booking_ids_format_blocks_orphan():
    line = _line(
        10,
        reservation_id=101,
        meta={"booking_ids": [101, 102], "billing_unit": "round_trip"},
    )
    inv = _inv(1, InvoiceStatus.PAID)
    claims = find_blocking_invoice_claims(
        {102}, invoice_lines_with_invoices=_pairs((line, inv))
    )
    assert 102 in claims
    assert claims[102].claim_source == "booking_ids"


def test_t11_route_group_alone_does_not_block():
    """Même route_group sans claim explicite → second non bloqué."""
    # Aucune ligne fournie → pas de claim
    claims = find_blocking_invoice_claims(
        {201, 202},
        invoice_lines_with_invoices=[],
    )
    assert claims == {}
    open_b = filter_bookings_without_active_invoice_claim(
        [
            _booking(201, invoice_line_id=None, route_group_id="g1"),
            _booking(202, invoice_line_id=None, route_group_id="g1"),
        ],
        invoice_lines_with_invoices=[],
    )
    assert {x.id for x in open_b} == {201, 202}


def test_filter_open_and_unclaimed_via_round_trip_lock():
    from application.invoices.round_trip_billing_lock import (
        filter_bookings_open_for_new_invoice_line,
    )
    from application.invoices.active_invoice_claim import (
        filter_bookings_open_and_unclaimed,
    )

    line = _line(
        4648,
        reservation_id=37127,
        meta={"round_trip_merge_partner_reservation_id": 37128},
    )
    inv = _inv(1773, "PAID")
    # FK NULL uniquement : évite le lookup DB de booking_open_for_new_invoice_line.
    bookings = [
        _booking(37128, invoice_line_id=None, parent_booking_id=37127),
    ]
    out = filter_bookings_open_and_unclaimed(
        bookings, invoice_lines_with_invoices=_pairs((line, inv))
    )
    assert out == []
    assert filter_bookings_open_for_new_invoice_line([]) == []


def test_b12_rca_fixture_then_intentional_release():
    line = _line(
        4648,
        reservation_id=37127,
        meta={
            "round_trip_merge_partner_reservation_id": 37128,
            "amount_overridden": True,
        },
    )
    inv = _inv(1773, InvoiceStatus.PAID)
    pairs = _pairs((line, inv))
    assert booking_has_active_invoice_claim(
        37128, invoice_lines_with_invoices=pairs
    ) is True

    released = _line(
        4648,
        reservation_id=37127,
        meta={
            "intentional_single_leg": True,
            "released_round_trip_booking_ids": [37128],
            "booking_ids": [37127],
        },
    )
    pairs2 = _pairs((released, inv))
    assert booking_has_active_invoice_claim(
        37128, invoice_lines_with_invoices=pairs2
    ) is False


def test_t7_patient_preview_path_uses_open_filter_claim():
    """Même corruption : B absent via filter open+claim (chemin patient)."""
    from application.invoices.active_invoice_claim import (
        filter_bookings_open_and_unclaimed,
    )
    from application.invoices.round_trip_billing_lock import (
        filter_bookings_open_for_new_invoice_line,
    )

    line = _line(
        10,
        reservation_id=101,
        meta={"round_trip_merge_partner_reservation_id": 102},
    )
    inv = _inv(1, InvoiceStatus.SENT)
    bookings = [
        _booking(102, invoice_line_id=None, parent_booking_id=101),
    ]
    out = filter_bookings_open_and_unclaimed(
        bookings, invoice_lines_with_invoices=_pairs((line, inv))
    )
    assert [x.id for x in out] == []
    assert filter_bookings_open_for_new_invoice_line([]) == []


def test_t8_opportunity_counts_exclude_claimed_return():
    """B ne contribue ni au count ni au montant (filtre avant grouping)."""
    line = _line(
        10,
        reservation_id=101,
        meta={"booking_ids": [101, 102]},
    )
    inv = _inv(1, InvoiceStatus.PAID)
    candidates = [
        _booking(102, invoice_line_id=None),
        _booking(103, invoice_line_id=None),
    ]
    open_b = filter_bookings_without_active_invoice_claim(
        candidates, invoice_lines_with_invoices=_pairs((line, inv))
    )
    assert [x.id for x in open_b] == [103]


def test_t9_t10_generation_excludes_claimed_booking():
    """Génération patient/S2 : B revendiqué non générable."""
    line = _line(
        4648,
        reservation_id=37127,
        meta={"round_trip_merge_partner_reservation_id": 37128},
    )
    inv = _inv(1773, InvoiceStatus.PAID)
    gen_candidates = [_booking(37128, invoice_line_id=None)]
    assert (
        filter_bookings_without_active_invoice_claim(
            gen_candidates, invoice_lines_with_invoices=_pairs((line, inv))
        )
        == []
    )


def test_overdue_and_partially_paid_block():
    for status in (InvoiceStatus.OVERDUE, InvoiceStatus.PARTIALLY_PAID):
        line = _line(
            1, reservation_id=10, meta={"round_trip_merge_partner_reservation_id": 11}
        )
        inv = _inv(2, status)
        assert 11 in find_blocking_invoice_claims(
            {11}, invoice_lines_with_invoices=_pairs((line, inv))
        )


def test_r5_multitenant_foreign_invoice_does_not_block():
    """Claim d'une facture company B n'affecte pas un booking company A."""
    from application.invoices.active_invoice_claim import (
        find_all_blocking_invoice_claims,
    )

    line_b = _line(
        90,
        reservation_id=500,
        meta={"round_trip_merge_partner_reservation_id": 102},
    )
    inv_b = _inv(91, InvoiceStatus.PAID, company_id=999)
    bookings_a = [_booking(102, invoice_line_id=None, company_id=1)]
    claims = find_blocking_invoice_claims(
        {102},
        context_bookings=bookings_a,
        invoice_lines_with_invoices=_pairs((line_b, inv_b)),
    )
    assert claims == {}
    assert (
        find_all_blocking_invoice_claims(
            {102},
            context_bookings=bookings_a,
            invoice_lines_with_invoices=_pairs((line_b, inv_b)),
        )
        == {}
    )


def test_r6_parent_invoiced_without_claim_keeps_return_open():
    """A facturé sans revendiquer B → B non bloqué (pas de heuristique parent)."""
    line = _line(
        10,
        reservation_id=101,
        meta={"booking_ids": [101], "billing_unit": "single"},
    )
    inv = _inv(1, InvoiceStatus.PAID)
    assert covered_booking_ids(line) == {101}
    claims = find_blocking_invoice_claims(
        {102},
        context_bookings=[
            _booking(101, invoice_line_id=10, parent_booking_id=None),
            _booking(102, invoice_line_id=None, parent_booking_id=101),
        ],
        invoice_lines_with_invoices=_pairs((line, inv)),
    )
    assert 102 not in claims
    open_b = filter_bookings_without_active_invoice_claim(
        [_booking(102, invoice_line_id=None, parent_booking_id=101)],
        invoice_lines_with_invoices=_pairs((line, inv)),
    )
    assert [x.id for x in open_b] == [102]


def test_r6_route_group_invoiced_peer_without_claim_open():
    """Même route_group + A facturé sans claim B → B ouvert."""
    line = _line(10, reservation_id=201, meta={"booking_ids": [201]})
    inv = _inv(1, InvoiceStatus.PAID)
    open_b = filter_bookings_without_active_invoice_claim(
        [
            _booking(201, invoice_line_id=10, route_group_id="rg-shared"),
            _booking(202, invoice_line_id=None, route_group_id="rg-shared"),
        ],
        invoice_lines_with_invoices=_pairs((line, inv)),
    )
    assert [x.id for x in open_b] == [202]


def test_r7_cancelled_fk_ignored_active_claim_blocks():
    """FK résiduelle CANCELLED ignorée ; nouvelle claim active bloque."""
    old_line = _line(
        1, reservation_id=50, meta={"round_trip_merge_partner_reservation_id": 51}
    )
    old_inv = _inv(10, InvoiceStatus.CANCELLED)
    new_line = _line(2, reservation_id=51, meta={"booking_ids": [51]})
    new_inv = _inv(20, InvoiceStatus.SENT)
    claims = find_blocking_invoice_claims(
        {51},
        invoice_lines_with_invoices=_pairs((old_line, old_inv), (new_line, new_inv)),
    )
    assert 51 in claims
    assert claims[51].invoice_line_id == 2
    assert claims[51].invoice_status in ("sent", "SENT", InvoiceStatus.SENT.value)


def test_r8_multiple_active_claims_still_block():
    from application.invoices.active_invoice_claim import (
        find_all_blocking_invoice_claims,
    )

    line_x = _line(
        1, reservation_id=10, meta={"round_trip_merge_partner_reservation_id": 11}
    )
    line_y = _line(2, reservation_id=11, meta={"booking_ids": [11]})
    inv_x = _inv(100, InvoiceStatus.PAID)
    inv_y = _inv(101, InvoiceStatus.DRAFT)
    all_c = find_all_blocking_invoice_claims(
        {11},
        invoice_lines_with_invoices=_pairs((line_x, inv_x), (line_y, inv_y)),
    )
    assert len(all_c[11]) == 2
    claims = find_blocking_invoice_claims(
        {11},
        invoice_lines_with_invoices=_pairs((line_x, inv_x), (line_y, inv_y)),
    )
    assert 11 in claims
    assert claims[11].claim_count == 2
    assert (
        filter_bookings_without_active_invoice_claim(
            [_booking(11, invoice_line_id=None)],
            invoice_lines_with_invoices=_pairs((line_x, inv_x), (line_y, inv_y)),
        )
        == []
    )


def test_r3_blocking_statuses_aligned_with_round_trip_lock():
    from application.invoices.active_invoice_claim import (
        BLOCKING_INVOICE_STATUSES_FOR_CLAIM,
    )
    from application.invoices import round_trip_billing_lock as rtl

    assert rtl._BLOCKING_INVOICE_STATUSES is BLOCKING_INVOICE_STATUSES_FOR_CLAIM
    assert InvoiceStatus.CANCELLED not in BLOCKING_INVOICE_STATUSES_FOR_CLAIM
