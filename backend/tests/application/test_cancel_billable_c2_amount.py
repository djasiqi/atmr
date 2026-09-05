"""C2 — montant d'annulation. Tests first, pas de correctif C1/C3/C4."""

from __future__ import annotations

from decimal import Decimal

from application.invoices.billable_amount import SOURCE_CANCELLATION_UNRESOLVED
from ext import db
from tests.application.helpers.cancel_billable_c2_world import (
    FEE_FULL,
    FEE_PARTIAL,
    FEE_ZERO,
    RIDE_HT,
    add_canceled_booking,
    add_completed_booking,
    build_c2_world,
    canonical_amount,
    canonical_billable,
    generate_clinic_amount,
    preview_clinic_amount,
    registry_clinic_amount,
)


def _assert_resolved_surfaces(
    world, booking, *, expected: Decimal, allow_ride_source: bool
) -> None:
    """Registre et preview avant generate : l'émission pose invoice_line_id."""
    db.session.flush()
    bid = int(booking.id)
    billed = canonical_billable(booking)
    canonical, source = canonical_amount(booking)
    registry = registry_clinic_amount(world)
    preview = preview_clinic_amount(world, bid)
    generate = generate_clinic_amount(world, bid)
    assert billed.resolved is True
    assert registry == expected, f"registre={registry}"
    assert preview == expected, f"preview={preview}"
    assert generate == expected, f"generate={generate}"
    assert canonical == expected, f"canonique={canonical} source={source}"
    assert registry == preview == generate
    if allow_ride_source:
        assert source == "booking.amount"
    else:
        assert source == "cancellation_fee_amount"


def test_c2_completed_ride_uses_booking_amount(db):
    world = build_c2_world(db)
    booking = add_completed_booking(db, world)
    _assert_resolved_surfaces(world, booking, expected=RIDE_HT, allow_ride_source=True)


def test_c2_canceled_partial_fee(db):
    world = build_c2_world(db)
    booking = add_canceled_booking(
        db,
        world,
        billed_to_type="clinic",
        is_cancellation_billable=True,
        cancellation_fee_amount=FEE_PARTIAL,
    )
    _assert_resolved_surfaces(
        world, booking, expected=FEE_PARTIAL, allow_ride_source=False
    )


def test_c2_canceled_explicit_full_fare(db):
    world = build_c2_world(db)
    booking = add_canceled_booking(
        db,
        world,
        billed_to_type="clinic",
        is_cancellation_billable=True,
        cancellation_fee_amount=FEE_FULL,
    )
    _assert_resolved_surfaces(
        world, booking, expected=FEE_FULL, allow_ride_source=False
    )


def test_c2_canceled_explicit_zero_fee(db):
    world = build_c2_world(db)
    booking = add_canceled_booking(
        db,
        world,
        billed_to_type="clinic",
        is_cancellation_billable=True,
        cancellation_fee_amount=FEE_ZERO,
    )
    _assert_resolved_surfaces(
        world, booking, expected=FEE_ZERO, allow_ride_source=False
    )


def test_c2_canceled_unresolved_fee_never_falls_back_to_ride(db):
    """P0 : NULL = frais non résolu, jamais 90 par fallback sur booking.amount."""
    world = build_c2_world(db)
    booking = add_canceled_booking(
        db,
        world,
        billed_to_type="clinic",
        is_cancellation_billable=True,
        cancellation_fee_amount=None,
    )
    db.session.flush()
    bid = int(booking.id)
    billed = canonical_billable(booking)
    canonical, source = canonical_amount(booking)
    registry = registry_clinic_amount(world)
    preview = preview_clinic_amount(world, bid)
    generate = generate_clinic_amount(world, bid)

    assert billed.resolved is False
    assert source == SOURCE_CANCELLATION_UNRESOLVED
    assert source != "booking.amount"
    assert canonical != RIDE_HT
    assert registry == preview == generate
    for label, amount in (
        ("canonique", canonical),
        ("registre", registry),
        ("preview", preview),
        ("generate", generate),
    ):
        assert amount != RIDE_HT, f"{label}={amount} (fallback course interdit)"
