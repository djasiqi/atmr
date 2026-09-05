"""C3 — libellés d'annulation. Tests first, pas de correctif C1/C2/C4."""

from __future__ import annotations

from tests.application.helpers.cancel_billable_c3_world import (
    HISTORICAL_FALLBACK,
    add_canceled_labeled_booking,
    build_c3_world,
    canonical_cancellation_label,
    generate_clinic_description,
    preview_clinic_description,
)

GENERIC_LAST_MINUTE = "Annulation dernière minute"
NO_SHOW_LABEL = "Client ne s'est pas présenté"
CLIENT_REQUEST_LABEL = "Client a demandé l'annulation"
CLIENT_REQUEST_WITH_FEE = f"{CLIENT_REQUEST_LABEL} — frais 50 %"
OTHER_COMMENT = "Patient hospitalisé en urgence"


def _assert_c3_surfaces(
    world, booking, *, expected: str, forbidden: tuple[str, ...] = ()
):
    bid = int(booking.id)
    preview = preview_clinic_description(world, bid)
    generate = generate_clinic_description(world, bid)
    assert preview == generate, f"preview={preview!r} generate={generate!r}"
    assert preview == expected, f"preview={preview!r} attendu={expected!r}"
    assert generate == expected, f"generate={generate!r} attendu={expected!r}"
    for token in forbidden:
        assert token not in (preview or ""), f"preview contient {token!r}: {preview!r}"
    assert " → " not in (preview or "")
    assert " → " not in (generate or "")


def test_c3_last_minute_keeps_real_motif(db):
    world = build_c3_world(db)
    booking = add_canceled_labeled_booking(db, world, reason_code="LAST_MINUTE")
    expected = canonical_cancellation_label(
        reason_code="LAST_MINUTE",
        reason_text=None,
        persisted_label=booking.cancellation_display_label,
    )
    assert expected == GENERIC_LAST_MINUTE
    _assert_c3_surfaces(world, booking, expected=expected)


def test_c3_no_show_never_generic_last_minute(db):
    world = build_c3_world(db)
    booking = add_canceled_labeled_booking(db, world, reason_code="NO_SHOW")
    expected = canonical_cancellation_label(
        reason_code="NO_SHOW",
        reason_text=None,
        persisted_label=booking.cancellation_display_label,
    )
    assert expected == NO_SHOW_LABEL
    _assert_c3_surfaces(
        world,
        booking,
        expected=expected,
        forbidden=(GENERIC_LAST_MINUTE,),
    )


def test_c3_client_request_keeps_motif_and_fee_percent(db):
    world = build_c3_world(db)
    booking = add_canceled_labeled_booking(
        db,
        world,
        reason_code="CLIENT_REQUEST",
        fee_percent=50,
        fee_tier_id="< 12h",
    )
    assert booking.cancellation_display_label == CLIENT_REQUEST_LABEL
    _assert_c3_surfaces(
        world,
        booking,
        expected=CLIENT_REQUEST_WITH_FEE,
        forbidden=("Annulation (< 12h)",),
    )


def test_c3_other_uses_business_comment_not_raw_code(db):
    world = build_c3_world(db)
    booking = add_canceled_labeled_booking(
        db,
        world,
        reason_code="OTHER",
        reason_text=OTHER_COMMENT,
    )
    expected = canonical_cancellation_label(
        reason_code="OTHER",
        reason_text=OTHER_COMMENT,
        persisted_label=booking.cancellation_display_label,
    )
    assert OTHER_COMMENT in expected
    assert "OTHER:" not in expected
    _assert_c3_surfaces(
        world,
        booking,
        expected=expected,
        forbidden=("OTHER:", "OTHER"),
    )


def test_c3_historical_without_reason_uses_explicit_fallback(db):
    world = build_c3_world(db)
    booking = add_canceled_labeled_booking(
        db,
        world,
        reason_code=None,
        persist_display_label=False,
    )
    expected = canonical_cancellation_label(
        reason_code=None,
        reason_text=None,
        persisted_label=booking.cancellation_display_label,
    )
    assert expected == HISTORICAL_FALLBACK
    _assert_c3_surfaces(
        world,
        booking,
        expected=expected,
        forbidden=(GENERIC_LAST_MINUTE,),
    )
