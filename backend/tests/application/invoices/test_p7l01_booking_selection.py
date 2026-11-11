"""P7-L01 — sélection déterministe et empreinte de prévisualisation."""

from __future__ import annotations

import pytest

from application.invoices.booking_selection import (
    BookingSelectionRefusal,
    apply_excluded,
    derive_reservation_ids_from_client_map,
    enforce_strict_selection,
    normalize_id_list,
    selection_fingerprint,
)


class _Row:
    def __init__(self, booking_id: int) -> None:
        self.id = booking_id


def test_normalize_and_fingerprint_stables() -> None:
    assert normalize_id_list([3, "2", 3, 0, -1, "x"]) == [3, 2]
    assert selection_fingerprint([2, 1, 2]) == selection_fingerprint([1, 2])
    assert selection_fingerprint([1]) != selection_fingerprint([1, 2])


def test_apply_excluded() -> None:
    rows = [_Row(10), _Row(11), _Row(12)]
    kept = apply_excluded(rows, [11, 99])
    assert [row.id for row in kept] == [10, 12]


def test_derive_client_reservations() -> None:
    assert derive_reservation_ids_from_client_map({39847: [53360]}, 39847) == [53360]
    assert derive_reservation_ids_from_client_map({"39847": [1, 2]}, 39847) == [1, 2]
    assert derive_reservation_ids_from_client_map({1: [9]}, 2) is None


def test_strict_refuse_foreign_or_already_billed_mixed() -> None:
    with pytest.raises(BookingSelectionRefusal) as refused:
        enforce_strict_selection(
            requested_ids=[53360, 99999],
            eligible_ids=[53360],
            excluded_ids=[],
            final_ids=[53360],
            confirmed_fingerprint=None,
        )
    assert refused.value.code == "reservation_non_eligible"


def test_strict_refuse_silent_extra() -> None:
    with pytest.raises(BookingSelectionRefusal) as refused:
        enforce_strict_selection(
            requested_ids=[53360],
            eligible_ids=[53360, 53359],
            excluded_ids=[],
            final_ids=[53360, 53359],
            confirmed_fingerprint=None,
        )
    assert refused.value.code == "selection_elargie"


def test_strict_fingerprint_mismatch() -> None:
    final = [53360]
    good = selection_fingerprint(final)
    with pytest.raises(BookingSelectionRefusal) as refused:
        enforce_strict_selection(
            requested_ids=final,
            eligible_ids=final,
            excluded_ids=[],
            final_ids=final,
            confirmed_fingerprint="0" * 64,
        )
    assert refused.value.code == "previsualisation_perimee"
    enforce_strict_selection(
        requested_ids=final,
        eligible_ids=final,
        excluded_ids=[],
        final_ids=final,
        confirmed_fingerprint=good,
    )


def test_driver_deposit_lines_use_imputed_amount_not_gross(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from decimal import Decimal
    from types import SimpleNamespace

    from services.documents import driver_deposits

    monkeypatch.setattr(
        driver_deposits,
        "_collection_booking_map",
        lambda _ids: {1: {"booking_id": 53360, "scheduled_time": None}},
    )
    invoice = SimpleNamespace(
        payments=[
            SimpleNamespace(
                id=224,
                amount="20.00",
                method=SimpleNamespace(value="cash"),
                cash_collection_id=1,
                paid_at=None,
            )
        ]
    )
    lines = driver_deposits.persisted_driver_deposit_lines(invoice)
    assert len(lines) == 1
    assert lines[0].amount_chf == Decimal("20.00")
    assert lines[0].booking_id == 53360
    assert lines[0].mode_label == "Espèces chauffeur"
    assert driver_deposits.driver_deposit_total(invoice) == Decimal("20.00")
