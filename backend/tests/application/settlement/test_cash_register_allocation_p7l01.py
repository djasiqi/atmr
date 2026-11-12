"""P7-L01 — Règlements : solde TTC et imputation depuis la facture liée."""

from __future__ import annotations

from decimal import Decimal

from application.settlement.cash_ledger import _apply_invoice_allocation_to_view


def test_allocation_absente_reste_non_impute():
    view = {
        "amount_due_chf": "45.00",
        "balance_chf": "25.00",
        "remaining_due_chf": "25.00",
        "allocated_invoice_payment_id": None,
        "invoice_number": None,
        "invoice_issued": False,
    }

    def _none(_cid: int, booking_id=None):  # noqa: ARG001
        return None

    import application.settlement.cash_ledger as ledger

    original = ledger._invoice_allocation_snapshot
    ledger._invoice_allocation_snapshot = _none  # type: ignore[assignment]
    try:
        out = _apply_invoice_allocation_to_view(view, collection_id=2)
    finally:
        ledger._invoice_allocation_snapshot = original  # type: ignore[assignment]

    assert out["allocation_status"] == "not_allocated"
    assert out["allocated_invoice_payment_id"] is None
    assert out["invoice_number"] is None
    assert out["balance_chf"] == "25.00"
    assert out["amount_due_chf"] == "45.00"


def test_allocation_facture_aligne_ttc_et_solde():
    view = {
        "amount_due_chf": "45.00",
        "balance_chf": "25.00",
        "remaining_due_chf": "25.00",
        "recognized_net_chf": "20.00",
        "allocated_invoice_payment_id": None,
        "invoice_number": None,
        "invoice_issued": False,
    }

    def _snap(_cid: int, booking_id=None):  # noqa: ARG001
        return {
            "payment_id": 225,
            "invoice_number": "EM-2026-11-0004",
            "total_amount": Decimal("48.45"),
            "line_ttc": Decimal("48.45"),
            "balance_due": Decimal("28.45"),
            "amount_paid": Decimal("20.00"),
            "imputed_amount": Decimal("20.00"),
        }

    import application.settlement.cash_ledger as ledger

    original = ledger._invoice_allocation_snapshot
    ledger._invoice_allocation_snapshot = _snap  # type: ignore[assignment]
    try:
        out = _apply_invoice_allocation_to_view(view, collection_id=2)
    finally:
        ledger._invoice_allocation_snapshot = original  # type: ignore[assignment]

    assert out["allocation_status"] == "allocated"
    assert out["allocated_invoice_payment_id"] == 225
    assert out["invoice_number"] == "EM-2026-11-0004"
    assert out["invoice_issued"] is True
    assert out["amount_due_chf"] == "48.45"
    assert out["balance_chf"] == "28.45"
    assert out["remaining_due_chf"] == "28.45"
    assert out["recognized_net_chf"] == "20.00"


def test_allocation_indisponible_ne_recalcule_pas():
    view = {
        "amount_due_chf": "45.00",
        "balance_chf": "25.00",
        "remaining_due_chf": "25.00",
    }

    def _boom(_cid: int, booking_id=None):  # noqa: ARG001
        raise RuntimeError("db")

    import application.settlement.cash_ledger as ledger

    original = ledger._invoice_allocation_snapshot
    ledger._invoice_allocation_snapshot = _boom  # type: ignore[assignment]
    try:
        out = _apply_invoice_allocation_to_view(view, collection_id=2)
    finally:
        ledger._invoice_allocation_snapshot = original  # type: ignore[assignment]

    assert out["allocation_status"] == "unavailable"
    assert out["balance_chf"] == "25.00"
    assert out["amount_due_chf"] == "45.00"
