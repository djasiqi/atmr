"""Tests éditeur / replace facture plateforme."""

from decimal import Decimal

import pytest

from services.platform_billing.invoice_replace import (
    InvoiceReplaceError,
    compute_totals,
    normalize_editor_lines,
)
from services.platform_billing.dossier_status import (
    ACTION_CORRECT_INVOICE,
    ACTION_CREDIT,
    ACTION_EDIT_INVOICE,
    STATUS_A_ENVOYER,
    STATUS_PAID,
    resolve_actions,
)


def test_normalize_unit_price_and_fixed():
    lines = normalize_editor_lines(
        [
            {
                "calculation_mode": "UNIT_PRICE",
                "label": "Support",
                "quantity": "2.5",
                "unit_amount": "120.00",
                "line_type": "SUPPORT",
            },
            {
                "calculation_mode": "FIXED_AMOUNT",
                "label": "Remise",
                "amount": "-20.00",
                "line_type": "DISCOUNT",
            },
        ]
    )
    assert lines[0]["amount"] == "300.00"
    assert lines[1]["amount"] == "-20.00"
    totals = compute_totals(lines, Decimal("0"))
    assert totals["total_amount"] == Decimal("280.00")


def test_normalize_rejects_empty():
    with pytest.raises(InvoiceReplaceError):
        normalize_editor_lines([])


def test_edit_action_when_a_envoyer_unpaid():
    class Inv:
        sent_at = None
        amount_paid = Decimal("0")

    actions = resolve_actions(
        status=STATUS_A_ENVOYER,
        statement=None,
        primary_invoice=Inv(),
        credit_note_id=None,
        issuable=False,
        issuer_errors=[],
        caps=set(),
    )
    assert ACTION_EDIT_INVOICE in actions["allowed_actions"]


def test_paid_blocks_credit_and_correct():
    class Inv:
        sent_at = object()
        amount_paid = Decimal("94.00")

    actions = resolve_actions(
        status=STATUS_PAID,
        statement=None,
        primary_invoice=Inv(),
        credit_note_id=None,
        issuable=False,
        issuer_errors=[],
        caps=set(),
    )
    assert ACTION_CREDIT not in actions["allowed_actions"]
    assert ACTION_CORRECT_INVOICE not in actions["allowed_actions"]
    assert ACTION_CREDIT in actions["blocked_actions"]
