"""Tests SSOT statut opérationnel dossiers facturation plateforme."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

from models.enums import (
    PlatformBillingPeriodStatus,
    PlatformIssuedDocumentType,
    PlatformIssuedInvoiceStatus,
    PlatformStatementStatus,
)
from services.platform_billing.dossier_status import (
    STATUS_A_CALCULER,
    STATUS_A_CONTROLER,
    STATUS_A_ENCAISSER,
    STATUS_A_ENVOYER,
    STATUS_OVERDUE,
    STATUS_PARTIALLY_PAID,
    STATUS_PRETE_A_CLOTURER,
    STATUS_PRETE_A_EMETTRE,
    ACTION_ISSUE,
    ACTION_RECALCULATE_DOSSIER,
    ACTION_VIEW,
    dossier_key,
    operational_status,
    resolve_actions,
    zero_charge_flags,
)


def _period(**kwargs):
    defaults = dict(id=42, status=PlatformBillingPeriodStatus.DRAFT.value)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _statement(**kwargs):
    defaults = dict(
        id=91,
        company_id=18,
        statement_status=PlatformStatementStatus.DRAFT.value,
        total_amount=Decimal("94.00"),
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _issued(**kwargs):
    defaults = dict(
        id=8,
        document_type=PlatformIssuedDocumentType.INVOICE.value,
        status=PlatformIssuedInvoiceStatus.ISSUED.value,
        total_amount=Decimal("94.00"),
        amount_paid=Decimal("0.00"),
        sent_at=None,
        due_at=datetime(2026, 9, 1, tzinfo=UTC),
        paid_at=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestOperationalStatus:
    def test_no_statement_a_calculer(self):
        assert (
            operational_status(
                statement=None, period=_period(), primary_invoice=None
            )
            == STATUS_A_CALCULER
        )

    def test_draft_a_calculer(self):
        assert (
            operational_status(
                statement=_statement(),
                period=_period(),
                primary_invoice=None,
            )
            == STATUS_A_CALCULER
        )

    def test_calculated_a_controler_even_zero(self):
        st = _statement(
            statement_status=PlatformStatementStatus.CALCULATED.value,
            total_amount=Decimal("0.00"),
        )
        assert (
            operational_status(
                statement=st, period=_period(), primary_invoice=None
            )
            == STATUS_A_CONTROLER
        )
        zc, reason = zero_charge_flags(st)
        assert zc is True
        assert reason

    def test_validated_prete_a_cloturer(self):
        assert (
            operational_status(
                statement=_statement(
                    statement_status=PlatformStatementStatus.VALIDATED.value
                ),
                period=_period(status=PlatformBillingPeriodStatus.DRAFT.value),
                primary_invoice=None,
            )
            == STATUS_PRETE_A_CLOTURER
        )

    def test_locked_prete_a_emettre(self):
        assert (
            operational_status(
                statement=_statement(
                    statement_status=PlatformStatementStatus.LOCKED.value
                ),
                period=_period(status=PlatformBillingPeriodStatus.LOCKED.value),
                primary_invoice=None,
            )
            == STATUS_PRETE_A_EMETTRE
        )

    def test_issued_a_envoyer(self):
        assert (
            operational_status(
                statement=_statement(
                    statement_status=PlatformStatementStatus.LOCKED.value
                ),
                period=_period(status=PlatformBillingPeriodStatus.LOCKED.value),
                primary_invoice=_issued(),
            )
            == STATUS_A_ENVOYER
        )

    def test_sent_a_encaisser(self):
        inv = _issued(
            status=PlatformIssuedInvoiceStatus.SENT.value,
            sent_at=datetime(2026, 8, 2, tzinfo=UTC),
            due_at=datetime(2026, 12, 1, tzinfo=UTC),
        )
        assert (
            operational_status(
                statement=_statement(
                    statement_status=PlatformStatementStatus.LOCKED.value
                ),
                period=_period(status=PlatformBillingPeriodStatus.LOCKED.value),
                primary_invoice=inv,
                now=datetime(2026, 8, 4, tzinfo=UTC),
            )
            == STATUS_A_ENCAISSER
        )

    def test_overdue_before_partial(self):
        inv = _issued(
            status=PlatformIssuedInvoiceStatus.OVERDUE.value,
            sent_at=datetime(2026, 7, 1, tzinfo=UTC),
            due_at=datetime(2026, 7, 15, tzinfo=UTC),
            amount_paid=Decimal("40.00"),
        )
        assert (
            operational_status(
                statement=_statement(
                    statement_status=PlatformStatementStatus.LOCKED.value
                ),
                period=_period(status=PlatformBillingPeriodStatus.LOCKED.value),
                primary_invoice=inv,
                now=datetime(2026, 8, 4, tzinfo=UTC),
            )
            == STATUS_OVERDUE
        )

    def test_partial_not_overdue(self):
        inv = _issued(
            status=PlatformIssuedInvoiceStatus.SENT.value,
            sent_at=datetime(2026, 8, 1, tzinfo=UTC),
            due_at=datetime(2026, 12, 1, tzinfo=UTC),
            amount_paid=Decimal("40.00"),
        )
        assert (
            operational_status(
                statement=_statement(
                    statement_status=PlatformStatementStatus.LOCKED.value
                ),
                period=_period(status=PlatformBillingPeriodStatus.LOCKED.value),
                primary_invoice=inv,
                now=datetime(2026, 8, 4, tzinfo=UTC),
            )
            == STATUS_PARTIALLY_PAID
        )


class TestActions:
    def test_cloturer_blocks_issue(self):
        out = resolve_actions(
            status=STATUS_PRETE_A_CLOTURER,
            statement=_statement(
                statement_status=PlatformStatementStatus.VALIDATED.value
            ),
            primary_invoice=None,
            credit_note_id=None,
            issuable=False,
            issuer_errors=["Période non verrouillée"],
            caps=set(),
        )
        assert out["primary_action"] == ACTION_VIEW
        assert ACTION_ISSUE not in out["allowed_actions"]
        assert ACTION_ISSUE in out["blocked_actions"]

    def test_emettre_when_issuable(self):
        out = resolve_actions(
            status=STATUS_PRETE_A_EMETTRE,
            statement=_statement(
                statement_status=PlatformStatementStatus.LOCKED.value
            ),
            primary_invoice=None,
            credit_note_id=None,
            issuable=True,
            issuer_errors=[],
            caps=set(),
        )
        assert out["primary_action"] == ACTION_ISSUE

    def test_emettre_blocked_zero_charge(self):
        out = resolve_actions(
            status=STATUS_PRETE_A_EMETTRE,
            statement=_statement(total_amount=Decimal("0")),
            primary_invoice=None,
            credit_note_id=None,
            issuable=False,
            issuer_errors=["Montant total doit être > 0 pour QR"],
            caps=set(),
        )
        assert out["primary_action"] == ACTION_VIEW
        assert ACTION_ISSUE in out["blocked_actions"]

    def test_a_calculer_primary(self):
        out = resolve_actions(
            status=STATUS_A_CALCULER,
            statement=None,
            primary_invoice=None,
            credit_note_id=None,
            issuable=False,
            issuer_errors=[],
            caps=set(),
        )
        assert out["primary_action"] == ACTION_RECALCULATE_DOSSIER


def test_dossier_key():
    assert dossier_key(42, 18) == "42:18"
