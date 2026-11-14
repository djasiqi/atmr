"""CONVENTION-SIGNATURE-REAL-01 — cycle de vie et gate différé (sans DB)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from domain.end_of_trip_settlement.closure_options import (
    OPTION_CASH,
    OPTION_DEFERRED,
    OPTION_INCIDENT,
    ClosureContext,
    build_closure_menu,
)
from domain.end_of_trip_settlement.convention_real_lifecycle import (
    ConventionLifecycleRefusal,
    assert_transition,
    deferred_billing_gate,
    frozen_document_must_not_change,
    signature_alone_is_not_applicable,
)


def test_transitions_coherentes():
    assert_transition("draft", "awaiting_signature")
    assert_transition("awaiting_signature", "signed_pending_validation")
    assert_transition("signed_pending_validation", "applicable")
    assert_transition("applicable", "revoked")
    with pytest.raises(ConventionLifecycleRefusal):
        assert_transition("revoked", "applicable")


def test_signature_ecran_pas_suffisante_seule():
    assert (
        signature_alone_is_not_applicable(
            modality="on_screen_trace",
            consent_acknowledged=True,
            document_sha256="abc",
            archive_reference="conventions/1/1-v1.pdf",
        )
        is True
    )


def test_gate_differe_ferme_sans_activation():
    gate = deferred_billing_gate(
        status="applicable",
        legally_effective=False,
        legal_activation_unlocked=False,
    )
    assert gate.authorizes_deferred_billing is False
    assert gate.legally_effective is False
    assert gate.reason == "legal_activation_closed"


def test_gate_demo_jamais_differe_reel():
    gate = deferred_billing_gate(
        status="applicable",
        legally_effective=True,
        legal_activation_unlocked=True,
        is_demo_row=True,
    )
    assert gate.authorizes_deferred_billing is False
    assert gate.reason == "demo_convention_not_legally_effective"


def test_document_fige_refuse_alteration():
    with pytest.raises(ConventionLifecycleRefusal) as caught:
        frozen_document_must_not_change(
            previous_sha256="aaa",
            candidate_sha256="bbb",
            status="applicable",
        )
    assert caught.value.code == "document_fige"


def test_cloture_paiement_immediat_sans_signature():
    menu = build_closure_menu(
        ClosureContext(
            booking_id=1,
            company_id=10,
            driver_company_id=10,
            billed_to_type="patient",
            mandate_kind="absent",
            amount_due_chf=Decimal("40.00"),
            payment_received=False,
            convention_legally_effective=False,
        )
    )
    codes = {o.code: o for o in menu.options}
    assert codes[OPTION_CASH].available is True
    assert codes[OPTION_DEFERRED].authorizes_deferred_billing is False
    assert codes[OPTION_INCIDENT].available is True


def test_cloture_applicable_sans_legal_bloque_differe():
    menu = build_closure_menu(
        ClosureContext(
            booking_id=1,
            company_id=10,
            driver_company_id=10,
            billed_to_type="patient",
            mandate_kind="applicable",
            amount_due_chf=Decimal("40.00"),
            payment_received=False,
            convention_legally_effective=False,
        )
    )
    deferred = next(o for o in menu.options if o.code == OPTION_DEFERRED)
    assert deferred.authorizes_deferred_billing is False
    assert deferred.available is False
