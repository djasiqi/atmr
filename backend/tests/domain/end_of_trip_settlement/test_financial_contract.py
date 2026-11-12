"""Contrat financier pur, sans base."""

from decimal import Decimal

import pytest

from domain.end_of_trip_settlement.cash_settlement import money
from domain.end_of_trip_settlement.financial_contract import (
    KIND_COLLECTION,
    KIND_COMPENSATION,
    KIND_HANDOVER,
    CashCustody,
    CashLine,
    CashOperation,
    LegAmount,
    accumulate_cash,
    apply_handover,
    credit_room,
    imputable,
    recognized_net,
    settle,
    settle_legs,
    within_cap,
)


def _cash(
    operation_id: str,
    *,
    net: str,
    kind: str = KIND_COLLECTION,
    state: str = "received",
    disposition: str | None = None,
    due: str = "45.00",
    leg_id: str = "aller",
) -> CashOperation:
    return CashOperation(
        operation_id=operation_id,
        kind=kind,
        state=state,
        disposition=disposition,
        net_chf=Decimal(net),
        due_chf=Decimal(due),
        leg_id=leg_id,
    )


def test_transport_45_sans_paiement():
    result = settle(Decimal("45.00"), (), (), ())
    assert result.claim_chf == Decimal("45.00")
    assert result.balance_due_chf == Decimal("45.00")
    assert result.refundable_chf == Decimal("0.00")


def test_transport_45_paiement_20_solde_25():
    result = settle(Decimal("45.00"), (), (Decimal("20.00"),), ())
    assert result.claim_chf == Decimal("45.00")
    assert result.net_paid_chf == Decimal("20.00")
    assert result.balance_due_chf == Decimal("25.00")
    assert result.refundable_chf == Decimal("0.00")


def test_transport_45_paiements_20_et_25_solde_nul():
    recognized = accumulate_cash(
        Decimal("45.00"),
        (
            _cash("a", disposition="partial", net="20.00"),
            _cash("b", net="25.00", due="25.00"),
        ),
    )
    assert recognized == Decimal("45.00")
    result = settle(
        Decimal("45.00"),
        (),
        (Decimal("20.00"), Decimal("25.00")),
        (),
    )
    assert result.balance_due_chf == Decimal("0.00")
    assert result.refundable_chf == Decimal("0.00")


def test_transport_45_paiements_20_et_30_depassement():
    with pytest.raises(ValueError, match="depassement_autorise"):
        accumulate_cash(
            Decimal("45.00"),
            (
                _cash("a", disposition="partial", net="20.00"),
                _cash("b", disposition="partial", net="30.00"),
            ),
        )


def test_montant_negatif_injustifie():
    with pytest.raises(ValueError, match="montant_incoherent"):
        accumulate_cash(
            Decimal("45.00"),
            (_cash("a", net="-5.00"),),
        )
    with pytest.raises(ValueError, match="montant_incoherent"):
        settle(Decimal("45.00"), (), (Decimal("-20.00"),), ())


def test_correction_compensatrice_autorisee():
    lines = (
        CashLine(
            KIND_COLLECTION,
            "received",
            "partial",
            Decimal("20.00"),
            Decimal("45.00"),
        ),
        CashLine(
            KIND_COMPENSATION,
            "received",
            None,
            Decimal("-5.00"),
            Decimal("45.00"),
        ),
    )
    assert recognized_net(lines) == Decimal("15.00")
    assert within_cap(recognized_net(lines), Decimal("45.00")) is True


def test_compensation_sous_zero_refusee():
    with pytest.raises(ValueError, match="depassement_autorise"):
        accumulate_cash(
            Decimal("45.00"),
            (
                _cash("a", disposition="partial", net="20.00"),
                _cash(
                    "b",
                    kind=KIND_COMPENSATION,
                    net="-25.00",
                ),
            ),
        )


def test_encaissement_incertain_non_imputable():
    assert (
        imputable(
            state="uncertain",
            disposition=None,
            net=Decimal("20.00"),
            due=Decimal("45.00"),
        )
        is False
    )
    recognized = accumulate_cash(
        Decimal("45.00"),
        (_cash("u", state="uncertain", net="20.00"),),
    )
    assert recognized == Decimal("0.00")


def test_encaissement_conteste_non_imputable():
    assert (
        imputable(
            state="administrative",
            disposition="contested",
            net=Decimal("20.00"),
            due=Decimal("45.00"),
        )
        is False
    )
    line = CashLine(
        KIND_COLLECTION,
        "administrative",
        "contested",
        Decimal("20.00"),
        Decimal("45.00"),
    )
    assert recognized_net((line,)) == Decimal("0.00")


def test_avoir_45_sur_facture_90_impayee():
    result = settle(Decimal("90.00"), (Decimal("45.00"),), (), ())
    assert result.claim_chf == Decimal("45.00")
    assert result.net_paid_chf == Decimal("0.00")
    assert result.balance_due_chf == Decimal("45.00")
    assert result.refundable_chf == Decimal("0.00")


def test_avoir_total_sur_facture_impayee():
    result = settle(Decimal("90.00"), (Decimal("90.00"),), (), ())
    assert result.claim_chf == Decimal("0.00")
    assert result.balance_due_chf == Decimal("0.00")
    assert result.net_paid_chf == Decimal("0.00")


def test_avoir_sur_facture_partiellement_payee():
    result = settle(
        Decimal("90.00"),
        (Decimal("45.00"),),
        (Decimal("20.00"),),
        (),
    )
    assert result.claim_chf == Decimal("45.00")
    assert result.net_paid_chf == Decimal("20.00")
    assert result.balance_due_chf == Decimal("25.00")
    assert result.refundable_chf == Decimal("0.00")


def test_avoir_sur_facture_integralement_payee_trop_percu():
    result = settle(
        Decimal("90.00"),
        (Decimal("45.00"),),
        (Decimal("90.00"),),
        (),
    )
    assert result.claim_chf == Decimal("45.00")
    assert result.balance_due_chf == Decimal("0.00")
    assert result.refundable_chf == Decimal("45.00")
    assert result.net_paid_chf == Decimal("90.00")


def test_remboursement_distinct_de_l_avoir():
    credited = settle(
        Decimal("90.00"),
        (Decimal("45.00"),),
        (Decimal("90.00"),),
        (),
    )
    refunded = settle(
        Decimal("90.00"),
        (Decimal("45.00"),),
        (Decimal("90.00"),),
        (Decimal("45.00"),),
    )
    assert credited.claim_chf == refunded.claim_chf
    assert credited.refundable_chf == Decimal("45.00")
    assert refunded.refundable_chf == Decimal("0.00")
    assert refunded.net_paid_chf == Decimal("45.00")
    assert refunded.balance_due_chf == Decimal("0.00")


def test_plafond_des_remboursements():
    with pytest.raises(ValueError, match="depassement_autorise"):
        settle(
            Decimal("90.00"),
            (Decimal("45.00"),),
            (Decimal("90.00"),),
            (Decimal("46.00"),),
        )
    with pytest.raises(ValueError, match="depassement_autorise"):
        settle(Decimal("90.00"), (), (), (Decimal("10.00"),))


def test_remise_sans_nouveau_paiement_patient():
    before = CashCustody(
        contractual_chf=Decimal("45.00"),
        patient_paid_chf=Decimal("20.00"),
        driver_held_chf=Decimal("20.00"),
        company_held_chf=Decimal("0.00"),
    )
    after = apply_handover(before, Decimal("20.00"))
    assert after.contractual_chf == before.contractual_chf
    assert after.patient_paid_chf == before.patient_paid_chf
    assert after.driver_held_chf == Decimal("0.00")
    assert after.company_held_chf == Decimal("20.00")
    recognized = accumulate_cash(
        Decimal("45.00"),
        (
            _cash("enc", disposition="partial", net="20.00"),
            _cash("remise", kind=KIND_HANDOVER, net="20.00"),
        ),
    )
    assert recognized == Decimal("20.00")


def test_aller_retour_montants_differents():
    settled = settle_legs(
        (
            LegAmount("aller", Decimal("45.00")),
            LegAmount("retour", Decimal("30.00")),
        ),
        {"aller": (Decimal("20.00"),)},
    )
    assert settled["aller"].claim_chf == Decimal("45.00")
    assert settled["aller"].balance_due_chf == Decimal("25.00")
    assert settled["retour"].claim_chf == Decimal("30.00")
    assert settled["retour"].balance_due_chf == Decimal("30.00")
    half = (Decimal("45.00") + Decimal("30.00")) / 2
    assert settled["aller"].claim_chf != half
    assert settled["retour"].claim_chf != half


def test_centimes_exacts_et_millimes_refus():
    result = settle(
        Decimal("20.15"),
        (),
        (Decimal("10.10"), Decimal("10.05")),
        (),
    )
    assert result.balance_due_chf == Decimal("0.00")
    assert result.net_paid_chf == Decimal("20.15")
    with pytest.raises(ValueError, match="montant_illisible"):
        money("10.125")


def test_idempotence_d_une_operation_repetee():
    operations = (
        _cash("a", disposition="partial", net="20.00"),
        _cash("a", disposition="partial", net="20.00"),
    )
    assert accumulate_cash(Decimal("45.00"), operations) == Decimal("20.00")
    with pytest.raises(ValueError, match="montant_incoherent"):
        accumulate_cash(
            Decimal("45.00"),
            (
                _cash("a", disposition="partial", net="20.00"),
                _cash("a", disposition="partial", net="10.00"),
            ),
        )


def test_plafond_des_avoirs_sur_le_facture_restant():
    assert credit_room(Decimal("90.00"), (Decimal("40.00"),)) == Decimal("50.00")
    result = settle(
        Decimal("90.00"),
        (Decimal("40.00"), Decimal("50.00")),
        (),
        (),
    )
    assert result.balance_due_chf == Decimal("0.00")
    with pytest.raises(ValueError, match="avoir_au_dela_du_facture"):
        settle(
            Decimal("90.00"),
            (Decimal("40.00"), Decimal("60.00")),
            (),
            (),
        )


def test_avoir_ne_fabrique_pas_de_paiement_negatif():
    notes = (Decimal("45.00"),)
    result = settle(Decimal("90.00"), notes, (), ())
    assert result.claim_chf == Decimal("45.00")
    assert all(note > 0 for note in notes)
    assert result.net_paid_chf == Decimal("0.00")
