"""CONVENTION-REGLEMENT-01 — scénarios A–J (domaine pur, base non requise)."""

from __future__ import annotations

from decimal import Decimal

from application.settlement.settlement_incident_registry import (
    SettlementIncidentRegistry,
    attempt_card_payment,
    cash_payment_recorded,
)
from domain.end_of_trip_settlement.closure_options import (
    OPTION_CARD,
    OPTION_CASH,
    OPTION_DEFERRED,
    OPTION_INCIDENT,
    ClosureContext,
    build_closure_menu,
    preserve_debtor_on_recipient_change,
)
from domain.end_of_trip_settlement.mandate_applicability import (
    assess_mandate,
    real_invoice_blocked,
)
from domain.end_of_trip_settlement.settlement_incident import (
    MOTIF_AUTRE,
    MOTIF_REFUS_SIGNER_PAYER,
    STATUS_A_TRAITER,
    STATUS_CLOTURE,
    STATUS_EN_COURS,
    STATUS_REGULARISE,
    SettlementIncidentDraft,
)


def _ctx(**overrides) -> ClosureContext:
    base = {
        "booking_id": 100,
        "company_id": 7,
        "driver_company_id": 7,
        "billed_to_type": "patient",
        "mandate_kind": "absent",
        "amount_due_chf": Decimal("45.00"),
        "payment_received": False,
        "card_provider_operational": False,
        "debtor_id": 55,
        "debtor_label": "Client Test",
    }
    base.update(overrides)
    return ClosureContext(**base)


def _draft(**overrides) -> SettlementIncidentDraft:
    base = {
        "booking_id": 100,
        "company_id": 7,
        "driver_id": 3,
        "client_id": 55,
        "debtor_id": 55,
        "debtor_label": "Client Test",
        "amount_due_chf": Decimal("45.00"),
        "transport_date": "2026-11-14",
        "motif": MOTIF_REFUS_SIGNER_PAYER,
        "observation": "Client refuse de signer et de payer. Constat factuel.",
        "idempotency_key": "idemp-100-1",
    }
    base.update(overrides)
    return SettlementIncidentDraft(**base)


# --- A ---
def test_a_convention_valide_facture_differee_autorisee():
    menu = build_closure_menu(_ctx(mandate_kind="applicable_demonstration"))
    deferred = next(o for o in menu.options if o.code == OPTION_DEFERRED)
    assert deferred.available is True
    assert deferred.authorizes_deferred_billing is True
    assert deferred.records_payment is False
    mandate = assess_mandate(
        mandate_company_id=7,
        booking_company_id=7,
        status="demo_valid_simulated",
        covered_tariff_versions=("demo-2026.1",),
        booking_tariff_version=None,
        valid_from=None,
        valid_until=None,
        on_date=__import__("datetime").date(2026, 11, 14),
        billed_to_type="patient",
        already_paid=False,
    )
    assert mandate.kind == "applicable_demonstration"
    # Valeur juridique toujours refusée tant que non confirmée.
    assert (
        real_invoice_blocked(
            kind=mandate.kind,
            legally_effective=False,
            real_signature_collected=False,
        )
        == "mandat_demonstration"
    )


# --- B ---
def test_b_aucune_convention_especes_enregistrees():
    menu = build_closure_menu(_ctx(mandate_kind="absent"))
    deferred = next(o for o in menu.options if o.code == OPTION_DEFERRED)
    cash = next(o for o in menu.options if o.code == OPTION_CASH)
    assert deferred.available is False
    assert cash.available is True
    assert cash_payment_recorded(
        amount_due_chf=Decimal("45.00"), tendered_chf=Decimal("45.00")
    )


# --- C ---
def test_c_carte_non_operationnelle_reste_non_payee():
    menu = build_closure_menu(_ctx(mandate_kind="absent"))
    card = next(o for o in menu.options if o.code == OPTION_CARD)
    assert card.available is False
    assert card.reason == "card_provider_not_operational"
    result = attempt_card_payment(provider_operational=False)
    assert result.payment_recorded is False
    assert result.status == "unpaid"


# --- D ---
def test_d_refus_signature_paiement_mission_terminee_incident():
    registry = SettlementIncidentRegistry()
    decision = registry.open(_draft(), actor_company_id=7)
    assert decision.accepted is True
    assert decision.incident is not None
    assert decision.incident.mission_close_allowed is True
    assert decision.incident.payment_recorded is False
    assert decision.incident.signature_recorded is False
    assert decision.incident.deferred_billing_authorized is False
    assert decision.incident.status == STATUS_A_TRAITER


# --- E ---
def test_e_administration_consulte_et_traite():
    registry = SettlementIncidentRegistry()
    opened = registry.open(_draft(), actor_company_id=7)
    incident_id = opened.incident.id  # type: ignore[union-attr]
    queue = registry.list_for_company(7)
    assert len(queue) == 1
    assert queue[0].booking_id == 100
    mid = registry.admin_set_status(
        company_id=7,
        incident_id=incident_id,
        new_status=STATUS_EN_COURS,
        actor_id=9,
        justification="Examen démarré",
    )
    assert mid.accepted is True
    done = registry.admin_set_status(
        company_id=7,
        incident_id=incident_id,
        new_status=STATUS_REGULARISE,
        actor_id=9,
        justification="Paiement reçu au guichet le lendemain",
    )
    assert done.accepted is True
    noted = registry.admin_note_payment(
        company_id=7,
        incident_id=incident_id,
        actor_id=9,
        note="Virement 45.00 CHF reçu",
    )
    assert noted.accepted is True
    assert noted.incident is not None
    assert (
        noted.incident.payment_recorded is False
    )  # pas d'encaissement chauffeur fictif
    assert any(e.action == "later_payment_noted" for e in noted.incident.history)
    closed = registry.admin_set_status(
        company_id=7,
        incident_id=incident_id,
        new_status=STATUS_CLOTURE,
        actor_id=9,
        justification="Créance soldée",
    )
    assert closed.accepted is True
    assert closed.incident is not None
    assert closed.incident.status == STATUS_CLOTURE
    assert registry.driver_delete_refused(closed.incident) is True
    assert registry.driver_admin_status_refused(closed.incident) is True


# --- F ---
def test_f_repetition_reseau_pas_de_double_incident():
    registry = SettlementIncidentRegistry()
    first = registry.open(_draft(), actor_company_id=7)
    second = registry.open(_draft(), actor_company_id=7)
    assert first.accepted is True
    assert second.accepted is True
    assert second.duplicate is True
    assert len(registry.list_for_company(7)) == 1
    assert cash_payment_recorded(
        amount_due_chf=Decimal("45.00"), tendered_chf=Decimal("45.00")
    )
    # Un second « paiement » exact n'invente pas un double encaissement ici :
    # la couche caisse existante porte l'idempotence opérationnelle.


# --- G ---
def test_g_chauffeur_autre_entreprise_refuse():
    menu = build_closure_menu(_ctx(driver_company_id=99))
    assert menu.applicable is False
    assert "company_isolation_refused" in menu.reasons
    registry = SettlementIncidentRegistry()
    denied = registry.open(_draft(), actor_company_id=99)
    assert denied.accepted is False
    assert denied.reason == "company_isolation_refused"


# --- H ---
def test_h_changement_destinataire_conserve_debiteur():
    result = preserve_debtor_on_recipient_change(
        debtor_id=55,
        debtor_type="patient",
        new_recipient_id=88,
        new_recipient_label="Famille / Curatelle",
    )
    assert result["debtor_id"] == 55
    assert result["debtor_changed"] is False
    assert result["recipient_changed"] is True
    assert result["invoice_recipient_id"] == 88


# --- I ---
def test_i_convention_non_applicable_pas_de_differe_auto():
    for kind in ("absent", "prepared", "expired", "amendment_required", "specimen"):
        menu = build_closure_menu(_ctx(mandate_kind=kind))
        deferred = next(o for o in menu.options if o.code == OPTION_DEFERRED)
        assert deferred.available is False
        assert deferred.authorizes_deferred_billing is False


# --- J ---
def test_j_institution_hors_matrice_client():
    menu = build_closure_menu(_ctx(billed_to_type="clinic", mandate_kind="absent"))
    assert menu.applicable is False
    assert menu.institution_uses_own_rules is True
    assert "institution_own_rules" in menu.reasons


def test_motif_autre_exige_observation():
    registry = SettlementIncidentRegistry()
    bad = registry.open(
        _draft(motif=MOTIF_AUTRE, observation="  ", idempotency_key="x-autre"),
        actor_company_id=7,
    )
    assert bad.accepted is False
    assert bad.reason == "observation_requise_pour_autre"


def test_menu_inclut_signalement_quand_particulier():
    menu = build_closure_menu(_ctx())
    codes = {o.code for o in menu.options if o.available}
    assert OPTION_CASH in codes
    assert OPTION_INCIDENT in codes
    assert OPTION_DEFERRED not in codes
