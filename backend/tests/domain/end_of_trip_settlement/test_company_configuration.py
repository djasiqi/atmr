"""L'entreprise configure les moyens. Le formulaire ne les choisit pas."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from domain.end_of_trip_settlement.company_configuration import (
    CompanySettlementSettings,
    _company_modes,
    acceptance_write_plan,
    capability_change_conflict,
    declare_patient_notice,
    prepare_acceptance_snapshot,
    settings_for_acceptance,
)

WHEN = datetime(2026, 10, 26, tzinfo=UTC)


def notice(**overrides: object):
    base = {
        "confirmed": True,
        "auto_checked": False,
        "author_user_id": 4,
        "confirmed_at": WHEN,
        "conditions_version": "projet-2026-10",
        "amount_chf": Decimal("45.00"),
        "debtor_label": "Patient",
        "channel": "verbal",
    }
    base.update(overrides)
    return declare_patient_notice(**base)  # type: ignore[arg-type]


def settings(**overrides: object) -> CompanySettlementSettings:
    base: dict[str, object] = {
        "tap_to_pay_enabled": False,
        "cash_enabled": False,
        "direct_patient_invoice_enabled": False,
        "periodic_accounts_enabled": False,
        "config_version": "entreprise-a",
        "invoice_signature_approved": False,
    }
    base.update(overrides)
    return CompanySettlementSettings(**base)  # type: ignore[arg-type]


def prepare(**overrides: object):
    base = {
        "company_assigned": True,
        "payer": "patient",
        "created_via": "client_app",
        "financial_responsibility_locked": False,
        "client_has_established_account": False,
        "settings": settings(cash_enabled=True),
        "already_paid": False,
        "invoice_issued": False,
        "is_round_trip": False,
        "round_trip_collection_approved": False,
        "return_time_defined": None,
        "patient_notice": notice(),
    }
    base.update(overrides)
    return prepare_acceptance_snapshot(**base)  # type: ignore[arg-type]


def test_especes_seules() -> None:
    decision = prepare()
    assert decision.modes == frozenset({"cash"})
    assert decision.collection_allowed is False
    assert decision.lock_applied is False


def test_tap_seul() -> None:
    decision = prepare(settings=settings(tap_to_pay_enabled=True))
    assert decision.modes == frozenset({"tap_to_pay"})


def test_carte_et_especes() -> None:
    decision = prepare(
        settings=settings(tap_to_pay_enabled=True, cash_enabled=True)
    )
    assert decision.modes == frozenset({"tap_to_pay", "cash"})


def test_facture_directe_exige_un_texte_approuve() -> None:
    refused = prepare(
        settings=settings(direct_patient_invoice_enabled=True)
    )
    approved = prepare(
        settings=settings(
            direct_patient_invoice_enabled=True,
            invoice_signature_approved=True,
        )
    )
    assert refused.reason == "company_has_no_enabled_mode"
    assert approved.modes == frozenset({"invoice"})
    assert approved.modality == "patient_invoice_after"


def test_aucun_moyen_active() -> None:
    decision = prepare(settings=settings())
    assert decision.modes == frozenset()
    assert decision.collection_allowed is False


def test_clinique_payeuse_ne_propose_pas_d_encaissement() -> None:
    decision = prepare(payer="institution", created_via="institution_portal")
    assert decision.reason == "institution_payer"
    assert decision.modes == frozenset()
    assert decision.financial_lock_required is False


def test_patient_institution_non_verrouille() -> None:
    decision = prepare(
        created_via="institution_portal",
        financial_responsibility_locked=False,
        settings=settings(tap_to_pay_enabled=True, cash_enabled=True),
    )
    assert decision.reason == "payer_still_mutable"
    assert decision.modes == frozenset()
    assert decision.financial_lock_required is True
    assert decision.lock_applied is False


def test_patient_institution_verrouille_prepare_les_moyens() -> None:
    decision = prepare(
        created_via="institution_portal",
        financial_responsibility_locked=True,
        settings=settings(cash_enabled=True),
    )
    assert decision.modes == frozenset({"cash"})
    assert decision.lock_applied is False
    assert decision.collection_allowed is False


def test_compte_mensuel_prime_sur_les_moyens_generaux() -> None:
    decision = prepare(
        client_has_established_account=True,
        settings=settings(
            tap_to_pay_enabled=True,
            cash_enabled=True,
            periodic_accounts_enabled=False,
        ),
    )
    assert decision.reason == "established_account"
    assert decision.modes == frozenset()


def test_facturation_periodique_ne_couvre_pas_tous_les_clients() -> None:
    decision = prepare(
        client_has_established_account=False,
        settings=settings(periodic_accounts_enabled=True, cash_enabled=True),
    )
    assert decision.modes == frozenset({"cash"})


def test_invite_deja_paye() -> None:
    decision = prepare(created_via="public_guest", already_paid=True)
    assert decision.reason == "already_settled"
    assert decision.modes == frozenset()
    assert decision.record is False


def test_aller_retour_paye_ne_redemande_rien() -> None:
    decision = prepare(already_paid=True, is_round_trip=True)
    assert decision.reason == "already_settled"
    assert decision.modes == frozenset()


def test_entreprise_absente_n_invente_pas_de_moyens() -> None:
    decision = prepare(company_assigned=False)
    assert decision.reason == "company_not_assigned"
    assert decision.modes == frozenset()


def test_changement_de_capacite_ne_reecrit_pas_la_politique() -> None:
    conflict = capability_change_conflict(
        frozenset({"cash", "tap_to_pay"}),
        settings(tap_to_pay_enabled=True),
    )
    assert conflict == "capability_withdrawn_needs_review"
    assert "cash" not in _remaining(settings(tap_to_pay_enabled=True))


def _remaining(current: CompanySettlementSettings) -> frozenset[str]:
    return _company_modes(current)


def test_confirmation_non_cochee_bloque_l_encaissement() -> None:
    missing = prepare(patient_notice=notice(confirmed=False))
    automatic = prepare(patient_notice=notice(auto_checked=True))
    assert missing.reason == "patient_notice_missing"
    assert automatic.reason == "patient_notice_missing"


def test_ancienne_confirmation_ne_suit_pas_un_nouveau_montant() -> None:
    stale = prepare(announced_amount_chf=Decimal("90.00"))
    current = prepare(announced_amount_chf=Decimal("45.00"))
    assert stale.reason == "patient_notice_stale"
    assert stale.collection_allowed is False
    assert stale.record is False
    assert current.reason == "compatible_modes_prepared"


def test_chauffeur_ne_peut_pas_ajouter_un_moyen() -> None:
    decision = prepare(driver_requested_modes=frozenset({"tap_to_pay"}))
    assert decision.modes == frozenset({"cash"})
    assert decision.institution_modes_ignored is True


def test_institution_ne_definit_pas_un_moyen() -> None:
    decision = prepare(institution_requested_modes=frozenset({"tap_to_pay"}))
    assert "tap_to_pay" not in decision.modes
    assert decision.institution_modes_ignored is True


def test_deux_entreprises_gardent_leurs_reglages() -> None:
    cash_only = prepare(settings=settings(cash_enabled=True, config_version="a"))
    card_only = prepare(
        settings=settings(tap_to_pay_enabled=True, config_version="b")
    )
    assert cash_only.modes == frozenset({"cash"})
    assert card_only.config_version == "b"
    assert cash_only.config_version == "a"


def test_retour_sans_heure_n_autorise_pas_l_encaissement() -> None:
    decision = prepare(is_round_trip=True, return_time_defined=False)
    assert decision.reason == "return_time_missing"
    assert decision.collection_allowed is False


def test_acceptation_reelle_n_ecrit_pas_la_politique() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "application"
        / "institutions"
        / "accept_offer.py"
    ).read_text(encoding="utf-8")
    assert "prepare_acceptance_snapshot" not in source
    assert "booking_settlement_policy" not in source


def test_preference_ne_rend_pas_le_moyen_disponible() -> None:
    effective = settings_for_acceptance(
        preferences=settings(
            cash_preferred=True,
            tap_to_pay_preferred=True,
            direct_patient_invoice_preferred=True,
        ),
        authorized=settings(),
    )
    decision = prepare(settings=effective)
    assert decision.modes == frozenset()
    assert decision.collection_allowed is False
    assert decision.lock_applied is False
    assert acceptance_write_plan(decision) == "no_policy_row"


def test_acceptation_reelle_reste_gatee() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "application"
        / "companies"
        / "accept_reservation.py"
    ).read_text(encoding="utf-8")
    assert "_review_settlement" in source
    from application.billing.settlement_acceptance import (
        note_acceptance_financial_review,
    )
    from domain.end_of_trip_settlement.activation import (
        END_OF_TRIP_COLLECTION_ENABLED,
        SETTLEMENT_ACCEPTANCE_SNAPSHOT_ENABLED,
        SETTLEMENT_COMPANY_SETTINGS_ENABLED,
    )

    assert END_OF_TRIP_COLLECTION_ENABLED is False
    assert SETTLEMENT_ACCEPTANCE_SNAPSHOT_ENABLED is False
    assert SETTLEMENT_COMPANY_SETTINGS_ENABLED is False

    class _Booking:
        id = 1

    assert (
        note_acceptance_financial_review(_Booking(), company_id=3) == "skipped"
    )
