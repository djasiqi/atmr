"""Décision réelle et fixtures isolées. Aucune base, aucun prestataire."""

from __future__ import annotations

from decimal import Decimal

from domain.end_of_trip_settlement.activation import (
    END_OF_TRIP_COLLECTION_ENABLED,
    SETTLEMENT_DRIVER_PANEL_ENABLED,
    driver_panel_enabled,
    simulation_preview_allowed,
)
from domain.end_of_trip_settlement.mission_facts import (
    MissionFacts,
    decide_mission,
    enabled_company_modes,
    policy_at,
    simulation_catalog,
    simulation_scenario,
)
from domain.end_of_trip_settlement.prototypes import (
    TAP_IN_PROGRESS,
    TAP_INTERRUPTED,
    open_tap_attempt,
    resolve_tap_simulation,
)


def facts(**overrides: object) -> MissionFacts:
    base: dict[str, object] = {
        "booking_id": 1,
        "company_id": 10,
        "executing_company_id": 10,
        "reader_driver_id": 7,
        "reader_company_id": 10,
        "assigned_driver_id": 7,
        "created_via": "client_app",
        "billed_to_type": "patient",
        "billing_intent": None,
        "billing_party_id": 5,
        "operational_status": "COMPLETED",
        "leg_amount_chf": Decimal("45.00"),
        "group_amount_chf": None,
        "is_round_trip": False,
        "is_return_leg": False,
        "priced_leg_count": 1,
        "invoice_already_issued": False,
        "completed_payment_amount_chf": Decimal("0.00"),
        "debtor_display_name": "Patient fictif",
        "policy": policy_at(
            modality="on_site",
            modes=frozenset({"tap_to_pay", "cash", "invoice"}),
            locked=True,
        ),
        "tap_to_pay_enabled": False,
        "cash_enabled": False,
        "direct_patient_invoice_enabled": False,
        "tap_to_pay_preferred": True,
        "cash_preferred": True,
        "direct_patient_invoice_preferred": True,
        "attestation_recorded": True,
    }
    base.update(overrides)
    return MissionFacts(**base)  # type: ignore[arg-type]


def test_panneau_et_encaissement_restent_coupes() -> None:
    assert END_OF_TRIP_COLLECTION_ENABLED is False
    assert SETTLEMENT_DRIVER_PANEL_ENABLED is False
    assert driver_panel_enabled() is False
    assert (
        simulation_preview_allowed(
            panel_enabled=True, production=True, recipe_database=True
        )
        is False
    )
    assert (
        simulation_preview_allowed(
            panel_enabled=True, production=False, recipe_database=False
        )
        is False
    )
    assert (
        simulation_preview_allowed(
            panel_enabled=False, production=False, recipe_database=True
        )
        is False
    )


def test_preferences_ne_sont_pas_des_autorisations() -> None:
    reading = decide_mission(facts())
    assert enabled_company_modes(facts()) == frozenset()
    assert reading["modes"] == []
    assert reading["payment_executable"] is False
    assert reading["reason"] == "no_authorized_mode"
    assert reading["simulation"] is False


def test_scenarios_reels_sans_bouton() -> None:
    paid = decide_mission(
        facts(
            policy=None,
            completed_payment_amount_chf=Decimal("45.00"),
            tap_to_pay_preferred=False,
        )
    )
    institution = decide_mission(
        facts(
            policy=None,
            billed_to_type="clinic",
            billing_intent="institution",
            created_via="institution_portal",
        )
    )
    account = decide_mission(
        facts(
            policy=policy_at(
                modality="established_account", modes=frozenset(), locked=True
            )
        )
    )
    unlocked = decide_mission(
        facts(
            created_via="institution_portal",
            billed_to_type="patient",
            billing_intent="patient",
            policy=policy_at(
                modality="on_site", modes=frozenset({"cash"}), locked=False
            ),
        )
    )
    missing = decide_mission(facts(policy=None))
    assert paid["kind"] == "already_paid"
    assert institution["reason"] == "institution_payer"
    assert account["reason"] == "established_account"
    assert unlocked["reason"] == "payer_still_mutable"
    assert missing["reason"] == "settlement_policy_missing"
    for reading in (paid, institution, account, unlocked, missing):
        assert reading["modes"] == []
        assert reading["writes_completed_payment"] is False


def test_aller_retour_et_acces() -> None:
    paid_policy = policy_at(
        modality="prepaid",
        modes=frozenset(),
        locked=True,
        payment_scope="group",
        amount_chf="90.00",
        round_trip_collection_approved=True,
    )
    paid_group = facts(
        is_round_trip=True,
        priced_leg_count=2,
        leg_amount_chf=Decimal("45.00"),
        group_amount_chf=Decimal("90.00"),
        completed_payment_amount_chf=Decimal("90.00"),
        policy=paid_policy,
    )
    first = decide_mission(paid_group)
    second = decide_mission(
        facts(
            is_round_trip=True,
            priced_leg_count=2,
            leg_amount_chf=Decimal("45.00"),
            group_amount_chf=Decimal("90.00"),
            completed_payment_amount_chf=Decimal("90.00"),
            policy=paid_policy,
            reader_driver_id=8,
            assigned_driver_id=9,
        )
    )
    outbound = decide_mission(
        facts(
            is_round_trip=True,
            priced_leg_count=2,
            group_amount_chf=Decimal("90.00"),
            policy=policy_at(
                modality="on_site",
                modes=frozenset({"cash"}),
                locked=True,
                payment_scope="group",
                amount_chf="90.00",
            ),
        )
    )
    unscheduled = decide_mission(
        facts(
            is_round_trip=True,
            is_return_leg=True,
            operational_status="ACCEPTED",
            priced_leg_count=2,
            group_amount_chf=Decimal("90.00"),
            policy=policy_at(
                modality="on_site",
                modes=frozenset({"cash"}),
                locked=True,
                payment_scope="group",
                amount_chf="90.00",
            ),
        )
    )
    other_company = decide_mission(
        facts(
            is_round_trip=True,
            is_return_leg=True,
            executing_company_id=99,
            priced_leg_count=2,
            group_amount_chf=Decimal("90.00"),
            operational_status="COMPLETED",
        )
    )
    assert first["kind"] == "already_paid"
    assert second["reason"] == "driver_unassigned"
    assert second["amount_chf"] is None
    assert outbound["modes"] == []
    assert unscheduled["payment_executable"] is False
    assert other_company["reason"] == "company_not_owner"
    assert first["attestation_separate"] is True


def test_simulation_isolee_n_ecrit_rien() -> None:
    catalog = {item["id"] for item in simulation_catalog()}
    assert "round-trip-paid" in catalog
    refused = simulation_scenario("card-refused")
    paid = simulation_scenario("already-paid")
    institution = simulation_scenario("institution")
    assert refused["preview_screen"] == "tap-refused"
    assert refused["payment_executable"] is False
    assert refused["writes_completed_payment"] is False
    assert paid["modes"] == []
    assert institution["reason"] == "institution_payer"
    try:
        simulation_scenario("reservation-reelle")
    except KeyError:
        denied = True
    else:
        denied = False
    assert denied is True
    started = open_tap_attempt(
        current_state="ready",
        current_attempt_id=None,
        amount_chf=Decimal("45.00"),
        currency="CHF",
        merchant_company_id=10,
        new_attempt_id="sim-stop",
    )
    stopped = resolve_tap_simulation(started, TAP_INTERRUPTED)
    assert started.state == TAP_IN_PROGRESS
    assert stopped.state == TAP_INTERRUPTED
    assert stopped.may_start_another_debit is False


def test_le_serveur_declare_le_panneau() -> None:
    from services.recipe_environment import (
        disclose_driver_settlement_panel,
        recipe_target_payload,
    )

    closed = disclose_driver_settlement_panel(
        recipe_target_payload("atmr_test"), panel_enabled=False
    )
    assert closed["writable_recipe"] is True
    assert closed["driver_settlement_panel"] is False
    assert driver_panel_enabled() is False


def test_parcours_fictifs_sans_ecriture() -> None:
    institution = decide_mission(
        facts(
            policy=None,
            billed_to_type="clinic",
            billing_intent="institution",
            created_via="institution_portal",
            operational_status="IN_PROGRESS",
        )
    )
    unlocked = decide_mission(
        facts(
            created_via="institution_portal",
            billed_to_type="patient",
            billing_intent="patient",
            operational_status="IN_PROGRESS",
            policy=policy_at(
                modality="on_site", modes=frozenset({"cash"}), locked=False
            ),
        )
    )
    prepaid = decide_mission(
        facts(
            policy=None,
            created_via="public_guest",
            completed_payment_amount_chf=Decimal("45.00"),
            operational_status="COMPLETED",
        )
    )
    monthly = decide_mission(
        facts(
            policy=policy_at(
                modality="established_account", modes=frozenset(), locked=True
            ),
            operational_status="COMPLETED",
        )
    )
    missing = decide_mission(facts(policy=None, operational_status="IN_PROGRESS"))
    group = decide_mission(
        facts(
            is_round_trip=True,
            priced_leg_count=2,
            leg_amount_chf=Decimal("45.00"),
            group_amount_chf=Decimal("90.00"),
            completed_payment_amount_chf=Decimal("90.00"),
            operational_status="COMPLETED",
            policy=policy_at(
                modality="prepaid",
                modes=frozenset(),
                locked=True,
                payment_scope="group",
                amount_chf="90.00",
                round_trip_collection_approved=True,
            ),
        )
    )
    capable = decide_mission(
        facts(
            tap_to_pay_enabled=True,
            cash_enabled=True,
            direct_patient_invoice_enabled=True,
            operational_status="IN_PROGRESS",
        )
    )
    mocked = simulation_scenario("several-means")
    assert institution["reason"] == "institution_payer"
    assert unlocked["reason"] == "payer_still_mutable"
    assert prepaid["kind"] == "already_paid"
    assert monthly["reason"] == "established_account"
    assert missing["reason"] == "settlement_policy_missing"
    assert group["kind"] == "already_paid"
    assert capable["modes"] == []
    assert mocked["simulation"] is True
    assert mocked["modes"] == ["tap_to_pay", "cash", "invoice"]
    assert mocked["writes_completed_payment"] is False
    for reading in (institution, unlocked, prepaid, monthly, missing, group, capable):
        assert reading["modes"] == []
        assert reading["payment_executable"] is False
        assert reading["completion_independent"] is True
        assert reading["attestation_separate"] is True
