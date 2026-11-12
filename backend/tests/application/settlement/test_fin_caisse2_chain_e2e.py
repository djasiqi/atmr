"""FIN-CAISSE-2 — chaîne caisse bout-en-bout sur base jetable.

Parcours : ouverture → déclaration → remise → confirmation → écart.
Flags produit restent OFF ; seule la préparation locale de recette s'ouvre.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from application.settlement.cash_ledger import (
    confirm_handover,
    declare_cash,
    declare_handover,
    open_collection,
    reconcile,
)
from application.settlement.production_gates import require_prep
from domain.end_of_trip_settlement.activation import END_OF_TRIP_COLLECTION_ENABLED
from domain.end_of_trip_settlement.production_prep_barrier import (
    PrepClosed,
    PrepRefusal,
    prep_open,
)
from ext import db as app_db
from models import CashCollection, CashHandover, CashReconciliation
from services.recipe_environment import database_name
from tests.application.settlement.test_cash_position_p4l01 import _world

pytestmark = pytest.mark.integration


@pytest.fixture
def recipe_db():
    current = database_name() or ""
    if not current.startswith(("lirie_e2e_", "lirie_sim_", "atmr_etape")):
        pytest.skip(f"Base refusée pour cette recette : {current or 'absente'}")


def test_flags_produit_restent_eteints(recipe_db):
    assert END_OF_TRIP_COLLECTION_ENABLED is False


def test_prep_fermee_bloque_les_ecritures(db, recipe_db, monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "0")
    assert prep_open() is False
    world = _world(db, due="40.00")
    with pytest.raises(PrepClosed):
        require_prep()
    with pytest.raises(PrepClosed):
        open_collection(
            world["booking"],
            driver_id=world["driver"].id,
            company_id=world["company"].id,
            actor_id=world["driver_user"].id,
        )


def test_chaine_complete_partielle_remise_ecart_idempotence(db, recipe_db, monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")
    # Flags produit OFF : la lecture chauffeur n'offre pas encore les espèces.
    # On isole la chaîne ledger (idempotence / remise / écart) sans activer le produit.
    monkeypatch.setattr(
        "application.settlement.cash_ledger.require_new_cash_declaration",
        lambda *args, **kwargs: None,
    )
    assert prep_open() is True
    world = _world(db, due="40.00")
    booking = world["booking"]
    driver_id = world["driver"].id
    company_id = world["company"].id
    actor = world["driver_user"].id
    owner = world["owner"].id

    opened = open_collection(
        booking,
        driver_id=driver_id,
        company_id=company_id,
        actor_id=actor,
    )
    collection_id = int(opened["id"])

    # Encaissement complet (état received) puis remise partielle : la remise
    # n'exige pas un second encaissement pour le même booking.
    declared = declare_cash(
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": "fc2-declare",
            "tendered_chf": "40.00",
            "change_chf": "0.00",
            "change_confirmed": True,
        },
        driver_id=driver_id,
        company_id=company_id,
        actor_id=actor,
    )
    assert declared.get("replay") is not True
    assert declared["state"] == "received"
    assert Decimal(str(declared["net_chf"])) == Decimal("40.00")

    replay_declare = declare_cash(
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": "fc2-declare",
            "tendered_chf": "40.00",
            "change_chf": "0.00",
            "change_confirmed": True,
        },
        driver_id=driver_id,
        company_id=company_id,
        actor_id=actor,
    )
    assert replay_declare.get("replay") is True
    assert (
        CashCollection.query.filter_by(booking_id=booking.id, kind="collection").count()
        == 1
    )

    handover = declare_handover(
        booking,
        driver_id=driver_id,
        company_id=company_id,
        actor_id=actor,
        idempotency_key="fc2-remise",
        amount_chf="25.00",
    )
    assert handover["handover_state"] == "declared"
    assert handover.get("replay") is not True
    assert CashHandover.query.filter_by(collection_id=collection_id).count() == 1
    handover_id = int(handover["handover_id"])

    replay_handover = declare_handover(
        booking,
        driver_id=driver_id,
        company_id=company_id,
        actor_id=actor,
        idempotency_key="fc2-remise",
        amount_chf="25.00",
    )
    assert replay_handover.get("replay") is True
    assert CashHandover.query.filter_by(collection_id=collection_id).count() == 1

    confirmed = confirm_handover(
        collection_id,
        company_id=company_id,
        actor_id=owner,
        idempotency_key="fc2-remise",
    )
    assert confirmed.get("replay") is not True
    assert confirmed["handover_state"] == "confirmed"

    again = confirm_handover(
        collection_id,
        company_id=company_id,
        actor_id=owner,
        idempotency_key="fc2-remise",
    )
    assert again.get("replay") is True

    counted = reconcile(
        collection_id,
        {
            "counted_chf": "24.00",
            "idempotency_key": "fc2-ecart",
            "handover_id": handover_id,
            "reason": "manque une piece",
        },
        company_id=company_id,
        actor_id=owner,
    )
    assert counted.get("replay") is not True
    assert counted["reconciliation_variance_chf"] == "-1.00"
    assert CashReconciliation.query.filter_by(collection_id=collection_id).count() == 1

    # Remise du solde restant (15.00) pour couvrir encaissement partiel déclaré ≠ remis total
    second = declare_handover(
        booking,
        driver_id=driver_id,
        company_id=company_id,
        actor_id=actor,
        idempotency_key="fc2-remise-reste",
        amount_chf="15.00",
    )
    assert second["handover_state"] == "declared"
    assert Decimal(str(second["handover_amount_chf"])) == Decimal("15.00")

    foreign = _world(db, due="40.00")
    with pytest.raises(PrepRefusal) as refused:
        confirm_handover(
            collection_id,
            company_id=foreign["company"].id,
            actor_id=foreign["owner"].id,
            idempotency_key="fc2-intrus",
        )
    assert refused.value.status in {403, 404}
