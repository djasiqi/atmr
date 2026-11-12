"""Étape 33. Couverture d'un aller-retour prépayé et arrondi TVA au 5 centimes."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal

import pytest

from infrastructure.invoices.invoice_calculator import (
    InvoiceCalculator,
    round_to_5_cents,
)
from models.enums import PaymentStatus
from tests.application.settlement.test_billing_certification_etape31 import (
    _booking,
    _generate,
    _preview,
    prep,
    world,
)
from tests.application.settlement.test_billing_certification_etape32 import (
    _ids,
    _lines,
    _pay,
)

pytestmark = pytest.mark.integration
_ = (prep, world)


def test_aller_retour_groupe_sans_ventiler_un_excedent_independant(client, world):
    """90 CHF sur un aller-retour lié retirent les deux trajets.

    90 CHF sur une course sans retour ne soldent pas une autre réservation.
    Un écart de somme suspend le groupe, sans ventilation.
    """
    outbound = _booking(
        world,
        amount=Decimal("45.00"),
        is_round_trip=True,
        pickup_location="Aller groupe 1, 1000 Recette",
        dropoff_location="Aller groupe 2, 1000 Recette",
    )
    retour = _booking(
        world,
        amount=Decimal("45.00"),
        parent_booking_id=outbound.id,
        is_return=True,
        scheduled_time=datetime(2026, 10, 15, 18, 0, tzinfo=UTC),
        pickup_location="Aller groupe 2, 1000 Recette",
        dropoff_location="Aller groupe 1, 1000 Recette",
    )
    _pay(outbound, 90.0, PaymentStatus.COMPLETED)

    lone = _booking(
        world,
        amount=Decimal("45.00"),
        pickup_location="Course seule 9, 1000 Recette",
        dropoff_location="Course seule 3, 1000 Recette",
    )
    _pay(lone, 90.0, PaymentStatus.COMPLETED)
    neighbor = _booking(
        world,
        amount=Decimal("45.00"),
        pickup_location="Voisine 4, 1000 Recette",
        dropoff_location="Voisine 8, 1000 Recette",
    )

    odd_out = _booking(
        world,
        amount=Decimal("45.00"),
        is_round_trip=True,
        pickup_location="Flou 1, 1000 Recette",
        dropoff_location="Flou 2, 1000 Recette",
    )
    odd_back = _booking(
        world,
        amount=Decimal("30.00"),
        parent_booking_id=odd_out.id,
        is_return=True,
        scheduled_time=datetime(2026, 10, 16, 18, 0, tzinfo=UTC),
        pickup_location="Flou 2, 1000 Recette",
        dropoff_location="Flou 1, 1000 Recette",
    )
    _pay(odd_out, 90.0, PaymentStatus.COMPLETED)

    preview = _preview(client, world, year=2026, month=10, client_id=world["client"].id)
    selected = _ids(preview)
    text = " ".join(preview.get("warnings") or [])
    assert outbound.id not in selected
    assert retour.id not in selected
    assert lone.id not in selected
    assert neighbor.id in selected
    assert odd_out.id not in selected
    assert odd_back.id not in selected
    assert "n'est pas affecté" in text
    assert "suspendue" in text

    blocked = _generate(client, world, [retour.id])
    assert blocked.status_code in {400, 409, 422}
    assert _lines(retour.id) == []
    assert _lines(odd_back.id) == []


def test_tva_arrondie_au_cinq_centimes_puis_residuelle():
    """7,7 % de 12,00 font 0,924, soit 0,92 au centime.

    Le total payable est ensuite arrondi au 5 centimes : 12,92 devient 12,90.
    La TVA affichée est le résidu 0,90, pas un arrondi séparé à 0,92.
    """
    base = Decimal("12.00")
    rate = Decimal("7.7")
    exact = base * rate / Decimal("100")
    assert exact == Decimal("0.924")
    to_cent = exact.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    assert to_cent == Decimal("0.92")
    payable_before = base + to_cent
    assert payable_before == Decimal("12.92")
    assert round_to_5_cents(payable_before) == Decimal("12.90")
    vat, total = InvoiceCalculator().calculate_vat(base, rate)
    assert total == Decimal("12.90")
    assert vat == total - base
    assert vat == Decimal("0.90")
