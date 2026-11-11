"""Étape 35 : même clôture TVA pour l'aperçu et la préparation, identité du spécimen."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from infrastructure.invoices.invoice_calculator import (
    InvoiceCalculator,
    finalize_invoice_vat,
)
from services.documents.invoice_recipient import _patient_self_party


def test_merged_round_trip_uses_the_same_five_cent_close_as_preparation():
    calculator = InvoiceCalculator()
    rate = Decimal("7.7")
    separate = sum(
        calculator.calculate_vat(Decimal("45.00"), rate)[0] for _ in range(2)
    )
    merged = calculator.calculate_vat(Decimal("90.00"), rate)[0]
    assert separate == Decimal("6.90")
    assert merged == Decimal("6.95")

    lines = [
        Decimal("45.00"),
        Decimal("90.00"),
        Decimal("45.00"),
        Decimal("40.00"),
        Decimal("20.00"),
    ]
    vat_sum = sum(calculator.calculate_vat(amount, rate)[0] for amount in lines)
    ht, vat_final, total = finalize_invoice_vat(sum(lines, Decimal("0.00")), vat_sum)
    assert (ht, vat_final, total) == (
        Decimal("240.00"),
        Decimal("18.50"),
        Decimal("258.50"),
    )


def test_institution_and_partner_samples_stay_exact():
    calculator = InvoiceCalculator()
    rate = Decimal("7.7")

    def close(amounts: list[str]) -> tuple[Decimal, Decimal, Decimal]:
        values = [Decimal(amount) for amount in amounts]
        vat_sum = sum(calculator.calculate_vat(amount, rate)[0] for amount in values)
        return finalize_invoice_vat(sum(values, Decimal("0.00")), vat_sum)

    assert close(["80.00", "70.00"]) == (
        Decimal("150.00"),
        Decimal("11.55"),
        Decimal("161.55"),
    )
    assert close(["40.00", "25.00", "12.00"]) == (
        Decimal("77.00"),
        Decimal("5.95"),
        Decimal("82.95"),
    )


def test_specimen_pdf_uses_billing_identity_not_the_technical_login():
    invoice = SimpleNamespace(
        client=SimpleNamespace(
            is_institution=False,
            user=SimpleNamespace(
                first_name="",
                last_name="",
                username="patient-recette-15",
                address=None,
            ),
            domicile_address=None,
            domicile_zip=None,
            domicile_city=None,
            billing_address=None,
            residence_facility=None,
        ),
        lines=[],
    )
    party = SimpleNamespace(
        display_name="Patient brouillon chauffeur",
        billing_address="Adresse fictive, recette",
    )
    billed = _patient_self_party(invoice, bookings_by_id={}, billing_party=party)
    assert billed.addressee == "Patient brouillon chauffeur"
    assert "patient-recette-15" not in billed.addressee


def test_long_and_missing_billing_names_do_not_invent_a_person():
    long_name = "Patient " + ("Très-long " * 12)
    invoice = SimpleNamespace(
        client=SimpleNamespace(
            is_institution=False,
            user=SimpleNamespace(
                first_name=long_name[:100],
                last_name="",
                username="patient-recette-99",
                address=None,
            ),
            domicile_address=None,
            domicile_zip=None,
            domicile_city=None,
            billing_address=None,
            residence_facility=None,
        ),
        lines=[],
    )
    party = SimpleNamespace(display_name=long_name.strip(), billing_address="")
    billed = _patient_self_party(invoice, bookings_by_id={}, billing_party=party)
    assert billed.addressee.startswith("Patient")
    assert "patient-recette-99" not in billed.addressee

    empty = SimpleNamespace(
        client=SimpleNamespace(
            is_institution=False,
            user=SimpleNamespace(
                first_name="",
                last_name="",
                username="patient-recette-0",
                address=None,
            ),
            domicile_address=None,
            domicile_zip=None,
            domicile_city=None,
            billing_address=None,
            residence_facility=None,
        ),
        lines=[],
    )
    missing = _patient_self_party(
        empty,
        bookings_by_id={},
        billing_party=SimpleNamespace(display_name="Patient", billing_address=""),
    )
    assert missing.addressee == "Patient"
