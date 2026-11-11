"""P6-L02 — situation financière lue, jamais réécrite."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from application.invoices.invoice_statement import (
    assemble_statement,
    statement_for_invoice,
)
from models import Invoice, InvoiceCreditNote, InvoiceLine, InvoicePayment, User
from models.enums import InvoiceLineType, InvoiceStatus, PaymentMethod, UserRole
from tests.routes.test_invoices_list_lot6 import _company_headers


def _moment(day: int) -> datetime:
    return datetime(2026, 3, day, 10, 0, tzinfo=UTC)


def _statement(**overrides):
    payload = {
        "total": Decimal("90.00"),
        "issued_at": _moment(1),
        "invoice_number": "F-90",
        "pdf_url": "/uploads/invoices/f-90.pdf",
        "payments": [],
        "credit_notes": [],
        "refunds": [],
    }
    payload.update(overrides)
    return assemble_statement(**payload)


def test_facture_sans_acompte():
    statement = _statement()
    assert statement["prestations_chf"] == "90.00"
    assert statement["avoirs_chf"] == "0.00"
    assert statement["net_facture_chf"] == "90.00"
    assert statement["acomptes_chf"] == "0.00"
    assert statement["paiements_complementaires_chf"] == "0.00"
    assert statement["reste_a_payer_chf"] == "90.00"
    assert statement["a_rembourser_chf"] == "0.00"


def test_acompte_chauffeur_exemple_90_30_20():
    statement = _statement(
        payments=[
            {
                "amount": Decimal("20.00"),
                "paid_at": _moment(2),
                "reference": "ESP-20",
                "cash_collection_id": 15,
            }
        ],
        credit_notes=[
            {
                "amount": Decimal("30.00"),
                "issued_at": _moment(4),
                "number": "AV-1-2026-1",
                "pdf_url": "/uploads/credit_notes/AV-1-2026-1.pdf",
            }
        ],
    )
    assert statement["prestations_chf"] == "90.00"
    assert statement["avoirs_chf"] == "-30.00"
    assert statement["net_facture_chf"] == "60.00"
    assert statement["acomptes_chf"] == "-20.00"
    assert statement["paiements_complementaires_chf"] == "0.00"
    assert statement["reste_a_payer_chf"] == "40.00"
    assert statement["a_rembourser_chf"] == "0.00"
    labels = [event["label"] for event in statement["events"]]
    assert labels == [
        "Facture émise",
        "Acompte chauffeur déjà reçu",
        "Avoir émis",
    ]
    assert "Paiement complémentaire" not in labels


def test_paiement_ulterieur():
    statement = _statement(
        payments=[
            {
                "amount": Decimal("20.00"),
                "paid_at": _moment(3),
                "reference": "VIR-20",
                "cash_collection_id": None,
            }
        ]
    )
    assert statement["acomptes_chf"] == "0.00"
    assert statement["paiements_complementaires_chf"] == "-20.00"
    assert statement["reste_a_payer_chf"] == "70.00"


def test_avoir_partiel_et_total():
    partial = _statement(
        credit_notes=[
            {
                "amount": Decimal("45.00"),
                "issued_at": _moment(4),
                "number": "AV-1-2026-2",
                "pdf_url": None,
            }
        ]
    )
    assert partial["avoirs_chf"] == "-45.00"
    assert partial["net_facture_chf"] == "45.00"
    assert partial["reste_a_payer_chf"] == "45.00"
    assert partial["paiements_complementaires_chf"] == "0.00"
    assert all(
        event["label"] != "Paiement complémentaire" for event in partial["events"]
    )

    full = _statement(
        credit_notes=[
            {
                "amount": Decimal("90.00"),
                "issued_at": _moment(4),
                "number": "AV-1-2026-3",
                "pdf_url": "/uploads/credit_notes/total.pdf",
            }
        ]
    )
    assert full["net_facture_chf"] == "0.00"
    assert full["reste_a_payer_chf"] == "0.00"
    assert full["a_rembourser_chf"] == "0.00"


def test_fixture_45_20_25():
    statement = _statement(
        total=Decimal("45.00"),
        payments=[
            {
                "amount": Decimal("20.00"),
                "paid_at": _moment(2),
                "reference": "VIR-45",
                "cash_collection_id": None,
            }
        ],
    )
    assert statement["prestations_chf"] == "45.00"
    assert statement["paiements_complementaires_chf"] == "-20.00"
    assert statement["reste_a_payer_chf"] == "25.00"


def test_facture_integralement_payee_puis_avoir_partiel():
    statement = _statement(
        payments=[
            {
                "amount": Decimal("90.00"),
                "paid_at": _moment(2),
                "reference": "VIR-90",
                "cash_collection_id": None,
            }
        ],
        credit_notes=[
            {
                "amount": Decimal("30.00"),
                "issued_at": _moment(5),
                "number": "AV-1-2026-4",
                "pdf_url": "/uploads/credit_notes/partiel.pdf",
            }
        ],
    )
    assert statement["reste_a_payer_chf"] == "0.00"
    assert statement["a_rembourser_chf"] == "30.00"
    assert statement["net_facture_chf"] == "60.00"


def test_trop_percu_puis_remboursements():
    overpaid = _statement(
        payments=[
            {
                "amount": Decimal("100.00"),
                "paid_at": _moment(2),
                "reference": "VIR-100",
                "cash_collection_id": None,
            }
        ]
    )
    assert overpaid["reste_a_payer_chf"] == "0.00"
    assert overpaid["a_rembourser_chf"] == "10.00"

    partial = _statement(
        payments=[
            {
                "amount": Decimal("100.00"),
                "paid_at": _moment(2),
                "reference": "VIR-100",
                "cash_collection_id": None,
            }
        ],
        refunds=[
            {
                "amount": Decimal("4.00"),
                "status": "confirmed",
                "confirmed_at": _moment(6),
                "created_at": _moment(6),
                "reference": "R-4",
            }
        ],
    )
    assert partial["remboursements_confirmes_chf"] == "4.00"
    assert partial["a_rembourser_chf"] == "6.00"
    assert partial["reste_a_payer_chf"] == "0.00"

    full = _statement(
        payments=[
            {
                "amount": Decimal("100.00"),
                "paid_at": _moment(2),
                "reference": "VIR-100",
                "cash_collection_id": None,
            }
        ],
        refunds=[
            {
                "amount": Decimal("10.00"),
                "status": "confirmed",
                "confirmed_at": _moment(7),
                "created_at": _moment(7),
                "reference": "R-10",
            }
        ],
    )
    assert full["remboursements_confirmes_chf"] == "10.00"
    assert full["a_rembourser_chf"] == "0.00"
    assert full["reste_a_payer_chf"] == "0.00"


def test_remboursement_non_confirme_ne_diminue_pas_le_trop_percu():
    statement = _statement(
        payments=[
            {
                "amount": Decimal("100.00"),
                "paid_at": _moment(2),
                "reference": "VIR-100",
                "cash_collection_id": None,
            }
        ],
        refunds=[
            {
                "amount": Decimal("10.00"),
                "status": "requested",
                "confirmed_at": None,
                "created_at": _moment(6),
                "reference": "R-attente",
            }
        ],
    )
    assert statement["remboursements_confirmes_chf"] == "0.00"
    assert statement["a_rembourser_chf"] == "10.00"
    assert statement["events"][-1]["label"] == "Remboursement enregistré, non confirmé"


def test_historique_ordonne_et_remise_ignoree():
    statement = _statement(
        payments=[
            {
                "amount": Decimal("80.00"),
                "paid_at": _moment(3),
                "reference": "VIR-80",
                "cash_collection_id": None,
            },
            {
                "amount": Decimal("20.00"),
                "paid_at": _moment(2),
                "reference": "ESP-20",
                "cash_collection_id": 4,
            },
            {
                "amount": Decimal("15.00"),
                "paid_at": _moment(2),
                "reference": "REMISE",
                "kind": "handover",
            },
        ],
        credit_notes=[
            {
                "amount": Decimal("5.00"),
                "issued_at": _moment(4),
                "number": "AV-1-2026-5",
                "pdf_url": "/uploads/credit_notes/av.pdf",
            }
        ],
        refunds=[
            {
                "amount": Decimal("1.00"),
                "status": "confirmed",
                "confirmed_at": _moment(8),
                "created_at": _moment(8),
                "reference": "R-1",
            }
        ],
    )
    assert [event["label"] for event in statement["events"]] == [
        "Facture émise",
        "Acompte chauffeur déjà reçu",
        "Paiement complémentaire",
        "Avoir émis",
        "Remboursement confirmé",
    ]
    assert all(event["reference"] != "REMISE" for event in statement["events"])
    assert statement["acomptes_chf"] == "-20.00"
    assert statement["paiements_complementaires_chf"] == "-80.00"
    assert statement["reste_a_payer_chf"] == "0.00"


def test_documents_et_montant_inconnu():
    statement = _statement(pdf_url=None, credit_notes=None, refunds=None)
    assert statement["documents"][0]["disponible"] is False
    assert statement["avoirs_chf"] is None
    assert statement["net_facture_chf"] is None
    assert statement["a_rembourser_chf"] is None

    missing_note = _statement(
        credit_notes=[
            {
                "amount": Decimal("30.00"),
                "issued_at": _moment(4),
                "number": "AV-1-2026-6",
                "pdf_url": None,
            }
        ]
    )
    avoir = missing_note["documents"][1]
    assert avoir["disponible"] is False
    assert avoir["pdf_url"] is None
    assert missing_note["documents"][0]["pdf_url"] == "/uploads/invoices/f-90.pdf"


def test_statement_for_invoice_ne_reecrit_pas(monkeypatch):
    invoice = SimpleNamespace(
        id=3,
        company_id=1,
        total_amount=Decimal("90.00"),
        balance_due=Decimal("40.00"),
        refundable_chf=Decimal("0.00"),
        issued_at=_moment(1),
        invoice_number="F-90",
        pdf_url="/uploads/invoices/f-90.pdf",
        payments=[
            SimpleNamespace(
                amount=Decimal("20.00"),
                paid_at=_moment(2),
                reference="ESP-20",
                cash_collection_id=15,
            )
        ],
    )

    def _forbid():
        raise AssertionError("écriture interdite")

    invoice.update_balance = _forbid
    monkeypatch.setattr(
        "application.invoices.invoice_statement._load_credit_notes",
        lambda _invoice: [
            {
                "amount": Decimal("30.00"),
                "issued_at": _moment(4),
                "number": "AV-1-2026-1",
                "pdf_url": "/uploads/credit_notes/AV-1-2026-1.pdf",
            }
        ],
    )
    monkeypatch.setattr(
        "application.invoices.invoice_statement._load_refunds",
        lambda _invoice: [],
    )
    result = statement_for_invoice(invoice)
    assert result["reste_a_payer_chf"] == "40.00"
    assert result["net_facture_chf"] == "60.00"
    assert invoice.pdf_url == "/uploads/invoices/f-90.pdf"


def test_montants_stockes_incoherents(monkeypatch):
    invoice = SimpleNamespace(
        id=4,
        company_id=1,
        total_amount=Decimal("90.00"),
        balance_due=Decimal("90.00"),
        refundable_chf=Decimal("0.00"),
        issued_at=_moment(1),
        invoice_number="F-90",
        pdf_url="/uploads/invoices/f-90.pdf",
        payments=[],
    )
    monkeypatch.setattr(
        "application.invoices.invoice_statement._load_credit_notes",
        lambda _invoice: [
            {
                "amount": Decimal("30.00"),
                "issued_at": _moment(4),
                "number": "AV-1-2026-1",
                "pdf_url": None,
            }
        ],
    )
    monkeypatch.setattr(
        "application.invoices.invoice_statement._load_refunds",
        lambda _invoice: [],
    )
    result = statement_for_invoice(invoice)
    assert result["disponible"] is False
    assert "ne concordent pas" in result["message"]
    assert result["reste_a_payer_chf"] is None
    assert result["documents"][0]["pdf_url"] == "/uploads/invoices/f-90.pdf"


def _issued(company, client, user) -> Invoice:
    suffix = uuid.uuid4().hex[:8]
    invoice = Invoice(
        company=company,
        client=client,
        invoice_number=f"P6L02-{suffix}",
        period_year=2026,
        period_month=3,
        status=InvoiceStatus.SENT,
        issued_at=_moment(1),
        due_date=_moment(20),
        subtotal_amount=Decimal("90.00"),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal("90.00"),
        amount_paid=Decimal("0.00"),
        balance_due=Decimal("90.00"),
        refundable_chf=Decimal("0.00"),
        pdf_url="/uploads/invoices/p6l02-original.pdf",
    )
    from ext import db

    db.session.add(invoice)
    db.session.flush()
    db.session.add(
        InvoiceLine(
            invoice=invoice,
            type=InvoiceLineType.RIDE,
            description="Transport contractuel",
            qty=Decimal("1.00"),
            unit_price=Decimal("90.00"),
            line_total=Decimal("90.00"),
            vat_rate=Decimal("0.00"),
            vat_amount=Decimal("0.00"),
            total_with_vat=Decimal("90.00"),
        )
    )
    db.session.add(
        InvoicePayment(
            invoice_id=invoice.id,
            amount=Decimal("20.00"),
            paid_at=_moment(3),
            method=PaymentMethod.BANK_TRANSFER,
            reference=f"VIR-{suffix}",
        )
    )
    db.session.add(
        InvoiceCreditNote(
            company_id=company.id,
            invoice_id=invoice.id,
            credit_number=f"AV-{company.id}-2026-{invoice.id}",
            amount_chf=Decimal("30.00"),
            reason="Geste commercial",
            status="issued",
            issued_at=_moment(4),
            created_by_user_id=user.id,
            idempotency_key=uuid.uuid4().hex,
            pdf_url=None,
        )
    )
    db.session.flush()
    invoice.update_balance()
    db.session.flush()
    return invoice


@pytest.mark.integration
def test_consultation_get_ne_cree_rien(
    client, db, sample_company, sample_user, sample_client
):
    invoice = _issued(sample_company, sample_client, sample_user)
    before = InvoicePayment.query.filter_by(invoice_id=invoice.id).count()
    original_pdf = invoice.pdf_url
    original_total = invoice.total_amount
    headers = _company_headers(client, sample_user, sample_company.id)
    url = f"/api/v1/invoices/companies/{sample_company.id}/invoices/{invoice.id}"
    response = client.get(url, headers=headers)
    assert response.status_code == 200
    body = response.get_json()["data"]["statement"]
    assert body["prestations_chf"] == "90.00"
    assert body["avoirs_chf"] == "-30.00"
    assert body["net_facture_chf"] == "60.00"
    assert body["paiements_complementaires_chf"] == "-20.00"
    assert body["reste_a_payer_chf"] == "40.00"
    assert body["documents"][1]["disponible"] is False
    db.session.refresh(invoice)
    assert invoice.pdf_url == original_pdf
    assert invoice.total_amount == original_total
    assert InvoicePayment.query.filter_by(invoice_id=invoice.id).count() == before


@pytest.mark.integration
def test_autre_entreprise_et_chauffeur_refuses(
    client, db, sample_company, sample_user, sample_client
):
    invoice = _issued(sample_company, sample_client, sample_user)
    headers = _company_headers(client, sample_user, sample_company.id)
    foreign = client.get(
        f"/api/v1/invoices/companies/{sample_company.id + 900000}/invoices/{invoice.id}",
        headers=headers,
    )
    assert foreign.status_code == 403
    assert "refus" in json.dumps(foreign.get_json()).lower()

    driver = User()
    driver.username = f"p6l02_driver_{uuid.uuid4().hex[:8]}"
    driver.email = f"p6l02-{uuid.uuid4().hex[:8]}@test.ch"
    driver.role = UserRole.DRIVER
    driver.public_id = str(uuid.uuid4())
    driver.set_password("password123", force_change=False)
    db.session.add(driver)
    db.session.flush()
    refused = client.get(
        f"/api/v1/invoices/companies/{sample_company.id}/invoices/{invoice.id}",
        headers=_company_headers(client, driver, sample_company.id),
    )
    assert refused.status_code == 403
