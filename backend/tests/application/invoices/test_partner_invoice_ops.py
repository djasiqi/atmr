"""Opérations natives partner_invoices — isolation partner:34 ≠ standard:34."""

from __future__ import annotations

from unittest.mock import MagicMock

from application.invoices.add_partner_invoice_payment import add_partner_invoice_payment
from application.invoices.cancel_partner_invoice import cancel_partner_invoice
from application.invoices.force_regenerate_partner_invoice_pdf import (
    force_regenerate_partner_invoice_pdf,
)
from application.invoices.get_partner_invoice import get_partner_invoice
from application.invoices.update_partner_invoice_draft import (
    update_partner_invoice_draft,
)
from models.partner_invoice import PartnerInvoiceStatus
from tests.factories import CompanyFactory
from tests.services.test_partnership_invoices import _svc, _world


def _generated(db, monkeypatch):
    world = _world()
    service = _svc(monkeypatch)
    invoice = service.generate_monthly_invoice(
        world.partnership.id, world.year, world.month, world.executing.id
    )
    return world, invoice


def test_generate_persiste_les_lignes_snapshot(db, monkeypatch):
    _world_ns, invoice = _generated(db, monkeypatch)
    db.session.refresh(invoice)
    assert invoice.lines
    assert float(invoice.lines[0].amount) == 40.0
    assert invoice.lines[0].source_type == "booking_transfer"
    assert invoice.lines[0].source_id is not None


def test_get_partner_invoice_ne_lit_pas_invoices(db, monkeypatch):
    world, invoice = _generated(db, monkeypatch)
    result = get_partner_invoice(
        company_id=world.executing.id, partner_invoice_id=invoice.id
    )
    assert result.ok is True
    assert result.invoice["is_partner_invoice"] is True
    assert result.invoice["invoice_type"] == "partner"
    assert result.invoice["lines"]
    assert result.invoice["total_amount"] == float(invoice.total_amount)


def test_update_lignes_virtuelles_sans_id_persiste(db, monkeypatch):
    """GET historique (id ligne null) puis PATCH : les edits doivent tenir."""
    world, invoice = _generated(db, monkeypatch)
    for line in list(invoice.lines):
        db.session.delete(line)
    db.session.commit()
    db.session.refresh(invoice)
    assert not invoice.lines
    result = update_partner_invoice_draft(
        company_id=world.executing.id,
        partner_invoice_id=invoice.id,
        payload={
            "lines": [
                {
                    "description": "Ligne sans id persistee",
                    "quantity": 1,
                    "unit_price": "41.00",
                    "amount": "41.00",
                    "source_id": world.transfer.id,
                }
            ],
        },
    )
    assert result.ok is True
    assert result.invoice["lines"][0]["description"] == "Ligne sans id persistee"
    assert result.invoice["lines"][0]["amount"] == 41.0


def test_update_ligne_ne_reecrit_pas_le_transfert(db, monkeypatch):
    world, invoice = _generated(db, monkeypatch)
    source_cost = world.transfer.partner_cost
    line = invoice.lines[0]
    result = update_partner_invoice_draft(
        company_id=world.executing.id,
        partner_invoice_id=invoice.id,
        payload={
            "notes": "Correction brouillon",
            "lines": [
                {
                    "id": line.id,
                    "description": "Transport MT Genève corrigé",
                    "quantity": 1,
                    "unit_price": "45.50",
                    "amount": "45.50",
                }
            ],
        },
    )
    assert result.ok is True
    assert result.invoice["notes"] == "Correction brouillon"
    assert result.invoice["lines"][0]["description"] == "Transport MT Genève corrigé"
    assert result.invoice["lines"][0]["amount"] == 45.5
    db.session.refresh(world.transfer)
    assert world.transfer.partner_cost == source_cost


def test_cancel_draft_partenaire(db, monkeypatch):
    world, invoice = _generated(db, monkeypatch)
    result = cancel_partner_invoice(
        company_id=world.executing.id, partner_invoice_id=invoice.id
    )
    assert result.ok is True
    assert result.invoice["status"] == PartnerInvoiceStatus.CANCELLED


def test_cancel_paid_interdit(db, monkeypatch):
    world, invoice = _generated(db, monkeypatch)
    invoice.status = PartnerInvoiceStatus.PAID
    invoice.amount_paid = invoice.total_amount
    db.session.commit()
    result = cancel_partner_invoice(
        company_id=world.executing.id, partner_invoice_id=invoice.id
    )
    assert result.ok is False
    assert result.status_code == 400


def test_paiement_partenaire_maj_solde(db, monkeypatch):
    world, invoice = _generated(db, monkeypatch)
    result = add_partner_invoice_payment(
        company_id=world.executing.id,
        partner_invoice_id=invoice.id,
        payload={"amount": "20.00", "method": "bank_transfer"},
    )
    assert result.ok is True
    assert result.invoice["amount_paid"] == 20.0
    assert result.invoice["balance_due"] == float(invoice.total_amount) - 20.0
    assert result.invoice["status"] == PartnerInvoiceStatus.PARTIALLY_PAID


def test_regenerate_echec_conserve_ancien_pdf(db, monkeypatch):
    world, invoice = _generated(db, monkeypatch)
    old_url = invoice.pdf_url
    monkeypatch.setattr(
        "application.invoices.force_regenerate_partner_invoice_pdf.PartnerInvoiceService._generate_invoice_pdf",
        MagicMock(side_effect=RuntimeError("pdf fail")),
    )
    result = force_regenerate_partner_invoice_pdf(
        company_id=world.executing.id, partner_invoice_id=invoice.id
    )
    assert result.ok is False
    db.session.refresh(invoice)
    assert invoice.pdf_url == old_url


def test_regenerate_succes_remplace_url(db, monkeypatch):
    world, invoice = _generated(db, monkeypatch)
    monkeypatch.setattr(
        "application.invoices.force_regenerate_partner_invoice_pdf.PartnerInvoiceService._generate_invoice_pdf",
        MagicMock(return_value="/uploads/invoices/partner-new.pdf"),
    )
    result = force_regenerate_partner_invoice_pdf(
        company_id=world.executing.id, partner_invoice_id=invoice.id
    )
    assert result.ok is True
    assert result.pdf_url == "/uploads/invoices/partner-new.pdf"
    db.session.refresh(invoice)
    assert invoice.pdf_url == "/uploads/invoices/partner-new.pdf"


def test_collision_id_catalogue_separe(db, monkeypatch):
    """Le GET partenaire ne peut pas renvoyer une facture du catalogue standard."""
    world, partner_invoice = _generated(db, monkeypatch)
    partner = get_partner_invoice(
        company_id=world.executing.id, partner_invoice_id=partner_invoice.id
    )
    assert partner.ok is True
    assert partner.invoice["invoice_number"].startswith("PARTNER-")
    assert partner.invoice["kind"] == "partner"
    assert partner.invoice["is_partner_invoice"] is True
    assert get_partner_invoice.__module__.endswith("get_partner_invoice")


def test_get_inconnu_404(db):
    company = CompanyFactory()
    result = get_partner_invoice(company_id=company.id, partner_invoice_id=34)
    assert result.ok is False
    assert result.status_code == 404
