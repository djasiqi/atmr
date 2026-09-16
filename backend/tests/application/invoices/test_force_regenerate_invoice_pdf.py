"""Anti-régression permanente — contrat figé « Régénérer PDF » (CLOSED).

Voir docs/facturation/regenerer-pdf-contrat.md.
Ne pas retirer : PATIENT live, tiers payeur indépendant, remplacement
atomique du pdf_url, conservation de l'ancien PDF en cas d'échec.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from application.invoices.force_regenerate_invoice_pdf import (
    ForceRegenerateInvoicePdfUseCase,
    sync_patient_billing_party_from_live,
)
from models import Client, Company, Invoice, InvoiceLine, User
from models.billing_party import BillingParty, ClientBillingParty
from models.enums import (
    BillingPartyType,
    InvoiceLineType,
    InvoiceStatus,
    UserRole,
)
from services.documents.pdf import _get_billed_to


def _ensure_password(user: User) -> None:
    if not getattr(user, "public_id", None):
        user.public_id = str(uuid.uuid4())
    if not getattr(user, "password", None):
        user.set_password("password123", force_change=False)


def _world(db):
    suf = uuid.uuid4().hex[:8]
    owner = User(username=f"own_{suf}", email=f"own_{suf}@test.example")
    owner.role = UserRole.company
    _ensure_password(owner)
    db.session.add(owner)
    db.session.flush()

    company = Company(name="ATMR Test", uid_ide="CHE-111.222.333")
    company.user_id = owner.id
    db.session.add(company)
    db.session.flush()

    client_user = User(
        username=f"cli_{suf}",
        email=f"cli_{suf}@test.example",
        first_name="Ancien",
        last_name="NOM",
    )
    client_user.role = UserRole.client
    _ensure_password(client_user)
    db.session.add(client_user)
    db.session.flush()

    client = Client(user=client_user, company=company)
    client.domicile_address = "Rue Ancienne 1"
    client.domicile_zip = "1200"
    client.domicile_city = "Genève"
    db.session.add(client)
    db.session.flush()

    bp = BillingParty()
    bp.company_id = company.id
    bp.type = BillingPartyType.PATIENT
    bp.display_name = "Ancien NOM"
    bp.billing_address = "Rue Ancienne 1\n1200 Genève"
    bp.external_ref = f"patient_client:{client.id}"
    bp.is_active = True
    db.session.add(bp)
    db.session.flush()

    invoice = Invoice(
        company=company,
        client=client,
        invoice_number=f"INV-REGEN-{suf}",
        period_year=2026,
        period_month=9,
        status=InvoiceStatus.DRAFT,
        issued_at=datetime.now(UTC),
        due_date=datetime.now(UTC),
        subtotal_amount=Decimal("80.00"),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal("80.00"),
        billing_party_id=bp.id,
    )
    invoice.pdf_url = "/uploads/invoices/old_invoice.pdf"
    db.session.add(invoice)
    db.session.flush()

    line = InvoiceLine(
        invoice=invoice,
        type=InvoiceLineType.CUSTOM,
        description="Prestation test",
        qty=Decimal("1.00"),
        unit_price=Decimal("80.00"),
        line_total=Decimal("80.00"),
        vat_rate=Decimal("0.00"),
        vat_amount=Decimal("0.00"),
        total_with_vat=Decimal("80.00"),
    )
    db.session.add(line)
    db.session.commit()
    return company, client, client_user, bp, invoice


def test_get_billed_to_uses_live_client_after_rename(db):
    _company, client, client_user, bp, invoice = _world(db)
    name_a, addr_a = _get_billed_to(invoice)
    assert "Ancien" in name_a
    assert "Rue Ancienne 1" in addr_a

    client_user.first_name = "Nouveau"
    client_user.last_name = "PRENOM"
    client.domicile_address = "Avenue Nouvelle 9"
    client.domicile_zip = "1205"
    client.domicile_city = "Genève"
    db.session.commit()
    db.session.expire_all()

    invoice = Invoice.query.get(invoice.id)
    name_b, addr_b = _get_billed_to(invoice)
    assert "Nouveau" in name_b
    assert "PRENOM" in name_b
    assert "Ancien" not in name_b
    assert "Avenue Nouvelle 9" in addr_b
    assert "Rue Ancienne 1" not in addr_b
    # Le snapshot BP n'a pas encore été réécrit : le PDF ne doit pas s'y fier.
    assert bp.display_name == "Ancien NOM"


def test_third_party_uses_current_billing_party_and_contact(db):
    company, client, _user, _patient_bp, invoice = _world(db)
    hospice = BillingParty()
    hospice.company_id = company.id
    hospice.type = BillingPartyType.CURATORSHIP
    hospice.display_name = "Hospice général"
    hospice.billing_address = "Rue Ancienne 10\n1201 Genève"
    hospice.is_active = True
    db.session.add(hospice)
    db.session.flush()

    link = ClientBillingParty()
    link.client_id = client.id
    link.billing_party_id = hospice.id
    link.contact_name = "Amandine HAUSER"
    link.role = "gestionnaire"
    db.session.add(link)
    invoice.billing_party_id = hospice.id
    db.session.commit()
    db.session.expire_all()

    invoice = Invoice.query.get(invoice.id)
    name, addr = _get_billed_to(invoice)
    assert "Hospice" in name
    assert "À l'att. de Amandine HAUSER" in name
    assert "curateur" not in name.lower()
    assert "Rue Ancienne 10" in addr

    hospice.display_name = "Hospice général — Siège"
    hospice.billing_address = "Boulevard Nouveau 2\n1202 Genève"
    link.contact_name = "Amandine HAUSER"
    db.session.commit()
    db.session.expire_all()

    invoice = Invoice.query.get(invoice.id)
    name_b, addr_b = _get_billed_to(invoice)
    assert "siège" in name_b.casefold()
    assert "À l'att. de Amandine HAUSER" in name_b
    assert "curateur" not in name_b.lower()
    assert "Boulevard Nouveau 2" in addr_b
    assert "Rue Ancienne 10" not in addr_b


def test_sync_patient_billing_party_from_live(db):
    _company, client, client_user, _bp, invoice = _world(db)
    client_user.first_name = "Nouveau"
    client_user.last_name = "PRENOM"
    client.domicile_address = "Avenue Nouvelle 9"
    client.domicile_zip = "1205"
    client.domicile_city = "Genève"
    db.session.commit()
    db.session.refresh(invoice)
    db.session.refresh(invoice.client)
    db.session.refresh(invoice.client.user)
    db.session.refresh(invoice.billing_party)

    sync_patient_billing_party_from_live(invoice)
    assert "Nouveau" in invoice.billing_party.display_name
    assert "Avenue Nouvelle 9" in (invoice.billing_party.billing_address or "")


def test_force_regenerate_success_replaces_pdf_url(db):
    company, _client, _user, _bp, invoice = _world(db)
    old_url = invoice.pdf_url
    invoice_id = invoice.id

    with (
        patch(
            "application.invoices.force_regenerate_invoice_pdf.GenerateInvoicePdfUseCase"
        ) as uc_cls,
        patch(
            "application.invoices.force_regenerate_invoice_pdf._delete_replaced_invoice_pdf"
        ) as del_old,
    ):
        uc_cls.return_value.execute.return_value = SimpleNamespace(
            ok=True,
            pdf_url="/uploads/invoices/new_invoice.pdf",
            error=None,
            status_code=None,
        )
        result = ForceRegenerateInvoicePdfUseCase().execute(
            company_id=company.id, invoice_id=invoice_id
        )

    assert result.ok is True
    assert result.pdf_url == "/uploads/invoices/new_invoice.pdf"
    assert result.previous_pdf_url == old_url
    del_old.assert_called_once_with(old_url, "/uploads/invoices/new_invoice.pdf")
    reloaded = Invoice.query.get(invoice_id)
    assert reloaded.pdf_url == "/uploads/invoices/new_invoice.pdf"


def test_force_regenerate_failure_keeps_old_pdf(db):
    company, _client, _user, _bp, invoice = _world(db)
    old_url = invoice.pdf_url
    invoice_id = invoice.id

    with patch(
        "application.invoices.force_regenerate_invoice_pdf.GenerateInvoicePdfUseCase"
    ) as uc_cls:
        uc_cls.return_value.execute.return_value = SimpleNamespace(
            ok=False,
            pdf_url=None,
            error={"error": "PDF_FAIL"},
            status_code=500,
        )
        result = ForceRegenerateInvoicePdfUseCase().execute(
            company_id=company.id, invoice_id=invoice_id
        )

    assert result.ok is False
    assert result.previous_pdf_url == old_url
    reloaded = Invoice.query.get(invoice_id)
    assert reloaded.pdf_url == old_url
    assert (reloaded.meta or {}).get("pdf", {}).get("status") != "failed"


def test_pdf_content_picks_up_renamed_client(db):
    """Le pipeline PDF (bloc Facturé à) reprend le nom/adresse live après mutation."""
    _company, client, client_user, _bp, invoice = _world(db)
    name_a, addr_a = _get_billed_to(invoice)
    assert "Ancien" in name_a
    assert "Rue Ancienne 1" in addr_a

    client_user.first_name = "Nouveau"
    client_user.last_name = "PRENOM"
    client.domicile_address = "Avenue Nouvelle 9"
    client.domicile_zip = "1205"
    client.domicile_city = "Genève"
    db.session.commit()
    db.session.expire_all()
    invoice = Invoice.query.get(invoice.id)

    name_b, addr_b = _get_billed_to(invoice)
    assert "Nouveau" in name_b
    assert "PRENOM" in name_b
    assert "Avenue Nouvelle 9" in addr_b
    assert "Ancien" not in name_b
    assert "Rue Ancienne 1" not in addr_b
