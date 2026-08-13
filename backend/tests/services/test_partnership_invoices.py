"""Couverture de ``services.partnerships.invoices`` (factures mensuelles partenaires)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ext import db
from models.booking_transfer import BookingTransfer
from models.enums import (
    BookingStatus,
    PartnershipStatus,
    TransferModel,
    TransferStatus,
)
from models.partner_invoice import PartnerInvoice, PartnerInvoiceStatus
from models.partnership import Partnership
from services.partnerships.invoices import PartnerInvoiceService
from tests.factories import BookingFactory, ClientFactory, CompanyFactory


def _svc(monkeypatch, *, vat_rate: str = "0", vat_applicable: bool = False):
    billing = SimpleNamespace(
        payment_terms_days=30,
        vat_rate=Decimal(vat_rate),
        vat_applicable=vat_applicable,
    )
    seq = SimpleNamespace(id=1)
    service = PartnerInvoiceService(
        billing_settings_repo=MagicMock(),
        invoice_sequence_repo=MagicMock(),
        invoice_number_generator=MagicMock(),
        pdf_service=MagicMock(),
    )
    service.billing_settings_repo.find_or_create.return_value = billing
    service.invoice_sequence_repo.find_or_create.return_value = seq
    service.invoice_sequence_repo.increment_sequence.return_value = seq
    service.invoice_number_generator.generate.return_value = uuid.uuid4().hex[:12]
    monkeypatch.setattr(
        PartnerInvoiceService,
        "_generate_invoice_pdf",
        lambda *_a, **_k: "http://test/invoices/x.pdf",
    )
    return service


def _partnership(owner, partner):
    row = Partnership(
        owner_company_id=owner.id,
        partner_company_id=partner.id,
        status=PartnershipStatus.ACCEPTED,
        default_transfer_model=TransferModel.SUBCONTRACT,
        payment_terms_days=30,
    )
    db.session.add(row)
    db.session.flush()
    return row


def _transfer(
    *,
    booking,
    partnership,
    owner,
    executing,
    when: datetime,
    cost: Decimal | None = Decimal("40.00"),
):
    transfer = BookingTransfer(
        booking_id=booking.id,
        partnership_id=partnership.id,
        transfer_model=TransferModel.SUBCONTRACT,
        status=TransferStatus.COMPLETED,
        is_validated=True,
        validated_at=when.replace(tzinfo=UTC) if when.tzinfo is None else when,
        client_price=Decimal("45.00"),
        partner_cost=cost,
        currency="CHF",
        owner_company_id=owner.id,
        executing_company_id=executing.id,
    )
    db.session.add(transfer)
    db.session.flush()
    return transfer


def _booking(company, when: datetime):
    client = ClientFactory(company=company)
    return BookingFactory(
        company=company,
        client=client,
        scheduled_time=when,
        status=BookingStatus.PENDING,
        amount=45.0,
    )


def _world(month: int = 6, year: int = 2030, *, executing_is_partner: bool = True):
    owner = CompanyFactory()
    partner = CompanyFactory()
    partnership = _partnership(owner, partner)
    executing = partner if executing_is_partner else owner
    when = datetime(year, month, 15, 10, 0, 0)
    booking = _booking(executing, when)
    transfer = _transfer(
        booking=booking,
        partnership=partnership,
        owner=owner,
        executing=executing,
        when=when,
    )
    return SimpleNamespace(
        owner=owner,
        partner=partner,
        partnership=partnership,
        executing=executing,
        booking=booking,
        transfer=transfer,
        when=when,
        year=year,
        month=month,
    )


def test_init_default_dependencies(db):
    service = PartnerInvoiceService()
    assert service.billing_settings_repo is not None
    assert service.invoice_calculator is not None
    assert service.pdf_service is not None


def test_generate_partnership_introuvable(db, monkeypatch):
    service = _svc(monkeypatch)
    with pytest.raises(ValueError, match="introuvable"):
        service.generate_monthly_invoice(999999, 2030, 6, 1)


def test_generate_refuse_si_ni_partenaire_ni_owner(db, monkeypatch):
    world = _world()
    stranger = CompanyFactory()
    service = _svc(monkeypatch)
    with pytest.raises(ValueError, match="Seule l'entreprise"):
        service.generate_monthly_invoice(
            world.partnership.id, world.year, world.month, stranger.id
        )


def test_generate_refuse_owner_sans_transfert_executant(db, monkeypatch):
    world = _world(executing_is_partner=True)
    service = _svc(monkeypatch)
    with pytest.raises(ValueError, match="Seule l'entreprise"):
        service.generate_monthly_invoice(
            world.partnership.id, world.year, world.month, world.owner.id
        )


def test_generate_ok_partenaire_et_overrides(db, monkeypatch):
    world = _world()
    extra = _booking(world.executing, world.when + timedelta(days=1))
    t2 = _transfer(
        booking=extra,
        partnership=world.partnership,
        owner=world.owner,
        executing=world.executing,
        when=world.when + timedelta(days=1),
        cost=None,
    )
    t3 = _transfer(
        booking=_booking(world.executing, world.when + timedelta(days=2)),
        partnership=world.partnership,
        owner=world.owner,
        executing=world.executing,
        when=world.when + timedelta(days=2),
        cost=None,
    )
    service = _svc(monkeypatch, vat_rate="8.1", vat_applicable=True)
    invoice = service.generate_monthly_invoice(
        world.partnership.id,
        world.year,
        world.month,
        world.executing.id,
        transfer_ids=[world.transfer.id, t2.id, t3.id],
        overrides={world.transfer.id: {"amount": "12.33"}, t2.id: {"amount": 0}},
    )
    assert invoice.invoice_number.startswith("PARTNER-")
    assert invoice.status == PartnerInvoiceStatus.DRAFT
    assert invoice.pdf_url == "http://test/invoices/x.pdf"
    assert invoice.period_year == world.year
    assert invoice.subtotal_amount == Decimal("12.35")


def test_generate_transferts_invalides(db, monkeypatch):
    world = _world()
    service = _svc(monkeypatch)
    with pytest.raises(ValueError, match="déjà facturés"):
        service.generate_monthly_invoice(
            world.partnership.id,
            world.year,
            world.month,
            world.executing.id,
            transfer_ids=[world.transfer.id, 424242],
        )


def test_generate_aucun_transfert(db, monkeypatch):
    world = _world()
    service = _svc(monkeypatch)
    with pytest.raises(ValueError, match="Aucun transfert"):
        service.generate_monthly_invoice(
            world.partnership.id, 2031, 1, world.executing.id
        )


def test_generate_owner_executant_et_credit_et_pdf_erreur(db, monkeypatch):
    world = _world(executing_is_partner=False)
    prev = PartnerInvoice(
        partnership_id=world.partnership.id,
        executing_company_id=world.executing.id,
        period_year=world.year,
        period_month=5,
        invoice_number=f"PARTNER-PREV-{uuid.uuid4().hex[:8]}",
        subtotal_amount=Decimal("50.00"),
        vat_amount=Decimal("0"),
        total_amount=Decimal("50.00"),
        currency="CHF",
        status=PartnerInvoiceStatus.PAID,
        credit_balance=Decimal("100.00"),
        issued_at=datetime(world.year, 5, 1, tzinfo=UTC),
        due_date=datetime(world.year, 6, 1, tzinfo=UTC),
    )
    prev2 = PartnerInvoice(
        partnership_id=world.partnership.id,
        executing_company_id=world.executing.id,
        period_year=world.year,
        period_month=4,
        invoice_number=f"PARTNER-PREV2-{uuid.uuid4().hex[:8]}",
        subtotal_amount=Decimal("20.00"),
        vat_amount=Decimal("0"),
        total_amount=Decimal("20.00"),
        currency="CHF",
        status=PartnerInvoiceStatus.PARTIALLY_PAID,
        credit_balance=Decimal("5.00"),
        issued_at=datetime(world.year, 4, 1, tzinfo=UTC),
        due_date=datetime(world.year, 5, 1, tzinfo=UTC),
    )
    prev3 = PartnerInvoice(
        partnership_id=world.partnership.id,
        executing_company_id=world.executing.id,
        period_year=world.year,
        period_month=3,
        invoice_number=f"PARTNER-PREV3-{uuid.uuid4().hex[:8]}",
        subtotal_amount=Decimal("10.00"),
        vat_amount=Decimal("0"),
        total_amount=Decimal("10.00"),
        currency="CHF",
        status=PartnerInvoiceStatus.PAID,
        credit_balance=Decimal("1.00"),
        issued_at=datetime(world.year, 6, 1, tzinfo=UTC),
        due_date=datetime(world.year, 7, 1, tzinfo=UTC),
    )
    db.session.add_all([prev, prev2, prev3])
    db.session.flush()

    def _boom(*_a, **_k):
        raise RuntimeError("pdf boom")

    service = _svc(monkeypatch)
    monkeypatch.setattr(PartnerInvoiceService, "_generate_invoice_pdf", _boom)
    invoice = service.generate_monthly_invoice(
        world.partnership.id, world.year, world.month, world.executing.id
    )
    assert invoice.pdf_url is None
    assert invoice.total_amount == Decimal("0.00")
    db.session.refresh(prev2)
    db.session.refresh(prev)
    db.session.refresh(prev3)
    assert prev2.credit_balance == Decimal("0.00")
    assert prev.credit_balance == Decimal("65.00")
    assert prev3.credit_balance == Decimal("1.00")


def test_generate_decembre(db, monkeypatch):
    world = _world(month=12, year=2030)
    service = _svc(monkeypatch)
    invoice = service.generate_monthly_invoice(
        world.partnership.id, 2030, 12, world.executing.id
    )
    assert invoice.period_month == 12


def test_generate_pdf_reel_mocke(db, monkeypatch):
    world = _world()
    service = PartnerInvoiceService(
        billing_settings_repo=MagicMock(),
        invoice_sequence_repo=MagicMock(),
        invoice_number_generator=MagicMock(),
    )
    service.billing_settings_repo.find_or_create.return_value = SimpleNamespace(
        payment_terms_days=None, vat_rate=None, vat_applicable=False
    )
    seq = SimpleNamespace(id=2)
    service.invoice_sequence_repo.find_or_create.return_value = seq
    service.invoice_sequence_repo.increment_sequence.return_value = seq
    service.invoice_number_generator.generate.return_value = uuid.uuid4().hex[:12]

    monkeypatch.setattr(
        "services.partnerships.invoices_pdf.generate_partner_invoice_pdf_content",
        lambda *_a, **_k: b"%PDF-fake",
    )
    monkeypatch.setattr(
        "shared.upload_write.write_upload_bytes", lambda *_a, **_k: None
    )
    invoice = service.generate_monthly_invoice(
        world.partnership.id, world.year, world.month, world.executing.id
    )
    assert "/invoices/" in invoice.pdf_url
    assert invoice.pdf_url.endswith(".pdf")


def test_regenerate_pdf_erreurs_et_succes(db, monkeypatch):
    world = _world()
    service = _svc(monkeypatch)
    invoice = service.generate_monthly_invoice(
        world.partnership.id, world.year, world.month, world.executing.id
    )

    with pytest.raises(ValueError, match="introuvable"):
        service.regenerate_pdf(999999)

    empty = PartnerInvoice(
        partnership_id=world.partnership.id,
        executing_company_id=world.executing.id,
        period_year=2031,
        period_month=1,
        invoice_number=f"PARTNER-EMPTY-{uuid.uuid4().hex[:8]}",
        subtotal_amount=Decimal("0"),
        vat_amount=Decimal("0"),
        total_amount=Decimal("0"),
        currency="CHF",
        status=PartnerInvoiceStatus.DRAFT,
        issued_at=datetime.now(UTC),
        due_date=datetime.now(UTC) + timedelta(days=30),
    )
    db.session.add(empty)
    db.session.flush()
    with pytest.raises(ValueError, match="Aucun transfert"):
        service.regenerate_pdf(empty.id)

    url = service.regenerate_pdf(invoice.id)
    assert url == "http://test/invoices/x.pdf"

    def _fail(*_a, **_k):
        raise RuntimeError("fail")

    monkeypatch.setattr(PartnerInvoiceService, "_generate_invoice_pdf", _fail)
    with pytest.raises(ValueError, match="régénération"):
        service.regenerate_pdf(invoice.id)


def test_mark_as_sent_et_paid(db, monkeypatch):
    world = _world()
    service = _svc(monkeypatch)
    invoice = service.generate_monthly_invoice(
        world.partnership.id, world.year, world.month, world.executing.id
    )

    with pytest.raises(ValueError, match="introuvable"):
        service.mark_as_sent(999999, world.executing.id)
    with pytest.raises(ValueError, match="exécutante"):
        service.mark_as_sent(invoice.id, world.owner.id)

    sent = service.mark_as_sent(invoice.id, world.executing.id)
    assert sent.status == PartnerInvoiceStatus.SENT
    assert sent.sent_at is not None
    with pytest.raises(ValueError, match="Déjà envoyée"):
        service.mark_as_sent(invoice.id, world.executing.id)

    with pytest.raises(ValueError, match="introuvable"):
        service.mark_as_paid(999999)
    paid = service.mark_as_paid(invoice.id)
    assert paid.status == PartnerInvoiceStatus.PAID
    assert paid.paid_at is not None


def test_get_monthly_et_pending(db, monkeypatch):
    world = _world()
    service = _svc(monkeypatch)
    assert (
        service.get_monthly_invoice(world.partnership.id, world.year, world.month)
        is None
    )
    invoice = service.generate_monthly_invoice(
        world.partnership.id, world.year, world.month, world.executing.id
    )
    found = service.get_monthly_invoice(world.partnership.id, world.year, world.month)
    assert found is not None
    assert found.id == invoice.id

    pending = service.get_pending_transfers_count(
        world.partnership.id, world.year, world.month
    )
    assert pending == 0
    amount = service.get_pending_amount(world.partnership.id, world.year, world.month)
    assert amount == Decimal("0")

    later = datetime(2030, 12, 10, 8, 0, 0)
    extra_booking = _booking(world.executing, later)
    _transfer(
        booking=extra_booking,
        partnership=world.partnership,
        owner=world.owner,
        executing=world.executing,
        when=later,
        cost=Decimal("15.50"),
    )
    _transfer(
        booking=_booking(world.executing, later + timedelta(hours=2)),
        partnership=world.partnership,
        owner=world.owner,
        executing=world.executing,
        when=later + timedelta(hours=2),
        cost=None,
    )
    assert service.get_pending_transfers_count(world.partnership.id, 2030, 12) == 2
    assert service.get_pending_amount(world.partnership.id, 2030, 12) == Decimal(
        "15.50"
    )
