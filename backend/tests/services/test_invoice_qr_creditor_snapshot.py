"""Immutabilité du créancier QR (compte + identité).

Le débiteur est déjà figé. Ici on prouve qu'une facture SENT régénérée conserve
l'IBAN A et l'adresse créancier A après mutation des paramètres entreprise vers B.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from models import Company, CompanyBillingSettings, User
from models.enums import InvoiceStatus, UserRole
from services.billing import BillingProfileService
from services.documents.invoice_recipient import (
    freeze_billed_to_snapshot,
    refresh_billed_to_snapshot_if_draft,
)
from services.documents.pdf import QRBillService
from services.documents.qr_creditor import (
    QR_CREDITOR_SNAPSHOT_KEY,
    read_qr_creditor_snapshot,
    resolve_invoice_qr_creditor,
    resolve_qr_creditor_live,
)
from tests.services.test_invoice_billed_to_snapshot import (
    _arnaud_curatelle_invoice,
    _reload,
)


@pytest.fixture
def company(db):
    suf = uuid.uuid4().hex[:8]
    owner = User(username=f"own_{suf}", email=f"own_{suf}@test.example")
    owner.role = UserRole.company
    owner.public_id = str(uuid.uuid4())
    owner.set_password("password123", force_change=False)
    db.session.add(owner)
    db.session.flush()
    company = Company(name="ATMR Test", uid_ide="CHE-111.222.333")
    company.user_id = owner.id
    company.domicile_country = "CH"
    db.session.add(company)
    db.session.flush()
    return company


IBAN_A = "CH9300762011623852957"
IBAN_B = "CH4431999123000889012"
NAME_A = "ATMR Ancien SA"
NAME_B = "ATMR Nouveau SA"
STREET_A = "Rue Ancienne 1"
STREET_B = "Rue Nouvelle 99"


def _profile(*, iban: str, name: str, street: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        qr_iban=None,
        iban=iban,
        legal_name=name,
        street_name=street,
        building_number="",
        postal_code="1200",
        city="Genève",
        country_code="CH",
    )


def _patch_profile(monkeypatch, profile) -> None:
    monkeypatch.setattr(
        BillingProfileService,
        "get_by_company_id",
        Mock(return_value=profile),
    )


def test_live_creditor_reads_profile_iban(db, company, monkeypatch):
    invoice, _ = _arnaud_curatelle_invoice(db, company)
    _patch_profile(monkeypatch, _profile(iban=IBAN_A, name=NAME_A, street=STREET_A))
    live = resolve_qr_creditor_live(invoice)
    assert live.account == IBAN_A
    assert live.name == NAME_A
    assert live.street == STREET_A
    assert live.source == "live"


def test_freeze_then_iban_mutation_keeps_account_a(db, company, monkeypatch):
    invoice, _ = _arnaud_curatelle_invoice(db, company)
    _patch_profile(monkeypatch, _profile(iban=IBAN_A, name=NAME_A, street=STREET_A))
    refresh_billed_to_snapshot_if_draft(invoice, reason="pdf_generation")
    invoice.mark_as_sent()
    db.session.commit()
    invoice = _reload(invoice.id)

    snap = read_qr_creditor_snapshot(invoice)
    assert snap is not None
    assert snap["account"] == IBAN_A
    assert snap["name"] == NAME_A
    assert snap["frozen_at"]

    _patch_profile(monkeypatch, _profile(iban=IBAN_B, name=NAME_B, street=STREET_B))
    frozen = resolve_invoice_qr_creditor(invoice)
    assert frozen.source == "snapshot"
    assert frozen.account == IBAN_A
    assert frozen.name == NAME_A
    assert frozen.street == STREET_A
    assert frozen.account != IBAN_B

    account, address = QRBillService()._resolve_qr_account_and_creditor(invoice)
    assert account == IBAN_A
    assert address["name"] == NAME_A
    assert address["street"] == STREET_A


def test_freeze_then_creditor_address_mutation_keeps_snapshot(db, company, monkeypatch):
    invoice, _ = _arnaud_curatelle_invoice(db, company)
    _patch_profile(monkeypatch, _profile(iban=IBAN_A, name=NAME_A, street=STREET_A))
    freeze_billed_to_snapshot(invoice, reason="status:draft->sent")
    invoice.status = InvoiceStatus.SENT
    db.session.commit()
    invoice = _reload(invoice.id)
    _patch_profile(monkeypatch, _profile(iban=IBAN_B, name=NAME_B, street=STREET_B))
    cred = resolve_invoice_qr_creditor(invoice)
    assert cred.name == NAME_A
    assert cred.street == STREET_A
    assert cred.pcode == "1200"
    assert cred.city == "Genève"
    assert cred.country == "CH"


def test_settings_fallback_iban_is_snapshotted(db, company, monkeypatch):
    invoice, _ = _arnaud_curatelle_invoice(db, company)
    _patch_profile(monkeypatch, None)
    db.session.add(
        CompanyBillingSettings(
            company_id=company.id,
            iban=IBAN_A,
            payment_terms_days=10,
        )
    )
    db.session.commit()
    invoice = _reload(invoice.id)
    freeze_billed_to_snapshot(invoice, reason="status:draft->sent")
    invoice.mark_as_sent()
    db.session.commit()
    settings = CompanyBillingSettings.query.filter_by(company_id=company.id).first()
    settings.iban = IBAN_B
    db.session.commit()
    invoice = _reload(invoice.id)
    assert resolve_invoice_qr_creditor(invoice).account == IBAN_A


def test_legacy_live_when_frozen_without_creditor_snapshot(db, company, monkeypatch):
    invoice, _ = _arnaud_curatelle_invoice(db, company)
    _patch_profile(monkeypatch, _profile(iban=IBAN_A, name=NAME_A, street=STREET_A))
    invoice.mark_as_sent()
    db.session.commit()
    invoice = _reload(invoice.id)
    meta = dict(invoice.meta or {})
    meta.pop(QR_CREDITOR_SNAPSHOT_KEY, None)
    invoice.meta = meta
    db.session.commit()
    invoice = _reload(invoice.id)
    _patch_profile(monkeypatch, _profile(iban=IBAN_B, name=NAME_B, street=STREET_B))
    cred = resolve_invoice_qr_creditor(invoice)
    assert cred.source == "legacy_live"
    assert cred.account == IBAN_B


def test_draft_refresh_follows_live_iban(db, company, monkeypatch):
    invoice, _ = _arnaud_curatelle_invoice(db, company)
    _patch_profile(monkeypatch, _profile(iban=IBAN_A, name=NAME_A, street=STREET_A))
    refresh_billed_to_snapshot_if_draft(invoice)
    assert read_qr_creditor_snapshot(invoice)["account"] == IBAN_A
    _patch_profile(monkeypatch, _profile(iban=IBAN_B, name=NAME_B, street=STREET_B))
    refresh_billed_to_snapshot_if_draft(invoice)
    assert read_qr_creditor_snapshot(invoice)["account"] == IBAN_B
    assert invoice.status == InvoiceStatus.DRAFT


def test_invoice_immutable_fields_not_duplicated_in_creditor_snapshot(
    db, company, monkeypatch
):
    invoice, _ = _arnaud_curatelle_invoice(db, company)
    invoice.qr_reference = "RF123456789012345678901234"
    _patch_profile(monkeypatch, _profile(iban=IBAN_A, name=NAME_A, street=STREET_A))
    freeze_billed_to_snapshot(invoice)
    snap = read_qr_creditor_snapshot(invoice)
    assert "reference" not in snap
    assert "amount" not in snap
    assert "currency" not in snap
    assert invoice.qr_reference == "RF123456789012345678901234"
    assert invoice.total_amount is not None
