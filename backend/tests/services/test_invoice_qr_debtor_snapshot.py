"""Immutabilité du débiteur QR-facture (``invoice.meta["qr_debtor_snapshot"]``).

Le débiteur QR n'est pas le bloc « Facturé à » : patient + domicile structuré,
jamais le c/o. Snapshot dédié, gelé à la sortie de DRAFT, relu à la régénération.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import event

from models import Company, Invoice, User
from models.billing_party import BillingParty
from models.enums import (
    BillingPartyType,
    InvoiceBillingStrategy,
    InvoiceStatus,
    UserRole,
)
from models.invoice import _invoice_freeze_identity_snapshots
from services.documents.invoice_recipient import (
    BILLED_TO_SNAPSHOT_KEY,
    refresh_billed_to_snapshot_if_draft,
)
from services.documents.pdf import QRBillService
from services.documents.qr_debtor import (
    QR_DEBTOR_SNAPSHOT_KEY,
    read_qr_debtor_snapshot,
    resolve_invoice_qr_debtor,
    resolve_qr_debtor_live,
)
from tests.services.test_invoice_billed_to_resolver import (
    _invoice,
    _link,
    _opad,
    _party,
    _patient,
)
from tests.services.test_invoice_billed_to_snapshot import (
    MUTATED_STREET,
    _arnaud_curatelle_invoice,
    _mutate_master_data,
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


def _qr(invoice):
    return QRBillService()._get_debtor_info(invoice)


def test_qr_debtor_patient_curator_is_patient_domicile_not_care_of(db, company):
    """Règle QR : patient + domicile — distinct du bloc c/o curatrice."""
    invoice, _guylene = _arnaud_curatelle_invoice(db, company)
    debtor = resolve_qr_debtor_live(invoice)
    assert debtor.rule == "patient_direct"
    assert debtor.name == "Arnaud JACQUEMOUD"
    assert debtor.street == "Avenue du Plateau 4C"
    assert debtor.pcode == "1213"
    assert debtor.city == "Petit-Lancy"
    assert _qr(invoice)["name"] == "Arnaud JACQUEMOUD"
    assert "Guylène" not in _qr(invoice)["name"]
    assert "Patru" not in _qr(invoice)["street"]


def test_frozen_qr_survives_patient_mutation(db, company):
    invoice, _ = _arnaud_curatelle_invoice(db, company)
    refresh_billed_to_snapshot_if_draft(invoice, reason="pdf_generation")
    before = _qr(invoice)
    invoice.mark_as_sent()
    db.session.commit()
    _mutate_master_data(db, _reload(invoice.id))
    invoice = _reload(invoice.id)
    live = resolve_invoice_qr_debtor(invoice, use_snapshot=False)
    assert "NOUVEAU" in live.name
    frozen = resolve_invoice_qr_debtor(invoice)
    assert frozen.source == "snapshot"
    assert _qr(invoice) == before
    assert "NOUVEAU" not in _qr(invoice)["name"]
    assert "Chemin Nouveau" not in _qr(invoice)["street"]


def test_frozen_qr_survives_payer_mutation(db, company):
    invoice, _ = _arnaud_curatelle_invoice(db, company)
    refresh_billed_to_snapshot_if_draft(invoice)
    invoice.mark_as_sent()
    db.session.commit()
    invoice = _reload(invoice.id)
    before = _qr(invoice)
    guylene = BillingParty.query.get(invoice.billing_party_id)
    guylene.display_name = "Autre Payeur"
    guylene.billing_address = f"{MUTATED_STREET}, 1200, Genève"
    db.session.commit()
    invoice = _reload(invoice.id)
    assert _qr(invoice) == before
    assert MUTATED_STREET not in _qr(invoice)["street"]


def test_draft_qr_follows_live_patient(db, company):
    invoice, _ = _arnaud_curatelle_invoice(db, company)
    refresh_billed_to_snapshot_if_draft(invoice)
    invoice.client.user.last_name = "JACQUEMOUD-BROUILLON"
    invoice.client.domicile_address = "Rue Brouillon 1"
    db.session.commit()
    invoice = _reload(invoice.id)
    assert invoice.status == InvoiceStatus.DRAFT
    debtor = resolve_invoice_qr_debtor(invoice)
    assert debtor.source == "live"
    assert debtor.name == "Arnaud JACQUEMOUD-BROUILLON"
    assert debtor.street == "Rue Brouillon 1"
    snap = refresh_billed_to_snapshot_if_draft(invoice)
    assert snap is not None
    assert read_qr_debtor_snapshot(invoice)["street"] == "Rue Brouillon 1"


def test_frozen_qr_without_snapshot_is_legacy_live_no_backfill(db, company):
    invoice, _ = _arnaud_curatelle_invoice(db, company)
    invoice.mark_as_sent()
    db.session.commit()
    invoice = _reload(invoice.id)
    meta = dict(invoice.meta or {})
    meta.pop(QR_DEBTOR_SNAPSHOT_KEY, None)
    meta.pop(BILLED_TO_SNAPSHOT_KEY, None)
    invoice.meta = meta
    db.session.commit()
    invoice = _reload(invoice.id)
    debtor = resolve_invoice_qr_debtor(invoice)
    assert debtor.source == "legacy_live"
    assert debtor.name == "Arnaud JACQUEMOUD"
    assert read_qr_debtor_snapshot(invoice) is None
    refresh_billed_to_snapshot_if_draft(invoice)
    assert read_qr_debtor_snapshot(invoice) is None


def test_current_code_cannot_leave_draft_without_both_snapshots(db, company):
    invoice, _ = _arnaud_curatelle_invoice(db, company)
    invoice.mark_as_sent()
    db.session.commit()
    invoice = _reload(invoice.id)
    assert invoice.status != InvoiceStatus.DRAFT
    assert invoice.meta[BILLED_TO_SNAPSHOT_KEY]["frozen_at"]
    assert invoice.meta[QR_DEBTOR_SNAPSHOT_KEY]["frozen_at"]
    assert invoice.meta[QR_DEBTOR_SNAPSHOT_KEY]["rule"] == "patient_direct"
    assert invoice.meta.get("qr_creditor_snapshot", {}).get("frozen_at")


def test_invoice_mapper_listener_registered_without_factory():
    """Le hook est sur le mapper Invoice — tout process qui importe le modèle l'a."""
    assert event.contains(Invoice, "before_update", _invoice_freeze_identity_snapshots)


def test_celery_worker_path_loads_create_app_and_mapper_hook():
    """Preuve code : ContextTask → get_flask_app → create_app ; mapper déjà accroché."""
    import inspect

    from celery_app import ContextTask, get_flask_app

    assert "create_app" in inspect.getsource(get_flask_app)
    assert "get_flask_app" in inspect.getsource(ContextTask.__call__)
    assert event.contains(Invoice, "before_update", _invoice_freeze_identity_snapshots)


def test_opad_qr_debtor_remains_patient_not_opad(db, company):
    astrid = _patient(
        db,
        company,
        first="Astrid-Jacqueline",
        last="SCHURTER",
        street="Cité Vieusseux 8",
        zip_code="1203",
        city="Genf",
    )
    opad = _opad(db, company)
    _link(db, astrid, opad)
    invoice = _invoice(db, company, astrid, party=opad)
    debtor = resolve_qr_debtor_live(invoice)
    assert debtor.rule == "patient_direct"
    assert "SCHURTER" in debtor.name
    assert "OPAD" not in debtor.name
    assert "Jeunes" not in debtor.street


def test_s2_qr_debtor_uses_billing_party(db, company):
    jean = _patient(
        db,
        company,
        first="Jean",
        last="Dupont",
        street="Rue X 1",
        zip_code="1200",
        city="Genève",
    )
    clinic = _party(
        db,
        company,
        party_type=BillingPartyType.CLINIC,
        name="Clinique les Hauts d'Anières",
        address="Chemin des Courbes 9, 1247 Anières, Suisse",
    )
    invoice = _invoice(
        db,
        company,
        jean,
        party=clinic,
        strategy=InvoiceBillingStrategy.S2_CLINIC_MONTHLY,
    )
    invoice.billed_to_company_id = company.id
    db.session.commit()
    invoice = _reload(invoice.id)
    debtor = resolve_qr_debtor_live(invoice)
    assert debtor.rule == "s2_clinic"
    assert "Hauts" in debtor.name
    assert "1247" in debtor.pcode or "Courbes" in debtor.street
