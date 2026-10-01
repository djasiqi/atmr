"""Immutabilité du bloc « Facturé à » (``invoice.meta["billed_to_snapshot"]``).

Architecture testée :
- brouillon → résolution courante, snapshot rafraîchi à chaque PDF construit ;
- sortie de DRAFT (envoi e-mail / papier / lot, paiement, annulation) → snapshot figé ;
- facture figée → PDF et HTML rendus depuis le snapshot, jamais depuis les master data ;
- ancienne facture figée sans snapshot → fallback ``legacy_live`` explicite.

Test obligatoire : facture patient + curatrice (Arnaud JACQUEMOUD / c/o Mme Lucia
GUYLÈNE / Rue Patru 2 / 1205 Genève), figée, puis master data mutées (nom et
adresse du payeur « Rue Exemple 99 », nom et adresse du patient, contact) et
régénération : PDF et HTML conservent exactement les valeurs snapshotées.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from application.invoices.billed_to_snapshot_guard import (
    freeze_billed_to_for_session,
    invoice_leaves_draft,
)
from application.invoices.force_regenerate_invoice_pdf import (
    ForceRegenerateInvoicePdfUseCase,
    refresh_recipient_snapshot_meta,
    sync_patient_billing_party_from_live,
)
from application.invoices.invoice_pdf_state import mark_pdf_ready
from ext import db as _db
from models import Company, CompanyBillingSettings, Invoice, User
from models.billing_party import BillingParty, ClientBillingParty
from models.enums import BillingPartyType, InvoiceStatus, UserRole
from services.documents.invoice_recipient import (
    BILLED_TO_SNAPSHOT_KEY,
    BilledToParty,
    billed_to_name_lines,
    billed_to_party_from_snapshot,
    billed_to_snapshot_from_party,
    explicit_recipient_mode,
    freeze_billed_to_snapshot,
    invoice_billed_to_is_frozen,
    read_billed_to_snapshot,
    refresh_billed_to_snapshot_if_draft,
    resolve_invoice_billed_to,
    structure_postal_address,
)
from services.documents.invoice_template_builder import InvoiceTemplateBuilder
from services.documents.pdf import PDFService, _get_billed_to
from tests.services.test_invoice_billed_to_resolver import (
    _guylene,
    _invoice,
    _jacquemoud,
    _link,
    _opad,
    _party,
    _patient,
)
from tests.services.test_invoice_pdf_s2_gates_helpers import extract_text_per_page

EXPECTED_NAME = "Arnaud JACQUEMOUD\nc/o Mme Lucia GUYLÈNE"
EXPECTED_ADDR = "Rue Patru 2<br/>1205 Genève"
MUTATED_STREET = "Rue Exemple 99"


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


def _html_block(invoice) -> tuple[str, str]:
    return InvoiceTemplateBuilder()._resolve_billed_to(invoice)


def _reload(invoice_id: int) -> Invoice:
    _db.session.expire_all()
    return Invoice.query.get(invoice_id)


def _arnaud_curatelle_invoice(
    db, company, *, contact: str | None = None
) -> tuple[Invoice, BillingParty]:
    arnaud = _jacquemoud(db, company)
    guylene = _guylene(db, company)
    _link(db, arnaud, guylene, role="Curatrice", contact_name=contact)
    return _invoice(db, company, arnaud, party=guylene), guylene


def _mutate_master_data(db, invoice: Invoice) -> None:
    """Mutations post-émission : payeur (nom + adresse), patient (nom + adresse), contact."""
    guylene = BillingParty.query.get(invoice.billing_party_id)
    guylene.display_name = "Mme Lucia Guylène-Renommée"
    guylene.billing_address = f"{MUTATED_STREET}, 1200, Genève"
    guylene.contact_email = "nouveau@example.test"
    client = invoice.client
    client.user.last_name = "JACQUEMOUD-NOUVEAU"
    client.domicile_address = "Chemin Nouveau 7"
    client.billing_address = "Chemin Nouveau 7, 1227, Carouge"
    link = ClientBillingParty.query.filter_by(
        client_id=client.id, billing_party_id=guylene.id
    ).first()
    link.contact_name = "Nouveau Contact"
    db.session.commit()


# ───────────────────── brouillon : le snapshot suit les master data ─────────────────────


def test_draft_snapshot_follows_master_data(db, company):
    invoice, guylene = _arnaud_curatelle_invoice(db, company)
    assert invoice_billed_to_is_frozen(invoice) is False
    assert read_billed_to_snapshot(invoice) is None

    snap = refresh_billed_to_snapshot_if_draft(invoice, reason="pdf_generation")
    db.session.commit()
    invoice = _reload(invoice.id)
    assert read_billed_to_snapshot(invoice) == snap
    assert snap["mode"] == "patient_care_of"
    assert snap["mode_origin"] == "inferred"
    assert snap["addressee"] == "Arnaud JACQUEMOUD"
    assert snap["care_of"] == "Mme Lucia Guylène"
    assert snap["patient_display_name"] == "Arnaud JACQUEMOUD"
    assert snap["payer_display_name"] == "Mme Lucia Guylène"
    assert snap["payer_type"] == "curatorship"
    assert snap["address"]["raw"] == "Rue Patru, 2, 1205, Genève"
    assert snap["address"]["street"] == "Rue Patru 2"
    assert snap["address"]["postal_code"] == "1205"
    assert snap["address"]["city"] == "Genève"
    assert snap["billing_party_id"] == guylene.id
    assert snap["frozen_at"] is None

    # Brouillon : le rendu reste live, et le snapshot suit la mutation au prochain PDF.
    guylene = BillingParty.query.get(guylene.id)
    guylene.billing_address = f"{MUTATED_STREET}, 1200, Genève"
    db.session.commit()
    invoice = _reload(invoice.id)
    assert resolve_invoice_billed_to(invoice).source == "live"
    assert _get_billed_to(invoice)[1] == f"{MUTATED_STREET}<br/>1200 Genève"
    snap2 = refresh_billed_to_snapshot_if_draft(invoice)
    assert snap2["address"]["street"] == MUTATED_STREET


# ───────────────── TEST OBLIGATOIRE : master data mutées après gel ─────────────────


def test_frozen_invoice_block_survives_master_data_mutation(db, company):
    """Facture figée → PDF/HTML depuis le snapshot, même après mutation des master data."""
    invoice, _guylene = _arnaud_curatelle_invoice(db, company)

    # 1. Dernier PDF construit en brouillon → snapshot capturé.
    refresh_billed_to_snapshot_if_draft(invoice, reason="pdf_generation")
    pdf_before = _get_billed_to(invoice)
    html_before = _html_block(invoice)
    assert pdf_before == (EXPECTED_NAME, EXPECTED_ADDR)

    # 2. Gel : la facture quitte DRAFT (envoi) → garde before_flush.
    invoice.mark_as_sent()
    db.session.commit()
    invoice = _reload(invoice.id)
    snap = read_billed_to_snapshot(invoice)
    assert invoice.status == InvoiceStatus.SENT
    assert snap is not None
    assert snap["frozen_at"]
    assert snap["frozen_reason"] == "status:draft->sent"

    # 3. Master data mutées (payeur, patient, contact).
    _mutate_master_data(db, invoice)
    invoice = _reload(invoice.id)
    # Preuve que la mutation est effective côté master data :
    live = resolve_invoice_billed_to(invoice, use_snapshot=False)
    assert live.addressee == "Arnaud JACQUEMOUD-NOUVEAU"
    assert live.care_of == "Mme Lucia Guylène-Renommée"
    assert MUTATED_STREET in live.address

    # 4. Rendu PDF + HTML : exactement les valeurs snapshotées.
    frozen = resolve_invoice_billed_to(invoice)
    assert frozen.source == "snapshot"
    assert _get_billed_to(invoice) == pdf_before
    assert _html_block(invoice) == html_before
    assert MUTATED_STREET not in _get_billed_to(invoice)[1]
    assert "Renommée" not in _get_billed_to(invoice)[0]
    assert "NOUVEAU" not in _get_billed_to(invoice)[0]

    # 5. Régénération forcée (SENT autorisé) : même bloc, snapshot intact.
    seen: dict[str, tuple[str, str]] = {}

    def _fake_execute(*, invoice, force_regenerate):
        seen["pdf"] = _get_billed_to(invoice)
        seen["html"] = _html_block(invoice)
        return SimpleNamespace(
            ok=True, pdf_url="/uploads/invoices/regen.pdf", error=None, status_code=None
        )

    with (
        patch(
            "application.invoices.force_regenerate_invoice_pdf.GenerateInvoicePdfUseCase"
        ) as uc_cls,
        patch(
            "application.invoices.force_regenerate_invoice_pdf._delete_replaced_invoice_pdf"
        ),
    ):
        uc_cls.return_value.execute.side_effect = _fake_execute
        result = ForceRegenerateInvoicePdfUseCase().execute(
            company_id=company.id, invoice_id=invoice.id
        )
    assert result.ok is True
    assert seen["pdf"] == pdf_before
    assert seen["html"] == html_before
    invoice = _reload(invoice.id)
    assert read_billed_to_snapshot(invoice) == snap


def _emit_then_mutate_and_regenerate(db, company) -> tuple[list[str], list[str]]:
    """Pipeline PDF réel : (pages du PDF émis, pages du PDF régénéré après mutation)."""
    db.session.add(
        CompanyBillingSettings(
            company_id=company.id,
            iban="CH6509000000152631289",
            payment_terms_days=10,
        )
    )
    db.session.commit()
    invoice, _guylene = _arnaud_curatelle_invoice(db, company)
    invoice.pdf_url = None
    db.session.commit()

    service = PDFService()
    url = service.generate_invoice_pdf(invoice, force_regenerate=True)
    assert url
    invoice = Invoice.query.get(invoice.id)
    mark_pdf_ready(invoice, url)
    db.session.commit()
    invoice = _reload(invoice.id)
    assert read_billed_to_snapshot(invoice)["captured_reason"] == "pdf_generation"
    first_pdf = Path(service.invoices_dir, url.rsplit("/", 1)[-1]).read_bytes()

    invoice.mark_as_sent()
    db.session.commit()
    _mutate_master_data(db, _reload(invoice.id))

    with patch(
        "application.invoices.force_regenerate_invoice_pdf._delete_replaced_invoice_pdf"
    ):
        result = ForceRegenerateInvoicePdfUseCase().execute(
            company_id=company.id, invoice_id=invoice.id
        )
    assert result.ok is True
    assert result.pdf_url
    second_pdf = Path(
        service.invoices_dir, result.pdf_url.rsplit("/", 1)[-1]
    ).read_bytes()
    return extract_text_per_page(first_pdf), extract_text_per_page(second_pdf)


def _invoice_page(pages: list[str]) -> str:
    page = next(p for p in pages if "Facturé à" in p)
    return page.split("Numéro de facture")[0]


@pytest.mark.integration
def test_real_pdf_regeneration_after_mutation_keeps_snapshotted_block(db, company):
    """Pipeline PDF réel : le bloc « Facturé à » du PDF régénéré = celui du PDF émis."""
    first_pages, second_pages = _emit_then_mutate_and_regenerate(db, company)
    first_block = _invoice_page(first_pages)
    second_block = _invoice_page(second_pages)

    for block in (first_block, second_block):
        assert (
            "Arnaud JACQUEMOUD\nc/o Mme Lucia GUYLÈNE\nRue Patru 2\n1205 Genève"
            in block
        )
    assert MUTATED_STREET not in second_block
    assert "NOUVEAU" not in second_block
    assert "Renomm" not in second_block
    assert "Nouveau Contact" not in second_block
    assert first_block == second_block


@pytest.mark.integration
def test_real_pdf_qr_bill_debtor_is_also_frozen(db, company):
    first_pages, second_pages = _emit_then_mutate_and_regenerate(db, company)
    first_qr = next(p for p in first_pages if "Payable par" in p)
    second_qr = next(p for p in second_pages if "Payable par" in p)
    assert "Arnaud JACQUEMOUD" in first_qr
    assert "JACQUEMOUD-NOUVEAU" not in second_qr
    assert "Chemin Nouveau 7" not in second_qr
    assert MUTATED_STREET not in second_qr
    assert first_qr == second_qr


# ───────────── facture figée sans snapshot (ancien modèle) → fallback explicite ─────────────


def test_frozen_invoice_without_snapshot_uses_legacy_live_fallback(db, company):
    invoice, _guylene = _arnaud_curatelle_invoice(db, company)
    invoice.mark_as_sent()
    db.session.commit()
    # Simuler une facture émise avant l'introduction du snapshot.
    invoice = _reload(invoice.id)
    meta = dict(invoice.meta or {})
    meta.pop(BILLED_TO_SNAPSHOT_KEY, None)
    meta.pop("qr_debtor_snapshot", None)
    invoice.meta = meta
    db.session.commit()
    invoice = _reload(invoice.id)
    assert read_billed_to_snapshot(invoice) is None

    party = resolve_invoice_billed_to(invoice)
    assert party.source == "legacy_live"
    assert party.mode == "patient_care_of"
    assert _get_billed_to(invoice) == (EXPECTED_NAME, EXPECTED_ADDR)

    # Pas de backfill implicite : une régénération ne crée pas de snapshot a posteriori.
    assert refresh_billed_to_snapshot_if_draft(invoice) is None
    assert read_billed_to_snapshot(invoice) is None


# ───────────────────── gel à chaque transition hors DRAFT ─────────────────────


@pytest.mark.parametrize(
    "transition",
    ["mark_as_sent", "status_sent", "status_paid", "status_partially_paid", "cancel"],
)
def test_every_exit_from_draft_freezes_snapshot(db, company, transition):
    invoice, _guylene = _arnaud_curatelle_invoice(db, company)
    if transition == "mark_as_sent":
        invoice.mark_as_sent()
    elif transition == "status_sent":
        invoice.status = InvoiceStatus.SENT
    elif transition == "status_paid":
        invoice.status = InvoiceStatus.PAID
    elif transition == "status_partially_paid":
        invoice.status = InvoiceStatus.PARTIALLY_PAID
    else:
        invoice.cancel()
    leaves, old, new = invoice_leaves_draft(invoice)
    assert (leaves, old) == (True, "draft")
    db.session.commit()
    invoice = _reload(invoice.id)
    snap = read_billed_to_snapshot(invoice)
    assert snap is not None
    assert snap["frozen_at"]
    assert snap["frozen_reason"] == f"status:draft->{new}"
    assert snap["addressee"] == "Arnaud JACQUEMOUD"
    assert snap["care_of"] == "Mme Lucia Guylène"


def test_non_draft_transitions_do_not_rewrite_snapshot(db, company):
    invoice, _guylene = _arnaud_curatelle_invoice(db, company)
    invoice.mark_as_sent()
    db.session.commit()
    invoice = _reload(invoice.id)
    snap = read_billed_to_snapshot(invoice)

    _mutate_master_data(db, invoice)
    invoice = _reload(invoice.id)
    invoice.status = InvoiceStatus.OVERDUE
    assert invoice_leaves_draft(invoice)[0] is False
    assert freeze_billed_to_for_session(db.session) == 0
    db.session.commit()
    invoice = _reload(invoice.id)
    assert read_billed_to_snapshot(invoice) == snap
    # Un gel explicite sur un snapshot déjà figé ne le réécrit pas non plus.
    assert freeze_billed_to_snapshot(invoice) == snap


def test_frozen_invoice_never_syncs_master_data_nor_payer_snapshot(db, company):
    """Facture figée : ni le BillingParty PATIENT ni ``recipient_snapshot`` ne sont réécrits."""
    jean = _patient(
        db,
        company,
        first="Jean",
        last="Dupont",
        street="Rue X 1",
        zip_code="1200",
        city="Genève",
    )
    self_party = _party(
        db,
        company,
        party_type=BillingPartyType.PATIENT,
        name="Jean Dupont",
        address="Rue X 1\n1200 Genève",
    )
    meta = {
        "recipient_snapshot": {
            "type": "patient",
            "display_name": "Jean Dupont",
            "billing_address": "Rue X 1\n1200 Genève",
            "billing_party_id": self_party.id,
        }
    }
    invoice = _invoice(db, company, jean, party=self_party, meta=meta)
    invoice.mark_as_sent()
    db.session.commit()
    invoice = _reload(invoice.id)

    jean = invoice.client
    jean.user.last_name = "Durand"
    jean.domicile_address = MUTATED_STREET
    db.session.commit()
    invoice = _reload(invoice.id)

    sync_patient_billing_party_from_live(invoice)
    refresh_recipient_snapshot_meta(invoice)
    db.session.commit()
    invoice = _reload(invoice.id)
    assert invoice.billing_party.display_name == "Jean Dupont"
    assert invoice.billing_party.billing_address == "Rue X 1\n1200 Genève"
    assert invoice.meta["recipient_snapshot"]["display_name"] == "Jean Dupont"
    assert _get_billed_to(invoice) == ("Jean DUPONT", "Rue X 1<br/>1200 Genève")


# ───────────── snapshot autonome : reproduction sans master data ─────────────


def test_snapshot_round_trip_reproduces_block_without_master_data():
    party = BilledToParty(
        mode="patient_care_of",
        addressee="Arnaud JACQUEMOUD",
        address="Rue Patru, 2, 1205, Genève",
        care_of="Mme Lucia Guylène",
        attention="Jean Dupont",
        address_owner="Mme Lucia Guylène",
        client_reference="123.456",
        client_reference_label="No. SPC",
        patient_name="Arnaud JACQUEMOUD",
        payer_contact_email="curatelle@example.test",
        billing_party=SimpleNamespace(
            display_name="Mme Lucia Guylène", type="curatorship"
        ),
    )
    invoice_stub = SimpleNamespace(
        id=1, billing_party_id=42, client_id=7, institution_patient_id=None
    )
    snap = billed_to_snapshot_from_party(
        party, invoice_stub, reason="pdf_generation", frozen=True
    )
    assert snap["version"] == 1
    assert snap["billing_party_id"] == 42
    assert snap["client_id"] == 7
    assert snap["frozen_at"]

    rebuilt = billed_to_party_from_snapshot(snap)
    assert rebuilt.source == "snapshot"
    assert rebuilt.mode_origin == "snapshot"
    assert rebuilt.billing_party is None  # aucune dépendance aux master data
    assert billed_to_name_lines(rebuilt) == billed_to_name_lines(party)
    assert rebuilt.address == party.address
    assert rebuilt.client_reference == "123.456"
    assert rebuilt.client_reference_label == "No. SPC"
    assert rebuilt.payer_contact_email == "curatelle@example.test"

    # Snapshot sans ``raw`` (édité / migré) : l'adresse est recomposée depuis la structure.
    without_raw = dict(snap)
    without_raw["address"] = {**snap["address"], "raw": ""}
    assert (
        billed_to_party_from_snapshot(without_raw).address == "Rue Patru 2\n1205 Genève"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "Rue Patru, 2, 1205, Genève",
            {
                "street": "Rue Patru 2",
                "postal_code": "1205",
                "city": "Genève",
                "country": None,
            },
        ),
        (
            "Rte des Jeunes 1c, 1227 Genève, Suisse",
            {
                "street": "Rte des Jeunes 1c",
                "postal_code": "1227",
                "city": "Genève",
                "country": "CH",
            },
        ),
        (
            "Rue Exemple 99\n1200 Genève",
            {
                "street": "Rue Exemple 99",
                "postal_code": "1200",
                "city": "Genève",
                "country": None,
            },
        ),
        ("", {"street": None, "postal_code": None, "city": None, "country": None}),
    ],
)
def test_structure_postal_address(raw, expected):
    structured = structure_postal_address(raw)
    assert structured["raw"] == raw.strip()
    assert {k: structured[k] for k in expected} == expected


# ───────────── rôle de destinataire explicite (Hospice général A / B) ─────────────


def test_explicit_recipient_mode_skips_auto_and_reads_first_explicit():
    assert explicit_recipient_mode(None) is None
    assert explicit_recipient_mode(SimpleNamespace(recipient_mode="auto")) is None
    assert explicit_recipient_mode(SimpleNamespace(recipient_mode="care_of")) == (
        "patient_care_of"
    )
    assert explicit_recipient_mode(SimpleNamespace(recipient_mode="DEBTOR")) == (
        "organization_debtor"
    )
    enum_like = SimpleNamespace(recipient_mode=SimpleNamespace(value="debtor"))
    assert explicit_recipient_mode(enum_like) == "organization_debtor"
    # Precedence métier : le lien (1er argument) l'emporte sur le payeur.
    assert (
        explicit_recipient_mode(
            SimpleNamespace(recipient_mode="care_of"),
            SimpleNamespace(recipient_mode="debtor"),
        )
        == "patient_care_of"
    )


def test_hospice_general_explicit_debtor_vs_inferred_care_of(db, company):
    """Type ``other`` : sans attribut → A (c/o, provisoire) ; ``debtor`` déclaré → B."""
    patient = _patient(
        db,
        company,
        first="Ancien",
        last="NOM",
        street="Rue Ancienne 1",
        zip_code="1200",
        city="Genève",
    )
    hospice = _party(
        db,
        company,
        party_type=BillingPartyType.OTHER,
        name="Hospice général",
        address="Cours de Rive 12\n1204 Genève",
    )
    _link(db, patient, hospice, role="Coordinatrice", contact_name="Amandine HAUSER")
    invoice = _invoice(db, company, patient, party=hospice)

    # A — modèle actuel (inférence par type) : tiers de correspondance.
    inferred = resolve_invoice_billed_to(invoice)
    assert (inferred.mode, inferred.mode_origin) == ("patient_care_of", "inferred")
    assert _get_billed_to(invoice)[0].split("\n") == [
        "Ancien NOM",
        "c/o Hospice GÉNÉRAL",
        "À l'att. de Amandine HAUSER",
    ]

    # B — rôle déclaré « debtor » : organisme facturé en nom propre, contact conservé.
    hospice = invoice.billing_party
    hospice.recipient_mode = "debtor"  # attribut explicite (colonne proposée)
    explicit = resolve_invoice_billed_to(invoice)
    assert (explicit.mode, explicit.mode_origin) == ("organization_debtor", "explicit")
    assert explicit.patient_name == "Ancien NOM"  # informatif, non imprimé
    assert _get_billed_to(invoice)[0].split("\n") == [
        "Hospice GÉNÉRAL",
        "À l'att. de Amandine HAUSER",
    ]
    assert _html_block(invoice)[0].split("<br/>") == _get_billed_to(invoice)[0].split(
        "\n"
    )
    # Le snapshot mémorise l'origine du mode.
    snap = billed_to_snapshot_from_party(explicit, invoice, reason="test", frozen=False)
    assert snap["mode_origin"] == "explicit"


def test_explicit_care_of_overrides_organization_type_and_address_check(db, company):
    """« care_of » déclaré l'emporte sur le type établissement et sur l'adresse commune."""
    patient = _patient(
        db,
        company,
        first="Paul",
        last="MARTIN",
        street="Chemin Gustave-Rochette 14",
        zip_code="1213",
        city="Onex",
    )
    ems = _party(
        db,
        company,
        party_type=BillingPartyType.EMS,
        name="EMS Résidence Butini",
        address="Chemin Gustave-Rochette 14, 1213, Onex",
    )
    link = _link(db, patient, ems, contact_name="Service facturation")
    invoice = _invoice(db, company, patient, party=ems)
    assert resolve_invoice_billed_to(invoice).mode == "organization_debtor"

    link = ClientBillingParty.query.get(link.id)
    link.recipient_mode = "care_of"
    party = resolve_invoice_billed_to(invoice)
    assert (party.mode, party.mode_origin) == ("patient_care_of", "explicit")
    assert billed_to_name_lines(party) == [
        "Paul MARTIN",
        "c/o EMS Résidence Butini",
        "À l'att. de Service facturation",
    ]


def test_other_auto_inferred_care_of(db, company):
    patient = _patient(
        db,
        company,
        first="Léa",
        last="DURAND",
        street="Rue A 1",
        zip_code="1200",
        city="Genève",
    )
    party = _party(
        db,
        company,
        party_type=BillingPartyType.OTHER,
        name="Service social X",
        address="Rue B 2, 1201 Genève",
    )
    party.recipient_mode = "auto"
    _link(db, patient, party)
    invoice = _invoice(db, company, patient, party=party)
    resolved = resolve_invoice_billed_to(invoice)
    assert (resolved.mode, resolved.mode_origin) == ("patient_care_of", "inferred")


def test_other_explicit_care_of(db, company):
    patient = _patient(
        db,
        company,
        first="Léa",
        last="DURAND",
        street="Rue A 1",
        zip_code="1200",
        city="Genève",
    )
    party = _party(
        db,
        company,
        party_type=BillingPartyType.OTHER,
        name="Service social X",
        address="Rue B 2, 1201 Genève",
    )
    party.recipient_mode = "care_of"
    _link(db, patient, party)
    invoice = _invoice(db, company, patient, party=party)
    resolved = resolve_invoice_billed_to(invoice)
    assert (resolved.mode, resolved.mode_origin) == ("patient_care_of", "explicit")
    assert resolved.care_of == "Service social X"


def test_other_explicit_debtor(db, company):
    patient = _patient(
        db,
        company,
        first="Léa",
        last="DURAND",
        street="Rue A 1",
        zip_code="1200",
        city="Genève",
    )
    party = _party(
        db,
        company,
        party_type=BillingPartyType.OTHER,
        name="Service social X",
        address="Rue B 2, 1201 Genève",
    )
    party.recipient_mode = "debtor"
    _link(db, patient, party, contact_name="Comptabilité")
    invoice = _invoice(db, company, patient, party=party)
    resolved = resolve_invoice_billed_to(invoice)
    assert (resolved.mode, resolved.mode_origin) == ("organization_debtor", "explicit")
    assert resolved.care_of is None
    assert "DURAND" not in _get_billed_to(invoice)[0]


def test_opad_explicit_care_of(db, company):
    patient = _patient(
        db,
        company,
        first="Marie",
        last="OLIVIER",
        street="Rue C 3",
        zip_code="1203",
        city="Genève",
    )
    opad = _opad(db, company)
    opad.recipient_mode = "care_of"
    _link(db, patient, opad)
    invoice = _invoice(db, company, patient, party=opad)
    resolved = resolve_invoice_billed_to(invoice)
    assert (resolved.mode, resolved.mode_origin) == ("patient_care_of", "explicit")
    assert resolved.care_of.startswith("OPAD")


def test_organization_explicit_debtor(db, company):
    patient = _patient(
        db,
        company,
        first="Marc",
        last="FAVRE",
        street="Rue D 4",
        zip_code="1204",
        city="Genève",
    )
    clinic = _party(
        db,
        company,
        party_type=BillingPartyType.CLINIC,
        name="Clinique des Acacias",
        address="Route E 5, 1205 Genève",
    )
    clinic.recipient_mode = "debtor"
    _link(db, patient, clinic)
    invoice = _invoice(db, company, patient, party=clinic)
    resolved = resolve_invoice_billed_to(invoice)
    assert (resolved.mode, resolved.mode_origin) == ("organization_debtor", "explicit")


def test_link_recipient_mode_overrides_payer(db, company):
    """Conflit : le lien (ce patient) l'emporte sur le payeur."""
    patient = _patient(
        db,
        company,
        first="Nina",
        last="PERRET",
        street="Rue F 6",
        zip_code="1206",
        city="Genève",
    )
    hospice = _party(
        db,
        company,
        party_type=BillingPartyType.OTHER,
        name="Hospice général",
        address="Cours de Rive 12\n1204 Genève",
    )
    hospice.recipient_mode = "debtor"
    link = _link(db, patient, hospice, contact_name="Amandine HAUSER")
    link.recipient_mode = "care_of"
    invoice = _invoice(db, company, patient, party=hospice)
    resolved = resolve_invoice_billed_to(invoice)
    assert (resolved.mode, resolved.mode_origin) == ("patient_care_of", "explicit")
    assert "c/o Hospice" in _get_billed_to(invoice)[0]


def test_snapshot_timestamp_is_utc_iso():
    party = BilledToParty(
        mode="patient_self", addressee="Jean Dupont", address="Rue X 1"
    )
    snap = billed_to_snapshot_from_party(
        party, SimpleNamespace(id=1), reason="pdf_generation", frozen=False
    )
    captured = datetime.fromisoformat(snap["captured_at"])
    assert captured.tzinfo is not None
    assert captured.utcoffset() == datetime.now(UTC).utcoffset()
