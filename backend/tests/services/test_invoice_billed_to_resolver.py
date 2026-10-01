"""Bloc « Facturé à » — source de vérité commune (``resolve_invoice_billed_to``).

Cas réel : EM-2026-09-0013 (patient Arnaud JACQUEMOUD, payeur curatrice
« Mme Lucia Guylène ») rendait « Mme Lucia GUYLÈNE / Rue Patru, 2 / 1205 Genève »
au lieu de « Arnaud JACQUEMOUD / c/o Mme Lucia GUYLÈNE / Rue Patru 2 / 1205 Genève »
(format historique EM-2026-05-0033 : « Astrid-Jacqueline SCHURTER / c/o OPAD … »).

Règle : ``payer != patient`` ne remplace jamais le patient par le payeur. Le patient
reste la personne facturée ; un tiers de correspondance apparaît en « c/o » avec son
adresse ; un établissement facturé en nom propre reste seul destinataire ; le payeur
reste identifiable séparément (snapshots, registre).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

from application.invoices.billing_opportunities import client_display_name
from application.invoices.force_regenerate_invoice_pdf import (
    ForceRegenerateInvoicePdfUseCase,
    refresh_recipient_snapshot_meta,
    sync_patient_billing_party_from_live,
)
from models import Client, Company, Invoice, InvoiceLine, User
from models.billing_party import BillingParty, ClientBillingParty
from models.enums import (
    BillingPartyType,
    InvoiceBillingStrategy,
    InvoiceLineType,
    InvoiceStatus,
    UserRole,
)
from services.documents.invoice_recipient import (
    BilledToParty,
    billed_to_name_lines,
    resolve_invoice_billed_to,
    same_postal_address,
    split_postal_address_lines,
)
from services.documents.invoice_template_builder import InvoiceTemplateBuilder
from services.documents.pdf import _build_recipient_block_flowable, _get_billed_to

# ───────────────────────────── monde de test ─────────────────────────────


def _ensure_password(user: User) -> None:
    if not getattr(user, "public_id", None):
        user.public_id = str(uuid.uuid4())
    if not getattr(user, "password", None):
        user.set_password("password123", force_change=False)


@pytest.fixture
def company(db):
    suf = uuid.uuid4().hex[:8]
    owner = User(username=f"own_{suf}", email=f"own_{suf}@test.example")
    owner.role = UserRole.company
    _ensure_password(owner)
    db.session.add(owner)
    db.session.flush()
    company = Company(name="ATMR Test", uid_ide="CHE-111.222.333")
    company.user_id = owner.id
    company.domicile_country = "CH"
    db.session.add(company)
    db.session.flush()
    return company


def _patient(
    db,
    company,
    *,
    first: str,
    last: str,
    street: str,
    zip_code: str,
    city: str,
    billing_address: str | None = None,
    residence: str | None = None,
) -> Client:
    suf = uuid.uuid4().hex[:8]
    user = User(
        username=f"cli_{suf}",
        email=f"cli_{suf}@test.example",
        first_name=first,
        last_name=last,
    )
    user.role = UserRole.client
    _ensure_password(user)
    db.session.add(user)
    db.session.flush()
    client = Client(user=user, company=company)
    client.domicile_address = street
    client.domicile_zip = zip_code
    client.domicile_city = city
    if billing_address:
        client.billing_address = billing_address
    if residence:
        client.residence_facility = residence
    db.session.add(client)
    db.session.flush()
    return client


def _party(
    db, company, *, party_type: BillingPartyType, name: str, address: str
) -> BillingParty:
    bp = BillingParty()
    bp.company_id = company.id
    bp.type = party_type
    bp.display_name = name
    bp.billing_address = address
    bp.is_active = True
    db.session.add(bp)
    db.session.flush()
    return bp


def _link(
    db,
    client: Client,
    party: BillingParty,
    *,
    role: str | None = None,
    contact_name: str | None = None,
    client_reference: str | None = None,
) -> ClientBillingParty:
    link = ClientBillingParty()
    link.client_id = client.id
    link.billing_party_id = party.id
    link.role = role
    link.contact_name = contact_name
    link.client_reference = client_reference
    link.is_default = True
    db.session.add(link)
    db.session.flush()
    return link


def _invoice(
    db,
    company,
    client: Client,
    *,
    party: BillingParty | None,
    strategy: InvoiceBillingStrategy = InvoiceBillingStrategy.S1_PATIENT,
    meta: dict | None = None,
) -> Invoice:
    suf = uuid.uuid4().hex[:8]
    invoice = Invoice(
        company=company,
        client=client,
        invoice_number=f"INV-BT-{suf}",
        period_year=2026,
        period_month=9,
        status=InvoiceStatus.DRAFT,
        issued_at=datetime.now(UTC),
        due_date=datetime.now(UTC),
        subtotal_amount=Decimal("180.00"),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal("180.00"),
        billing_party_id=party.id if party is not None else None,
    )
    invoice.billing_strategy = strategy
    invoice.pdf_url = "/uploads/invoices/old_invoice.pdf"
    if meta is not None:
        invoice.meta = meta
    db.session.add(invoice)
    db.session.flush()
    line = InvoiceLine(
        invoice=invoice,
        type=InvoiceLineType.CUSTOM,
        description="Transport test",
        qty=Decimal("1.00"),
        unit_price=Decimal("180.00"),
        line_total=Decimal("180.00"),
        vat_rate=Decimal("0.00"),
        vat_amount=Decimal("0.00"),
        total_with_vat=Decimal("180.00"),
    )
    db.session.add(line)
    db.session.commit()
    db.session.expire_all()
    return Invoice.query.get(invoice.id)


def _jacquemoud(db, company) -> Client:
    return _patient(
        db,
        company,
        first="Arnaud",
        last="JACQUEMOUD",
        street="Avenue du Plateau 4C",
        zip_code="1213",
        city="Petit-Lancy",
        billing_address="Avenue du Plateau 4C, 1213, Petit-Lancy",
    )


def _guylene(db, company) -> BillingParty:
    # Données réelles (prod) : type curatorship, personne physique, adresse « rue, numéro ».
    return _party(
        db,
        company,
        party_type=BillingPartyType.CURATORSHIP,
        name="Mme Lucia Guylène",
        address="Rue Patru, 2, 1205, Genève",
    )


def _opad(db, company) -> BillingParty:
    return _party(
        db,
        company,
        party_type=BillingPartyType.CURATORSHIP,
        name="OPAD (Office de Protection de l'Adulte)",
        address="Rte des Jeunes 1c, 1227 Genève, Suisse",
    )


def _html_name_lines(invoice) -> list[str]:
    name, _addr = InvoiceTemplateBuilder()._resolve_billed_to(invoice)
    return name.split("<br/>")


def _html_addr_lines(invoice) -> list[str]:
    _name, addr = InvoiceTemplateBuilder()._resolve_billed_to(invoice)
    return addr.split("<br/>")


# ───────────────────────── 1. patient = payeur ─────────────────────────


def test_patient_is_payer_no_care_of_line(db, company):
    """Patient = payeur (BillingParty PATIENT ou aucun payeur) → aucune ligne c/o."""
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
    for party in (self_party, None):
        invoice = _invoice(db, company, jean, party=party)
        resolved = resolve_invoice_billed_to(invoice)
        assert resolved.mode == "patient_self"
        assert resolved.care_of is None
        name, addr = _get_billed_to(invoice)
        assert name == "Jean DUPONT"
        assert "c/o" not in name.lower()
        assert addr == "Rue X 1<br/>1200 Genève"


# ─────────────── 2. patient + curatrice personne physique ───────────────


def test_patient_with_natural_person_curator_care_of(db, company):
    """EM-2026-09-0013 : patient, c/o curatrice, adresse de la curatrice."""
    arnaud = _jacquemoud(db, company)
    guylene = _guylene(db, company)
    _link(db, arnaud, guylene, role="Curatrice")
    invoice = _invoice(db, company, arnaud, party=guylene)

    resolved = resolve_invoice_billed_to(invoice)
    assert resolved.mode == "patient_care_of"
    assert resolved.addressee == "Arnaud JACQUEMOUD"
    assert resolved.care_of == "Mme Lucia Guylène"
    assert resolved.attention is None  # le rôle « Curatrice » n'est jamais imprimé

    name, addr = _get_billed_to(invoice)
    assert name == "Arnaud JACQUEMOUD\nc/o Mme Lucia GUYLÈNE"
    assert addr == "Rue Patru 2<br/>1205 Genève"
    assert "curat" not in name.lower()
    # Le payeur reste identifiable séparément dans les données comptables.
    assert invoice.billing_party.display_name == "Mme Lucia Guylène"


# ───────────────────────── 3. patient + OPAD ─────────────────────────


def test_patient_with_opad_care_of(db, company):
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

    resolved = resolve_invoice_billed_to(invoice)
    assert resolved.mode == "patient_care_of"
    name, addr = _get_billed_to(invoice)
    assert name.split("\n")[0] == "Astrid-Jacqueline SCHURTER"
    assert name.split("\n")[1].startswith("c/o OPAD")
    assert "Rte des Jeunes 1c" in addr
    assert "1227 Genève" in addr
    assert "Cité Vieusseux" not in addr


# ─────── 4. adresse de facturation appartenant au patient → pas de c/o ───────


def test_payer_address_belongs_to_patient_no_artificial_care_of(db, company):
    """Payeur distinct mais domicilié à l'adresse du patient : pas de c/o artificiel."""
    mireille = _patient(
        db,
        company,
        first="Mireille",
        last="LIENHARDT",
        street="Avenue du Gros-Chêne 14",
        zip_code="1213",
        city="Onex",
    )
    spouse = _party(
        db,
        company,
        party_type=BillingPartyType.FAMILY,
        name="Eric Lienhardt",
        address="Avenue du Gros-Chêne 14, 1213, Onex, Suisse",
    )
    _link(db, mireille, spouse, role="Époux")
    invoice = _invoice(db, company, mireille, party=spouse)

    resolved = resolve_invoice_billed_to(invoice)
    assert resolved.mode == "patient_self"
    assert resolved.care_of is None
    name, addr = _get_billed_to(invoice)
    assert name == "Mireille LIENHARDT"
    assert "c/o" not in name.lower()
    assert addr == "Avenue du Gros-Chêne 14<br/>1213 Onex"
    # Le payeur reste le conjoint dans les données comptables.
    assert invoice.billing_party.display_name == "Eric Lienhardt"


def test_payer_address_matches_explicit_client_billing_address(db, company):
    """Adresse de facturation explicite du client = adresse du payeur → pas de c/o."""
    patient = _patient(
        db,
        company,
        first="Paul",
        last="MARTIN",
        street="Chemin Gustave-Rochette 14",
        zip_code="1213",
        city="Onex",
        billing_address="EMS Résidence Butini, Chemin Gustave-Rochette 14, 1213 Onex",
        residence="EMS Résidence Butini",
    )
    son = _party(
        db,
        company,
        party_type=BillingPartyType.FAMILY,
        name="Luc Martin",
        address="EMS Résidence Butini\nChemin Gustave-Rochette 14\n1213 Onex",
    )
    _link(db, patient, son, role="Fils")
    invoice = _invoice(db, company, patient, party=son)

    name, _addr = _get_billed_to(invoice)
    assert name.split("\n")[0] == "Paul MARTIN"
    assert "c/o" not in name.lower()
    assert "EMS Résidence Butini" in name


# ──────────── 5. régénération → même identité / adresse ────────────


def test_regeneration_keeps_identity_and_payer_snapshot(db, company):
    """Régénérer ne change ni le bloc ni le payeur du snapshot (toujours la curatrice)."""
    arnaud = _jacquemoud(db, company)
    guylene = _guylene(db, company)
    _link(db, arnaud, guylene, role="Curatrice")
    meta = {
        "recipient_snapshot": {
            "type": "curatorship",
            "display_name": "Mme Lucia Guylène",
            "billing_address": "Rue Patru, 2, 1205, Genève",
            "billing_party_id": guylene.id,
            "recipient_status": "ready",
        },
        "billing_subject_snapshot": {
            "type": "client",
            "client_id": arnaud.id,
            "display_name": "Arnaud JACQUEMOUD",
        },
    }
    invoice = _invoice(db, company, arnaud, party=guylene, meta=meta)
    before = _get_billed_to(invoice)

    # Étapes exécutées par ForceRegenerateInvoicePdfUseCase avant la génération.
    sync_patient_billing_party_from_live(invoice)
    refresh_recipient_snapshot_meta(invoice)
    db.session.commit()
    db.session.expire_all()
    invoice = Invoice.query.get(invoice.id)

    after = _get_billed_to(invoice)
    assert after == before
    assert after[0] == "Arnaud JACQUEMOUD\nc/o Mme Lucia GUYLÈNE"
    snap = invoice.meta["recipient_snapshot"]
    assert (
        snap["display_name"] == "Mme Lucia Guylène"
    )  # payeur, jamais écrasé par le patient
    assert snap["billing_party_id"] == guylene.id
    assert (
        invoice.meta["billing_subject_snapshot"]["display_name"] == "Arnaud JACQUEMOUD"
    )

    # Régénération complète : le PDF est construit depuis la même facture / même bloc.
    seen: dict[str, tuple[str, str]] = {}

    def _fake_execute(*, invoice, force_regenerate):
        assert force_regenerate is True
        seen["block"] = _get_billed_to(invoice)
        return SimpleNamespace(
            ok=True, pdf_url="/uploads/invoices/new.pdf", error=None, status_code=None
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
    assert seen["block"] == before


# ──────────── 6. aperçu HTML et PDF officiel → même bloc ────────────


def test_html_builder_and_pdf_share_same_recipient_block(db, company):
    arnaud = _jacquemoud(db, company)
    guylene = _guylene(db, company)
    _link(db, arnaud, guylene, role="Curatrice")
    invoice = _invoice(db, company, arnaud, party=guylene)

    pdf_name, pdf_addr = _get_billed_to(invoice)
    assert pdf_name.split("\n") == _html_name_lines(invoice)
    assert pdf_addr.split("<br/>") == _html_addr_lines(invoice)
    assert _html_name_lines(invoice) == ["Arnaud JACQUEMOUD", "c/o Mme Lucia GUYLÈNE"]
    assert _html_addr_lines(invoice) == ["Rue Patru 2", "1205 Genève"]


def test_html_builder_and_pdf_agree_for_patient_and_organization(db, company):
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
    for party in (None, clinic):
        invoice = _invoice(db, company, jean, party=party)
        pdf_name, _ = _get_billed_to(invoice)
        assert pdf_name.split("\n") == _html_name_lines(invoice)


# ──────── 7. téléchargement / envoi / régénération → même résultat ────────


def test_recipient_block_flowable_uses_resolver_for_every_entry_point(db, company):
    """Le flowable (PDF officiel, régénération, pièce jointe e-mail) = resolver."""
    arnaud = _jacquemoud(db, company)
    guylene = _guylene(db, company)
    _link(db, arnaud, guylene, role="Curatrice")
    invoice = _invoice(db, company, arnaud, party=guylene)
    style = ParagraphStyle(
        "Normal",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=10,
        fontName="Helvetica",
    )
    expected = [
        "Arnaud JACQUEMOUD",
        "c/o Mme Lucia GUYLÈNE",
        "Rue Patru 2",
        "1205 Genève",
    ]
    # Pipeline PDF (bookings préchargés) et appel isolé (régénération / envoi).
    for bookings_by_id in ({}, None):
        para, lines = _build_recipient_block_flowable(
            invoice, style, bookings_by_id=bookings_by_id
        )
        assert para is not None
        assert lines == expected
    # Première ligne (patient) en gras, « c/o » comme l'adresse (pas en gras).
    para, _ = _build_recipient_block_flowable(invoice, style)
    assert "<b>Arnaud JACQUEMOUD</b>" in para.text
    assert "<b>c/o" not in para.text


# ───────────── 8. non-régression du cas historique OPAD ─────────────


def test_historical_opad_block_non_regression(db, company):
    """EM-2026-05-0033 : « Astrid-Jacqueline SCHURTER / c/o OPAD (…) / Rte des Jeunes 1c / 1227 Genève »."""
    astrid = _patient(
        db,
        company,
        first="Astrid-Jacqueline",
        last="SCHURTER",
        street="Cité Vieusseux 8",
        zip_code="1203",
        city="Genf",
        residence="IEPA des Franchises",
    )
    opad = _opad(db, company)
    _link(db, astrid, opad)
    invoice = _invoice(db, company, astrid, party=opad)

    name, addr = _get_billed_to(invoice)
    assert name == (
        "Astrid-Jacqueline SCHURTER\nc/o OPAD (Office de Protection de L'ADULTE)"
    )
    assert addr == "Rte des Jeunes 1c<br/>1227 Genève"
    # La résidence du patient n'est pas une adresse de correspondance ici.
    assert "IEPA" not in name


def test_opad_with_contact_adds_attention_line_never_curator(db, company):
    maryse = _patient(
        db,
        company,
        first="Maryse",
        last="BERSET",
        street="Rue A 1",
        zip_code="1201",
        city="Genève",
    )
    opad = _opad(db, company)
    _link(db, maryse, opad, role="Curatrice", contact_name="Getou Christianne MUSANGU")
    invoice = _invoice(db, company, maryse, party=opad)

    name, _addr = _get_billed_to(invoice)
    assert name.split("\n") == [
        "Maryse BERSET",
        "c/o OPAD (Office de Protection de L'ADULTE)",
        "À l'att. de Getou Christianne MUSANGU",
    ]
    assert "curat" not in name.lower()


# ───────────────────────── cas complémentaires ─────────────────────────


def test_organization_debtor_is_sole_addressee(db, company):
    """Clinique / EMS / S2 : l'établissement est facturé en nom propre, jamais de c/o."""
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
    _link(db, jean, clinic, contact_name="Comptabilité")
    for strategy in (
        InvoiceBillingStrategy.S1_PATIENT,
        InvoiceBillingStrategy.S2_CLINIC_MONTHLY,
    ):
        invoice = _invoice(db, company, jean, party=clinic, strategy=strategy)
        resolved = resolve_invoice_billed_to(invoice)
        assert resolved.mode == "organization_debtor"
        name, addr = _get_billed_to(invoice)
        assert "DUPONT" not in name.upper()
        assert "c/o" not in name.lower()
        assert name.startswith("Clinique les Hauts")
        assert addr == "Chemin des Courbes 9<br/>1247 Anières"


def test_other_type_organization_with_contact_keeps_patient_and_attention(db, company):
    """Organisme typé « other » (ex. Hospice général) : patient, c/o organisme, À l'att. du contact."""
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
        address="Rue Ancienne 10\n1201 Genève",
    )
    _link(db, patient, hospice, role="Coordinatrice", contact_name="Amandine HAUSER")
    invoice = _invoice(db, company, patient, party=hospice)

    name, addr = _get_billed_to(invoice)
    assert name.split("\n") == [
        "Ancien NOM",
        "c/o Hospice GÉNÉRAL",
        "À l'att. de Amandine HAUSER",
    ]
    assert "curateur" not in name.lower()
    assert addr == "Rue Ancienne 10<br/>1201 Genève"


def test_explicit_care_of_in_payer_name_is_not_duplicated(db, company):
    patient = _patient(
        db,
        company,
        first="Léa",
        last="ROUX",
        street="Rue B 2",
        zip_code="1202",
        city="Genève",
    )
    party = _party(
        db,
        company,
        party_type=BillingPartyType.CURATORSHIP,
        name="c/o Service des curatelles",
        address="Rue des Curatelles 5, 1204 Genève",
    )
    _link(db, patient, party)
    invoice = _invoice(db, company, patient, party=party)

    name, _addr = _get_billed_to(invoice)
    assert name.split("\n") == ["Léa ROUX", "c/o Service des CURATELLES"]
    assert "c/o c/o" not in name.lower()


def test_company_name_payer_is_not_concatenated_with_patient(db, company):
    patient = _patient(
        db,
        company,
        first="Marc",
        last="FAVRE",
        street="Rue C 3",
        zip_code="1203",
        city="Genève",
    )
    firm = _party(
        db,
        company,
        party_type=BillingPartyType.CURATORSHIP,
        name="GLZ Conseil et Curatelle Sàrl",
        address="GLZ conseil & curatelle Sàrl, Rue de l'Est 8, 1207, Genève",
    )
    _link(db, patient, firm, role="Curateur", contact_name="Marc FAVRE")
    invoice = _invoice(db, company, patient, party=firm)

    name, addr = _get_billed_to(invoice)
    lines = name.split("\n")
    assert lines[0] == "Marc FAVRE"
    assert "GLZ" not in lines[0]
    assert lines[1].startswith("c/o GLZ Conseil et Curatelle")
    assert "FAVRE" not in lines[1]
    # Contact identique au patient : pas de ligne « À l'att. de » redondante.
    assert len(lines) == 2
    assert "Rue de l'Est 8" in addr
    assert "1207 Genève" in addr


def test_payer_without_client_link_falls_back_to_patient_domicile(db, company):
    patient = _patient(
        db,
        company,
        first="Nora",
        last="KELLER",
        street="Rue D 4",
        zip_code="1205",
        city="Genève",
    )
    stranger = _party(
        db,
        company,
        party_type=BillingPartyType.LAWYER,
        name="MMB - Avocats",
        address="Rue du Nant 6, 1207, Genève",
    )
    invoice = _invoice(db, company, patient, party=stranger)  # aucun lien

    resolved = resolve_invoice_billed_to(invoice)
    assert resolved.mode == "patient_self"
    name, addr = _get_billed_to(invoice)
    assert name == "Nora KELLER"
    assert addr == "Rue D 4<br/>1205 Genève"


def test_spc_reference_kept_under_care_of_block(db, company):
    patient = _patient(
        db,
        company,
        first="Charles",
        last="Dupuis",
        street="Rue E 5",
        zip_code="1206",
        city="Genève",
    )
    spc = _party(
        db,
        company,
        party_type=BillingPartyType.OTHER,
        name="SPC",
        address="Route de Chêne 54, 1208, Genève",
    )
    _link(db, patient, spc, client_reference="123.456")
    invoice = _invoice(db, company, patient, party=spc)

    name, addr = _get_billed_to(invoice)
    assert name == "Charles DUPUIS\nc/o SPC"
    assert addr.endswith("No. SPC : 123.456")
    assert _html_addr_lines(invoice)[-1] == "No. SPC : 123.456"


def test_name_lines_formatter_never_touches_prefixes():
    party = BilledToParty(
        mode="patient_care_of",
        addressee="Arnaud Jacquemoud",
        care_of="Mme Lucia Guylène",
        attention="Jean Dupont",
        address="Rue Patru, 2, 1205, Genève",
    )
    lines = billed_to_name_lines(party, name_formatter=str.upper)
    assert lines == [
        "ARNAUD JACQUEMOUD",
        "c/o MME LUCIA GUYLÈNE",
        "À l'att. de Jean Dupont",
    ]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("Rue Patru, 2, 1205, Genève", "Avenue du Plateau 4C\n1213 Petit-Lancy", False),
        (
            "Avenue du Plateau 4C, 1213, Petit-Lancy, Suisse",
            "Avenue du Plateau 4C\n1213 Petit-Lancy",
            True,
        ),
        (
            "Rte des Jeunes 1c, 1227 Genève, Suisse",
            "Cité Vieusseux 8\n1203 Genf",
            False,
        ),
        ("", "Rue X 1\n1200 Genève", False),
        ("1200", "1200", False),
    ],
)
def test_same_postal_address(left, right, expected):
    assert same_postal_address(left, right) is expected


def test_split_postal_address_lines_joins_number_and_postal_code():
    assert split_postal_address_lines("Rue Patru, 2, 1205, Genève") == [
        "Rue Patru 2",
        "1205 Genève",
    ]
    assert split_postal_address_lines("Rte des Jeunes 1c, 1227 Genève, Suisse") == [
        "Rte des Jeunes 1c",
        "1227 Genève",
        "Suisse",
    ]
    assert split_postal_address_lines(
        "Avenue du Grand-Salève 2, 1255, Veyrier\n1255 Veyrier"
    ) == [
        "Avenue du Grand-Salève 2",
        "1255 Veyrier",
    ]
    assert split_postal_address_lines("") == []


def test_client_display_name_is_patient_not_payer(db, company):
    """Snapshot sujet : le nom du patient, jamais celui du payeur."""
    arnaud = _jacquemoud(db, company)
    assert client_display_name(arnaud) == "Arnaud JACQUEMOUD"
