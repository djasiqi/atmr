"""LIRIE — étape 22 : immutabilité des factures émises (correctif conservatoire).

Les tests démontrent l'**absence de modification** (fichier PDF, hash, lignes,
totaux, TVA, destinataire, QR, meta, ``updated_at``) et pas seulement un code
HTTP de refus.

Cas couverts :
- A brouillon (flux existant conservé) ; B envoyée ; C partiellement payée ;
  D payée ; E en retard ; (+ annulée) ;
- F facture patient ; G facture institutionnelle ; H facture partenaire ;
  I facture avec QR ; J facture avec frais papier ; K taux TVA historique figé ;
  L facture émise sans PDF original ;
- deux régénérations simultanées ; régénération pendant un changement de statut ;
  erreur entre écriture et validation ; fichier absent ; réponse perdue /
  opération répétée ; utilisateur non autorisé ; autre entreprise ; rappel.

Base PostgreSQL jetable (``atmr_etape22``), fichiers PDF factices dans un
répertoire temporaire (``UPLOADS_DIR`` surchargé), aucune notification réelle.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import threading
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import text
from sqlalchemy.orm import selectinload

from application.invoices.edit_draft_invoice import (
    add_draft_custom_line,
    apply_draft_global_discount,
    invoice_allows_line_editing,
    remove_draft_invoice_line,
    repair_draft_invoice_if_line_totals_inconsistent,
    update_draft_invoice_line,
)
from application.invoices.force_regenerate_invoice_pdf import (
    ForceRegenerateInvoicePdfUseCase,
    force_regenerate_invoice_pdf,
)
from application.invoices.force_regenerate_partner_invoice_pdf import (
    force_regenerate_partner_invoice_pdf,
)
from application.invoices.invoice_pdf_state import get_pdf_state
from application.invoices.issued_invoice_guard import (
    ERROR_CODE_ISSUED_INVOICE_IMMUTABLE,
    ERROR_CODE_ISSUED_INVOICE_PDF_MISSING,
    ISSUED_INVOICE_STATUSES,
    IssuedInvoiceImmutableError,
    assert_invoice_mutable,
    invoice_is_issued,
    invoice_is_mutable,
    partner_invoice_is_issued,
)
from application.invoices.send_invoice_by_email import (
    SendInvoiceByEmailInput,
    SendInvoiceByEmailUseCase,
)
from application.invoices.update_partner_invoice_draft import (
    update_partner_invoice_draft,
)
from models import Client, CompanyBillingSettings, Invoice, InvoiceLine, User
from models.billing_party import BillingParty
from models.enums import (
    BillingPartyType,
    InvoiceBillingStrategy,
    InvoiceLineType,
    InvoiceStatus,
)
from models.partner_invoice import PartnerInvoice, PartnerInvoiceStatus
from services.documents.pdf import PDFService, _vat_status_text_for_invoice
from services.partnerships.invoices import PartnerInvoiceService
from tests.application.invoices.test_force_regenerate_invoice_pdf import _world
from tests.routes.test_invoices_list_lot6 import _company_headers

_REGEN_UC = (
    "application.invoices.force_regenerate_invoice_pdf.GenerateInvoicePdfUseCase"
)
_REGEN_DELETE = (
    "application.invoices.force_regenerate_invoice_pdf._delete_replaced_invoice_pdf"
)
_BREVO_SEND = "services.email.brevo_provider.BrevoEmailProvider.send_invoice_email"
_BREVO_TX = "services.email.brevo_provider.BrevoEmailProvider.send_transactional"
# Classe provider patchée dans le use case d'envoi : aucune construction réelle,
# aucun appel externe possible.
_SEND_BREVO_CLASS = "application.invoices.send_invoice_by_email.BrevoEmailProvider"

ISSUED_CASES = [
    pytest.param(InvoiceStatus.SENT, id="B-sent"),
    pytest.param(InvoiceStatus.PARTIALLY_PAID, id="C-partially_paid"),
    pytest.param(InvoiceStatus.PAID, id="D-paid"),
    pytest.param(InvoiceStatus.OVERDUE, id="E-overdue"),
    pytest.param(InvoiceStatus.CANCELLED, id="X-cancelled"),
]


# ───────────────────────────── outillage ─────────────────────────────


@pytest.fixture
def uploads_tmp(app, tmp_path):
    """Répertoire uploads isolé : les PDF factices ne touchent pas le dépôt."""
    previous = {key: app.config.get(key) for key in ("UPLOADS_DIR", "UPLOAD_FOLDER")}
    app.config["UPLOADS_DIR"] = str(tmp_path)
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    (tmp_path / "invoices").mkdir(parents=True, exist_ok=True)
    try:
        yield tmp_path
    finally:
        for key, value in previous.items():
            if value is None:
                app.config.pop(key, None)
            else:
                app.config[key] = value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fake_pdf(uploads_tmp: Path, name: str) -> tuple[str, Path, str]:
    """Écrit un « PDF original » factice ; retourne (pdf_url, chemin, sha256)."""
    path = uploads_tmp / "invoices" / name
    path.write_bytes(
        b"%PDF-1.4\n% original LIRIE etape22 " + name.encode() + b"\n%%EOF\n"
    )
    return f"/uploads/invoices/{name}", path, _sha256(path)


def _uploads_listing(uploads_tmp: Path) -> set[str]:
    return {p.name for p in (uploads_tmp / "invoices").iterdir()}


def _dec(value) -> str | None:
    return None if value is None else str(Decimal(str(value)))


def _financial_state(db, invoice_id: int) -> dict:
    """Photographie complète de l'état financier/documentaire d'une facture."""
    db.session.expire_all()
    inv = Invoice.query.options(selectinload(Invoice.lines)).get(invoice_id)
    assert inv is not None
    lines = sorted(
        (
            (
                ln.id,
                ln.description,
                _dec(ln.qty),
                _dec(ln.unit_price),
                _dec(ln.line_total),
                _dec(ln.vat_rate),
                _dec(ln.vat_amount),
                _dec(ln.total_with_vat),
                copy.deepcopy(ln.line_meta),
                ln.reservation_id,
            )
            for ln in inv.lines
        ),
        key=lambda t: t[0],
    )
    return {
        "status": inv.status.value,
        "pdf_url": inv.pdf_url,
        "subtotal": _dec(inv.subtotal_amount),
        "vat_total": _dec(inv.vat_total_amount),
        "total": _dec(inv.total_amount),
        "balance_due": _dec(inv.balance_due),
        "amount_paid": _dec(inv.amount_paid),
        "qr_reference": inv.qr_reference,
        "billing_party_id": inv.billing_party_id,
        "bill_to_client_id": inv.bill_to_client_id,
        "billed_to_company_id": inv.billed_to_company_id,
        "vat_breakdown": copy.deepcopy(inv.vat_breakdown),
        "meta": copy.deepcopy(inv.meta),
        "updated_at": inv.updated_at.isoformat() if inv.updated_at else None,
        "lines": lines,
        "line_count": len(lines),
    }


def _issue(db, invoice: Invoice, status: InvoiceStatus, *, pdf_url: str | None):
    """Passe une facture brouillon au statut émis ``status`` avec snapshots figés."""
    meta = dict(invoice.meta or {})
    meta["vat"] = {
        "applicable": True,
        "default_rate": 7.7,
        "label": "TVA 7.7 %",
        "number": None,
    }
    meta["qr_debtor_snapshot"] = {
        "name": "Ancien NOM",
        "street": "Rue Ancienne 1",
        "pcode": "1200",
        "town": "Genève",
    }
    meta["qr_creditor_snapshot"] = {
        "name": "ATMR Test",
        "iban": "CH6509000000152631289",
    }
    invoice.meta = meta
    invoice.qr_reference = "000000000000000000000000012"
    invoice.pdf_url = pdf_url
    invoice.status = status
    if status in (InvoiceStatus.PARTIALLY_PAID, InvoiceStatus.PAID):
        paid = (
            invoice.total_amount if status == InvoiceStatus.PAID else Decimal("30.00")
        )
        invoice.amount_paid = paid
        invoice.balance_due = (invoice.total_amount or Decimal("0")) - paid
    db.session.commit()
    db.session.expire_all()
    return Invoice.query.get(invoice.id)


def _issued_world(db, uploads_tmp: Path, status: InvoiceStatus, suffix: str = ""):
    """Monde patient (F) : facture émise + PDF original factice sur disque."""
    company, client, client_user, bp, invoice = _world(db)
    company.is_approved = True
    db.session.commit()
    name = f"invoice_{invoice.invoice_number}{suffix}_20260101_120000.pdf"
    pdf_url, path, digest = _write_fake_pdf(uploads_tmp, name)
    invoice = _issue(db, invoice, status, pdf_url=pdf_url)
    return SimpleNamespace(
        company=company,
        company_id=company.id,
        owner_user_id=company.user_id,
        client=client,
        client_id=client.id,
        client_user=client_user,
        client_user_id=client_user.id,
        billing_party=bp,
        invoice=invoice,
        invoice_id=invoice.id,
        line_id=invoice.lines[0].id,
        pdf_url=pdf_url,
        pdf_path=path,
        sha256=digest,
    )


def _assert_untouched(db, world, before: dict) -> None:
    """Fichier présent, hash identique, état financier strictement identique."""
    assert world.pdf_path.exists(), "le PDF original a été supprimé"
    assert _sha256(world.pdf_path) == world.sha256, "le PDF original a été modifié"
    after = _financial_state(db, world.invoice_id)
    assert after == before
    assert after["pdf_url"] == world.pdf_url
    assert after["meta"]["vat"]["default_rate"] == 7.7
    assert after["qr_reference"] == before["qr_reference"]
    assert after["meta"]["qr_debtor_snapshot"] == before["meta"]["qr_debtor_snapshot"]
    assert (
        after["meta"]["qr_creditor_snapshot"] == before["meta"]["qr_creditor_snapshot"]
    )


def _assert_refusal(result, *, code: str = ERROR_CODE_ISSUED_INVOICE_IMMUTABLE):
    assert result.ok is False
    assert result.status_code == 409
    assert isinstance(result.error, dict)
    assert result.error.get("error_code") == code
    assert (
        result.error.get("details", {}).get("administrative_procedure_required") is True
    )
    assert "administration" in (result.error.get("error") or "").lower()


# ───────────────────────── garde-fou (unitaire) ─────────────────────────


def test_guard_statuses_match_real_model():
    """Les statuts émis = tous les statuts réels sauf DRAFT (modèle, pas hypothèse)."""
    assert frozenset(InvoiceStatus) - {InvoiceStatus.DRAFT} == ISSUED_INVOICE_STATUSES


@pytest.mark.parametrize(
    ("status", "expected_issued"),
    [
        (InvoiceStatus.DRAFT, False),
        ("draft", False),
        (InvoiceStatus.SENT, True),
        (InvoiceStatus.PARTIALLY_PAID, True),
        (InvoiceStatus.PAID, True),
        (InvoiceStatus.OVERDUE, True),
        (InvoiceStatus.CANCELLED, True),
        ("UNPAID", True),  # valeur héritée inconnue du modèle Python → fail-closed
        ("", True),
        (None, True),
    ],
)
def test_guard_is_fail_closed(status, expected_issued):
    fake = SimpleNamespace(id=1, status=status)
    assert invoice_is_issued(fake) is expected_issued
    assert invoice_is_mutable(fake) is (not expected_issued)
    assert invoice_allows_line_editing(fake) is (not expected_issued)
    if expected_issued:
        with pytest.raises(IssuedInvoiceImmutableError) as exc:
            assert_invoice_mutable(fake, "unit")
        payload = exc.value.to_error_payload()
        assert exc.value.status_code == 409
        assert payload["error_code"] == ERROR_CODE_ISSUED_INVOICE_IMMUTABLE
        assert payload["details"]["operation"] == "unit"
        assert payload["details"]["administrative_procedure_required"] is True
    else:
        assert_invoice_mutable(fake, "unit")


def test_guard_has_no_bypass_parameter():
    """Aucune option de contournement : ni dans le garde-fou, ni dans la génération PDF."""
    assert (
        "force_bypass_locked"
        not in inspect.signature(PDFService.generate_invoice_pdf).parameters
    )
    assert set(inspect.signature(assert_invoice_mutable).parameters) == {
        "invoice",
        "operation",
    }


@pytest.mark.parametrize("status", [InvoiceStatus.OVERDUE, InvoiceStatus.CANCELLED])
def test_pdf_state_locks_overdue_and_cancelled(status):
    with_pdf = SimpleNamespace(
        status=status, pdf_url="/uploads/invoices/x.pdf", meta=None
    )
    without = SimpleNamespace(status=status, pdf_url=None, meta=None)
    assert get_pdf_state(with_pdf).status == "ready"
    missing = get_pdf_state(without)
    assert missing.status == "failed"
    assert missing.error == "MISSING_PDF"


# ─────────── B/C/D/E (+annulée) : régénération forcée refusée, rien modifié ───────────


@pytest.mark.integration
@pytest.mark.parametrize("status", ISSUED_CASES)
def test_force_regenerate_issued_invoice_is_refused_and_nothing_changes(
    db, uploads_tmp, status
):
    world = _issued_world(db, uploads_tmp, status)
    before = _financial_state(db, world.invoice_id)
    listing_before = _uploads_listing(uploads_tmp)

    with (
        patch(_REGEN_UC) as uc_cls,
        patch(_REGEN_DELETE) as del_old,
        patch(_BREVO_SEND) as brevo_send,
        patch(_BREVO_TX) as brevo_tx,
    ):
        result = force_regenerate_invoice_pdf(
            company_id=world.company_id, invoice_id=world.invoice_id
        )

    _assert_refusal(result)
    assert result.previous_pdf_url == world.pdf_url
    assert result.error["details"]["status"] == status.value
    uc_cls.return_value.execute.assert_not_called()
    del_old.assert_not_called()
    brevo_send.assert_not_called()
    brevo_tx.assert_not_called()
    _assert_untouched(db, world, before)
    assert _uploads_listing(uploads_tmp) == listing_before


@pytest.mark.integration
@pytest.mark.parametrize("status", ISSUED_CASES)
def test_pdf_service_never_writes_for_issued_invoice(db, uploads_tmp, status):
    """``generate_invoice_pdf`` : aucun recalcul, aucun fichier, PDF original retourné."""
    world = _issued_world(db, uploads_tmp, status)
    # Totaux volontairement incohérents avec Σ lignes : un recalcul silencieux
    # (ancien comportement) les aurait réécrits.
    world.invoice.total_amount = Decimal("999.95")
    world.invoice.subtotal_amount = Decimal("999.95")
    world.invoice.balance_due = Decimal("999.95") - (world.invoice.amount_paid or 0)
    db.session.commit()
    before = _financial_state(db, world.invoice_id)
    listing_before = _uploads_listing(uploads_tmp)

    with (
        patch("shared.upload_write.write_upload_bytes") as writer,
        patch.object(PDFService, "_create_invoice_pdf_content") as renderer,
    ):
        url_plain = PDFService().generate_invoice_pdf(world.invoice)
        url_forced = PDFService().generate_invoice_pdf(
            world.invoice, force_regenerate=True
        )

    assert url_plain == world.pdf_url
    assert url_forced == world.pdf_url
    writer.assert_not_called()
    renderer.assert_not_called()
    after = _financial_state(db, world.invoice_id)
    assert after == before
    assert after["total"] == "999.95"
    assert _uploads_listing(uploads_tmp) == listing_before
    assert _sha256(world.pdf_path) == world.sha256


@pytest.mark.integration
@pytest.mark.parametrize("status", ISSUED_CASES)
def test_line_edits_on_issued_invoice_are_refused(db, uploads_tmp, status):
    """Modification / ajout / suppression de ligne, remise : 409 et état intact."""
    world = _issued_world(db, uploads_tmp, status)
    line_id = world.line_id
    before = _financial_state(db, world.invoice_id)
    cid, iid = world.company_id, world.invoice_id

    results = [
        update_draft_invoice_line(cid, iid, line_id, line_total=1.0),
        add_draft_custom_line(cid, iid, description="Ajout", line_total=10.0),
        apply_draft_global_discount(cid, iid, global_discount_percent=10.0),
        remove_draft_invoice_line(cid, iid, line_id),
    ]
    for res in results:
        assert res.success is False
        assert res.status_code == 409
        assert res.error["error_code"] == ERROR_CODE_ISSUED_INVOICE_IMMUTABLE
    _assert_untouched(db, world, before)


@pytest.mark.integration
def test_repair_on_get_never_touches_issued_invoice(db, uploads_tmp):
    """Le « repair » TTC déclenché au GET (liste / détail) est réservé au brouillon."""
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT)
    world.invoice.total_amount = Decimal("123.45")  # ≠ Σ lignes (80.00)
    db.session.commit()
    before = _financial_state(db, world.invoice_id)
    inv = Invoice.query.options(selectinload(Invoice.lines)).get(world.invoice_id)
    assert repair_draft_invoice_if_line_totals_inconsistent(inv) is False
    db.session.commit()
    after = _financial_state(db, world.invoice_id)
    assert after == before
    assert after["total"] == "123.45"


# ───────────── A : le brouillon conserve le flux (régénération + remplacement) ─────────────


@pytest.mark.integration
def test_draft_regeneration_still_replaces_pdf_and_deletes_old_file(db, uploads_tmp):
    company, _client, _user, _bp, invoice = _world(db)
    old_url, old_path, _digest = _write_fake_pdf(
        uploads_tmp, f"invoice_{invoice.invoice_number}_20260101_120000.pdf"
    )
    new_url, new_path, _ = _write_fake_pdf(
        uploads_tmp, f"invoice_{invoice.invoice_number}_20260102_120000_abcd1234.pdf"
    )
    invoice.pdf_url = old_url
    db.session.commit()
    assert invoice.status == InvoiceStatus.DRAFT

    with patch(_REGEN_UC) as uc_cls:
        uc_cls.return_value.execute.return_value = SimpleNamespace(
            ok=True, pdf_url=new_url, error=None, status_code=None
        )
        result = ForceRegenerateInvoicePdfUseCase().execute(
            company_id=company.id, invoice_id=invoice.id
        )

    assert result.ok is True
    assert result.pdf_url == new_url
    assert result.previous_pdf_url == old_url
    reloaded = Invoice.query.get(invoice.id)
    assert reloaded.pdf_url == new_url
    assert (reloaded.meta or {}).get("pdf", {}).get("status") == "ready"
    # Chemin brouillon : l'ancien fichier est bien remplacé (suppression réelle).
    assert not old_path.exists()
    assert new_path.exists()


# ───────────── G : facture institutionnelle (tiers payeur établissement, S2) ─────────────


@pytest.mark.integration
def test_institution_invoice_issued_is_immutable(db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_inst")
    ems = BillingParty()
    ems.company_id = world.company_id
    ems.type = BillingPartyType.EMS
    ems.display_name = "EMS Les Tilleuls"
    ems.billing_address = "Route des Tilleuls 4\n1206 Genève"
    ems.is_active = True
    db.session.add(ems)
    db.session.flush()
    inv = Invoice.query.get(world.invoice_id)
    inv.billing_party_id = ems.id
    inv.billing_strategy = InvoiceBillingStrategy.S2_CLINIC_MONTHLY
    db.session.commit()
    before = _financial_state(db, inv.id)
    assert before["billing_party_id"] == ems.id

    with patch(_REGEN_UC) as uc_cls, patch(_REGEN_DELETE) as del_old:
        result = force_regenerate_invoice_pdf(
            company_id=world.company_id, invoice_id=inv.id
        )
    _assert_refusal(result)
    uc_cls.return_value.execute.assert_not_called()
    del_old.assert_not_called()
    _assert_untouched(db, world, before)
    # Le destinataire (tiers payeur) n'a pas été réécrit.
    reloaded = BillingParty.query.get(ems.id)
    assert reloaded.display_name == "EMS Les Tilleuls"
    assert reloaded.billing_address == "Route des Tilleuls 4\n1206 Genève"


# ───────────────── J : frais papier jamais ajoutés sur une facture émise ─────────────────


@pytest.mark.integration
@pytest.mark.parametrize("status", [InvoiceStatus.SENT, InvoiceStatus.OVERDUE])
def test_paper_fee_is_never_added_to_issued_invoice(db, uploads_tmp, status):
    world = _issued_world(db, uploads_tmp, status, suffix="_paper")
    client = Client.query.get(world.client_id)
    client.invoice_delivery_method = "paper"
    db.session.commit()
    before = _financial_state(db, world.invoice_id)
    assert before["line_count"] == 1
    cid, iid = world.company_id, world.invoice_id

    with patch(_REGEN_UC) as uc_cls, patch(_REGEN_DELETE):
        regen = force_regenerate_invoice_pdf(company_id=cid, invoice_id=iid)
    edit = add_draft_custom_line(cid, iid, description="X", line_total=1.0)

    _assert_refusal(regen)
    assert edit.status_code == 409
    uc_cls.return_value.execute.assert_not_called()
    after = _financial_state(db, iid)
    assert after == before
    assert after["line_count"] == 1
    assert all("papier" not in (ln[1] or "").lower() for ln in after["lines"])
    assert "paper_invoice_fee_chf" not in (after["meta"] or {})


@pytest.mark.integration
def test_paper_fee_still_applies_to_draft(db, uploads_tmp):
    """Non-régression : le brouillon papier reçoit toujours sa ligne CHF 3."""
    company, client, _user, _bp, invoice = _world(db)
    client.invoice_delivery_method = "paper"
    db.session.commit()
    res = add_draft_custom_line(
        company.id, invoice.id, description="Attente", line_total=5.0
    )
    assert res.success is True, res.error
    state = _financial_state(db, invoice.id)
    descriptions = [ln[1] for ln in state["lines"]]
    assert "Frais de facture papier" in descriptions
    assert state["subtotal"] == "88.00"  # HT : 80 + 3 (papier) + 5 (attente)


# ───────────────────── K : taux TVA historique figé dans le rendu ─────────────────────


@pytest.mark.integration
def test_historical_vat_rate_is_rendered_from_snapshot(db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_vat")
    db.session.add(
        CompanyBillingSettings(
            company_id=world.company_id,
            vat_applicable=True,
            vat_rate=Decimal("8.10"),
            iban="CH6509000000152631289",
            payment_terms_days=30,
        )
    )
    db.session.commit()
    settings = CompanyBillingSettings.query.filter_by(
        company_id=world.company_id
    ).first()
    issued = Invoice.query.options(selectinload(Invoice.lines)).get(world.invoice_id)
    before = _financial_state(db, issued.id)

    # Facture émise → snapshot (7.7) et non paramètres courants (8.1).
    assert _vat_status_text_for_invoice(issued, settings) == "TVA 7.7% incluse"

    # Brouillon → paramètres courants (comportement existant).
    _c2, _cl2, _u2, _bp2, draft = _world(db)
    draft_settings = SimpleNamespace(
        vat_applicable=True, vat_rate=Decimal("8.10"), vat_number=None
    )
    assert "8.1" in _vat_status_text_for_invoice(draft, draft_settings)

    # Snapshot « non assujetti » figé : aucune mention même si l'entreprise l'est devenue.
    non_vat = SimpleNamespace(
        id=0,
        status=InvoiceStatus.SENT,
        meta={"vat": {"applicable": False, "default_rate": 8.1}},
        lines=[],
    )
    assert _vat_status_text_for_invoice(non_vat, settings) == ""

    # Numéro TVA figé prioritaire sur le taux.
    numbered = SimpleNamespace(
        id=0,
        status=InvoiceStatus.PAID,
        meta={"vat": {"applicable": True, "default_rate": 7.7, "number": "CHE-1 TVA"}},
        lines=[],
    )
    assert _vat_status_text_for_invoice(numbered, settings) == "N° TVA : CHE-1 TVA"

    # Facture émise sans snapshot → taux figé des lignes, jamais le taux courant.
    legacy = SimpleNamespace(
        id=0,
        status=InvoiceStatus.SENT,
        meta={},
        lines=[SimpleNamespace(vat_rate=Decimal("7.70"))],
    )
    assert _vat_status_text_for_invoice(legacy, settings) == "TVA 7.7% incluse"

    # Pipeline réel : le rendu (lecture seule) de la facture émise passe par le
    # snapshot et n'écrit rien en base.
    seen: list[str] = []
    original = _vat_status_text_for_invoice

    def _spy(invoice, billing_settings):
        value = original(invoice, billing_settings)
        seen.append(value)
        return value

    with patch("services.documents.pdf._vat_status_text_for_invoice", side_effect=_spy):
        pdf_bytes, _rows = PDFService()._create_invoice_pdf_content(issued)
    assert pdf_bytes[:4] == b"%PDF"
    assert seen
    assert all(v == "TVA 7.7% incluse" for v in seen)
    db.session.commit()
    assert _financial_state(db, issued.id) == before


# ───────────── L : facture émise sans PDF original → signalement, pas de fabrication ─────────────


@pytest.mark.integration
def test_issued_invoice_without_original_pdf_is_flagged_not_fabricated(db, uploads_tmp):
    company, _client, _user, _bp, invoice = _world(db)
    company.is_approved = True
    invoice = _issue(db, invoice, InvoiceStatus.SENT, pdf_url=None)
    before = _financial_state(db, invoice.id)
    listing_before = _uploads_listing(uploads_tmp)

    # Génération directe : erreur explicite, aucun fichier.
    with pytest.raises(IssuedInvoiceImmutableError) as exc:
        PDFService().generate_invoice_pdf(invoice, force_regenerate=True)
    assert exc.value.code == ERROR_CODE_ISSUED_INVOICE_PDF_MISSING
    assert exc.value.status_code == 409

    # Régénération forcée : refus (facture émise), aucun fichier.
    with patch(_REGEN_DELETE) as del_old:
        regen = force_regenerate_invoice_pdf(
            company_id=company.id, invoice_id=invoice.id
        )
    _assert_refusal(regen)
    del_old.assert_not_called()

    # Envoi e-mail : refus « PDF original absent », aucune notification.
    with (
        patch(_SEND_BREVO_CLASS) as brevo_cls,
        patch.object(PDFService, "_create_invoice_pdf_content") as renderer,
    ):
        use_case = SendInvoiceByEmailUseCase()
        sent = use_case.execute(
            SendInvoiceByEmailInput(
                invoice_id=invoice.id,
                recipient_email="destinataire@test.example",
                force_regenerate_pdf=True,
            )
        )
    assert sent.success is False
    assert sent.status_code == 409
    assert sent.error_payload["error_code"] == ERROR_CODE_ISSUED_INVOICE_PDF_MISSING
    brevo_cls.return_value.send_invoice_email.assert_not_called()
    brevo_cls.return_value.send_transactional.assert_not_called()
    renderer.assert_not_called()

    after = _financial_state(db, invoice.id)
    assert after == before
    assert after["pdf_url"] is None
    assert after["status"] == "sent"
    assert _uploads_listing(uploads_tmp) == listing_before


@pytest.mark.integration
def test_issued_invoice_with_missing_file_on_disk_is_not_regenerated_on_send(
    db, uploads_tmp
):
    """pdf_url présent mais fichier absent : pas de régénération, pas d'envoi."""
    world = _issued_world(db, uploads_tmp, InvoiceStatus.OVERDUE, suffix="_lost")
    world.pdf_path.unlink()
    before = _financial_state(db, world.invoice_id)
    listing_before = _uploads_listing(uploads_tmp)
    db.session.add(
        CompanyBillingSettings(
            company_id=world.company_id,
            smtp_username="factures@test.example",
            from_name="ATMR Test",
            domain_verified=True,
            payment_terms_days=30,
        )
    )
    db.session.commit()

    with (
        patch(_SEND_BREVO_CLASS) as brevo_cls,
        patch.object(PDFService, "_create_invoice_pdf_content") as renderer,
    ):
        sent = SendInvoiceByEmailUseCase().execute(
            SendInvoiceByEmailInput(
                invoice_id=world.invoice_id,
                recipient_email="destinataire@test.example",
                force_regenerate_pdf=True,
            )
        )
    assert sent.success is False
    assert sent.status_code == 409
    assert sent.error_payload["error_code"] == ERROR_CODE_ISSUED_INVOICE_PDF_MISSING
    renderer.assert_not_called()
    brevo_cls.return_value.send_invoice_email.assert_not_called()
    brevo_cls.return_value.send_transactional.assert_not_called()
    assert _financial_state(db, world.invoice_id) == before
    assert _uploads_listing(uploads_tmp) == listing_before


# ───────────── erreur entre écriture temporaire et validation ─────────────


@pytest.mark.integration
def test_write_failure_on_draft_keeps_previous_pdf_and_file(db, uploads_tmp):
    """Brouillon : échec d'écriture → ancienne référence et ancien fichier conservés."""
    company, _client, _user, _bp, invoice = _world(db)
    old_url, old_path, old_digest = _write_fake_pdf(
        uploads_tmp, f"invoice_{invoice.invoice_number}_20260101_120000.pdf"
    )
    invoice.pdf_url = old_url
    db.session.commit()

    with (
        patch.object(
            PDFService, "_create_invoice_pdf_content", return_value=(b"%PDF-new", 1)
        ),
        patch(
            "shared.upload_write.write_upload_bytes",
            side_effect=OSError("disque plein pendant l'écriture temporaire"),
        ),
    ):
        result = force_regenerate_invoice_pdf(
            company_id=company.id, invoice_id=invoice.id
        )
    assert result.ok is False
    assert result.status_code == 500
    reloaded = Invoice.query.get(invoice.id)
    assert reloaded.pdf_url == old_url
    assert old_path.exists()
    assert _sha256(old_path) == old_digest


@pytest.mark.integration
def test_write_is_never_attempted_for_issued_invoice(db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.PARTIALLY_PAID, suffix="_w")
    before = _financial_state(db, world.invoice_id)
    with patch("shared.upload_write.write_upload_bytes") as writer:
        result = force_regenerate_invoice_pdf(
            company_id=world.company_id, invoice_id=world.invoice_id
        )
    _assert_refusal(result)
    writer.assert_not_called()
    _assert_untouched(db, world, before)


# ───────────── réponse backend perdue / opération répétée ─────────────


@pytest.mark.integration
def test_repeated_regeneration_attempts_are_idempotent_refusals(db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_rep")
    before = _financial_state(db, world.invoice_id)
    with patch(_REGEN_UC) as uc_cls, patch(_REGEN_DELETE) as del_old:
        results = [
            force_regenerate_invoice_pdf(
                company_id=world.company_id, invoice_id=world.invoice_id
            )
            for _ in range(3)
        ]
    for res in results:
        _assert_refusal(res)
    uc_cls.return_value.execute.assert_not_called()
    del_old.assert_not_called()
    _assert_untouched(db, world, before)


# ───────────────────────────── concurrence ─────────────────────────────


def _run_in_thread(app, fn, results: list, key: str) -> threading.Thread:
    def _target():
        with app.app_context():
            from ext import db as thread_db

            try:
                results.append((key, fn()))
            except Exception as exc:  # pragma: no cover - diagnostic
                results.append((key, exc))
            finally:
                thread_db.session.remove()

    return threading.Thread(target=_target, name=f"etape22-{key}")


@pytest.mark.integration
def test_two_concurrent_regenerations_of_issued_invoice_are_both_refused(
    app, db, uploads_tmp
):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_cc")
    before = _financial_state(db, world.invoice_id)
    cid, iid = world.company_id, world.invoice_id
    db.session.remove()

    outcomes: list = []
    with patch(_REGEN_UC) as uc_cls, patch(_REGEN_DELETE) as del_old:
        threads = [
            _run_in_thread(
                app,
                lambda: force_regenerate_invoice_pdf(company_id=cid, invoice_id=iid),
                outcomes,
                f"t{i}",
            )
            for i in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
    assert len(outcomes) == 2
    for _key, res in outcomes:
        assert not isinstance(res, Exception), res
        _assert_refusal(res)
    uc_cls.return_value.execute.assert_not_called()
    del_old.assert_not_called()
    _assert_untouched(db, world, before)


@pytest.mark.integration
def test_regeneration_during_status_change_waits_for_lock_then_refuses(
    app, db, uploads_tmp
):
    """Brouillon en cours d'émission (UPDATE non commité) : la régénération attend
    le verrou de ligne puis voit le statut émis → refus, PDF original intact."""
    company, _client, _user, _bp, invoice = _world(db)
    pdf_url, pdf_path, digest = _write_fake_pdf(
        uploads_tmp, f"invoice_{invoice.invoice_number}_20260101_120000.pdf"
    )
    invoice.pdf_url = pdf_url
    db.session.commit()
    cid, iid = company.id, invoice.id
    db.session.remove()

    lock_taken = threading.Event()
    release = threading.Event()
    timings: dict[str, float] = {}
    outcomes: list = []

    def _status_change_holder():
        with app.app_context():
            from ext import db as thread_db

            with thread_db.engine.connect() as conn:
                tx = conn.begin()
                conn.execute(
                    text("UPDATE invoices SET status = 'SENT' WHERE id = :id"),
                    {"id": iid},
                )
                lock_taken.set()
                release.wait(timeout=30)
                time.sleep(1.0)
                timings["committed_at"] = time.monotonic()
                tx.commit()

    def _regenerate():
        timings["started_at"] = time.monotonic()
        res = force_regenerate_invoice_pdf(company_id=cid, invoice_id=iid)
        timings["finished_at"] = time.monotonic()
        return res

    holder = threading.Thread(target=_status_change_holder, name="etape22-holder")
    with patch(_REGEN_UC) as uc_cls, patch(_REGEN_DELETE) as del_old:
        holder.start()
        assert lock_taken.wait(timeout=30)
        regen_thread = _run_in_thread(app, _regenerate, outcomes, "regen")
        regen_thread.start()
        time.sleep(0.3)
        release.set()
        regen_thread.join(timeout=60)
        holder.join(timeout=60)

    assert len(outcomes) == 1
    _key, res = outcomes[0]
    assert not isinstance(res, Exception), res
    _assert_refusal(res)
    # La régénération n'a pu lire le statut qu'après le commit du changement.
    assert timings["finished_at"] >= timings["committed_at"]
    uc_cls.return_value.execute.assert_not_called()
    del_old.assert_not_called()
    db.session.expire_all()
    reloaded = Invoice.query.get(iid)
    assert reloaded.status == InvoiceStatus.SENT
    assert reloaded.pdf_url == pdf_url
    assert pdf_path.exists()
    assert _sha256(pdf_path) == digest


# ───────────── droits : utilisateur non autorisé / autre entreprise (HTTP) ─────────────


@pytest.mark.integration
def test_http_regenerate_rights_and_409_for_issued(client, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_http")
    other = _issued_world(db, uploads_tmp, InvoiceStatus.DRAFT, suffix="_other")
    before = _financial_state(db, world.invoice_id)
    url = (
        f"/api/v1/invoices/companies/{world.company_id}/invoices/"
        f"{world.invoice_id}/regenerate-pdf"
    )
    owner = User.query.get(world.owner_user_id)
    other_owner = User.query.get(other.owner_user_id)

    with patch(_REGEN_UC) as uc_cls, patch(_REGEN_DELETE) as del_old:
        # Sans jeton.
        anonymous = client.post(url)
        assert anonymous.status_code == 401
        # Utilisateur au rôle client (non autorisé).
        client_headers = _company_headers(client, world.client_user, world.company_id)
        forbidden = client.post(url, headers=client_headers)
        assert forbidden.status_code == 403
        # Autre entreprise sur l'URL de l'entreprise propriétaire.
        other_headers = _company_headers(client, other_owner, other.company_id)
        cross = client.post(url, headers=other_headers)
        assert cross.status_code == 403
        # Autre entreprise via sa propre URL avec l'ID de la facture étrangère.
        cross_own_url = client.post(
            f"/api/v1/invoices/companies/{other.company_id}/invoices/"
            f"{world.invoice_id}/regenerate-pdf",
            headers=other_headers,
        )
        assert cross_own_url.status_code == 404
        # Entreprise propriétaire : facture émise → 409 normalisé.
        owner_headers = _company_headers(client, owner, world.company_id)
        refused = client.post(url, headers=owner_headers)
        assert refused.status_code == 409
        body = refused.get_json()
        assert body["error_code"] == ERROR_CODE_ISSUED_INVOICE_IMMUTABLE
        assert body["details"]["administrative_procedure_required"] is True
        # Le mode asynchrone ne contourne pas le refus (aucune tâche créée).
        refused_async = client.post(url + "?async=1", headers=owner_headers)
        assert refused_async.status_code == 409

    uc_cls.return_value.execute.assert_not_called()
    del_old.assert_not_called()
    _assert_untouched(db, world, before)


# ───────────────────────────── rappel ─────────────────────────────


@pytest.mark.integration
def test_reminder_generation_leaves_original_invoice_pdf_intact(db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.OVERDUE, suffix="_rem")
    before = _financial_state(db, world.invoice_id)
    with patch.object(
        PDFService,
        "_create_invoice_pdf_content",
        return_value=(b"%PDF-1.4 rappel factice", 1),
    ):
        reminder_url = PDFService().generate_reminder_pdf(world.invoice, 1)
    assert reminder_url
    assert "reminder_" in reminder_url
    assert reminder_url != world.pdf_url
    db.session.commit()
    after = _financial_state(db, world.invoice_id)
    assert after == before
    assert after["pdf_url"] == world.pdf_url
    assert world.pdf_path.exists()
    assert _sha256(world.pdf_path) == world.sha256


# ───────────────────────── H : facture partenaire ─────────────────────────


def _partner_state(db, partner_invoice_id: int) -> dict:
    db.session.expire_all()
    pi = PartnerInvoice.query.get(partner_invoice_id)
    assert pi is not None
    return {
        "status": pi.status,
        "pdf_url": pi.pdf_url,
        "subtotal": _dec(pi.subtotal_amount),
        "vat": _dec(pi.vat_amount),
        "total": _dec(pi.total_amount),
        "lines": sorted(
            (
                ln.id,
                ln.description,
                _dec(getattr(ln, "amount", None)),
                _dec(getattr(ln, "vat_rate", None)),
            )
            for ln in pi.lines
        ),
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    "status",
    [
        PartnerInvoiceStatus.SENT,
        PartnerInvoiceStatus.PARTIALLY_PAID,
        PartnerInvoiceStatus.PAID,
        PartnerInvoiceStatus.OVERDUE,
        PartnerInvoiceStatus.CANCELLED,
    ],
)
def test_partner_invoice_issued_is_immutable(db, monkeypatch, uploads_tmp, status):
    from tests.services.test_partnership_invoices import _svc
    from tests.services.test_partnership_invoices import _world as _partner_world

    pworld = _partner_world()
    service = _svc(monkeypatch)
    partner_invoice = service.generate_monthly_invoice(
        pworld.partnership.id, pworld.year, pworld.month, pworld.executing.id
    )
    pdf_url, pdf_path, digest = _write_fake_pdf(
        uploads_tmp, f"partner_{partner_invoice.invoice_number}.pdf"
    )
    partner_invoice.pdf_url = pdf_url
    partner_invoice.status = status
    db.session.commit()
    assert partner_invoice_is_issued(partner_invoice) is True
    before = _partner_state(db, partner_invoice.id)
    cid, pid = pworld.executing.id, partner_invoice.id

    with (
        patch(
            "application.invoices.force_regenerate_partner_invoice_pdf."
            "ensure_partner_invoice_lines"
        ) as ensure_lines,
        patch(
            "application.invoices.force_regenerate_partner_invoice_pdf."
            "PartnerInvoiceService._generate_invoice_pdf"
        ) as gen_pdf,
        patch(
            "application.invoices.force_regenerate_partner_invoice_pdf."
            "_delete_replaced_invoice_pdf"
        ) as del_old,
    ):
        regen = force_regenerate_partner_invoice_pdf(
            company_id=cid, partner_invoice_id=pid
        )
    assert regen.ok is False
    assert regen.status_code == 409
    assert regen.error["error_code"] == ERROR_CODE_ISSUED_INVOICE_IMMUTABLE
    assert regen.error["details"]["catalog"] == "partner_invoice"
    ensure_lines.assert_not_called()
    gen_pdf.assert_not_called()
    del_old.assert_not_called()

    edit = update_partner_invoice_draft(
        company_id=cid,
        partner_invoice_id=pid,
        payload={
            "command": "add_custom_line",
            "description": "Attente",
            "line_total": "10.00",
            "qty": 1,
        },
    )
    assert edit.ok is False
    assert edit.status_code == 409
    assert edit.error["error_code"] == ERROR_CODE_ISSUED_INVOICE_IMMUTABLE

    with pytest.raises(IssuedInvoiceImmutableError):
        PartnerInvoiceService().regenerate_pdf(pid)

    assert _partner_state(db, pid) == before
    assert pdf_path.exists()
    assert _sha256(pdf_path) == digest


@pytest.mark.integration
def test_partner_draft_regeneration_still_works(db, monkeypatch, uploads_tmp):
    """Non-régression : brouillon partenaire régénérable (contrat existant)."""
    from tests.services.test_partnership_invoices import _svc
    from tests.services.test_partnership_invoices import _world as _partner_world

    pworld = _partner_world()
    service = _svc(monkeypatch)
    partner_invoice = service.generate_monthly_invoice(
        pworld.partnership.id, pworld.year, pworld.month, pworld.executing.id
    )
    assert partner_invoice.status == PartnerInvoiceStatus.DRAFT
    with patch(
        "application.invoices.force_regenerate_partner_invoice_pdf."
        "PartnerInvoiceService._generate_invoice_pdf",
        return_value="/uploads/invoices/partner-new.pdf",
    ):
        regen = force_regenerate_partner_invoice_pdf(
            company_id=pworld.executing.id, partner_invoice_id=partner_invoice.id
        )
    assert regen.ok is True
    assert regen.pdf_url == "/uploads/invoices/partner-new.pdf"


# ───────────── E/F : taux de ligne et totaux restent ceux d'origine (contrôle direct) ─────────────


@pytest.mark.integration
def test_issued_invoice_line_vat_rate_is_not_rewritten(db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_rate")
    line = InvoiceLine.query.get(world.line_id)
    line.vat_rate = Decimal("7.70")
    line.vat_amount = Decimal("6.15")
    line.total_with_vat = Decimal("86.15")
    line.type = InvoiceLineType.CUSTOM
    inv = Invoice.query.get(world.invoice_id)
    inv.vat_total_amount = Decimal("6.15")
    inv.total_amount = Decimal("86.15")
    inv.balance_due = Decimal("86.15")
    db.session.commit()
    before = _financial_state(db, world.invoice_id)

    with patch(_REGEN_UC), patch(_REGEN_DELETE):
        force_regenerate_invoice_pdf(
            company_id=world.company_id, invoice_id=world.invoice_id
        )
    PDFService().generate_invoice_pdf(Invoice.query.get(world.invoice_id))

    after = _financial_state(db, world.invoice_id)
    assert after == before
    assert after["lines"][0][5] == "7.70"
    assert after["vat_total"] == "6.15"
    assert after["total"] == "86.15"
