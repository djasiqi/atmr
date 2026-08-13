"""Couverture de ``SendPartnerInvoiceByEmailUseCase``."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from application.invoices.send_partner_invoice_by_email import (
    SendPartnerInvoiceByEmailInput,
    SendPartnerInvoiceByEmailUseCase,
)
from models.partner_invoice import PartnerInvoiceStatus
from services.email.base import EmailResult

_MOD = "application.invoices.send_partner_invoice_by_email"


@pytest.fixture(autouse=True)
def _app_ctx(app):
    with (
        app.app_context(),
        patch(f"{_MOD}.BrevoEmailProvider"),
        patch(f"{_MOD}.PartnerInvoiceService"),
    ):
        yield


def _companies(*, owner_is_executing: bool = True):
    executing = SimpleNamespace(id=10, name="Exec SA")
    billed = SimpleNamespace(
        id=20,
        name="Billed SA",
        billing_email="billed@test.ch",
        contact_email="contact@test.ch",
    )
    partnership = SimpleNamespace(
        owner_company_id=10 if owner_is_executing else 20,
        partner_company=billed if owner_is_executing else executing,
        owner_company=executing if owner_is_executing else billed,
    )
    return executing, billed, partnership


def _invoice(partnership, **kwargs):
    data = {
        "id": 1,
        "invoice_number": "PI-001",
        "executing_company_id": 10,
        "pdf_url": "/uploads/partners/PI-001.pdf",
        "status": PartnerInvoiceStatus.DRAFT,
        "period_month": 1,
        "period_year": 2026,
        "total_amount": Decimal("12.50"),
        "due_date": datetime(2026, 3, 1),
        "partnership": partnership,
        "sent_at": None,
    }
    data.update(kwargs)
    return SimpleNamespace(**data)


def _settings(**kwargs):
    data = {
        "smtp_username": "noreply@exec.ch",
        "from_name": "Exec SA",
        "domain_verified": True,
        "invoice_message_template": None,
    }
    data.update(kwargs)
    return SimpleNamespace(**data)


def _input(**kwargs):
    data = {"partner_invoice_id": 1, "company_id": 10}
    data.update(kwargs)
    return SendPartnerInvoiceByEmailInput(**data)


def _run(
    *,
    invoice,
    executing,
    settings,
    input_data=None,
    send_result=None,
    path_exists=True,
    path_open_error=False,
    inject_logo=None,
    regenerate_url=None,
    regenerate_error=None,
    extra_env=None,
):
    input_data = input_data or _input()
    send_result = send_result or EmailResult(success=True, message_id="mid-1")
    uc = SendPartnerInvoiceByEmailUseCase()
    if regenerate_error:
        uc.partner_invoice_service.regenerate_pdf.side_effect = regenerate_error
    elif regenerate_url is not None:
        uc.partner_invoice_service.regenerate_pdf.return_value = regenerate_url

    path_inst = MagicMock()
    path_inst.exists.return_value = path_exists
    if path_open_error:
        path_inst.open.side_effect = OSError("lock")
    else:
        path_inst.open.return_value.__enter__.return_value.read.return_value = b"PDF"

    env = extra_env or {}
    with (
        patch(f"{_MOD}.PartnerInvoice.query") as inv_q,
        patch(f"{_MOD}.Company.query") as co_q,
        patch(f"{_MOD}.CompanyBillingSettings.query") as set_q,
        patch(f"{_MOD}.db.session"),
        patch(f"{_MOD}.Path", return_value=path_inst),
        patch(
            f"{_MOD}.inject_signature_into_html",
            side_effect=lambda html, **_kw: (html, inject_logo),
        ),
        patch.object(
            uc.brevo_provider, "send_invoice_email", return_value=send_result
        ) as send_email,
        patch.dict("os.environ", env, clear=False),
    ):
        inv_q.get.return_value = invoice
        co_q.get.return_value = executing
        set_q.filter_by.return_value.first.return_value = settings
        result = uc.execute(input_data)
        return result, send_email, uc


def test_erreurs_validation():
    executing, _billed, partnership = _companies()
    uc = SendPartnerInvoiceByEmailUseCase()

    with patch(f"{_MOD}.PartnerInvoice.query") as inv_q:
        inv_q.get.return_value = None
        r = uc.execute(_input(partner_invoice_id=9))
        assert r.status_code == 404

    inv = _invoice(partnership, executing_company_id=99)
    with patch(f"{_MOD}.PartnerInvoice.query") as inv_q:
        inv_q.get.return_value = inv
        r = uc.execute(_input())
        assert r.status_code == 403

    inv = _invoice(None)
    with patch(f"{_MOD}.PartnerInvoice.query") as inv_q:
        inv_q.get.return_value = inv
        r = uc.execute(_input())
        assert r.status_code == 404
        assert "Partenariat" in (r.error or "")

    inv = _invoice(partnership)
    with (
        patch(f"{_MOD}.PartnerInvoice.query") as inv_q,
        patch(f"{_MOD}.Company.query") as co_q,
    ):
        inv_q.get.return_value = inv
        co_q.get.return_value = None
        r = uc.execute(_input())
        assert r.status_code == 404
        assert "exécutante" in (r.error or "")

    partnership.partner_company = None
    inv = _invoice(partnership)
    with (
        patch(f"{_MOD}.PartnerInvoice.query") as inv_q,
        patch(f"{_MOD}.Company.query") as co_q,
    ):
        inv_q.get.return_value = inv
        co_q.get.return_value = executing
        r = uc.execute(_input())
        assert r.status_code == 404
        assert "destinataire" in (r.error or "")


def test_email_manquant_et_owner_inverse():
    executing, billed, partnership = _companies(owner_is_executing=False)
    billed.billing_email = None
    billed.contact_email = None
    billed.name = "Sans Mail"
    inv = _invoice(partnership)
    with (
        patch(f"{_MOD}.PartnerInvoice.query") as inv_q,
        patch(f"{_MOD}.Company.query") as co_q,
    ):
        inv_q.get.return_value = inv
        co_q.get.return_value = executing
        r = SendPartnerInvoiceByEmailUseCase().execute(_input())
        assert r.status_code == 400
        assert "Sans Mail" in (r.error or "")


def test_billing_settings_incomplets():
    executing, _billed, partnership = _companies()
    inv = _invoice(partnership)

    r, _, _ = _run(invoice=inv, executing=executing, settings=None)
    assert r.status_code == 400
    assert "Paramètres de facturation" in (r.error or "")

    r, _, _ = _run(
        invoice=inv,
        executing=executing,
        settings=_settings(smtp_username=None),
    )
    assert r.status_code == 400
    assert "Adresse email d'envoi" in (r.error or "")

    r, _, _ = _run(
        invoice=inv,
        executing=executing,
        settings=_settings(domain_verified=False),
    )
    assert r.status_code == 403
    assert "n'est pas vérifié" in (r.error or "")


def test_succes_pdf_uploads_et_draft_sent():
    executing, _billed, partnership = _companies()
    inv = _invoice(partnership)
    r, send_email, _uc = _run(invoice=inv, executing=executing, settings=_settings())
    assert r.success is True
    assert r.recipient == "billed@test.ch"
    assert inv.status == PartnerInvoiceStatus.SENT
    assert inv.sent_at is not None
    send_email.assert_called_once()
    assert (
        send_email.call_args.kwargs["attachments"][0]["filename"]
        == "facture_PI-001.pdf"
    )
    assert "janvier 2026" in send_email.call_args.kwargs["html_content"]


def test_pdf_url_variants_et_regeneration():
    executing, _billed, partnership = _companies()

    inv = _invoice(partnership, pdf_url="https://cdn.example/uploads/x/a.pdf")
    r, _, _ = _run(invoice=inv, executing=executing, settings=_settings())
    assert r.success is True

    inv = _invoice(partnership, pdf_url="s3://bucket/nope.pdf")
    r, _, _ = _run(invoice=inv, executing=executing, settings=_settings())
    assert r.success is True

    inv = _invoice(partnership, pdf_url=None, status=PartnerInvoiceStatus.SENT)
    r, _, uc = _run(
        invoice=inv,
        executing=executing,
        settings=_settings(),
        input_data=_input(force_regenerate_pdf=True, recipient_email="x@y.ch"),
        regenerate_url="/uploads/partners/new.pdf",
        path_exists=False,
    )
    uc.partner_invoice_service.regenerate_pdf.assert_called_once_with(1)
    assert inv.status == PartnerInvoiceStatus.SENT

    inv = _invoice(partnership, pdf_url=None)
    r, _, _ = _run(
        invoice=inv,
        executing=executing,
        settings=_settings(),
        regenerate_error=RuntimeError("pdf fail"),
        path_exists=False,
    )
    assert r.success is True


def test_lecture_pdf_erreur_et_template():
    executing, billed, partnership = _companies()
    billed.name = None
    executing.name = "Exec SA"
    inv = _invoice(
        partnership,
        total_amount=None,
        due_date=None,
        invoice_number=None,
    )
    settings = _settings(
        from_name=None,
        invoice_message_template=(
            "Hi {partner_name} {recipient_name} {invoice_number} {period} "
            "{amount} {due_date}\nfin"
        ),
    )
    r, send_email, _ = _run(
        invoice=inv,
        executing=executing,
        settings=settings,
        path_open_error=True,
        extra_env={"EMAIL_SIGNATURE_DEBUG": "1", "EMAIL_PROVIDER_MODE": "smtp"},
    )
    assert r.success is True
    html = send_email.call_args.kwargs["html_content"]
    assert "Partenaire" in html
    assert "0.00" in html
    assert "À définir" in html
    assert "<br>" in html

    inv_default = _invoice(partnership, due_date=None)
    r, send_email, _ = _run(
        invoice=inv_default,
        executing=executing,
        settings=_settings(),
        path_exists=False,
    )
    assert "À définir" in send_email.call_args.kwargs["html_content"]


def test_logo_inline_et_brevo_echec():
    executing, _billed, partnership = _companies()
    inv = _invoice(partnership)

    r, send_email, _ = _run(
        invoice=inv,
        executing=executing,
        settings=_settings(),
        inject_logo={"bytes": b"", "cid": "company_logo"},
    )
    assert r.success is True
    assert len(send_email.call_args.kwargs["attachments"]) == 1

    r, send_email, _ = _run(
        invoice=inv,
        executing=executing,
        settings=_settings(),
        inject_logo={"bytes": b"x", "cid": "other"},
    )
    assert len(send_email.call_args.kwargs["attachments"]) == 1

    r, send_email, _ = _run(
        invoice=inv,
        executing=executing,
        settings=_settings(),
        inject_logo={
            "bytes": b"PNG",
            "cid": "company_logo",
            "filename": "logo.png",
            "mime_type": "image/png",
        },
    )
    assert len(send_email.call_args.kwargs["attachments"]) == 2

    r, _, _ = _run(
        invoice=inv,
        executing=executing,
        settings=_settings(),
        send_result=EmailResult(success=False, error="api"),
    )
    assert r.status_code == 500
    assert "Erreur Brevo" in (r.error or "")


def test_exception_globale():
    with (
        patch(f"{_MOD}.PartnerInvoice.query") as inv_q,
        patch(f"{_MOD}.db.session") as session,
    ):
        inv_q.get.side_effect = RuntimeError("boom")
        r = SendPartnerInvoiceByEmailUseCase().execute(_input())
        assert r.status_code == 500
        assert "inattendue" in (r.error or "")
        session.rollback.assert_called_once()
