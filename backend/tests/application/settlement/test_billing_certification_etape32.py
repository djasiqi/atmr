"""Corrections étape 32. Base jetable, aucun paiement ni émission réelle."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import func, select

from ext import db
from models.enums import (
    BookingStatus,
    InvoiceBillingStrategy,
    InvoiceLineType,
    InvoiceStatus,
    PartnershipStatus,
    PaymentStatus,
    TransferModel,
    TransferStatus,
)
from models.invoice import Invoice, InvoiceLine
from services.billing.payment_reference_generator import (
    PaymentReferenceGenerator,
    ScorReferenceError,
    compact_scor,
)
from tests.application.settlement.test_billing_certification_etape31 import (
    PARTNER_GENERATE,
    PATIENT,
    PATIENT_TX,
    TEST_IBAN,
    _booking,
    _driver,
    _generate,
    _owner,
    _preview,
    prep,
    world,
)

pytestmark = pytest.mark.integration

PDF_DIR = Path("/tmp/lirie-etape32-pdf")
_ = (prep, world)


def test_scor_refuse_la_troncature() -> None:
    with pytest.raises(ScorReferenceError):
        PaymentReferenceGenerator.generate_scor("PARTNER-PC14FEF-2026-10-0002", company_id=14)
    short = PaymentReferenceGenerator.generate_scor("EM-2026-10-0001", company_id=1)
    assert PaymentReferenceGenerator.validate_scor(short)
    assert len(compact_scor(short)) <= 25


def _pay(booking, amount: float, status: PaymentStatus):
    from models.payment import Payment

    row = Payment(
        amount=amount,
        method="credit_card",
        status=status,
        booking_id=booking.id,
        user_id=booking.user_id,
        client_id=booking.client_id,
    )
    db.session.add(row)
    db.session.commit()
    return row


def _ids(preview: dict) -> set[int]:
    return {int(line["booking_id"]) for line in preview["preview_lines"]}


def test_saferpay_couverture_prix_et_liens(client, world):
    """Le filtre regarde le montant lié à la course, pas seulement le statut."""
    assert not hasattr(PaymentStatus, "REFUNDED")
    full = _booking(world, amount=Decimal("45.00"))
    _pay(full, 45.0, PaymentStatus.COMPLETED)
    pending = _booking(world, amount=Decimal("45.00"))
    _pay(pending, 45.0, PaymentStatus.PENDING)
    refused = _booking(world, amount=Decimal("45.00"))
    _pay(refused, 45.0, PaymentStatus.FAILED)
    partial = _booking(world, amount=Decimal("45.00"))
    _pay(partial, 20.0, PaymentStatus.COMPLETED)
    surplus = _booking(world, amount=Decimal("45.00"))
    _pay(surplus, 60.0, PaymentStatus.COMPLETED)
    other = _booking(world, amount=Decimal("45.00"))
    stranger = _booking(world, amount=Decimal("45.00"), company_id=world["other"].id)
    _pay(stranger, 45.0, PaymentStatus.COMPLETED)

    outbound = _booking(world, amount=Decimal("45.00"))
    retour = _booking(
        world,
        amount=Decimal("45.00"),
        parent_booking_id=outbound.id,
        is_return=True,
        scheduled_time=datetime(2026, 10, 15, 18, 0, tzinfo=UTC),
    )
    _pay(outbound, 90.0, PaymentStatus.COMPLETED)

    preview = _preview(client, world, year=2026, month=10, client_id=world["client"].id)
    selected = _ids(preview)
    text = " ".join(preview.get("warnings") or [])
    assert full.id not in selected
    assert surplus.id not in selected
    assert outbound.id not in selected
    assert retour.id not in selected
    assert pending.id in selected
    assert refused.id in selected
    assert partial.id in selected
    assert other.id in selected
    assert "couverture partielle" in text
    assert "n'est pas affecté" in text
    assert "Aller-retour" in text
    full.pickup_location = "Rue isolée 9, 1000 Recette"
    full.dropoff_location = "Chemin isolé 3, 1000 Recette"
    db.session.commit()
    blocked = _generate(client, world, [full.id])
    assert blocked.status_code in {400, 409, 422}
    assert _lines(full.id) == []

    opened = client.post(PATIENT.format(booking_id=full.id), json={}, headers=_driver(world))
    assert opened.status_code == 409
    assert opened.get_json()["error_code"] == "deja_paye"


def _lines(booking_id: int) -> list[InvoiceLine]:
    return list(
        db.session.scalars(
            select(InvoiceLine).where(InvoiceLine.reservation_id == booking_id)
        ).all()
    )


def test_tva_pdf_et_reference_scor_partenaire(client, world):
    from models.booking_transfer import BookingTransfer
    from models.invoice import CompanyBillingSettings
    from models.partner_invoice import PartnerInvoice
    from models.partnership import Partnership

    partnership = Partnership(
        owner_company_id=world["other"].id,
        partner_company_id=world["company"].id,
        status=PartnershipStatus.ACCEPTED,
        default_transfer_model=TransferModel.SUBCONTRACT,
        payment_terms_days=30,
    )
    db.session.add(partnership)
    db.session.flush()
    prefix = f"Q{world['suffix'][:6].upper()}"
    db.session.add(
        CompanyBillingSettings(
            company_id=world["company"].id,
            invoice_prefix=prefix,
            payment_terms_days=30,
            iban=TEST_IBAN,
            vat_applicable=True,
            vat_rate=Decimal("7.7"),
        )
    )
    ride = _booking(world, amount=Decimal("12.00"))
    transfer = BookingTransfer(
        booking_id=ride.id,
        partnership_id=partnership.id,
        transfer_model=TransferModel.SUBCONTRACT,
        status=TransferStatus.COMPLETED,
        is_validated=True,
        validated_at=ride.scheduled_time,
        client_price=ride.amount,
        partner_cost=Decimal("12.00"),
        currency="CHF",
        owner_company_id=world["other"].id,
        executing_company_id=world["company"].id,
    )
    db.session.add(transfer)
    db.session.commit()
    created = client.post(
        PARTNER_GENERATE.format(company_id=world["company"].id),
        json={
            "partnership_id": partnership.id,
            "period_year": 2026,
            "period_month": 10,
            "transfer_ids": [transfer.id],
        },
        headers=_owner(world),
    )
    assert created.status_code in {200, 201}, created.get_data(as_text=True)[:600]
    invoice = db.session.scalars(
        select(PartnerInvoice)
        .where(PartnerInvoice.executing_company_id == world["company"].id)
        .order_by(PartnerInvoice.id.desc())
    ).first()
    assert invoice is not None
    stored = compact_scor(invoice.payment_reference or "")
    assert PaymentReferenceGenerator.validate_scor(stored)
    assert len(stored) <= 25
    body = re.sub(r"[^A-Z0-9]", "", invoice.invoice_number.upper())
    body = body[len("PARTNER") :] if body.startswith("PARTNER") else body
    assert body in stored
    assert invoice.subtotal_amount == Decimal("12.00")
    assert invoice.total_amount == invoice.subtotal_amount + invoice.vat_amount

    from services.partnerships.invoices_pdf import generate_partner_invoice_pdf_content

    pdf_bytes = generate_partner_invoice_pdf_content(invoice, list(invoice.transfers))
    text = _pdf_text(pdf_bytes)
    compact_text = compact_scor(text)
    assert "en sus" in text
    assert "incluse" not in text
    assert "12.90" in text or "12,90" in text
    assert stored in compact_text
    decoded = _decode_partner_page(pdf_bytes)
    assert compact_scor(decoded) == stored or stored in compact_scor(decoded)
    assert "CH6509000000152631289" in compact_scor(decoded)
    assert _content_bands(_render_page(pdf_bytes, 1)) == 1
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    (PDF_DIR / "partner.pdf").write_bytes(pdf_bytes)
    _write_pages(pdf_bytes, "partner")


def test_pdf_patient_nom_long_sans_page_orpheline(db, world):
    company = world["company"]
    client_row = world["client"]
    long_name = "Bénéficiaire fictif à accents — Hôpital des Érables Nord Extrême"
    invoice = Invoice(
        company_id=company.id,
        client_id=client_row.id,
        billing_strategy=InvoiceBillingStrategy.S1_PATIENT,
        invoice_number=f"SPEC32-{world['suffix']}",
        period_year=2026,
        period_month=10,
        status=InvoiceStatus.DRAFT,
        issued_at=datetime(2026, 10, 27, tzinfo=UTC),
        due_date=datetime(2026, 11, 27, tzinfo=UTC),
        currency="CHF",
        subtotal_amount=Decimal("45.00"),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal("45.00"),
        amount_paid=Decimal("0.00"),
        balance_due=Decimal("45.00"),
    )
    db.session.add(invoice)
    db.session.flush()
    booking = _booking(
        world,
        customer_name=long_name,
        amount=Decimal("45.00"),
    )
    db.session.add(
        InvoiceLine(
            invoice_id=invoice.id,
            reservation_id=booking.id,
            type=InvoiceLineType.RIDE,
            description=f"Transport fictif — {long_name}",
            qty=Decimal("1.00"),
            unit_price=Decimal("45.00"),
            line_total=Decimal("45.00"),
            vat_rate=Decimal("0.00"),
            vat_amount=Decimal("0.00"),
            total_with_vat=Decimal("45.00"),
            adjustment_note="SPÉCIMEN — NON ÉMIS — NE PAS PAYER",
        )
    )
    user = client_row.user
    user.first_name = "Bénéficiaire fictif à accents"
    user.last_name = "Hôpital des Érables Nord Extrême"
    db.session.commit()
    from services.documents.pdf import PDFService

    pdf_bytes, _pages = PDFService()._create_invoice_pdf_content(invoice)
    text = _pdf_text(pdf_bytes)
    assert "Érables Nord Extrême" in text or "ÉRABLES NORD EXTRÊME" in text.upper()
    assert "QR-Bill non disponible" not in text
    assert _page_count(pdf_bytes) == 1
    assert b"NE PAS PAYER" in pdf_bytes or "NE PAS PAYER" in text
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    (PDF_DIR / "patient.pdf").write_bytes(pdf_bytes)
    _write_pages(pdf_bytes, "patient")
    _ = PATIENT_TX


def test_tva_non_applicable_et_incluse_si_deja_dans_le_total():
    from types import SimpleNamespace

    from services.partnerships.invoices_pdf import _partner_vat_header

    added = SimpleNamespace(
        subtotal_amount=Decimal("12.00"),
        vat_amount=Decimal("0.90"),
        total_amount=Decimal("12.90"),
    )
    settings = SimpleNamespace(vat_applicable=True, vat_rate=Decimal("7.7"), vat_number="")
    assert _partner_vat_header(settings, added) == "TVA 7.7% en sus"
    included = SimpleNamespace(
        subtotal_amount=Decimal("12.90"),
        vat_amount=Decimal("0.90"),
        total_amount=Decimal("12.90"),
    )
    assert "incluse" in _partner_vat_header(settings, included)
    off = SimpleNamespace(vat_applicable=False, vat_rate=Decimal("7.7"), vat_number="")
    assert _partner_vat_header(off, added) == "TVA non applicable"
    missing = SimpleNamespace(vat_applicable=True, vat_rate=None, vat_number="")
    assert "non documenté" in _partner_vat_header(missing, added)


def _pdf_text(pdf_bytes: bytes) -> str:
    import io

    from pdfminer.high_level import extract_text

    return extract_text(io.BytesIO(pdf_bytes)) or ""


def _page_count(pdf_bytes: bytes) -> int:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_bytes)
    try:
        return len(document)
    finally:
        document.close()


def _render_page(pdf_bytes: bytes, index: int):
    import pypdfium2 as pdfium
    from PIL import Image

    document = pdfium.PdfDocument(pdf_bytes)
    try:
        page = document[index]
        bitmap = page.render(scale=1.5)
        image = bitmap.to_pil()
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return image.convert("L")
    finally:
        document.close()


def _write_pages(pdf_bytes: bytes, stem: str) -> None:
    count = _page_count(pdf_bytes)
    for index in range(count):
        _render_page(pdf_bytes, index).save(PDF_DIR / f"{stem}-page-{index + 1}.png")


def _content_bands(image) -> int:
    width, height = image.size
    pixels = image.load()
    bands = 0
    active = False
    gap = 0
    for y in range(height):
        dark = sum(1 for x in range(0, width, 2) if pixels[x, y] < 250)
        if dark > 8:
            if not active:
                bands += 1
                active = True
            gap = 0
        elif active:
            gap += 1
            if gap > 40:
                active = False
    return bands


def _decode_partner_page(pdf_bytes: bytes) -> str:
    import zxingcpp

    image = _render_page(pdf_bytes, _page_count(pdf_bytes) - 1)
    found = zxingcpp.read_barcodes(image)
    assert found, "QR partenaire non décodé"
    return found[0].text


def test_chauffeur_et_entreprise_simultanes(app, world):
    """Deux sessions HTTP sur la même course : un seul engagement, pas de 500."""
    import threading

    booking = _booking(world, status=BookingStatus.COMPLETED, amount=Decimal("45.00"))
    opened = app.test_client().post(
        PATIENT.format(booking_id=booking.id), json={}, headers=_driver(world)
    )
    assert opened.status_code == 200, opened.get_data(as_text=True)[:400]
    basis = app.test_client().post(
        PATIENT_TX.format(booking_id=booking.id),
        json={
            "action": "declare_recipe_fixture",
            "idempotency_key": "fixture-32",
            "recipe_fixture": "fictif-identifie",
        },
        headers=_driver(world),
    )
    assert basis.status_code == 200, basis.get_data(as_text=True)[:400]
    booking_id = int(booking.id)
    client_id = int(world["client"].id)
    company_id = int(world["company"].id)
    driver_headers = _driver(world)
    owner_headers = _owner(world)
    db.session.commit()
    db.session.remove()
    barrier = threading.Barrier(2)
    results: list[tuple[int, dict]] = []

    def _run(method_headers_body):
        local = app.test_client()
        barrier.wait(timeout=8)
        response = local.post(
            method_headers_body[0],
            json=method_headers_body[1],
            headers=method_headers_body[2],
        )
        results.append((response.status_code, response.get_json(silent=True) or {}))

    company_body = {
        "client_id": client_id,
        "period_year": 2026,
        "period_month": 10,
        "reservation_ids": [booking_id],
    }
    driver_body = {"action": "prepare_draft", "idempotency_key": "course-32"}
    threads = [
        threading.Thread(
            target=_run,
            args=((PATIENT_TX.format(booking_id=booking_id), driver_body, driver_headers),),
        ),
        threading.Thread(
            target=_run,
            args=(
                (
                    f"/api/v1/invoices/companies/{company_id}/invoices/generate",
                    company_body,
                    owner_headers,
                ),
            ),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert [thread.is_alive() for thread in threads] == [False, False]
    assert all(code not in {500, 0} for code, _payload in results)
    with app.app_context():
        lines = list(
            db.session.scalars(
                select(InvoiceLine).where(InvoiceLine.reservation_id == booking_id)
            ).all()
        )
    assert len(lines) == 1
    invoices = {line.invoice_id for line in lines}
    assert len(invoices) == 1


def test_reprise_apres_redemarrage_backend(app, world):
    """Le brouillon survit à l'arrêt puis au redémarrage du processus."""
    import json
    import os
    import subprocess
    import sys
    import time
    import urllib.error
    import urllib.request

    booking = _booking(world, status=BookingStatus.COMPLETED, amount=Decimal("45.00"))
    booking_id = int(booking.id)
    previous_secret = app.config.get("JWT_SECRET_KEY")
    app.config["JWT_SECRET_KEY"] = "test-jwt-key"
    headers = _driver(world)
    app.config["JWT_SECRET_KEY"] = previous_secret
    db.session.commit()
    db.session.remove()
    port = 5073
    script = (
        "from app import create_app\n"
        "application = create_app('testing')\n"
        f"application.run(host='127.0.0.1', port={port}, use_reloader=False, threaded=True)\n"
    )

    def _start():
        with Path("/tmp/lirie-etape32-recipe.log").open("ab") as log:
            return subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )

    def _wait(process):
        deadline = time.time() + 90
        while time.time() < deadline:
            if process.poll() is not None:
                pytest.fail("Le processus backend s'est arrêté avant d'écouter.")
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/v1/auth/csrf-token", timeout=2
                )
                return
            except urllib.error.HTTPError:
                return
            except Exception:
                time.sleep(1)
        pytest.fail("Le processus backend n'a pas démarré.")

    def _stop(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    def _live(method, url, body=None):
        payload = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{url}",
            data=payload,
            headers={**headers, "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode() or "{}")

    process = _start()
    try:
        _wait(process)
        opened_status, _opened = _live("POST", PATIENT.format(booking_id=booking_id), {})
        assert opened_status == 200
        basis_status, _basis = _live(
            "POST",
            PATIENT_TX.format(booking_id=booking_id),
            {
                "action": "declare_recipe_fixture",
                "idempotency_key": "fixture-restart",
                "recipe_fixture": "fictif-identifie",
            },
        )
        assert basis_status == 200
        draft_status, draft = _live(
            "POST",
            PATIENT_TX.format(booking_id=booking_id),
            {"action": "prepare_draft", "idempotency_key": "draft-restart"},
        )
        assert draft_status == 200, draft
        draft_id = draft["draft_invoice_id"]
        _stop(process)
        assert process.poll() is not None
        process = _start()
        _wait(process)
        read_status, reread = _live("GET", PATIENT.format(booking_id=booking_id))
        assert read_status == 200, reread
        assert reread["draft_invoice_id"] == draft_id
        replay_status, replay = _live(
            "POST",
            PATIENT_TX.format(booking_id=booking_id),
            {"action": "prepare_draft", "idempotency_key": "draft-restart"},
        )
        assert replay_status == 200, replay
        assert replay.get("replay") is True
        assert replay["draft_invoice_id"] == draft_id
    finally:
        _stop(process)
    with app.app_context():
        lines = list(
            db.session.scalars(
                select(InvoiceLine).where(InvoiceLine.reservation_id == booking_id)
            ).all()
        )
    assert len(lines) == 1


def test_deux_paiements_completent_le_prix(client, world):
    booking = _booking(world, amount=Decimal("45.00"))
    _pay(booking, 20.0, PaymentStatus.COMPLETED)
    _pay(booking, 25.0, PaymentStatus.COMPLETED)
    preview = _preview(client, world, year=2026, month=10, client_id=world["client"].id)
    assert booking.id not in _ids(preview)
    assert (
        db.session.scalar(select(func.count()).select_from(InvoiceLine)) is not None
    )
    _ = uuid
