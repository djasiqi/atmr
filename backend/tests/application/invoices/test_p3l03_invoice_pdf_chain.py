"""P3-L03 — chaîne réelle émission, PDF, QR et régénération.

Base jetable au schéma applicatif complet. Aucune écriture sur atmr.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from flask import current_app
from pypdf import PdfReader
from sqlalchemy import text

from application.invoices.force_regenerate_invoice_pdf import (
    force_regenerate_invoice_pdf,
)
from application.invoices.generate_invoice import (
    GenerateInvoiceInput,
    GenerateInvoiceUseCase,
)
from domain.end_of_trip_settlement.cash_settlement import evaluate_cash
from ext import db as app_db
from models import (
    Booking,
    Client,
    Company,
    CompanyBillingSettings,
    Driver,
    Invoice,
    User,
)
from models.enums import BookingStatus, InvoiceStatus, UserRole
from models.settlement_production import CashCollection, CashCollectionEvent
from services.documents.pdf import PDFService
from services.partnerships.invoices_pdf import generate_partner_invoice_pdf_content
from tests.services.test_partner_invoice_pdf import (
    _create_test_partner_invoice_setup,
)

_UPLOADS = Path("/app/uploads")
_IBAN = "CH6509000000152631289"


def _user(prefix: str, role: UserRole, first: str, last: str) -> User:
    suffix = uuid.uuid4().hex[:8]
    user = User(
        username=f"{prefix}_{suffix}",
        email=f"{prefix}_{suffix}@test.example",
        first_name=first,
        last_name=last,
    )
    user.role = role
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    return user


def _prepare_uploads() -> None:
    current_app.config["UPLOAD_FOLDER"] = str(_UPLOADS)
    (_UPLOADS / "invoices").mkdir(parents=True, exist_ok=True)


def _company_world():
    owner = _user("own", UserRole.company, "Ada", "Lirie")
    app_db.session.add(owner)
    app_db.session.flush()
    company = Company(
        name="LIRIE Transport",
        uid_ide=f"CHE-{uuid.uuid4().int % 900 + 100}.{uuid.uuid4().int % 900 + 100}.{uuid.uuid4().int % 900 + 100}",
        address="Rue de la Gare 4",
        domicile_address_line1="Rue de la Gare 4",
        domicile_zip="1003",
        domicile_city="Lausanne",
        user_id=owner.id,
    )
    company.iban = _IBAN
    app_db.session.add(company)
    app_db.session.flush()
    client_user = _user("cli", UserRole.client, "Jean", "Dupont")
    app_db.session.add(client_user)
    app_db.session.flush()
    client = Client(user=client_user, company=company)
    client.domicile_address = "Avenue du Patient 2"
    client.domicile_zip = "1004"
    client.domicile_city = "Lausanne"
    client.is_institution = False
    app_db.session.add(client)
    app_db.session.flush()
    driver_user = _user("drv", UserRole.driver, "Paul", "Chauffeur")
    app_db.session.add(driver_user)
    app_db.session.flush()
    driver = Driver(user_id=driver_user.id, company_id=company.id)
    app_db.session.add(driver)
    app_db.session.flush()
    billing = CompanyBillingSettings(
        company_id=company.id,
        payment_terms_days=30,
        vat_applicable=False,
        vat_rate=Decimal("0.00"),
    )
    billing.iban = _IBAN
    app_db.session.add(billing)
    app_db.session.commit()
    return company, client, client_user, driver


def _booking(company, client, client_user, driver, amount: str, **extra) -> Booking:
    booking = Booking(
        company_id=company.id,
        client_id=client.id,
        user_id=client_user.id,
        driver_id=driver.id,
        customer_name="Jean Dupont",
        pickup_location=extra.pop("pickup_location", "Gare"),
        dropoff_location=extra.pop("dropoff_location", "Hopital"),
        scheduled_time=extra.pop("scheduled_time", datetime(2026, 11, 15, 10, 0)),
        amount=Decimal(amount),
        price_amount=Decimal(amount),
        status=BookingStatus.COMPLETED,
        is_round_trip=extra.pop("is_round_trip", False),
        billed_to_type="patient",
    )
    for key, value in extra.items():
        setattr(booking, key, value)
    app_db.session.add(booking)
    app_db.session.commit()
    return booking


def _collect(booking, driver, company, amount: str, key: str) -> None:
    """Acompte patient déjà enregistré, sans passer par le registre de caisse."""
    due = Decimal(str(booking.amount))
    tendered = Decimal(amount)
    decision = evaluate_cash(
        due=due,
        tendered=tendered,
        change=Decimal("0.00"),
        change_confirmed=True,
        disposition="partial",
    )
    row = CashCollection(
        booking_id=int(booking.id),
        company_id=int(company.id),
        driver_id=int(driver.id),
        debtor_label="Patient test",
        amount_due_chf=due,
        tendered_chf=tendered,
        change_chf=decision.change_chf,
        change_confirmed=True,
        net_chf=decision.net_chf,
        currency="CHF",
        state=decision.outcome if decision.outcome != "administrative" else "received",
        disposition="partial" if tendered < due else None,
        decision_version="fin-facture-1",
        client_operation_key=key,
        kind="collection",
        production_activation=False,
        legally_approved=False,
    )
    app_db.session.add(row)
    app_db.session.flush()
    app_db.session.add(
        CashCollectionEvent(
            collection_id=int(row.id),
            idempotency_key=key,
            action="record_collection",
            from_state="prepared",
            to_state=row.state,
            actor_role="driver",
            actor_id=int(driver.user_id),
            payload={"net_chf": f"{decision.net_chf:.2f}"},
        )
    )
    app_db.session.commit()


def _emit(company, client, booking_ids: list[int]):
    _prepare_uploads()
    return GenerateInvoiceUseCase().execute(
        GenerateInvoiceInput(
            company_id=int(company.id),
            client_id=int(client.id),
            period_year=2026,
            period_month=11,
            reservation_ids=booking_ids,
            strict_reservation_ids=True,
        )
    )


def _pdf_bytes(pdf_url: str) -> bytes:
    name = pdf_url.rstrip("/").split("/")[-1]
    return (_UPLOADS / "invoices" / name).read_bytes()


def _pages(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(BytesIO(pdf_bytes))
    return [page.extract_text() or "" for page in reader.pages]


def _sql_invoice(invoice_id: int) -> dict:
    row = app_db.session.execute(
        text(
            """
            SELECT invoice_number, total_amount, amount_paid, balance_due,
                   status, qr_reference
            FROM invoices WHERE id = :id
            """
        ),
        {"id": invoice_id},
    ).one()
    payments = app_db.session.execute(
        text(
            """
            SELECT amount, method, cash_collection_id
            FROM invoice_payments WHERE invoice_id = :id ORDER BY id
            """
        ),
        {"id": invoice_id},
    ).all()
    lines = app_db.session.execute(
        text(
            """
            SELECT id, line_total, unit_price, reservation_id
            FROM invoice_lines WHERE invoice_id = :id ORDER BY id
            """
        ),
        {"id": invoice_id},
    ).all()
    return {"invoice": row, "payments": payments, "lines": lines}


def test_emission_45_20_25_pdf_et_qr(db):
    company, client, client_user, driver = _company_world()
    booking = _booking(company, client, client_user, driver, "45.00")
    _collect(booking, driver, company, "20.00", f"k-{booking.id}")
    result = _emit(company, client, [booking.id])
    assert result.success, result.error
    snapshot = _sql_invoice(int(result.invoice_id))
    assert Decimal(str(snapshot["invoice"].total_amount)) == Decimal("45.00")
    assert Decimal(str(snapshot["invoice"].amount_paid)) == Decimal("20.00")
    assert Decimal(str(snapshot["invoice"].balance_due)) == Decimal("25.00")
    assert str(snapshot["invoice"].status).lower() == InvoiceStatus.PARTIALLY_PAID.value
    assert len(snapshot["payments"]) == 1
    assert Decimal(str(snapshot["payments"][0].amount)) == Decimal("20.00")
    assert snapshot["payments"][0].cash_collection_id is not None
    assert len(snapshot["lines"]) == 1
    assert Decimal(str(snapshot["lines"][0].line_total)) == Decimal("45.00")
    invoice = app_db.session.get(Invoice, result.invoice_id)
    pages = _pages(_pdf_bytes(invoice.pdf_url))
    assert "45.00" in pages[0]
    assert "20.00" in pages[0]
    assert "25.00" in pages[0]
    assert "Acomptes déjà encaissés" in pages[0] or "NET À PAYER" in pages[0]
    assert "Espèces chauffeur" in pages[0]
    assert len(pages) >= 2
    assert "25.00" in pages[-1]
    assert "45.00" not in pages[-1]


def test_facture_soldee_sans_qr(db):
    company, client, client_user, driver = _company_world()
    booking = _booking(company, client, client_user, driver, "45.00")
    _collect(booking, driver, company, "45.00", f"k-{booking.id}")
    result = _emit(company, client, [booking.id])
    assert result.success, result.error
    snapshot = _sql_invoice(int(result.invoice_id))
    assert Decimal(str(snapshot["invoice"].balance_due)) == Decimal("0.00")
    assert str(snapshot["invoice"].status).lower() == InvoiceStatus.PAID.value
    assert len(snapshot["payments"]) == 1
    invoice = app_db.session.get(Invoice, result.invoice_id)
    pages = _pages(_pdf_bytes(invoice.pdf_url))
    joined = "\n".join(pages)
    assert "Payée" in joined
    assert "Solde nul" in joined
    assert "45.00" in pages[0]
    assert len(pages) == 1


def test_sans_acompte_qr_du_total(db):
    company, client, client_user, driver = _company_world()
    booking = _booking(company, client, client_user, driver, "45.00")
    result = _emit(company, client, [booking.id])
    assert result.success, result.error
    snapshot = _sql_invoice(int(result.invoice_id))
    assert snapshot["payments"] == []
    assert Decimal(str(snapshot["invoice"].balance_due)) == Decimal("45.00")
    assert str(snapshot["invoice"].status).lower() == InvoiceStatus.DRAFT.value
    invoice = app_db.session.get(Invoice, result.invoice_id)
    pages = _pages(_pdf_bytes(invoice.pdf_url))
    assert "Déjà encaissé" not in "\n".join(pages)
    assert "45.00" in pages[-1]


def test_aller_retour_90_20_70(db):
    company, client, client_user, driver = _company_world()
    aller = _booking(
        company,
        client,
        client_user,
        driver,
        "45.00",
        pickup_location="Gare",
        dropoff_location="Hopital",
        is_round_trip=True,
        scheduled_time=datetime(2026, 11, 15, 9, 0),
    )
    retour = _booking(
        company,
        client,
        client_user,
        driver,
        "45.00",
        pickup_location="Hopital",
        dropoff_location="Gare",
        is_round_trip=True,
        is_return=True,
        parent_booking_id=aller.id,
        scheduled_time=datetime(2026, 11, 15, 16, 0),
    )
    _collect(aller, driver, company, "20.00", f"k-{aller.id}")
    result = _emit(company, client, [aller.id, retour.id])
    assert result.success, result.error
    snapshot = _sql_invoice(int(result.invoice_id))
    assert len(snapshot["lines"]) == 1
    assert Decimal(str(snapshot["lines"][0].line_total)) == Decimal("90.00")
    assert len(snapshot["payments"]) == 1
    assert Decimal(str(snapshot["invoice"].balance_due)) == Decimal("70.00")
    links = app_db.session.execute(
        text(
            "SELECT id, invoice_line_id, price_amount FROM booking WHERE id IN (:a, :b)"
        ),
        {"a": aller.id, "b": retour.id},
    ).all()
    assert {int(row.invoice_line_id) for row in links} == {int(snapshot["lines"][0].id)}
    assert {Decimal(str(row.price_amount)) for row in links} == {Decimal("45.00")}
    invoice = app_db.session.get(Invoice, result.invoice_id)
    pages = _pages(_pdf_bytes(invoice.pdf_url))
    assert "90.00" in pages[0]
    assert "20.00" in pages[0]
    assert "70.00" in pages[0]
    assert "70.00" in pages[-1]


def test_regeneration_refusee_apres_emission_et_rejeu(db):
    company, client, client_user, driver = _company_world()
    booking = _booking(company, client, client_user, driver, "45.00")
    line_before_collect = booking.price_amount
    _collect(booking, driver, company, "20.00", f"k-{booking.id}")
    first = _emit(company, client, [booking.id])
    assert first.success, first.error
    before = _sql_invoice(int(first.invoice_id))
    number = before["invoice"].invoice_number
    reference = before["invoice"].qr_reference
    line_id = int(before["lines"][0].id)
    pdf_url = app_db.session.get(Invoice, first.invoice_id).pdf_url
    regen = force_regenerate_invoice_pdf(
        company_id=int(company.id), invoice_id=int(first.invoice_id)
    )
    assert regen.ok is False
    assert regen.status_code == 409
    after = _sql_invoice(int(first.invoice_id))
    assert after["invoice"].invoice_number == number
    assert after["invoice"].qr_reference == reference
    assert len(after["payments"]) == 1
    assert int(after["lines"][0].id) == line_id
    assert Decimal(str(after["lines"][0].line_total)) == Decimal("45.00")
    assert app_db.session.get(Invoice, first.invoice_id).pdf_url == pdf_url
    booking_row = app_db.session.get(Booking, booking.id)
    assert booking_row.invoice_line_id == line_id
    assert booking_row.price_amount == line_before_collect
    second = _emit(company, client, [booking.id])
    assert second.success, second.error
    assert second.invoice_id == first.invoice_id
    assert len(_sql_invoice(int(first.invoice_id))["payments"]) == 1
    invoices = app_db.session.execute(
        text("SELECT count(*) FROM invoices WHERE company_id = :id"),
        {"id": company.id},
    ).scalar()
    assert invoices == 1


def test_echec_pdf_puis_reprise_sans_seconde_imputation(db, monkeypatch):
    company, client, client_user, driver = _company_world()
    booking = _booking(company, client, client_user, driver, "45.00")
    _collect(booking, driver, company, "20.00", f"k-{booking.id}")

    def _fail(*_args, **_kwargs):
        raise RuntimeError("pdf indisponible")

    monkeypatch.setattr(PDFService, "generate_invoice_pdf", _fail)
    failed = _emit(company, client, [booking.id])
    monkeypatch.undo()
    assert failed.success, failed.error
    during = _sql_invoice(int(failed.invoice_id))
    assert len(during["payments"]) == 1
    assert Decimal(str(during["invoice"].total_amount)) == Decimal("45.00")
    assert Decimal(str(during["invoice"].balance_due)) == Decimal("25.00")
    assert str(during["invoice"].status).lower() == InvoiceStatus.DRAFT.value
    _prepare_uploads()
    resumed = force_regenerate_invoice_pdf(
        company_id=int(company.id), invoice_id=int(failed.invoice_id)
    )
    assert resumed.ok, resumed.error
    after = _sql_invoice(int(failed.invoice_id))
    assert len(after["payments"]) == 1
    assert Decimal(str(after["lines"][0].line_total)) == Decimal("45.00")
    assert Decimal(str(after["invoice"].balance_due)) == Decimal("25.00")
    pages = _pages(_pdf_bytes(resumed.pdf_url))
    assert "25.00" in pages[0]
    assert "20.00" in pages[0]
    assert "25.00" in pages[-1]


def test_rollback_avant_commit(db, monkeypatch):
    company, client, client_user, driver = _company_world()
    booking = _booking(company, client, client_user, driver, "45.00")
    _collect(booking, driver, company, "20.00", f"k-{booking.id}")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("avant commit")

    monkeypatch.setattr(
        "application.invoices.generate_invoice.impute_confirmed_cash", _boom
    )
    result = _emit(company, client, [booking.id])
    assert result.success is False
    invoices = app_db.session.execute(
        text("SELECT count(*) FROM invoices WHERE company_id = :id"),
        {"id": company.id},
    ).scalar()
    payments = app_db.session.execute(
        text(
            """
            SELECT count(*) FROM invoice_payments p
            JOIN invoices i ON i.id = p.invoice_id
            WHERE i.company_id = :id
            """
        ),
        {"id": company.id},
    ).scalar()
    assert invoices == 0
    assert payments == 0
    assert app_db.session.get(Booking, booking.id).invoice_line_id is None


def test_institution_et_partenaire_sans_acompte(db):
    company, client, client_user, driver = _company_world()
    client.is_institution = True
    app_db.session.commit()
    booking = _booking(company, client, client_user, driver, "80.00")
    result = _emit(company, client, [booking.id])
    assert result.success, result.error
    snapshot = _sql_invoice(int(result.invoice_id))
    assert snapshot["payments"] == []
    assert Decimal(str(snapshot["lines"][0].line_total)) == Decimal("80.00")
    invoice = app_db.session.get(Invoice, result.invoice_id)
    text_pdf = "\n".join(_pages(_pdf_bytes(invoice.pdf_url)))
    assert "Déjà encaissé" not in text_pdf
    assert "80.00" in text_pdf

    partner_invoice, transfers = _create_test_partner_invoice_setup(db)
    partner_bytes = generate_partner_invoice_pdf_content(
        partner_invoice, list(transfers)
    )
    partner_text = "\n".join(_pages(partner_bytes))
    assert "Déjà encaissé" not in partner_text
    assert partner_bytes.startswith(b"%PDF")
    assert (
        str(partner_invoice.total_amount) in partner_text
        or "Déjà encaissé" not in partner_text
    )


def test_deux_emissions_concurrentes(app, db):
    company, client, client_user, driver = _company_world()
    booking = _booking(company, client, client_user, driver, "45.00")
    _collect(booking, driver, company, "20.00", f"k-{booking.id}")
    _prepare_uploads()
    company_id = int(company.id)
    client_id = int(client.id)
    booking_id = int(booking.id)
    app_db.session.remove()
    from threading import Thread

    outcomes: list = []

    def _run():
        with app.app_context():
            try:
                outcomes.append(
                    _emit(
                        SimpleNamespace(id=company_id),
                        SimpleNamespace(id=client_id),
                        [booking_id],
                    )
                )
            except Exception as exc:
                outcomes.append(exc)
            finally:
                app_db.session.remove()

    first = Thread(target=_run)
    second = Thread(target=_run)
    first.start()
    second.start()
    first.join(timeout=60)
    second.join(timeout=60)
    assert len(outcomes) == 2
    assert all(getattr(item, "success", False) for item in outcomes), outcomes
    assert outcomes[0].invoice_id == outcomes[1].invoice_id
    with app.app_context():
        snapshot = _sql_invoice(int(outcomes[0].invoice_id))
    assert len(snapshot["payments"]) == 1
    assert len(snapshot["lines"]) == 1
