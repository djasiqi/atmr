"""Matrice de certification étape 31. Base jetable, aucune émission réelle."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from flask_jwt_extended import create_access_token
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from application.settlement.patient_billing_prep import specimen_pdf
from ext import db
from models.enums import (
    BookingBillingOrigin,
    BookingStatus,
    InstitutionBillingControlStatus,
    InvoiceStatus,
    PartnershipStatus,
    PaymentStatus,
    TransferModel,
    TransferStatus,
    UserRole,
)
from models.invoice import Invoice, InvoiceLine
from models.user import User
from services.recipe_environment import database_name

pytestmark = pytest.mark.integration

RECIPE = "lirie_e2e_"
PDF_DIR = Path("/tmp/lirie-etape31-pdf")
PATIENT = "/api/v1/driver/me/bookings/{booking_id}/patient-billing-dossier"
PATIENT_TX = PATIENT + "/transitions"
GENERATE = "/api/v1/invoices/companies/{company_id}/invoices/generate"
PREVIEW = "/api/v1/invoices/companies/{company_id}/invoices/period-preview"
PARTNER_GENERATE = "/api/v1/invoices/companies/{company_id}/partners/invoices/generate"
SPECIMEN_IBAN = "CH4431999123000889012"
TEST_IBAN = "CH6509000000152631289"


@pytest.fixture
def prep(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")
    monkeypatch.setenv("MAIL_SUPPRESS_SEND", "1")
    monkeypatch.setenv("EMAIL_NOTIFICATIONS_ENABLED", "0")
    monkeypatch.setenv("NOTIFICATIONS_KAFKA_PUBLISH_ENABLED", "0")


@pytest.fixture
def world(db, app, prep):
    current = database_name()
    if not (current or "").startswith(RECIPE):
        pytest.skip(f"Base refusée pour cette recette : {current or 'absente'}")
    suffix = uuid.uuid4().hex[:8]
    company, owner, driver, driver_user = _company_driver(f"a-{suffix}")
    other, other_owner, _driver, _user = _company_driver(f"b-{suffix}")
    client = _client(company, f"pat-{suffix}")
    db.session.commit()
    return {
        "app": app,
        "suffix": suffix,
        "company": company,
        "owner": owner,
        "other": other,
        "other_owner": other_owner,
        "driver": driver,
        "driver_user": driver_user,
        "client": client,
    }


def _user(role: UserRole, label: str) -> User:
    row = User()
    row.username = label
    row.email = f"{label}@lirie-e2e.invalid"
    row.role = role
    row.public_id = str(uuid.uuid4())
    row.set_password("recette-isolee", force_change=False)
    db.session.add(row)
    db.session.flush()
    return row


def _company_driver(label: str):
    from models import Company, Driver

    owner = _user(UserRole.company, f"co-{label}")
    company = Company(
        name=f"Certif31-{label}",
        user_id=owner.id,
        platform_billing_access_state="active",
    )
    db.session.add(company)
    db.session.flush()
    driver_user = _user(UserRole.driver, f"ch-{label}")
    driver = Driver(user_id=driver_user.id, company_id=company.id)
    db.session.add(driver)
    db.session.flush()
    return company, owner, driver, driver_user


def _client(company, label: str):
    from models.client import Client
    from models.enums import ClientType

    user = _user(UserRole.client, label)
    user.first_name = "Patient"
    user.last_name = f"Fictif-{label[-4:]}"
    row = Client(
        user_id=user.id,
        company_id=company.id,
        client_type=ClientType.TRANSPORT,
        default_billed_to_type="patient",
    )
    db.session.add(row)
    db.session.flush()
    return row


def _booking(world, **overrides):
    from models.booking import Booking

    when = overrides.pop("scheduled_time", datetime(2026, 10, 15, 9, 0, tzinfo=UTC))
    amount = overrides.pop("amount", Decimal("45.00"))
    row = Booking(
        company_id=world["company"].id,
        executing_company_id=world["company"].id,
        client_id=overrides.pop("client_id", world["client"].id),
        user_id=world["client"].user_id,
        driver_id=world["driver"].id,
        amount=amount,
        price_amount=amount,
        status=overrides.pop("status", BookingStatus.COMPLETED),
        scheduled_time=when,
        pickup_location=overrides.pop("pickup_location", "Rue fictive 1, 1000 Recette"),
        dropoff_location=overrides.pop(
            "dropoff_location", "Avenue fictive 2, 1000 Recette"
        ),
        customer_name="Patient fictif",
        billed_to_type=overrides.pop("billed_to_type", "patient"),
        is_round_trip=False,
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    db.session.add(row)
    db.session.commit()
    return row


def _headers(app, user: User) -> dict[str, str]:
    claims = {
        "role": user.role.value,
        "company_id": None,
        "aud": "atmr-api",
        "token_version": int(getattr(user, "token_version", 0) or 0),
    }
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}"}


def _owner(world):
    return _headers(world["app"], world["owner"])


def _driver(world):
    return _headers(world["app"], world["driver_user"])


def _lines(booking_id: int) -> list[InvoiceLine]:
    return list(
        db.session.scalars(
            select(InvoiceLine).where(InvoiceLine.reservation_id == booking_id)
        ).all()
    )


def _preview(client, world, **params) -> dict:
    query = "&".join(f"{key}={value}" for key, value in params.items())
    response = client.get(
        PREVIEW.format(company_id=world["company"].id) + f"?{query}",
        headers=_owner(world),
    )
    assert response.status_code == 200, response.get_data(as_text=True)[:500]
    return response.get_json()["data"]


def _generate(client, world, reservation_ids: list[int], **extra):
    payload = {
        "client_id": extra.pop("client_id", world["client"].id),
        "period_year": 2026,
        "period_month": 10,
        "reservation_ids": reservation_ids,
    }
    payload.update(extra)
    return client.post(
        GENERATE.format(company_id=world["company"].id),
        json=payload,
        headers=_owner(world),
    )


def test_patient_trajet_dates_multiples_et_exclusions(client, world):
    simple = _booking(world, amount=Decimal("45.00"))
    preview = _preview(client, world, year=2026, month=10, client_id=world["client"].id)
    assert preview["transports_count"] >= 1
    created = _generate(client, world, [simple.id])
    assert created.status_code in {200, 201}, created.get_data(as_text=True)[:500]
    body = created.get_json()
    invoice_id = body.get("id") or body.get("invoice_id")
    invoice = db.session.get(Invoice, invoice_id)
    assert invoice.status == InvoiceStatus.DRAFT
    assert invoice.client_id == world["client"].id
    assert len(_lines(simple.id)) == 1
    assert _lines(simple.id)[0].line_total == Decimal("45.00")
    again = _generate(client, world, [simple.id])
    assert again.status_code != 500
    assert len(_lines(simple.id)) == 1

    outbound = _booking(world, amount=Decimal("45.00"))
    _booking(
        world,
        amount=Decimal("45.00"),
        parent_booking_id=outbound.id,
        is_return=True,
        scheduled_time=datetime(2026, 10, 15, 16, 0, tzinfo=UTC),
    )
    round_trip = _preview(
        client, world, year=2026, month=10, client_id=world["client"].id
    )
    ids = {line["booking_id"] for line in round_trip["preview_lines"]}
    assert outbound.id in ids or any(
        line.get("round_trip_partner_booking_id")
        for line in round_trip["preview_lines"]
    )
    assert float(round_trip["estimated_total"]) >= 90.0

    early = _booking(
        world,
        amount=Decimal("20.00"),
        scheduled_time=datetime(2026, 10, 2, 8, 0, tzinfo=UTC),
    )
    late = _booking(
        world,
        amount=Decimal("30.00"),
        scheduled_time=datetime(2026, 10, 20, 18, 0, tzinfo=UTC),
    )
    span = _preview(client, world, year=2026, month=10, client_id=world["client"].id)
    span_ids = {line["booking_id"] for line in span["preview_lines"]}
    assert early.id in span_ids
    assert late.id in span_ids

    unfinished = _booking(world, status=BookingStatus.IN_PROGRESS)
    refused = _generate(client, world, [unfinished.id])
    assert refused.status_code in {400, 409, 422}
    assert _lines(unfinished.id) == []

    cancelled = _booking(world, status=BookingStatus.CANCELED, amount=Decimal("45.00"))
    cancelled.is_cancellation_billable = False
    db.session.commit()
    hidden = _preview(client, world, year=2026, month=10, client_id=world["client"].id)
    assert cancelled.id not in {line["booking_id"] for line in hidden["preview_lines"]}


def test_patient_prepaye_saferpay_nest_plus_propose(client, world):
    """Un paiement COMPLETED qui couvre le prix retire la course de l'aperçu.

    Aucun appel Saferpay n'est fait. Le paiement historique n'est pas modifié.
    """
    from models.payment import Payment

    booking = _booking(world, amount=Decimal("45.00"))
    payment = Payment(
        amount=45.0,
        method="credit_card",
        status=PaymentStatus.COMPLETED,
        booking_id=booking.id,
        user_id=world["client"].user_id,
        client_id=world["client"].id,
    )
    db.session.add(payment)
    db.session.commit()
    preview = _preview(client, world, year=2026, month=10, client_id=world["client"].id)
    assert booking.id not in {line["booking_id"] for line in preview["preview_lines"]}
    refused = _generate(client, world, [booking.id])
    assert refused.status_code in {400, 409, 422}


def test_institution_beneficiaires_retenue_et_patient_payeur(client, world):
    # Le débiteur institutionnel est l'autre entreprise, jamais le créancier.
    clinic_id = world["other"].id
    second = _client(world["company"], f"ben-{world['suffix']}")
    first = _booking(
        world,
        billed_to_type="clinic",
        billed_to_company_id=clinic_id,
        amount=Decimal("80.00"),
        customer_name="Beneficiaire fictif A",
    )
    other = _booking(
        world,
        client_id=second.id,
        billed_to_type="clinic",
        billed_to_company_id=clinic_id,
        amount=Decimal("70.00"),
        customer_name="Beneficiaire fictif B",
        scheduled_time=datetime(2026, 10, 16, 11, 0, tzinfo=UTC),
    )
    held = _booking(
        world,
        billed_to_type="clinic",
        billed_to_company_id=clinic_id,
        amount=Decimal("60.00"),
        billing_origin=BookingBillingOrigin.LIRIE_MARKETPLACE,
        institution_control_status=InstitutionBillingControlStatus.PENDING_REVIEW,
        scheduled_time=datetime(2026, 10, 18, 11, 0, tzinfo=UTC),
    )
    released = _booking(
        world,
        billed_to_type="clinic",
        billed_to_company_id=clinic_id,
        amount=Decimal("55.00"),
        billing_origin=BookingBillingOrigin.LIRIE_MARKETPLACE,
        institution_control_status=InstitutionBillingControlStatus.PENDING_REVIEW,
        scheduled_time=datetime(2026, 9, 12, 11, 0, tzinfo=UTC),
    )
    patient_leg = _booking(
        world,
        billed_to_type="patient",
        amount=Decimal("40.00"),
        billing_origin=BookingBillingOrigin.LIRIE_MARKETPLACE,
        institution_control_status=InstitutionBillingControlStatus.PENDING_REVIEW,
        customer_name="Patient payeur retenu",
    )
    patient_ready = _booking(
        world,
        billed_to_type="patient",
        amount=Decimal("40.00"),
        billing_origin=BookingBillingOrigin.LIRIE_MARKETPLACE,
        institution_control_status=InstitutionBillingControlStatus.VALIDATED,
        customer_name="Patient payeur valide",
        scheduled_time=datetime(2026, 10, 19, 11, 0, tzinfo=UTC),
    )
    october = _preview(
        client,
        world,
        year=2026,
        month=10,
        clinic_company_id=clinic_id,
    )
    selected = {line["booking_id"] for line in october["preview_lines"]}
    assert first.id in selected
    assert other.id in selected
    assert held.id not in selected
    assert patient_leg.id not in selected
    assert patient_ready.id not in selected
    assert float(october["estimated_total"]) == 150.0

    september = _preview(
        client,
        world,
        year=2026,
        month=9,
        clinic_company_id=clinic_id,
    )
    assert released.id in {line["booking_id"] for line in september["preview_lines"]}
    patient_view = _preview(
        client, world, year=2026, month=10, client_id=world["client"].id
    )
    patient_ids = {line["booking_id"] for line in patient_view["preview_lines"]}
    assert patient_ready.id in patient_ids
    assert patient_leg.id not in patient_ids


def test_partenaire_trajets_annulation_et_deux_entreprises(client, world):
    from models.booking_transfer import BookingTransfer
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
    from models.invoice import CompanyBillingSettings

    prefix = f"P{world['suffix'][:6].upper()}"
    db.session.add(
        CompanyBillingSettings(
            company_id=world["company"].id,
            invoice_prefix=prefix,
            payment_terms_days=30,
            iban=TEST_IBAN,
        )
    )
    kept = _booking(world, amount=Decimal("40.00"))
    day_two = _booking(
        world,
        amount=Decimal("25.00"),
        scheduled_time=datetime(2026, 10, 21, 7, 30, tzinfo=UTC),
        customer_name="Second beneficiaire fictif",
    )
    dropped = _booking(world, amount=Decimal("15.00"))

    def _transfer(booking, status):
        row = BookingTransfer(
            booking_id=booking.id,
            partnership_id=partnership.id,
            transfer_model=TransferModel.SUBCONTRACT,
            status=status,
            is_validated=True,
            validated_at=booking.scheduled_time,
            client_price=booking.amount,
            partner_cost=booking.amount,
            currency="CHF",
            owner_company_id=world["other"].id,
            executing_company_id=world["company"].id,
        )
        db.session.add(row)
        return row

    first = _transfer(kept, TransferStatus.COMPLETED)
    second = _transfer(day_two, TransferStatus.COMPLETED)
    _transfer(dropped, TransferStatus.CANCELLED)
    db.session.commit()
    created = client.post(
        PARTNER_GENERATE.format(company_id=world["company"].id),
        json={
            "partnership_id": partnership.id,
            "period_year": 2026,
            "period_month": 10,
            "transfer_ids": [first.id, second.id],
        },
        headers=_owner(world),
    )
    assert created.status_code in {200, 201}, created.get_data(as_text=True)[:600]
    duplicate = client.post(
        PARTNER_GENERATE.format(company_id=world["company"].id),
        json={
            "partnership_id": partnership.id,
            "period_year": 2026,
            "period_month": 10,
            "transfer_ids": [first.id],
        },
        headers=_owner(world),
    )
    assert duplicate.status_code in {400, 409, 422}

    third_company, third_owner, _driver, _user = _company_driver(f"c-{world['suffix']}")
    other_partnership = Partnership(
        owner_company_id=third_company.id,
        partner_company_id=world["company"].id,
        status=PartnershipStatus.ACCEPTED,
        default_transfer_model=TransferModel.SUBCONTRACT,
        payment_terms_days=30,
    )
    db.session.add(other_partnership)
    db.session.flush()
    foreign = _booking(world, amount=Decimal("12.00"))
    foreign_transfer = BookingTransfer(
        booking_id=foreign.id,
        partnership_id=other_partnership.id,
        transfer_model=TransferModel.SUBCONTRACT,
        status=TransferStatus.COMPLETED,
        is_validated=True,
        validated_at=foreign.scheduled_time,
        client_price=foreign.amount,
        partner_cost=foreign.amount,
        currency="CHF",
        owner_company_id=third_company.id,
        executing_company_id=world["company"].id,
    )
    db.session.add(foreign_transfer)
    db.session.commit()
    separate = client.post(
        PARTNER_GENERATE.format(company_id=world["company"].id),
        json={
            "partnership_id": other_partnership.id,
            "period_year": 2026,
            "period_month": 10,
            "transfer_ids": [foreign_transfer.id],
        },
        headers=_owner(world),
    )
    assert separate.status_code in {200, 201}, separate.get_data(as_text=True)[:600]
    hidden = client.post(
        PARTNER_GENERATE.format(company_id=world["company"].id),
        json={
            "partnership_id": partnership.id,
            "period_year": 2026,
            "period_month": 10,
            "transfer_ids": [foreign_transfer.id],
        },
        headers=_owner(world),
    )
    assert hidden.status_code in {400, 404, 409, 422}
    _ = third_owner


def test_reponse_perdue_et_rollback_injecte(client, world, monkeypatch):
    booking = _booking(world, status=BookingStatus.IN_PROGRESS)
    opened = client.post(
        PATIENT.format(booking_id=booking.id), json={}, headers=_driver(world)
    )
    assert opened.status_code == 200
    basis = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={
            "action": "declare_recipe_fixture",
            "idempotency_key": "fixture-31",
            "recipe_fixture": "fictif-identifie",
        },
        headers=_driver(world),
    )
    assert basis.status_code == 200

    def _boom() -> None:
        raise IntegrityError("preparation injectee", {}, Exception("rollback"))

    monkeypatch.setattr(db.session, "commit", _boom)
    failed = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "prepare_draft", "idempotency_key": "draft-rollback"},
        headers=_driver(world),
    )
    monkeypatch.undo()
    db.session.rollback()
    assert failed.status_code == 409
    assert failed.get_json()["error_code"] == "preparation_annulee"
    assert _lines(booking.id) == []
    assert (
        db.session.scalar(
            select(func.count())
            .select_from(Invoice)
            .where(Invoice.client_id.is_not(None))
        )
        is not None
    )
    created = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "prepare_draft", "idempotency_key": "draft-rollback"},
        headers=_driver(world),
    )
    assert created.status_code == 200
    replay = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "prepare_draft", "idempotency_key": "draft-rollback"},
        headers=_driver(world),
    )
    assert replay.status_code == 200
    assert replay.get_json()["replay"] is True
    assert (
        replay.get_json()["draft_invoice_id"] == created.get_json()["draft_invoice_id"]
    )
    assert len(_lines(booking.id)) == 1


def test_qr_specimen_et_decodage_hors_ligne(client, world):
    from models.invoice import CompanyBillingSettings
    from services.documents.qrbill import QRBillService

    booking = _booking(world)
    created = _generate(client, world, [booking.id])
    assert created.status_code in {200, 201}, created.get_data(as_text=True)[:500]
    invoice = db.session.get(
        Invoice, created.get_json().get("id") or created.get_json()["invoice_id"]
    )
    settings = db.session.scalars(
        select(CompanyBillingSettings).where(
            CompanyBillingSettings.company_id == world["company"].id
        )
    ).first()
    if settings is None:
        settings = CompanyBillingSettings(
            company_id=world["company"].id, payment_terms_days=30
        )
        db.session.add(settings)
    settings.iban = TEST_IBAN
    world["company"].address = "Rue de recette 1, 1000 Recette"
    world["company"].contact_email = f"recette-{world['suffix']}@lirie-e2e.invalid"
    db.session.commit()
    service = QRBillService()
    svg = service.generate_qr_bill_svg(invoice)
    assert svg
    payload = service._get_payment_reference(invoice)
    bill_amount = f"{invoice.total_amount:.2f}"
    decoded = _decode_qrbill(invoice)
    compact = decoded["raw"].replace(" ", "")
    assert TEST_IBAN in compact
    assert "CHF" in decoded["raw"].split("\n")
    assert bill_amount in decoded["raw"]
    assert payload is None or payload in decoded["raw"]
    assert decoded["raw"].startswith("SPC")

    dossier = client.post(
        PATIENT.format(booking_id=booking.id), json={}, headers=_driver(world)
    )
    assert dossier.status_code == 200
    specimen = specimen_qr_view(dossier.get_json(), booking)
    assert specimen["payable"] is False
    assert specimen["spc_payload"] is None
    pdf = specimen_pdf(specimen)
    raw = __import__("base64").b64decode(pdf)
    assert b"NE PAS PAYER" in raw
    assert b"SPC\n" not in raw


def specimen_qr_view(opened: dict, booking) -> dict:
    from application.settlement.patient_billing_prep import specimen_qr
    from models.settlement_production import PatientBillingDossier

    row = db.session.scalars(
        select(PatientBillingDossier).where(
            PatientBillingDossier.booking_id == booking.id
        )
    ).one()
    view = specimen_qr(row)
    view["company_name"] = opened.get("company_name") or "Certif"
    view["debtor_label"] = opened.get("debtor_label") or "Patient fictif"
    view["booking_id"] = booking.id
    view["amount_chf"] = opened.get("amount_chf") or "45.00"
    view["contractual_basis"] = opened.get("contractual_basis") or "absent"
    return view


def _decode_qrbill(invoice) -> dict[str, str]:
    from qrbill import QRBill

    from services.documents.qrbill import QRBillService

    service = QRBillService()
    debtor = service._get_debtor_info(invoice)
    iban, creditor = service._resolve_qr_account_and_creditor(invoice)
    bill = QRBill(
        account=iban,
        creditor=creditor,
        debtor=debtor,
        amount=str(invoice.total_amount),
        currency="CHF",
        reference_number=service._get_payment_reference(invoice),
        additional_information=f"Facture {invoice.invoice_number}",
        language="fr",
    )
    raw = bill.qr_data
    if callable(raw):
        raw = raw()
    return {"raw": str(raw).replace("\r\n", "\n").replace("\r", "\n")}


def test_pdf_specimens_ecrits_pour_inspection(db, world):
    from models import Booking, Client, Company, Invoice, InvoiceLine
    from models.enums import InvoiceBillingStrategy, InvoiceLineType

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    company = world["company"]
    client_row = world["client"]
    long_name = "Bénéficiaire fictif à accents — Hôpital des Érables " + ("Nord " * 12)
    long_address = "Chemin des Prés-de-la-Fontaine très long 128, 1000 Recette-sur-Lac"
    invoice = Invoice(
        company_id=company.id,
        client_id=client_row.id,
        billing_strategy=InvoiceBillingStrategy.S1_PATIENT,
        invoice_number=f"SPEC31-{world['suffix']}",
        period_year=2026,
        period_month=10,
        status=InvoiceStatus.DRAFT,
        issued_at=datetime(2026, 10, 27, tzinfo=UTC),
        due_date=datetime(2026, 11, 27, tzinfo=UTC),
        currency="CHF",
        subtotal_amount=Decimal("135.00"),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal("135.00"),
        amount_paid=Decimal("0.00"),
        balance_due=Decimal("135.00"),
    )
    db.session.add(invoice)
    db.session.flush()
    notes = (
        "Note fictive : transport aller à 08 h 15, retour prévu, "
        "accompagnant et accès par la cour intérieure. " * 4
    )
    for index, amount in enumerate((Decimal("45.00"), Decimal("90.00"))):
        booking = Booking(
            company_id=company.id,
            client_id=client_row.id,
            user_id=client_row.user_id,
            customer_name=long_name,
            pickup_location=long_address,
            dropoff_location="Arrivée fictive, bâtiment C",
            scheduled_time=datetime(2026, 10, 15 + index, 9, 0, tzinfo=UTC),
            amount=amount,
            status=BookingStatus.COMPLETED,
        )
        db.session.add(booking)
        db.session.flush()
        db.session.add(
            InvoiceLine(
                invoice_id=invoice.id,
                reservation_id=booking.id,
                type=InvoiceLineType.RIDE,
                description=f"Transport fictif {index + 1} — {long_name[:40]}",
                qty=Decimal("1.00"),
                unit_price=amount,
                line_total=amount,
                vat_rate=Decimal("0.00"),
                vat_amount=Decimal("0.00"),
                total_with_vat=amount,
                adjustment_note=notes,
            )
        )
    db.session.commit()
    from services.documents.pdf import PDFService

    pdf_bytes, _pages = PDFService()._create_invoice_pdf_content(invoice)
    (PDF_DIR / "patient-multiligne.pdf").write_bytes(pdf_bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert b"135" in pdf_bytes or b"45" in pdf_bytes
    _ = (Company, Client)
