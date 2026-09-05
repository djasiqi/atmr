"""Monde C4 — émission PDF / QR d'annulation. Réutilise C1–C3, pas de mock PDF."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from application.invoices.billable_amount import calculate_billable_booking_amount
from application.invoices.generate_clinic_monthly_invoice import (
    GenerateClinicMonthlyInvoiceInput,
    GenerateClinicMonthlyInvoiceUseCase,
)
from application.invoices.institution_invoice_plan import build_institution_invoice_plan
from application.invoices.invoice_pdf_state import get_pdf_state
from application.invoices.period_invoice_preview import build_period_invoice_preview
from ext import db
from models import Invoice, InvoiceLine
from models.invoice import CompanyBillingSettings
from tests.application.helpers.cancel_billable_c1_world import (
    PERIOD_MONTH,
    PERIOD_YEAR,
    build_c1_world,
)
from tests.application.helpers.cancel_billable_c3_world import (
    add_canceled_labeled_booking,
    canonical_cancellation_label,
)
from tests.e2e.helpers.institution_billing_g4 import (
    QR_IBAN,
    QR_UNAVAILABLE,
    _extract_pdf_text,
    _qr_svg_amount,
    _read_generated_pdf,
)

ZURICH = ZoneInfo("Europe/Zurich")
EMIT_AT = datetime(2026, 9, 1, 0, 0, tzinfo=ZURICH)
MONEY = Decimal("0.01")
TRAJET_EFFECTUE_TOKEN = " → "


def money(value: Any) -> Decimal:
    return Decimal(str(value)).quantize(MONEY)


def build_c4_world(db_session) -> dict[str, Any]:
    """Monde C1 + IBAN / domiciliation pour QR réel (même contrat que G4)."""
    world = build_c1_world(db_session)
    transport = world["transport"]
    transport.domicile_address_line1 = transport.address or "Rue Transport 1"
    transport.domicile_zip = "1200"
    transport.domicile_city = "Genève"
    transport.domicile_country = "CH"
    transport.iban = QR_IBAN
    settings = CompanyBillingSettings.query.filter_by(company_id=transport.id).first()
    assert settings is not None
    settings.iban = QR_IBAN
    clinic = world["clinic"]
    clinic.domicile_address_line1 = clinic.address or "Clinique addr"
    clinic.domicile_zip = "1247"
    clinic.domicile_city = "Anières"
    clinic.domicile_country = "CH"
    db_session.session.flush()
    return world


def add_canceled_emission_booking(
    db_session,
    world: dict[str, Any],
    *,
    fee_amount: Decimal | None,
    reason_code: str | None = "NO_SHOW",
    reason_text: str | None = None,
    persist_display_label: bool = True,
    day: int = 12,
):
    return add_canceled_labeled_booking(
        db_session,
        world,
        reason_code=reason_code,
        reason_text=reason_text,
        persist_display_label=persist_display_label,
        fee_amount=fee_amount,
        day=day,
    )


def preview(world: dict[str, Any]):
    return build_period_invoice_preview(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        clinic_company_id=world["clinic"].id,
        include_line_details=True,
        now=EMIT_AT,
    )


def plan(world: dict[str, Any]):
    return build_institution_invoice_plan(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        clinic_company_id=world["clinic"].id,
        clinic_client_id=world["clinic_client"].id,
        now=EMIT_AT,
    )


def generate_real_invoice(world: dict[str, Any]) -> Invoice:
    result = GenerateClinicMonthlyInvoiceUseCase().execute(
        GenerateClinicMonthlyInvoiceInput(
            company_id=world["transport"].id,
            clinic_company_id=world["clinic"].id,
            period_year=PERIOD_YEAR,
            period_month=PERIOD_MONTH,
        ),
        now=EMIT_AT,
    )
    assert result.success is True, result.error
    assert result.invoice_id is not None
    invoice = db.session.get(Invoice, result.invoice_id)
    assert invoice is not None
    db.session.refresh(invoice)
    return invoice


def invoice_line_ids_and_total(invoice_id: int) -> tuple[set[int], Decimal, list[str]]:
    ids: set[int] = set()
    total = Decimal("0.00")
    descriptions: list[str] = []
    for line in InvoiceLine.query.filter_by(invoice_id=invoice_id).all():
        total += money(line.line_total)
        descriptions.append(str(line.description or ""))
        meta = line.line_meta if isinstance(line.line_meta, dict) else {}
        claimed = meta.get("booking_ids") or []
        if claimed:
            ids.update(int(i) for i in claimed)
        elif line.reservation_id is not None:
            ids.add(int(line.reservation_id))
    return ids, money(total), descriptions


def read_pdf_text(invoice: Invoice) -> str:
    pdf_state = get_pdf_state(invoice)
    assert pdf_state.status == "ready", pdf_state
    pdf_bytes = _read_generated_pdf(invoice)
    texts = [_extract_pdf_text(pdf_bytes)]
    try:
        from io import BytesIO

        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams

        mined = extract_text(BytesIO(pdf_bytes), laparams=LAParams()) or ""
        texts.append(mined)
    except Exception:
        pass
    text = "\n".join(t for t in texts if t)
    assert QR_UNAVAILABLE not in text
    return text


def qr_amount(invoice: Invoice) -> Decimal:
    return money(_qr_svg_amount(invoice))


def expected_cancellation_label(booking) -> str:
    return canonical_cancellation_label(
        reason_code=booking.cancellation_reason_code,
        reason_text=booking.cancellation_reason_text,
        persisted_label=booking.cancellation_display_label,
    )


def billable_source(booking) -> str:
    return calculate_billable_booking_amount(booking).source
