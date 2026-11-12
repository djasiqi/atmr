"""G4 — émission institution (PDF / QR). Réutilise le monde G1, n'étend pas le métier."""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from application.invoices.booking_dispute import g1_financials as g1
from application.invoices.booking_dispute.machine import snapshot
from application.invoices.booking_dispute.service import (
    add_carrier_evidence,
    carrier_respond,
    decide_dispute,
    ensure_open_dispute,
    submit_dispute_for_validation,
)
from application.invoices.generate_clinic_monthly_invoice import (
    GenerateClinicMonthlyInvoiceInput,
    GenerateClinicMonthlyInvoiceUseCase,
)
from application.invoices.institution_invoice_eligibility import (
    is_institution_invoice_eligible,
)
from application.invoices.institution_invoice_plan import build_institution_invoice_plan
from application.invoices.invoice_pdf_state import get_pdf_state
from application.invoices.period_invoice_preview import build_period_invoice_preview
from ext import db
from models import Invoice, InvoiceLine
from models.booking_dispute import BookingDispute
from models.invoice import CompanyBillingSettings
from services.documents.pdf import PDFService
from services.documents.qrbill import QRBillService, resolve_qr_bill_amount_decimal
from tests.application.helpers.g1_clinic360_world import (
    PERIOD_MONTH,
    PERIOD_YEAR,
    build_g1_clinic360_world,
)

ZURICH = ZoneInfo("Europe/Zurich")
EMIT_AT = datetime(2026, 9, 1, 0, 0, tzinfo=ZURICH)
QR_IBAN = "CH6509000000152631289"
MARIE_LABEL = "DUPONT"
QR_UNAVAILABLE = "QR-Bill non disponible"


def build_g4_world(db_session) -> dict[str, Any]:
    """Monde G1 + IBAN / domiciliation pour que le QR de l'image réelle existe."""
    world = build_g1_clinic360_world(db_session)
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


def latest_dispute(booking) -> BookingDispute:
    row = (
        BookingDispute.query.filter_by(booking_id=int(booking.id))
        .order_by(BookingDispute.id.desc())
        .first()
    )
    assert row is not None
    return row


def resolve_marie_carrier(marie) -> None:
    ensure_open_dispute(marie, actor_role="institution")
    carrier_respond(
        marie, stance="mission_done", actor_user_id=None, actor_role="COMPANY"
    )
    add_carrier_evidence(
        marie,
        kind="signed_transport_sheet",
        note="bon",
        actor_user_id=None,
        actor_role="COMPANY",
    )
    submit_dispute_for_validation(marie, actor_user_id=None, actor_role="COMPANY")
    result = decide_dispute(
        marie,
        decision="accept_carrier",
        note="ok",
        actor_user_id=None,
        actor_role="institution_admin",
    )
    assert result.ok is True, result.error


def resolve_marie_institution(marie) -> None:
    ensure_open_dispute(marie, actor_role="institution")
    result = carrier_respond(
        marie,
        stance="institution_right",
        exclusion_reason="created_by_error",
        actor_user_id=None,
        actor_role="COMPANY",
    )
    assert result.ok is True, result.error


def _plan(world):
    return build_institution_invoice_plan(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        clinic_company_id=world["clinic"].id,
        clinic_client_id=world["clinic_client"].id,
        now=EMIT_AT,
    )


def _preview(world):
    return build_period_invoice_preview(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        clinic_company_id=world["clinic"].id,
        include_line_details=True,
        now=EMIT_AT,
    )


def _booking_ids_from_preview(preview) -> set[int]:
    ids: set[int] = set()
    for line in preview.preview_lines:
        ids.add(int(line.booking_id))
        partner = getattr(line, "round_trip_partner_booking_id", None)
        if partner is not None:
            ids.add(int(partner))
    return ids


def _booking_ids_from_invoice(invoice_id: int) -> tuple[set[int], float]:
    ids: set[int] = set()
    total = 0.0
    for line in InvoiceLine.query.filter_by(invoice_id=invoice_id).all():
        total = round(total + float(line.line_total), 2)
        meta = line.line_meta if isinstance(line.line_meta, dict) else {}
        claimed = meta.get("booking_ids") or []
        if claimed:
            ids.update(int(i) for i in claimed)
        elif line.reservation_id is not None:
            ids.add(int(line.reservation_id))
    return ids, total


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_generated_pdf(invoice: Invoice) -> bytes:
    url = invoice.pdf_url or ""
    filename = Path(url.split("?", 1)[0]).name
    assert filename.endswith(".pdf"), f"pdf_url invalide: {url}"
    path = PDFService().invoices_dir / filename
    assert path.is_file(), f"PDF émis introuvable: {path}"
    return path.read_bytes()


def _qr_svg_amount(invoice: Invoice) -> Decimal:
    svg = QRBillService().generate_qr_bill_svg(invoice)
    assert svg, "QR SVG absent — l'émission n'a pas produit de QR-facture"
    text = svg.decode("utf-8") if isinstance(svg, bytes) else str(svg)
    assert QR_UNAVAILABLE not in text
    matches = re.findall(r"(\d+\.\d{2})", text)
    assert matches, "Aucun montant xx.xx dans le SVG QR"
    canonical = resolve_qr_bill_amount_decimal(invoice)
    formatted = f"{canonical.quantize(Decimal('0.01')):.2f}"
    assert formatted in matches, f"QR SVG sans {formatted} (trouvé {matches})"
    return canonical


def assert_emission_chain(
    world: dict[str, Any],
    *,
    expected_status: str,
    expected_total: float,
    marie_in: bool,
) -> Invoice:
    """state → eligibility → plan → preview → facture → PDF → QR."""
    db.session.flush()
    marie = world["marie"]
    db.session.refresh(marie)
    dispute = latest_dispute(marie)
    state = snapshot(marie, dispute)
    assert state["status"] == expected_status
    assert state["terminal"] is True
    assert state["clinic_line_in_invoice"] is marie_in

    line = g1.line_financials(marie, dispute)
    assert line["is_billable_to_institution"] is marie_in
    assert is_institution_invoice_eligible(marie, now=EMIT_AT) is marie_in

    surface = g1.institution_surface(world["all_clinic"])
    plan = _plan(world)
    preview = _preview(world)
    clinic_ht = float(plan.clinic.estimated_total) if plan.clinic else 0.0
    preview_ht = float(preview.estimated_total)
    preview_ids = _booking_ids_from_preview(preview)
    rec = plan.reconciliation or {}
    plan_ids = {
        int(i)
        for i in (rec.get("buckets") or {})
        .get("clinic_billable", {})
        .get("booking_ids")
        or []
    }

    assert surface["institution_total"] == expected_total
    assert clinic_ht == expected_total
    assert preview_ht == expected_total
    assert clinic_ht == preview_ht
    assert plan_ids == preview_ids
    if marie_in:
        assert int(marie.id) in surface["eligible_lines"]
        assert int(marie.id) in plan_ids
    else:
        assert int(marie.id) in surface["excluded_lines"]
        assert int(marie.id) not in plan_ids

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
    db.session.refresh(marie)

    issued_ids, issued_ht = _booking_ids_from_invoice(int(invoice.id))
    assert issued_ids == plan_ids
    assert issued_ids == preview_ids
    assert issued_ht == expected_total
    assert float(invoice.total_amount) == expected_total
    assert len(issued_ids) == len(set(issued_ids))
    assert int(marie.id) in issued_ids if marie_in else int(marie.id) not in issued_ids
    if marie_in:
        assert marie.invoice_line_id is not None
    else:
        assert marie.invoice_line_id is None

    pdf_state = get_pdf_state(invoice)
    assert pdf_state.status == "ready", pdf_state
    pdf_bytes = _read_generated_pdf(invoice)
    pdf_text = _extract_pdf_text(pdf_bytes)
    assert QR_UNAVAILABLE not in pdf_text
    dupont_hits = len(re.findall(re.escape(MARIE_LABEL), pdf_text, flags=re.IGNORECASE))
    if marie_in:
        assert dupont_hits == 1, (
            f"Marie doit apparaître une seule fois, trouvé {dupont_hits}"
        )
    else:
        assert dupont_hits == 0, "Marie exclue a réapparu sur le PDF institution"

    qr_amount = _qr_svg_amount(invoice)
    assert float(qr_amount) == expected_total
    assert float(resolve_qr_bill_amount_decimal(invoice)) == expected_total
    return invoice
