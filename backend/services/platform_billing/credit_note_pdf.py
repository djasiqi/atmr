"""PDF note de crédit plateforme (sans QR-Bill)."""

from __future__ import annotations

from decimal import Decimal
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from models.platform_billing import PlatformIssuedInvoice
from services.platform_billing.invoice_pdf import publish_platform_invoice_pdf


def build_and_store_credit_note_pdf(
    credit: PlatformIssuedInvoice,
    source: PlatformIssuedInvoice,
) -> tuple[str, str]:
    """Génère et publie le PDF d'avoir ; met à jour storage_key/checksum."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "CnTitle",
        parent=styles["Normal"],
        fontSize=14,
        fontName="Helvetica-Bold",
        spaceAfter=10,
    )
    normal = ParagraphStyle("CnNormal", parent=styles["Normal"], fontSize=9, leading=12)
    debtor = credit.debtor_snapshot or {}
    creditor = credit.creditor_snapshot or {}
    story = [
        Paragraph("Note de crédit LIRIE", title),
        Paragraph(f"N° <b>{credit.invoice_number}</b>", normal),
        Paragraph(f"Facture d'origine : <b>{source.invoice_number}</b>", normal),
        Spacer(1, 8),
        Paragraph(
            f"Émetteur : {creditor.get('legal_name') or 'LIRIE'}",
            normal,
        ),
        Paragraph(
            f"Destinataire : {debtor.get('legal_name') or ''}",
            normal,
        ),
        Spacer(1, 12),
        Paragraph(f"Motif : {credit.credit_reason or '—'}", normal),
        Spacer(1, 12),
    ]
    amount = abs(Decimal(str(credit.total_amount or 0)))
    data = [
        ["Libellé", "Montant"],
        [
            f"Avoir total sur {source.invoice_number}",
            f"-{amount:.2f} {credit.currency}",
        ],
    ]
    table = Table(data, colWidths=[120 * mm, 40 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    path, checksum = publish_platform_invoice_pdf(
        credit.invoice_number, pdf_bytes, previous_path=credit.pdf_storage_key
    )
    credit.pdf_storage_key = path
    credit.pdf_checksum = checksum
    return path, checksum
