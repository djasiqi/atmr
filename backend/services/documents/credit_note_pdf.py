"""PDF d'avoir, distinct du PDF de la facture d'origine.

Ce document ne contient pas de bulletin QR : il ne demande aucun règlement.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from services.documents.pdf import _ensure_dejavu_pdf_fonts
from shared.upload_path_resolver import confine_upload_destination, get_uploads_base


@dataclass(frozen=True, slots=True)
class CreditNoteDocument:
    """Contenu figé du PDF d'avoir."""

    credit_number: str
    issued_on: str
    amount_chf: str
    reason: str
    invoice_number: str
    invoice_issued_on: str
    company_name: str
    company_ide: str
    recipient_name: str
    recipient_address: str
    lines: tuple[tuple[str, str], ...]
    original_total: str
    credit_total: str
    net_invoiced: str
    amount_paid: str
    balance_due: str
    overpayment: str


def render_credit_note_pdf(document: CreditNoteDocument) -> bytes:
    """Rend l'avoir en PDF. Aucun QR de créance n'est ajouté."""
    font_name, font_bold = _ensure_dejavu_pdf_fonts()
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "CreditTitle",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0f172a"),
    )
    body = ParagraphStyle(
        "CreditBody",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
    )
    muted = ParagraphStyle(
        "CreditMuted",
        parent=body,
        textColor=colors.HexColor("#475569"),
    )
    story: list[object] = [
        Paragraph(document.company_name, title),
        Paragraph(document.company_ide or " ", muted),
        Spacer(1, 0.4 * cm),
        Paragraph("AVOIR", title),
        Paragraph(f"N° {document.credit_number}", body),
        Paragraph(f"Date de l'avoir : {document.issued_on}", body),
        Spacer(1, 0.3 * cm),
        Paragraph("Destinataire", muted),
        Paragraph(document.recipient_name, body),
        Paragraph(document.recipient_address.replace("\n", "<br/>"), body),
        Spacer(1, 0.3 * cm),
        Paragraph(
            f"Facture d'origine : {document.invoice_number} "
            f"du {document.invoice_issued_on}",
            body,
        ),
        Paragraph(f"Motif : {document.reason}", body),
        Spacer(1, 0.4 * cm),
        Paragraph("Prestations de la facture d'origine", muted),
    ]
    table_rows: list[list[str]] = [["Prestation", "Montant CHF"]]
    table_rows.extend(
        [[label, amount] for label, amount in document.lines] or [["—", "0.00"]]
    )
    table_rows.extend(
        [
            ["Total facturé d'origine", document.original_total],
            ["Montant de cet avoir", document.amount_chf],
            ["Total des avoirs", document.credit_total],
            ["Net exigible", document.net_invoiced],
            ["Déjà payé", document.amount_paid],
            ["Reste dû", document.balance_due],
            ["Trop-perçu à traiter", document.overpayment],
        ]
    )
    table = Table(table_rows, colWidths=[12 * cm, 4 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), font_bold),
                ("FONTNAME", (0, 1), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.5 * cm))
    story.append(
        Paragraph(
            "Cet avoir réduit la créance de la facture citée. "
            "Il n'ouvre aucune nouvelle créance et ne constitue pas un paiement.",
            muted,
        )
    )
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=f"Avoir {document.credit_number}",
        author=document.company_name,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )
    doc.build(story)
    return buffer.getvalue()


def write_credit_note_pdf(document: CreditNoteDocument) -> str:
    """Écrit le PDF sous uploads et retourne son URL relative."""
    payload = render_credit_note_pdf(document)
    relative = f"credit_notes/{document.credit_number}.pdf"
    destination = confine_upload_destination(relative, uploads_base=get_uploads_base())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return f"/uploads/{relative}"
