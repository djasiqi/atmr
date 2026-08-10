"""Bordereau de remise nominatif (sans SHA du ZIP ni SHA du bordereau)."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from services.platform_billing.partner_agreement_versions import (
    MANIFEST_DOCUMENT_VERSION,
)

LIRIE_GREEN = colors.HexColor("#00796B")


def _esc(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_delivery_manifest_pdf_bytes(
    *,
    reference: str,
    partner_name: str,
    finalized_at_fr: str,
    particular_version: str,
    particular_sha256: str,
    general_terms_version: str,
    general_terms_sha256: str,
    dpa_version: str,
    dpa_sha256: str,
    retention_policy_version: str,
    subprocessors_version: str,
    delivery_declaration: dict[str, Any] | None = None,
) -> bytes:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ManTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=LIRIE_GREEN,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    body = ParagraphStyle(
        "ManBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    small = ParagraphStyle(
        "ManSmall",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        spaceAfter=2,
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Bordereau de remise LIRIE",
        author="LIRIE",
        creator="LIRIE-delivery-manifest",
    )
    story: list = [
        Paragraph("Bordereau de remise — dossier partenaire LIRIE", title),
        Paragraph(f"Version documentaire : {_esc(MANIFEST_DOCUMENT_VERSION)}", body),
        Paragraph(f"Partenaire : <b>{_esc(partner_name)}</b>", body),
        Paragraph(f"Référence : <b>{_esc(reference)}</b>", body),
        Paragraph(
            f"Date de finalisation du dossier / remise déclarée par LIRIE : "
            f"<b>{_esc(finalized_at_fr)}</b>",
            body,
        ),
        Spacer(1, 6),
        Paragraph("Documents juridiques remis", body),
    ]

    rows = [
        [
            Paragraph("<b>Document</b>", small),
            Paragraph("<b>Version</b>", small),
            Paragraph("<b>SHA-256</b>", small),
        ],
        [
            Paragraph("01 — Contrat particulier", small),
            Paragraph(_esc(particular_version), small),
            Paragraph(_esc(particular_sha256), small),
        ],
        [
            Paragraph("02 — Conditions générales", small),
            Paragraph(_esc(general_terms_version), small),
            Paragraph(_esc(general_terms_sha256), small),
        ],
        [
            Paragraph("03 — Accord de traitement des données", small),
            Paragraph(_esc(dpa_version), small),
            Paragraph(_esc(dpa_sha256), small),
        ],
    ]
    tbl = Table(rows, colWidths=[5.5 * cm, 5 * cm, 6.5 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.93, 0.96, 0.95)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            f"Versions internes du DPA : politique de conservation "
            f"<b>{_esc(retention_policy_version)}</b> ; liste des prestataires "
            f"<b>{_esc(subprocessors_version)}</b>.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Le présent bordereau ne contient ni son propre SHA-256 ni le SHA-256 "
            "du fichier ZIP qui l'inclut. Ces empreintes sont conservées dans le "
            "système LIRIE (journal d'audit et snapshot de génération).",
            small,
        )
    )
    decl = delivery_declaration or {}
    if decl:
        channel = decl.get("channel") or "—"
        recipient = decl.get("recipient") or "—"
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                f"Déclaration de remise LIRIE — canal : {_esc(str(channel))} ; "
                f"destinataire déclaré : {_esc(str(recipient))}.",
                body,
            )
        )
    doc.build(story)
    return buffer.getvalue()
