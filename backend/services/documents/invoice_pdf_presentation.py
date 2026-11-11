"""Mise en page unique des factures LIRIE (client, institution, partenaire).

Le partenaire n'a pas de gabarit propre : même en-tête métier, même tableau
Date | Description | Montant, mêmes totaux, même message QR. L'heure de prise
en charge est une ligne de la prestation, pas une colonne.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

QR_BILL_UNAVAILABLE_LABEL = "QR-Bill non disponible"
# Même libellé que la facture institution (aperçu HTML et PDF).
GRAND_TOTAL_LABEL = "TOTAL À FACTURER :"


def client_meta_line_html(name: str) -> str:
    """Ligne « Client : NOM Prénom », même typographie que la facture institution."""
    from services.documents.pdf import FONT_SECONDARY, _xml_escape_for_paragraph

    safe = _xml_escape_for_paragraph(str(name or "").strip())
    return f'<font size="{int(FONT_SECONDARY)}" color="#475569">Client : {safe}</font>'


def pickup_meta_line_html(label: str) -> str:
    """Ligne « Prise en charge : … ». ``label`` est déjà le libellé métier (ou « — »)."""
    from services.documents.pdf import FONT_SECONDARY, _xml_escape_for_paragraph

    safe = _xml_escape_for_paragraph(str(label or "—").strip() or "—")
    return (
        f'<font size="{int(FONT_SECONDARY)}" color="#475569">'
        f"Prise en charge : {safe}</font>"
    )


def issuer_block_html(name: str, address_html: str, vat_status_text: str = "") -> str:
    """Bloc émetteur : même tailles que la facture client."""
    from services.documents.pdf import (
        COLOR_MUTED_PDF,
        FONT_BODY,
        FONT_HEADER_COMPANY,
        _xml_escape_for_paragraph,
    )

    vat_line = (
        f'<br/><font size="{FONT_BODY}" color="{COLOR_MUTED_PDF}">'
        f"{_xml_escape_for_paragraph(vat_status_text)}</font>"
        if vat_status_text
        else ""
    )
    return (
        f'<font size="{FONT_HEADER_COMPANY}"><b>'
        f"{_xml_escape_for_paragraph(name)}</b></font><br/>"
        f'<font size="{FONT_BODY}">{address_html}</font>'
        f"{vat_line}"
    )


def invoice_meta_html(
    *,
    invoice_number: str,
    issued_on: str,
    due_on: str,
    period_label: str,
    corrects_invoice_number: str | None = None,
    credit_number: str | None = None,
) -> str:
    """Métadonnées facture (numéro, émission, échéance, période, correction)."""
    from services.documents.pdf import (
        FONT_META_DATES,
        FONT_META_NUMBER,
        _xml_escape_for_paragraph,
    )

    number = _xml_escape_for_paragraph(str(invoice_number or ""))
    issued = _xml_escape_for_paragraph(issued_on)
    due = _xml_escape_for_paragraph(due_on)
    period = _xml_escape_for_paragraph(period_label)
    html = (
        f'<font size="{FONT_META_NUMBER}"><b>Numéro de facture :</b></font> '
        f'<font size="{FONT_META_NUMBER}">{number}</font><br/>'
        f'<font size="{FONT_META_DATES}"><b>Date d\'émission :</b></font> '
        f'<font size="{FONT_META_DATES}">{issued}</font><br/>'
        f'<font size="{FONT_META_DATES}"><b>Date d\'échéance :</b></font> '
        f'<font size="{FONT_META_DATES}">{due}</font><br/>'
        f'<font size="{FONT_META_DATES}"><b>Période de facturation :</b></font> '
        f'<font size="{FONT_META_DATES}">{period}</font>'
    )
    original = str(corrects_invoice_number or "").strip()
    credit = str(credit_number or "").strip()
    if original or credit:
        parts: list[str] = []
        if original:
            parts.append(f"Rectifie la facture {original}")
        if credit:
            parts.append(f"Avoir {credit}")
        correction_label = _xml_escape_for_paragraph(" · ".join(parts))
        html += (
            f'<br/><font size="{FONT_META_DATES}"><b>Correction :</b></font> '
            f'<font size="{FONT_META_DATES}">{correction_label}</font>'
        )
    return html


def prestation_header_row(*, show_date: bool, font_name_bold: str) -> list[Any]:
    """En-tête Date | Description | Montant. Jamais de colonne horaire."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph

    from services.documents.invoice_pdf_columns import (
        HEADER_AMOUNT,
        LINE_TIME_NONE,
        detail_headers,
    )
    from services.documents.pdf import FONT_TABLE_HEADER

    lead = round(FONT_TABLE_HEADER * 1.3)
    base = ParagraphStyle(
        "InvoiceCompactThead",
        fontName=font_name_bold,
        fontSize=FONT_TABLE_HEADER,
        leading=lead,
        textColor=colors.black,
        spaceBefore=0,
        spaceAfter=0,
    )
    row = []
    for label in detail_headers(show_date=show_date, line_time_mode=LINE_TIME_NONE):
        align = TA_RIGHT if label == HEADER_AMOUNT else TA_LEFT
        shown = f"<nobr>{label}</nobr>" if label == HEADER_AMOUNT else label
        row.append(
            Paragraph(shown, ParagraphStyle(f"Th{label}", parent=base, alignment=align))
        )
    return row


def finalize_prestation_table(
    table_data: list[list[Any]],
    *,
    available_width_pt: float | None,
    show_date_column: bool,
    font_name: str,
    font_name_bold: str,
    patient_separator_after_rows: list[int] | None = None,
) -> Any:
    """Applique largeurs et style du tableau de prestations institutionnel."""
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle

    from services.documents.pdf import FONT_BODY, FONT_TABLE_HEADER, LEVEL_THRESHOLD

    date_w = 2.95 * cm
    amount_w = 2.75 * cm
    if available_width_pt is not None and available_width_pt > 0:
        if show_date_column:
            desc_w = max(available_width_pt - date_w - amount_w, 1 * cm)
            col_widths = [date_w, desc_w, amount_w]
        else:
            desc_w = max(available_width_pt - amount_w, 1 * cm)
            col_widths = [desc_w, amount_w]
    elif show_date_column:
        col_widths = [date_w, 12 * cm, amount_w]
    else:
        col_widths = [13 * cm, amount_w]

    table = Table(table_data, colWidths=col_widths, repeatRows=1, splitInRow=0)
    header_bg = colors.HexColor("#f8fafc")
    row_sep = colors.HexColor("#f1f5f9")
    pad_tb_h = 6.0
    pad_tb_body = 6.0
    pad_lr = 7.5
    pad_date_r = 1.0
    pad_desc_l = 3.0
    if show_date_column:
        style_rules: list[tuple[Any, ...]] = [
            ("BACKGROUND", (0, 0), (-1, 0), header_bg),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("ALIGN", (2, 0), (2, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, 0), pad_tb_h),
            ("BOTTOMPADDING", (0, 0), (-1, 0), pad_tb_h),
            ("LEFTPADDING", (0, 0), (0, 0), pad_lr),
            ("RIGHTPADDING", (0, 0), (0, 0), pad_date_r),
            ("LEFTPADDING", (1, 0), (1, 0), pad_desc_l),
            ("RIGHTPADDING", (1, 0), (1, 0), pad_lr),
            ("LEFTPADDING", (2, 0), (2, 0), pad_lr),
            ("RIGHTPADDING", (2, 0), (2, 0), pad_lr),
            ("LINEBELOW", (0, 0), (-1, 0), 0.75, row_sep),
            ("FONTNAME", (0, 1), (-1, -1), font_name),
            ("FONTSIZE", (0, 1), (-1, -1), FONT_BODY),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), font_name_bold),
            ("FONTSIZE", (0, 0), (-1, 0), FONT_TABLE_HEADER),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("TOPPADDING", (0, 1), (-1, -1), pad_tb_body),
            ("BOTTOMPADDING", (0, 1), (-1, -1), pad_tb_body),
            ("LEFTPADDING", (0, 1), (0, -1), pad_lr),
            ("RIGHTPADDING", (0, 1), (0, -1), pad_date_r),
            ("LEFTPADDING", (1, 1), (1, -1), pad_desc_l),
            ("RIGHTPADDING", (1, 1), (1, -1), pad_lr),
            ("LEFTPADDING", (2, 1), (2, -1), pad_lr),
            ("RIGHTPADDING", (2, 1), (2, -1), pad_lr),
        ]
    else:
        style_rules = [
            ("BACKGROUND", (0, 0), (-1, 0), header_bg),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, 0), pad_tb_h),
            ("BOTTOMPADDING", (0, 0), (-1, 0), pad_tb_h),
            ("LEFTPADDING", (0, 0), (0, 0), pad_desc_l),
            ("RIGHTPADDING", (0, 0), (0, 0), pad_lr),
            ("LEFTPADDING", (1, 0), (1, 0), pad_lr),
            ("RIGHTPADDING", (1, 0), (1, 0), pad_lr),
            ("LINEBELOW", (0, 0), (-1, 0), 0.75, row_sep),
            ("FONTNAME", (0, 1), (-1, -1), font_name),
            ("FONTSIZE", (0, 1), (-1, -1), FONT_BODY),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), font_name_bold),
            ("FONTSIZE", (0, 0), (-1, 0), FONT_TABLE_HEADER),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("TOPPADDING", (0, 1), (-1, -1), pad_tb_body),
            ("BOTTOMPADDING", (0, 1), (-1, -1), pad_tb_body),
            ("LEFTPADDING", (0, 1), (0, -1), pad_desc_l),
            ("RIGHTPADDING", (0, 1), (0, -1), pad_lr),
            ("LEFTPADDING", (1, 1), (1, -1), pad_lr),
            ("RIGHTPADDING", (1, 1), (1, -1), pad_lr),
        ]
    n_rows = len(table_data)
    if n_rows > LEVEL_THRESHOLD:
        for row_index in range(1, n_rows - 1):
            style_rules.append(
                ("LINEBELOW", (0, row_index), (-1, row_index), 0.35, row_sep)
            )
    for row_index in patient_separator_after_rows or []:
        style_rules.append(
            ("LINEBELOW", (0, row_index), (-1, row_index), 0.15, colors.lightgrey)
        )
    table.setStyle(TableStyle(style_rules))
    return table


def prestation_description_html(
    *,
    client_name: str,
    departure: str,
    arrival: str,
    pickup_label: str | None,
    show_pickup: bool,
    font_name: str,
    desc_inner_pt: float,
) -> str:
    """Client, prise en charge facultative, trajet. Même wrap que le PDF institution."""
    from application.invoices.invoice_line_description import (
        format_patient_display_name_nom_prenom,
    )
    from services.documents.pdf import _pdf_format_transport_detail_inner_wrapped

    display_name = format_patient_display_name_nom_prenom(client_name) or (
        client_name or "Client"
    )
    parts = [client_meta_line_html(display_name)]
    if show_pickup:
        parts.append(pickup_meta_line_html(pickup_label or "—"))
    route = " → ".join(part for part in (departure.strip(), arrival.strip()) if part)
    if route:
        parts.append(
            _pdf_format_transport_detail_inner_wrapped(
                route,
                font_name=font_name,
                desc_inner_pt=desc_inner_pt,
                is_ride_line=True,
                is_material_delivery=False,
            )
        )
    return "<br/>".join(part for part in parts if part)


def build_prestation_detail_table(
    rows: list[dict[str, Any]],
    *,
    show_pickup: bool,
    available_width_pt: float,
    show_date: bool = True,
) -> Any:
    """Tableau de prestations partagé. ``show_pickup`` ajoute une ligne, pas une colonne."""
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph

    from services.documents.pdf import (
        FONT_BODY,
        _compact_private_date_paragraph,
        _ensure_dejavu_pdf_fonts,
        _pdf_s2_amount_only_paragraph,
        _pdf_s2_per_line_discount_suffix_html,
    )

    font_name, font_name_bold = _ensure_dejavu_pdf_fonts()
    lead = round(FONT_BODY * 1.3)
    body_style = ParagraphStyle(
        "S2Main",
        fontName=font_name,
        fontSize=FONT_BODY,
        leading=lead,
        spaceBefore=0,
        spaceAfter=0,
    )
    date_w = 2.95 * cm
    amount_w = 2.75 * cm
    desc_w = max(float(available_width_pt) - date_w - amount_w, 1 * cm)
    desc_inner_pt = max(desc_w - (7.5 + 3.0), 60.0)
    table_data: list[list[Any]] = [
        prestation_header_row(show_date=show_date, font_name_bold=font_name_bold)
    ]
    for row in rows:
        html = prestation_description_html(
            client_name=str(
                row.get("client_name") or row.get("description") or "Client"
            ),
            departure=str(row.get("departure") or ""),
            arrival=str(row.get("arrival") or ""),
            pickup_label=None if row.get("pickup") is None else str(row.get("pickup")),
            show_pickup=show_pickup,
            font_name=font_name,
            desc_inner_pt=desc_inner_pt,
        )
        amount_raw = str(row.get("amount") or 0).replace(" CHF", "").strip() or "0"
        catalog_raw = row.get("catalog_amount")
        if catalog_raw not in (None, ""):
            catalog = Decimal(
                str(catalog_raw).replace(" CHF", "").replace(",", ".").strip()
            )
            net = Decimal(amount_raw.replace(",", "."))
            if abs(catalog - net) > Decimal("0.02") and net < catalog:
                html += _pdf_s2_per_line_discount_suffix_html(
                    catalog, net, compact_private_sub=True
                )
        amount_cell = _pdf_s2_amount_only_paragraph(
            Decimal(amount_raw),
            body_style,
            is_round_trip=False,
            ht_column_plain=False,
        )
        desc_cell = Paragraph(html, body_style)
        if show_date:
            table_data.append(
                [
                    _compact_private_date_paragraph(
                        str(row.get("date") or ""), font_name
                    ),
                    desc_cell,
                    amount_cell,
                ]
            )
        else:
            table_data.append([desc_cell, amount_cell])
    return finalize_prestation_table(
        table_data,
        available_width_pt=available_width_pt,
        show_date_column=show_date,
        font_name=font_name,
        font_name_bold=font_name_bold,
    )
