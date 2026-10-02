"""Tableau de prestations partagé (institution et partenaire).

Le style reprend la grille du PDF institutionnel : mêmes libellés, mêmes polices,
répétition de l'en-tête en pagination. Le partenaire n'ajoute qu'une colonne.
"""

from __future__ import annotations

from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Table, TableStyle

from services.documents.invoice_pdf_columns import (
    LINE_TIME_PICKUP,
    detail_headers,
    normalize_line_time_mode,
)


def build_services_table(
    rows: list[dict[str, str]],
    *,
    line_time_mode: str,
    available_width_pt: float,
    show_date: bool = True,
) -> Any:
    """Construit le tableau Date | [Prise en charge] | Description | Montant."""
    from services.documents.pdf import (
        FONT_SECONDARY,
        FONT_TABLE_HEADER,
        _ensure_dejavu_pdf_fonts,
    )

    font_name, font_name_bold = _ensure_dejavu_pdf_fonts()
    mode = normalize_line_time_mode(line_time_mode)
    headers = detail_headers(show_date=show_date, line_time_mode=mode)
    show_pickup = mode == LINE_TIME_PICKUP and "Prise en charge" in headers
    lead = round(FONT_TABLE_HEADER * 1.3)
    header_style = ParagraphStyle(
        "InvoiceServicesThead",
        fontName=font_name_bold,
        fontSize=FONT_TABLE_HEADER,
        leading=lead,
        textColor=colors.black,
        spaceBefore=0,
        spaceAfter=0,
    )
    body_style = ParagraphStyle(
        "InvoiceServicesCell",
        fontName=font_name,
        fontSize=FONT_SECONDARY,
        leading=round(FONT_SECONDARY * 1.3),
        textColor=colors.black,
        spaceBefore=0,
        spaceAfter=0,
    )

    def _cell(text: str, *, align: int) -> Paragraph:
        style = ParagraphStyle(
            "InvoiceServicesCellAlign", parent=body_style, alignment=align
        )
        safe = (
            str(text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return Paragraph(safe or " ", style)

    header_row = []
    for label in headers:
        align = TA_RIGHT if label == "Montant" else TA_LEFT
        style = ParagraphStyle(
            f"Th{label}",
            parent=header_style,
            alignment=align,
        )
        shown = "<nobr>Montant</nobr>" if label == "Montant" else label
        header_row.append(Paragraph(shown, style))

    table_data: list[list[Any]] = [header_row]
    for row in rows:
        cells: list[Any] = []
        if show_date:
            cells.append(_cell(row.get("date") or "", align=TA_LEFT))
        if show_pickup:
            cells.append(_cell(row.get("pickup") or "—", align=TA_LEFT))
        cells.append(_cell(row.get("description") or "", align=TA_LEFT))
        cells.append(_cell(row.get("amount") or "", align=TA_RIGHT))
        table_data.append(cells)

    date_w = 2.6 * cm
    pickup_w = 3.15 * cm
    amount_w = 2.75 * cm
    fixed = amount_w
    if show_date:
        fixed += date_w
    if show_pickup:
        fixed += pickup_w
    desc_w = max(float(available_width_pt) - fixed, 3 * cm)
    widths: list[float] = []
    if show_date:
        widths.append(date_w)
    if show_pickup:
        widths.append(pickup_w)
    widths.append(desc_w)
    widths.append(amount_w)
    scale = float(available_width_pt) / sum(widths) if sum(widths) else 1
    widths = [width * scale for width in widths]

    table = Table(table_data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), font_name_bold),
                ("FONTSIZE", (0, 0), (-1, 0), FONT_TABLE_HEADER),
                ("ALIGN", (0, 0), (-2, 0), "LEFT"),
                ("ALIGN", (-1, 0), (-1, -1), "RIGHT"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                ("FONTNAME", (0, 1), (-1, -1), font_name),
                ("FONTSIZE", (0, 1), (-1, -1), FONT_SECONDARY),
                ("ALIGN", (0, 1), (-2, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                ("TOPPADDING", (0, 1), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.lightgrey),
            ]
        )
    )
    return table
